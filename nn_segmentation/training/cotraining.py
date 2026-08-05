"""
Deep co-training via Cross-Pseudo Supervision (CPS) — the documented
alternative/ablation to FixMatch (DESIGN.md section 4.2 / 4.3), and the
deep analogue of this repo's existing classical co_training.ipynb.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from nn_segmentation.config.base import BACKGROUND_CLASS, VEHICLE_CLASSES, BaseConfig
from nn_segmentation.config.cotraining import CoTrainingConfig
from nn_segmentation.config.model_tcn import TCNConfig
from nn_segmentation.data import cache_builder
from nn_segmentation.data.augmentations import weak_augment
from nn_segmentation.data.dataset import (
    DASPoolDataset,
    DASSegmentationDataset,
    collate_pool_batch,
    collate_segmentation_batch,
)
from nn_segmentation.data.raster_labels import load_site_configs
from nn_segmentation.models.tcn import DASMultiStageTCN
from nn_segmentation.training.loop_utils import (
    EarlyStopping,
    build_optimizer,
    compute_class_weights,
    cycle_loader,
    init_wandb_run,
    prefix_metric_keys,
    save_checkpoint,
)
from nn_segmentation.training.losses import masked_weighted_ce, multistage_loss, pool_targets
from nn_segmentation.training.pseudo_labels import cap_trust_mask, make_pseudo_labels
from nn_segmentation.training.supervised import run_supervised_epoch
from nn_segmentation.utils.logging import get_logger

logger = get_logger(__name__)


class CPSPair(torch.nn.Module):
    """
    Bundles the two Cross-Pseudo-Supervision branches into a single
    nn.Module so training/loop_utils.py's build_optimizer/save_checkpoint/
    load_checkpoint work unchanged (their combined .parameters()/
    .state_dict() naturally cover both branches).
    """

    def __init__(self, branch_a: DASMultiStageTCN, branch_b: DASMultiStageTCN) -> None:
        super().__init__()
        self.branch_a = branch_a
        self.branch_b = branch_b


def _build_branch(site_channel_counts: dict[str, int], tcn_config: TCNConfig, seed: int) -> DASMultiStageTCN:
    """Construct one branch with its own weight initialization, seeded independently."""
    torch.manual_seed(seed)
    return DASMultiStageTCN(site_channel_counts, tcn_config)


def _cross_pseudo_loss(
    own_logits: torch.Tensor,
    peer_logits: torch.Tensor,
    config: CoTrainingConfig,
    vehicle_class_ids: set[int],
    background_class_id: int,
) -> torch.Tensor:
    """
    Cross-entropy supervising ``own_logits`` (with grad, the branch being
    updated) against the peer branch's confident predictions on the same
    crop. ``peer_logits`` is detached before pseudo-labeling — argmax'd
    class indices carry no gradient anyway, but detaching up front avoids
    building an unused backward graph through the peer.
    """
    peer_logits = peer_logits.detach()
    pseudo_label, trust_mask = make_pseudo_labels(
        peer_logits, config.tau_vehicle, config.tau_background, config.confidence_margin,
        vehicle_class_ids, background_class_id, config.pseudo_label_background,
    )
    confidence = torch.softmax(peer_logits, dim=1).amax(dim=1)
    trust_mask = cap_trust_mask(trust_mask, confidence, config.max_pseudo_label_fraction_per_batch)
    return masked_weighted_ce(own_logits, pseudo_label, trust_mask)


def _train_epoch_cotraining(
    cps_pair: CPSPair,
    train_loader: DataLoader,
    pool_iter: Iterator,
    n_classes: int,
    mask_class_ids: set[int],
    vehicle_class_ids: set[int],
    background_class_id: int,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    config: CoTrainingConfig,
    device: torch.device,
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
) -> tuple[float, float]:
    cps_pair.train()
    total_sup_loss = 0.0
    total_cross_loss = 0.0
    n_batches = 0
    out_hop = cps_pair.branch_a.config.out_hop

    for batch in train_loader:
        pool_batch = next(pool_iter)
        optimizer.zero_grad()
        loss = torch.zeros((), device=device)

        # ---- supervised term: both branches trained on the same ground truth ----
        for site_name, tensors in batch.items():
            x = tensors["signal"].to(device)
            label_full = tensors["label"].to(device)
            mask_full = tensors["labeled_mask"].to(device)

            target, mask = pool_targets(label_full, mask_full, out_hop, n_classes)
            for cls_id in mask_class_ids:
                mask = mask & (target != cls_id)

            outputs_a = cps_pair.branch_a(x, site_name)
            outputs_b = cps_pair.branch_b(x, site_name)
            sup_loss_a = multistage_loss(
                outputs_a, target, mask, class_weights=class_weights,
                tmse_weight=config.tmse_weight, tmse_clip=config.tmse_clip,
            )
            sup_loss_b = multistage_loss(
                outputs_b, target, mask, class_weights=class_weights,
                tmse_weight=config.tmse_weight, tmse_clip=config.tmse_clip,
            )
            loss = loss + sup_loss_a + sup_loss_b
            total_sup_loss += float(sup_loss_a.item() + sup_loss_b.item())

        # ---- cross-pseudo-supervision term: independent per-branch augmentation ----
        for site_name, signal_batch in pool_batch.items():
            signal_np = signal_batch.numpy()  # (B, C, T)
            aug_a = np.stack([weak_augment(c, config, rng_a) for c in signal_np])
            aug_b = np.stack([weak_augment(c, config, rng_b) for c in signal_np])

            logits_a = cps_pair.branch_a(torch.from_numpy(aug_a).to(device), site_name)[-1]
            logits_b = cps_pair.branch_b(torch.from_numpy(aug_b).to(device), site_name)[-1]

            cross_loss_a = _cross_pseudo_loss(logits_a, logits_b, config, vehicle_class_ids, background_class_id)
            cross_loss_b = _cross_pseudo_loss(logits_b, logits_a, config, vehicle_class_ids, background_class_id)

            loss = loss + config.cross_supervision_weight * (cross_loss_a + cross_loss_b)
            total_cross_loss += float(cross_loss_a.item() + cross_loss_b.item())

        loss.backward()
        optimizer.step()
        n_batches += 1

    return total_sup_loss / max(n_batches, 1), total_cross_loss / max(n_batches, 1)


def train_cross_pseudo_supervision(
    run_name: str,
    base_config: BaseConfig,
    cotraining_config: CoTrainingConfig,
) -> DASMultiStageTCN:
    """
    Two identically-structured DASMultiStageTCN instances
    (branch_a, branch_b), seeded differently
    (cotraining_config.branch_a_seed / branch_b_seed) and fed independently
    augmented views of the same unlabeled crop, cross-supervising each
    other every batch:

    1. Supervised batch (DASSegmentationDataset) trains both branches
       independently via training/losses.py::multistage_loss(), same as
       train_supervised().
    2. Unlabeled batch (DASPoolDataset, training-range only per
       data/splits.py): branch_a's confident predictions (same asymmetric
       thresholds as fixmatch.py: tau_vehicle, tau_background,
       confidence_margin, vehicle-only per pseudo_label_background) become
       pseudo-labels supervising branch_b's prediction on the same crop
       (under its own augmentation), and vice versa.
    3. Total loss = sum of both branches' supervised loss + cross-pseudo-
       supervision terms, weighted by cotraining_config.
       cross_supervision_weight.
    4. At inference, either branch (or their averaged softmax) can be used;
       this implementation reports/evaluates branch_a as the primary model
       (an arbitrary but documented choice — the two branches are
       symmetric peers of equal expected quality).

    This differs from the classical co_train() loop in co_training.ipynb by
    cross-supervising every batch rather than alternating discrete
    retrain-then-pseudo-label rounds, since batch-level cross-supervision is
    the standard approach for jointly-trained deep networks (DESIGN.md
    section 4.2).

    Returns
    -------
    DASMultiStageTCN
        branch_a (also checkpointed to disk as part of the CPSPair wrapper,
        so branch_b remains available in the saved state_dict).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sites = load_site_configs(base_config)
    # Loaded/normalized ONCE and shared between train_dataset, test_dataset,
    # and pool_dataset below — see training/fixmatch.py's identical
    # comment; same 3x-redundant-load risk applies here.
    preloaded_caches = cache_builder.load_and_normalize_sites(sites, base_config)

    train_dataset = DASSegmentationDataset(
        sites, base_config, cotraining_config, split="train", preloaded_caches=preloaded_caches
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "No labeled training crops found for any site. Run `build-cache` first and "
            "confirm labelStudio/output or labeling/log*.csv contain real events."
        )
    test_dataset = DASSegmentationDataset(
        sites, base_config, cotraining_config, split="test", preloaded_caches=preloaded_caches
    )
    pool_dataset = DASPoolDataset(
        sites, base_config, cotraining_config,
        max_pool_size=cotraining_config.max_pool_size, preloaded_caches=preloaded_caches,
    )
    if len(pool_dataset) == 0:
        raise RuntimeError("Unlabeled pool is empty; cannot run co-training. Check build-cache output.")

    train_loader = DataLoader(
        train_dataset, batch_size=cotraining_config.batch_size, shuffle=True, collate_fn=collate_segmentation_batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cotraining_config.batch_size, shuffle=False, collate_fn=collate_segmentation_batch
    )
    pool_loader = DataLoader(
        pool_dataset, batch_size=cotraining_config.batch_size, shuffle=True, collate_fn=collate_pool_batch
    )
    pool_iter = cycle_loader(pool_loader)

    tcn_config = TCNConfig(n_classes=len(base_config.class_names))
    site_channel_counts = {name: sig.shape[1] for name, sig in train_dataset.site_signals.items()}
    for name, sig in pool_dataset.site_signals.items():
        site_channel_counts.setdefault(name, sig.shape[1])

    # torch.manual_seed() inside _build_branch mutates global RNG state, so
    # branch_a is always built before branch_b regardless of call order.
    branch_a = _build_branch(site_channel_counts, tcn_config, cotraining_config.branch_a_seed)
    branch_b = _build_branch(site_channel_counts, tcn_config, cotraining_config.branch_b_seed)
    cps_pair = CPSPair(branch_a, branch_b).to(device)

    class_weights = compute_class_weights(train_dataset, tcn_config.n_classes).to(device)
    mask_class_ids = {i for i, name in enumerate(base_config.class_names) if name in cotraining_config.mask_classes}
    vehicle_class_ids = {i for i, name in enumerate(base_config.class_names) if name in VEHICLE_CLASSES}
    background_class_id = base_config.class_names.index(BACKGROUND_CLASS)

    optimizer = build_optimizer(cps_pair, cotraining_config.learning_rate, cotraining_config.weight_decay)
    early_stopping = EarlyStopping(patience=cotraining_config.early_stopping_patience, mode="max")
    rng_a = np.random.default_rng(cotraining_config.branch_a_seed)
    rng_b = np.random.default_rng(cotraining_config.branch_b_seed)

    wandb_run = None
    if cotraining_config.use_wandb:
        wandb_run = init_wandb_run(
            run_name, base_config, cotraining_config,
            project=cotraining_config.wandb_project, mode=cotraining_config.wandb_mode,
        )

    try:
        best_f1 = -1.0
        for epoch in range(cotraining_config.epochs):
            sup_loss, cross_loss = _train_epoch_cotraining(
                cps_pair, train_loader, pool_iter, tcn_config.n_classes, mask_class_ids,
                vehicle_class_ids, background_class_id, optimizer, class_weights,
                cotraining_config, device, rng_a, rng_b,
            )
            # branch_a is the reported/evaluated model (documented choice above).
            test_loss, test_metrics = run_supervised_epoch(
                cps_pair.branch_a, test_loader, tcn_config.n_classes, mask_class_ids,
                None, class_weights, cotraining_config, device, class_names=base_config.class_names,
            )

            macro_f1 = test_metrics.get("macro_f1", float("nan"))
            is_best = not np.isnan(macro_f1) and macro_f1 > best_f1
            if is_best:
                best_f1 = macro_f1
            save_checkpoint(run_name, cps_pair, optimizer, epoch, test_metrics, base_config, is_best=is_best)

            logger.info(
                "epoch %d: sup_loss=%.4f cross_loss=%.4f test_loss=%.4f test_macro_f1=%.4f%s",
                epoch, sup_loss, cross_loss, test_loss, macro_f1, " (best)" if is_best else "",
            )

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch, "sup_loss": sup_loss, "cross_loss": cross_loss,
                    "test_loss": test_loss, "is_best": is_best,
                    **prefix_metric_keys(test_metrics, "test_"),
                })

            if not np.isnan(macro_f1) and early_stopping.step(macro_f1):
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, early_stopping.patience)
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return cps_pair.branch_a
