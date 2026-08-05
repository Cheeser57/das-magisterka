"""
FixMatch-style semi-supervised training — the primary recommended SSL
approach (DESIGN.md section 4.1 / 4.3).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from nn_segmentation.config.base import BACKGROUND_CLASS, VEHICLE_CLASSES, BaseConfig
from nn_segmentation.config.fixmatch import FixMatchConfig
from nn_segmentation.config.model_tcn import TCNConfig
from nn_segmentation.data import cache_builder
from nn_segmentation.data.augmentations import strong_augment, weak_augment
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
from nn_segmentation.training.losses import multistage_loss, pool_targets
from nn_segmentation.training.pseudo_labels import cap_trust_mask, make_pseudo_labels
from nn_segmentation.training.supervised import run_supervised_epoch
from nn_segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def _augment_pool_batch(
    pool_batch: dict[str, torch.Tensor],
    config: FixMatchConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Apply weak_augment/strong_augment per-sample to every site group in a pool batch."""
    augmented = {}
    for site_name, signal_batch in pool_batch.items():
        signal_np = signal_batch.numpy()  # (B, C, T)
        weak = np.stack([weak_augment(c, config, rng) for c in signal_np])
        strong = np.stack([strong_augment(c, config, rng) for c in signal_np])
        augmented[site_name] = (torch.from_numpy(weak).to(device), torch.from_numpy(strong).to(device))
    return augmented


def _train_epoch_fixmatch(
    model: DASMultiStageTCN,
    train_loader: DataLoader,
    pool_iter: Iterator,
    n_classes: int,
    mask_class_ids: set[int],
    vehicle_class_ids: set[int],
    background_class_id: int,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    config: FixMatchConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[float, float]:
    model.train()
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        pool_batch = next(pool_iter)
        optimizer.zero_grad()
        loss = torch.zeros((), device=device)

        # ---- supervised term (ground-truth labeled crops) ----
        for site_name, tensors in batch.items():
            x = tensors["signal"].to(device)
            label_full = tensors["label"].to(device)
            mask_full = tensors["labeled_mask"].to(device)

            outputs = model(x, site_name)
            target, mask = pool_targets(label_full, mask_full, model.config.out_hop, n_classes)
            for cls_id in mask_class_ids:
                mask = mask & (target != cls_id)

            sup_loss = multistage_loss(
                outputs, target, mask, class_weights=class_weights,
                tmse_weight=config.tmse_weight, tmse_clip=config.tmse_clip,
            )
            loss = loss + sup_loss
            total_sup_loss += float(sup_loss.item())

        # ---- unsupervised term (FixMatch weak/strong consistency) ----
        augmented = _augment_pool_batch(pool_batch, config, rng, device)
        for site_name, (weak_x, strong_x) in augmented.items():
            with torch.no_grad():
                weak_logits = model(weak_x, site_name)[-1]
            pseudo_label, trust_mask = make_pseudo_labels(
                weak_logits, config.tau_vehicle, config.tau_background, config.confidence_margin,
                vehicle_class_ids, background_class_id, config.pseudo_label_background,
            )
            confidence = torch.softmax(weak_logits, dim=1).amax(dim=1)
            trust_mask = cap_trust_mask(trust_mask, confidence, config.max_pseudo_label_fraction_per_batch)

            # Consistency is only enforced on the final (most-refined) stage's
            # prediction — passed as a single-item list to reuse
            # multistage_loss's masked-CE + T-MSE combinator without also
            # forcing every intermediate refinement stage to match a
            # possibly-noisy pseudo-label.
            strong_logits = model(strong_x, site_name)[-1]
            unsup_loss = multistage_loss(
                [strong_logits], pseudo_label, trust_mask, class_weights=None,
                tmse_weight=config.tmse_weight, tmse_clip=config.tmse_clip,
            )
            loss = loss + config.unsupervised_weight * unsup_loss
            total_unsup_loss += float(unsup_loss.item())

        loss.backward()
        optimizer.step()
        n_batches += 1

    return total_sup_loss / max(n_batches, 1), total_unsup_loss / max(n_batches, 1)


def train_fixmatch(
    run_name: str,
    base_config: BaseConfig,
    fixmatch_config: FixMatchConfig,
) -> DASMultiStageTCN:
    """
    Extends training/supervised.py::train_supervised() with a per-batch
    unsupervised consistency term:

    1. Draw a supervised batch from DASSegmentationDataset (labeled crops)
       and an unlabeled batch from DASPoolDataset (data/dataset.py), both
       restricted to each site's training range only
       (data/splits.py — DESIGN.md section 4.4).
    2. For the unlabeled batch: weak_augment() and strong_augment()
       (data/augmentations.py) to produce two views of each crop.
    3. Run the model on the weak view (no grad) to get a per-output-frame
       pseudo-label + confidence; keep only frames passing the asymmetric
       thresholds (fixmatch_config.tau_vehicle / tau_background +
       confidence_margin), and — per fixmatch_config.pseudo_label_background
       — only ever pseudo-label vehicle classes, never background
       (DESIGN.md section 4.1).
    4. Run the model on the strong view (with grad); compute
       training/losses.py::masked_weighted_ce() against the kept
       pseudo-labels, weighted by fixmatch_config.unsupervised_weight, and
       capped by max_pseudo_label_fraction_per_batch.
    5. Total loss = supervised multistage_loss + unsupervised term; same
       optimizer/checkpoint/early-stopping machinery as train_supervised().

    Returns
    -------
    DASMultiStageTCN
        The trained model (also checkpointed to disk).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sites = load_site_configs(base_config)
    # Loaded/normalized ONCE and shared between train_dataset, test_dataset,
    # and pool_dataset below — each otherwise independently reloads and
    # re-normalizes the same per-site signal from disk (3x total here),
    # needlessly tripling memory for wide/multi-site configs (observed to
    # OOM in practice on a 5-site config with a 188-channel site alone
    # needing ~11 GiB per copy).
    preloaded_caches = cache_builder.load_and_normalize_sites(sites, base_config)

    train_dataset = DASSegmentationDataset(
        sites, base_config, fixmatch_config, split="train", preloaded_caches=preloaded_caches
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "No labeled training crops found for any site. Run `build-cache` first and "
            "confirm labelStudio/output or labeling/log*.csv contain real events."
        )
    test_dataset = DASSegmentationDataset(
        sites, base_config, fixmatch_config, split="test", preloaded_caches=preloaded_caches
    )
    pool_dataset = DASPoolDataset(
        sites, base_config, fixmatch_config,
        max_pool_size=fixmatch_config.max_pool_size, preloaded_caches=preloaded_caches,
    )
    if len(pool_dataset) == 0:
        raise RuntimeError("Unlabeled pool is empty; cannot run FixMatch. Check build-cache output.")

    train_loader = DataLoader(
        train_dataset, batch_size=fixmatch_config.batch_size, shuffle=True, collate_fn=collate_segmentation_batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=fixmatch_config.batch_size, shuffle=False, collate_fn=collate_segmentation_batch
    )
    pool_loader = DataLoader(
        pool_dataset, batch_size=fixmatch_config.batch_size, shuffle=True, collate_fn=collate_pool_batch
    )
    pool_iter = cycle_loader(pool_loader)

    tcn_config = TCNConfig(n_classes=len(base_config.class_names))
    site_channel_counts = {name: sig.shape[1] for name, sig in train_dataset.site_signals.items()}
    for name, sig in pool_dataset.site_signals.items():
        site_channel_counts.setdefault(name, sig.shape[1])
    model = DASMultiStageTCN(site_channel_counts, tcn_config).to(device)

    class_weights = compute_class_weights(train_dataset, tcn_config.n_classes).to(device)
    mask_class_ids = {i for i, name in enumerate(base_config.class_names) if name in fixmatch_config.mask_classes}
    vehicle_class_ids = {i for i, name in enumerate(base_config.class_names) if name in VEHICLE_CLASSES}
    background_class_id = base_config.class_names.index(BACKGROUND_CLASS)

    optimizer = build_optimizer(model, fixmatch_config.learning_rate, fixmatch_config.weight_decay)
    early_stopping = EarlyStopping(patience=fixmatch_config.early_stopping_patience, mode="max")
    rng = np.random.default_rng(base_config.seed)

    wandb_run = None
    if fixmatch_config.use_wandb:
        wandb_run = init_wandb_run(
            run_name, base_config, fixmatch_config,
            project=fixmatch_config.wandb_project, mode=fixmatch_config.wandb_mode,
        )

    try:
        best_f1 = -1.0
        for epoch in range(fixmatch_config.epochs):
            sup_loss, unsup_loss = _train_epoch_fixmatch(
                model, train_loader, pool_iter, tcn_config.n_classes, mask_class_ids,
                vehicle_class_ids, background_class_id, optimizer, class_weights, fixmatch_config, device, rng,
            )
            test_loss, test_metrics = run_supervised_epoch(
                model, test_loader, tcn_config.n_classes, mask_class_ids, None, class_weights, fixmatch_config,
                device, class_names=base_config.class_names,
            )

            macro_f1 = test_metrics.get("macro_f1", float("nan"))
            is_best = not np.isnan(macro_f1) and macro_f1 > best_f1
            if is_best:
                best_f1 = macro_f1
            save_checkpoint(run_name, model, optimizer, epoch, test_metrics, base_config, is_best=is_best)

            logger.info(
                "epoch %d: sup_loss=%.4f unsup_loss=%.4f test_loss=%.4f test_macro_f1=%.4f%s",
                epoch, sup_loss, unsup_loss, test_loss, macro_f1, " (best)" if is_best else "",
            )

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch, "sup_loss": sup_loss, "unsup_loss": unsup_loss,
                    "test_loss": test_loss, "is_best": is_best,
                    **prefix_metric_keys(test_metrics, "test_"),
                })

            if not np.isnan(macro_f1) and early_stopping.step(macro_f1):
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, early_stopping.patience)
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return model
