"""
Plain supervised training loop for DASMultiStageTCN — the baseline every
semi-supervised variant (fixmatch.py, cotraining.py) is compared against.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from nn_segmentation.config.base import BaseConfig
from nn_segmentation.config.model_tcn import TCNConfig
from nn_segmentation.config.supervised import SupervisedConfig
from nn_segmentation.data import cache_builder
from nn_segmentation.data.dataset import DASSegmentationDataset, collate_segmentation_batch
from nn_segmentation.data.raster_labels import load_site_configs
from nn_segmentation.models.tcn import DASMultiStageTCN
from nn_segmentation.training.loop_utils import (
    EarlyStopping,
    build_optimizer,
    compute_class_weights,
    init_wandb_run,
    prefix_metric_keys,
    save_checkpoint,
)
from nn_segmentation.training.losses import multistage_loss, pool_targets
from nn_segmentation.training.metrics import framewise_macro_f1
from nn_segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def run_supervised_epoch(
    model: DASMultiStageTCN,
    loader: DataLoader,
    n_classes: int,
    mask_class_ids: set[int],
    optimizer: torch.optim.Optimizer | None,
    class_weights: torch.Tensor,
    train_config: SupervisedConfig,
    device: torch.device,
    class_names: Sequence[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Shared train/eval loop body; optimizer=None runs in eval mode without
    backprop. Public (not module-private) because training/fixmatch.py and
    training/cotraining.py reuse it verbatim for their held-out-test-range
    evaluation pass — semi-supervised training only changes the *training*
    step, not evaluation.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets, all_masks = [], [], []

    for batch in loader:
        with torch.set_grad_enabled(is_train):
            batch_loss = torch.zeros((), device=device)
            for site_name, tensors in batch.items():
                x = tensors["signal"].to(device)
                label_full = tensors["label"].to(device)
                mask_full = tensors["labeled_mask"].to(device)

                outputs = model(x, site_name)
                target, mask = pool_targets(label_full, mask_full, model.config.out_hop, n_classes)
                for cls_id in mask_class_ids:
                    mask = mask & (target != cls_id)

                batch_loss = batch_loss + multistage_loss(
                    outputs,
                    target,
                    mask,
                    class_weights=class_weights,
                    tmse_weight=train_config.tmse_weight,
                    tmse_clip=train_config.tmse_clip,
                )

                if not is_train:
                    pred = outputs[-1].argmax(dim=1)
                    all_preds.append(pred.detach().cpu().numpy().ravel())
                    all_targets.append(target.detach().cpu().numpy().ravel())
                    all_masks.append(mask.detach().cpu().numpy().ravel())

            if is_train:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        total_loss += float(batch_loss.item())
        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    if all_preds:
        metrics = framewise_macro_f1(
            np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_masks),
            n_classes, class_names=class_names,
        )
    else:
        metrics = {"macro_f1": float("nan")}
    return mean_loss, metrics


def train_supervised(
    run_name: str,
    base_config: BaseConfig,
    train_config: SupervisedConfig,
) -> DASMultiStageTCN:
    """
    Standard training loop:

    1. Build DASSegmentationDataset(split="train") / (split="test") per
       data/dataset.py, for every site in base_config.sites.
    2. Build DASMultiStageTCN (models/tcn.py), sized from each site's
       *actual* cached channel count (train_dataset.site_signals[name]
       .shape[1]) rather than SiteConfig.n_channels' formula estimate —
       distance-slice endpoints are inclusive, so the real loaded channel
       count can be one more than the formula's (dist_end-dist_start)/DX.
    3. Per epoch: forward -> training/losses.py::multistage_loss() (masked
       by labeled_mask, class-weighted by training-range inverse frequency,
       traffic excluded per train_config.mask_classes by default) ->
       backward -> optimizer step (training/loop_utils.py).
    4. Evaluate on the held-out test range each epoch
       (training/metrics.py::framewise_macro_f1); checkpoint best-so-far
       via training/loop_utils.py::save_checkpoint(). (event_level_f1 is
       left for a later pass — not required to close the training loop.)
    5. Early stop after train_config.early_stopping_patience epochs without
       improvement.

    Returns
    -------
    DASMultiStageTCN
        The trained model (also checkpointed to disk).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sites = load_site_configs(base_config)
    # Loaded/normalized ONCE and shared between train_dataset and
    # test_dataset below — each otherwise independently reloads and
    # re-normalizes the same per-site signal from disk (2x here; 3x once
    # a pool dataset is also involved in fixmatch.py/cotraining.py),
    # needlessly multiplying memory for wide/multi-site configs.
    preloaded_caches = cache_builder.load_and_normalize_sites(sites, base_config)

    train_dataset = DASSegmentationDataset(
        sites, base_config, train_config, split="train", preloaded_caches=preloaded_caches
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "No labeled training crops found for any site. Run `build-cache` first and "
            "confirm labelStudio/output or labeling/log*.csv contain real events."
        )
    test_dataset = DASSegmentationDataset(
        sites, base_config, train_config, split="test", preloaded_caches=preloaded_caches
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=collate_segmentation_batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_config.batch_size, shuffle=False, collate_fn=collate_segmentation_batch
    )

    tcn_config = TCNConfig(n_classes=len(base_config.class_names))
    site_channel_counts = {name: sig.shape[1] for name, sig in train_dataset.site_signals.items()}
    model = DASMultiStageTCN(site_channel_counts, tcn_config).to(device)

    class_weights = compute_class_weights(train_dataset, tcn_config.n_classes).to(device)
    mask_class_ids = {
        i for i, name in enumerate(base_config.class_names) if name in train_config.mask_classes
    }

    optimizer = build_optimizer(model, train_config.learning_rate, train_config.weight_decay)
    early_stopping = EarlyStopping(patience=train_config.early_stopping_patience, mode="max")

    wandb_run = None
    if train_config.use_wandb:
        wandb_run = init_wandb_run(
            run_name, base_config, train_config, project=train_config.wandb_project, mode=train_config.wandb_mode
        )

    try:
        best_f1 = -1.0
        for epoch in range(train_config.epochs):
            train_loss, _ = run_supervised_epoch(
                model, train_loader, tcn_config.n_classes, mask_class_ids, optimizer, class_weights, train_config,
                device, class_names=base_config.class_names,
            )
            test_loss, test_metrics = run_supervised_epoch(
                model, test_loader, tcn_config.n_classes, mask_class_ids, None, class_weights, train_config,
                device, class_names=base_config.class_names,
            )

            macro_f1 = test_metrics.get("macro_f1", float("nan"))
            is_best = not np.isnan(macro_f1) and macro_f1 > best_f1
            if is_best:
                best_f1 = macro_f1
            save_checkpoint(run_name, model, optimizer, epoch, test_metrics, base_config, is_best=is_best)

            logger.info(
                "epoch %d: sup_loss=%.4f test_loss=%.4f test_macro_f1=%.4f%s",
                epoch, train_loss, test_loss, macro_f1, " (best)" if is_best else "",
            )

            if wandb_run is not None:
                # "sup_loss"/"test_loss" naming is shared verbatim by
                # fixmatch.py/cotraining.py so runs across all three training
                # modes can be overlaid on the same W&B charts.
                wandb_run.log({
                    "epoch": epoch, "sup_loss": train_loss, "test_loss": test_loss, "is_best": is_best,
                    **prefix_metric_keys(test_metrics, "test_"),
                })

            if not np.isnan(macro_f1) and early_stopping.step(macro_f1):
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, early_stopping.patience)
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return model
