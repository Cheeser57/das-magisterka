"""
Shared checkpointing/optimizer/scheduler/logging helpers used by every
training mode (supervised, cotraining, fixmatch).

Checkpoints are written under config.checkpoint_dir
(nn_segmentation/models/training/<run_name>/ — see DESIGN.md section 5 for
why this is a distinct location from nn_segmentation/training/, which holds
this source code).
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from nn_segmentation.config.base import BaseConfig
from nn_segmentation.data.dataset import DASSegmentationDataset


def cycle_loader(loader: DataLoader) -> Iterator:
    """
    Infinite iterator over a DataLoader, reshuffling each pass through it.
    Used by training/fixmatch.py and training/cotraining.py to draw an
    unlabeled pool batch every supervised-batch step even when the pool
    DataLoader is shorter (or longer) than the labeled one.
    """
    while True:
        yield from loader


def prefix_metric_keys(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """
    Prefix every key in a metrics dict (e.g. training/metrics.py's
    framewise_macro_f1 output) for W&B logging scope — "macro_f1" ->
    "test_macro_f1", "f1_tram" -> "test_f1_tram". Used identically by
    supervised.py/fixmatch.py/cotraining.py so W&B charts use the same
    metric names across all three training modes and can be overlaid
    directly for comparison.
    """
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def compute_class_weights(dataset: DASSegmentationDataset, n_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights from confirmed (labeled_mask=True)
    training samples. Shared by supervised.py, fixmatch.py, and
    cotraining.py — all three weight the same way (DESIGN.md section 3).
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    for site_name, labels in dataset.site_labels.items():
        mask = dataset.site_labeled_masks[site_name]
        vals, cnts = np.unique(labels[mask], return_counts=True)
        counts[vals] += cnts
    counts = np.where(counts > 0, counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


def _checkpoint_dir(run_name: str, config: BaseConfig) -> Path:
    d = Path(config.checkpoint_dir) / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(
    run_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: BaseConfig,
    is_best: bool = False,
) -> Path:
    """Writes to ``{config.checkpoint_dir}/{run_name}/checkpoint_{last,best}.pt``."""
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    d = _checkpoint_dir(run_name, config)
    last_path = d / "checkpoint_last.pt"
    torch.save(state, last_path)
    if is_best:
        best_path = d / "checkpoint_best.pt"
        torch.save(state, best_path)
        return best_path
    return last_path


def load_checkpoint(run_name: str, config: BaseConfig, which: str = "best") -> dict[str, Any]:
    path = _checkpoint_dir(run_name, config) / f"checkpoint_{which}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No {which!r} checkpoint for run {run_name!r} at {path}")
    # weights_only=False: these are checkpoints we generate ourselves (trusted
    # local files, not third-party downloads), and contain plain dicts/floats
    # alongside the tensor state_dicts.
    return torch.load(path, map_location="cpu", weights_only=False)


class EarlyStopping:
    """Stops training when a monitored metric hasn't improved for ``patience`` epochs."""

    def __init__(self, patience: int, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.n_bad = 0

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = self.best is None or (
            value > self.best if self.mode == "max" else value < self.best
        )
        if improved:
            self.best = value
            self.n_bad = 0
        else:
            self.n_bad += 1
        return self.n_bad >= self.patience


def init_wandb_run(
    run_name: str,
    config: BaseConfig,
    hyperparams: Any,
    project: str = "das-nn-segmentation",
    mode: str = "offline",
):
    """
    Thin wrapper around wandb.init(), writing the run config snapshot to
    ``{config.run_dir}/{run_name}/config.json`` alongside W&B's own local
    run files (DESIGN.md section 5). main.py already imports wandb but
    leaves it unused; this is the intended integration point.

    Parameters
    ----------
    hyperparams : dataclass or dict
        The training config for this run (e.g. a SupervisedConfig /
        FixMatchConfig / CoTrainingConfig instance) — logged to W&B and
        snapshotted to disk so a run's exact settings are always
        recoverable, independent of whether W&B sync ever happens.
    """
    from nn_segmentation.utils.logging import wandb_init

    hyperparams_dict = asdict(hyperparams) if is_dataclass(hyperparams) else dict(hyperparams)

    run_snapshot_dir = Path(config.run_dir) / run_name
    run_snapshot_dir.mkdir(parents=True, exist_ok=True)
    (run_snapshot_dir / "config.json").write_text(json.dumps(hyperparams_dict, indent=2, default=str))

    run = wandb_init(
        project=project,
        run_name=run_name,
        config=hyperparams_dict,
        run_dir=config.run_dir,
        mode=mode,
    )
    (run_snapshot_dir / "wandb_run_id.txt").write_text(str(run.id))
    return run
