"""Hyperparameters for plain supervised training of DASMultiStageTCN."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SupervisedConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    chunk_len_sec: float = 30.0  # see DESIGN.md section 2.2

    # Classes to exclude from the supervised loss (still predicted, just not
    # supervised against) — "traffic" is excluded by default, see DESIGN.md
    # section 3. Configurable per-run for ablations.
    mask_classes: tuple[str, ...] = ("traffic",)

    # T-MSE smoothing loss (MS-TCN paper): truncated MSE between adjacent
    # frame log-probabilities, discourages single-frame flicker.
    tmse_clip: float = 4.0
    tmse_weight: float = 0.15

    # Minimum fraction of labeled_mask=True timesteps (at output resolution)
    # required for a training crop to be sampled for supervised training.
    min_labeled_fraction: float = 0.5

    early_stopping_patience: int = 15

    # Weights & Biases logging (utils/logging.py, training/loop_utils.py).
    # Off by default since it requires a wandb account/network; "offline"
    # mode writes local run files under base_config.run_dir without needing
    # login, and can be synced later with `wandb sync`.
    use_wandb: bool = False
    wandb_mode: str = "offline"  # "online" | "offline"
    wandb_project: str = "das-nn-segmentation"
