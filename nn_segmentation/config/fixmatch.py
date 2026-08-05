"""
FixMatch hyperparameters.

Confidence thresholds and margin reuse the tuned values already validated
for vehicle-only pseudo-labeling in co_training.ipynb
(CT_CONF_VEH=0.60, CT_CONF_MARGIN=0.15) — see DESIGN.md section 4.1.
"""

from __future__ import annotations

from dataclasses import dataclass

from nn_segmentation.config.supervised import SupervisedConfig


@dataclass
class FixMatchConfig(SupervisedConfig):
    unsupervised_weight: float = 1.0

    # Cap on unlabeled pool size (data/dataset.py::DASPoolDataset), mirrors
    # MAX_POOL_SIZE=20_000 in co_training.ipynb.
    max_pool_size: int = 20_000

    # Per-class confidence thresholds for trusting a pseudo-label. Vehicle
    # classes reuse co_training.ipynb's CT_CONF_VEH; background is gated much
    # harder since it's trivially easy/dominant early in training (same
    # reasoning as the existing notebook's comments).
    tau_vehicle: float = 0.60
    tau_background: float = 0.95
    confidence_margin: float = 0.15  # top1 - top2, mirrors CT_CONF_MARGIN

    # Never generate unsupervised loss from predicted-background frames —
    # only confirmed labeled_mask=True background contributes to supervised
    # CE. See DESIGN.md section 4.1 for the reasoning (carried over from
    # co_training.ipynb's vehicle-only pseudo-labeling design).
    pseudo_label_background: bool = False

    # Cap on pseudo-labeled vehicle-frames contributing to the unsupervised
    # loss per batch — continuous analogue of CT_MAX_VEH_PER_ITER=30/iter.
    max_pseudo_label_fraction_per_batch: float = 0.3

    # Weak augmentation
    weak_time_shift_sec: float = 0.1
    weak_channel_shift: int = 8
    weak_noise_std_frac: float = 0.03  # fraction of per-channel std

    # Strong augmentation
    strong_channel_mask_frac: float = 0.15  # max fraction of channels masked
    strong_time_mask_frac: float = 0.15  # max fraction of time masked
    strong_amplitude_scale_range: tuple[float, float] = (0.7, 1.3)
    strong_inject_boundary_jump: bool = True
