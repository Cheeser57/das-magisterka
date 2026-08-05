"""
Deep co-training (Cross-Pseudo Supervision) hyperparameters.

Implements Option B from DESIGN.md section 4.2: two identically-structured
DASMultiStageTCN instances (different init + augmentation) cross-supervising
each other's confident predictions on unlabeled crops every batch, rather
than the literal two-view (spectral/shape) branch split used by the
classical co_training.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass

from nn_segmentation.config.supervised import SupervisedConfig


@dataclass
class CoTrainingConfig(SupervisedConfig):
    cross_supervision_weight: float = 1.0

    # Same asymmetric-threshold, vehicle-only pseudo-labeling design as
    # FixMatch (DESIGN.md section 4.1) applies to which cross-predictions
    # from the peer network are trusted.
    tau_vehicle: float = 0.60
    tau_background: float = 0.95
    confidence_margin: float = 0.15
    pseudo_label_background: bool = False

    # Cap on cross-pseudo-labeled vehicle-frames contributing to the
    # cross-supervision loss per batch — continuous analogue of
    # CT_MAX_VEH_PER_ITER=30/iter, same reasoning as FixMatchConfig's
    # max_pseudo_label_fraction_per_batch.
    max_pseudo_label_fraction_per_batch: float = 0.3

    # Cap on unlabeled pool size (data/dataset.py::DASPoolDataset), mirrors
    # MAX_POOL_SIZE=20_000 in co_training.ipynb.
    max_pool_size: int = 20_000

    # Independent augmentation streams per network branch (in addition to
    # different random init) supply the diversity CPS relies on in place of
    # the classical two-view conditional-independence assumption. Each
    # branch applies data/augmentations.py::weak_augment with its own RNG
    # (seeded from branch_a_seed/branch_b_seed below) to its view of the
    # same unlabeled crop — no strong/SpecAugment-style masking here, unlike
    # FixMatch, since CPS's diversity comes from init+augmentation rather
    # than a weak/strong teacher-student split (DESIGN.md section 4.2).
    weak_time_shift_sec: float = 0.1
    weak_channel_shift: int = 8
    weak_noise_std_frac: float = 0.03

    branch_a_seed: int = 42
    branch_b_seed: int = 1337
