"""
PyTorch Dataset classes drawing crops from the per-site caches.

DESIGN.md section 2.2. DASSegmentationDataset draws labeled crops (filtered
to sufficient labeled_mask coverage) for supervised training;
DASPoolDataset draws crops from anywhere in the train range (labeled or
not) for the FixMatch/CPS unlabeled stream. Both must respect
data/splits.py's train/test cutoff — see DESIGN.md section 4.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from nn_segmentation.config.base import FS, BaseConfig, SiteConfig
from nn_segmentation.config.supervised import SupervisedConfig
from nn_segmentation.data import cache_builder, normalization, splits
from nn_segmentation.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentationSample:
    signal: torch.Tensor  # (C, T) normalized
    label: torch.Tensor  # (T,) int64, at input resolution (pooled in the model)
    labeled_mask: torch.Tensor  # (T,) bool
    site_name: str


def _find_valid_crop_starts(
    range_mask: np.ndarray,
    labeled_mask: np.ndarray,
    chunk_samples: int,
    min_labeled_fraction: float,
    stride: int,
) -> np.ndarray:
    """
    Vectorized search for crop start indices that (a) fall entirely inside
    ``range_mask``'s contiguous block (the train or test half of a
    chronological split — always a single contiguous run of True) and
    (b) have at least ``min_labeled_fraction`` of their samples
    labeled_mask=True, using a prefix-sum sliding-window coverage
    computation rather than a per-index Python loop.
    """
    n = len(range_mask)
    if n < chunk_samples:
        return np.array([], dtype=np.int64)

    range_idx = np.flatnonzero(range_mask)
    if len(range_idx) == 0:
        return np.array([], dtype=np.int64)
    range_start, range_end = int(range_idx[0]), int(range_idx[-1]) + 1  # half-open

    last_start = range_end - chunk_samples
    if last_start < range_start:
        return np.array([], dtype=np.int64)

    cumsum = np.concatenate([[0], np.cumsum(labeled_mask.astype(np.int64))])
    candidate_starts = np.arange(range_start, last_start + 1, stride)
    coverage = (cumsum[candidate_starts + chunk_samples] - cumsum[candidate_starts]) / chunk_samples
    return candidate_starts[coverage >= min_labeled_fraction]


class DASSegmentationDataset(Dataset):
    """
    Supervised training crops for one or more sites, drawn only from each
    site's training range (data/splits.py::get_temporal_split), filtered to
    crops with at least ``train_config.min_labeled_fraction`` labeled_mask
    coverage.
    """

    def __init__(
        self,
        sites: list[SiteConfig],
        base_config: BaseConfig,
        train_config: SupervisedConfig,
        split: str = "train",
        preloaded_caches: dict[str, dict] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        split : {"train", "test"}
            Which half of data/splits.py's temporal split to draw crops from.
        preloaded_caches : optional
            Result of cache_builder.load_and_normalize_sites(sites, base_config),
            already-normalized and shared with any sibling dataset objects
            (DASSegmentationDataset(split="test"), DASPoolDataset) built for
            the same training run — avoids each dataset independently
            reloading and re-normalizing the same site's signal from disk.
            If omitted, this dataset loads and normalizes its own copy per
            site (standalone-usage fallback, e.g. tests/notebooks).
        """
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.chunk_samples = int(round(train_config.chunk_len_sec * FS))
        self.site_signals: dict[str, np.ndarray] = {}
        self.site_labels: dict[str, np.ndarray] = {}
        self.site_labeled_masks: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        stride = max(1, self.chunk_samples // 2)

        for site in sites:
            if preloaded_caches is not None:
                cache = preloaded_caches.get(site.name)
                if cache is None:
                    logger.warning("No cache for site %r, skipping (run build-cache first).", site.name)
                    continue
                normalized_signal = cache["signal"]  # already normalized by load_and_normalize_sites
            else:
                try:
                    cache = cache_builder.load_site_cache(site, base_config)
                except FileNotFoundError:
                    logger.warning("No cache for site %r, skipping (run build-cache first).", site.name)
                    continue
                normalized_signal = normalization.apply_zscore(cache["signal"], cache["norm_stats"])

            range_mask = (
                splits.in_train_range(cache["time_index"], cache["split"])
                if split == "train"
                else splits.in_test_range(cache["time_index"], cache["split"])
            )
            starts = _find_valid_crop_starts(
                range_mask,
                cache["labeled_mask"],
                self.chunk_samples,
                train_config.min_labeled_fraction,
                stride,
            )
            if len(starts) == 0:
                continue

            self.site_signals[site.name] = normalized_signal
            self.site_labels[site.name] = cache["label"]
            self.site_labeled_masks[site.name] = cache["labeled_mask"]
            self.samples.extend((site.name, int(s)) for s in starts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SegmentationSample:
        site_name, start = self.samples[index]
        end = start + self.chunk_samples
        signal_crop = self.site_signals[site_name][start:end]  # (T, C)
        label_crop = self.site_labels[site_name][start:end]
        mask_crop = self.site_labeled_masks[site_name][start:end]
        return SegmentationSample(
            signal=torch.from_numpy(np.ascontiguousarray(signal_crop.T)),
            label=torch.from_numpy(label_crop.astype(np.int64)),
            labeled_mask=torch.from_numpy(mask_crop.astype(bool)),
            site_name=site_name,
        )


def collate_segmentation_batch(
    samples: list[SegmentationSample],
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Group a (possibly mixed-site) batch by site name. Each site's Conv1d
    input adapter requires a uniform channel count and DASMultiStageTCN
    dispatches one adapter per forward() call (DESIGN.md section 2.5), so
    the model consumes one site's stacked tensors at a time; the training
    loop iterates over this dict's groups per batch.
    """
    grouped: dict[str, list[SegmentationSample]] = {}
    for s in samples:
        grouped.setdefault(s.site_name, []).append(s)
    return {
        site_name: {
            "signal": torch.stack([s.signal for s in group]),
            "label": torch.stack([s.label for s in group]),
            "labeled_mask": torch.stack([s.labeled_mask for s in group]),
        }
        for site_name, group in grouped.items()
    }


@dataclass
class PoolSample:
    signal: torch.Tensor  # (C, T) normalized, boundary-clean, pre-augmentation
    site_name: str


class DASPoolDataset(Dataset):
    """
    Unlabeled crops for semi-supervised training (FixMatch's weak/strong
    pair, or CPS's two branches), drawn from anywhere in a site's training
    range regardless of labeled_mask — but never from the test range
    (DESIGN.md section 4.4).

    Augmentation is deliberately NOT applied here: FixMatch needs a
    weak/strong pair of the same crop while CPS needs two independently
    augmented views for two different branches, so the training loops
    (training/fixmatch.py, training/cotraining.py) apply
    data/augmentations.py themselves after drawing a raw crop from here.
    """

    def __init__(
        self,
        sites: list[SiteConfig],
        base_config: BaseConfig,
        train_config: SupervisedConfig,
        max_pool_size: int | None = None,
        preloaded_caches: dict[str, dict] | None = None,
    ) -> None:
        """
        preloaded_caches : optional
            Result of cache_builder.load_and_normalize_sites(sites, base_config),
            shared with the sibling DASSegmentationDataset objects built for
            the same training run — see that class's docstring for why.
        """
        self.chunk_samples = int(round(train_config.chunk_len_sec * FS))
        self.site_signals: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        stride = max(1, self.chunk_samples // 2)

        for site in sites:
            if preloaded_caches is not None:
                cache = preloaded_caches.get(site.name)
                if cache is None:
                    logger.warning("No cache for site %r, skipping (run build-cache first).", site.name)
                    continue
                normalized_signal = cache["signal"]  # already normalized by load_and_normalize_sites
            else:
                try:
                    cache = cache_builder.load_site_cache(site, base_config)
                except FileNotFoundError:
                    logger.warning("No cache for site %r, skipping (run build-cache first).", site.name)
                    continue
                normalized_signal = normalization.apply_zscore(cache["signal"], cache["norm_stats"])

            # Only the train range is ever eligible for unlabeled pool crops
            # (DESIGN.md section 4.4) — no labeled_mask coverage requirement,
            # so pass an all-True mask through the same coverage search used
            # by DASSegmentationDataset with min_labeled_fraction=0.
            range_mask = splits.in_train_range(cache["time_index"], cache["split"])
            always_labeled = np.ones(len(range_mask), dtype=bool)
            starts = _find_valid_crop_starts(range_mask, always_labeled, self.chunk_samples, 0.0, stride)
            if len(starts) == 0:
                continue

            self.site_signals[site.name] = normalized_signal
            self.samples.extend((site.name, int(s)) for s in starts)

        if max_pool_size is not None and len(self.samples) > max_pool_size:
            rng = np.random.default_rng(base_config.seed)
            keep_idx = rng.choice(len(self.samples), size=max_pool_size, replace=False)
            self.samples = [self.samples[i] for i in keep_idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PoolSample:
        """Returns a single (C, T) normalized, boundary-clean crop (pre-augmentation)."""
        site_name, start = self.samples[index]
        end = start + self.chunk_samples
        signal_crop = self.site_signals[site_name][start:end]  # (T, C)
        return PoolSample(
            signal=torch.from_numpy(np.ascontiguousarray(signal_crop.T)),
            site_name=site_name,
        )


def collate_pool_batch(samples: list[PoolSample]) -> dict[str, torch.Tensor]:
    """Group a (possibly mixed-site) unlabeled batch by site name, mirroring collate_segmentation_batch."""
    grouped: dict[str, list[PoolSample]] = {}
    for s in samples:
        grouped.setdefault(s.site_name, []).append(s)
    return {site_name: torch.stack([s.signal for s in group]) for site_name, group in grouped.items()}
