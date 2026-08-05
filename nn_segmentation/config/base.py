"""
Shared physical/dataset constants for the DAS segmentation pipeline.

Reuses (does not redefine) the physical parameters already established in
the repo's classical pipeline (``das_loader.py``) so the two stay in sync.
See ``nn_segmentation/DESIGN.md`` sections 1-2 for the reasoning behind each
default below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import das_loader

# DAS physical parameters (kept identical to das_loader.py — do not diverge).
FS: float = das_loader.FS  # Hz
DX: float = das_loader.DX  # m per channel

# Segmentation classes. "traffic" is an aggregate high-vehicle-count period,
# not a single coherent signal — see DESIGN.md section 3, masked out of the
# loss by default via SupervisedConfig.mask_classes.
BACKGROUND_CLASS = "background"
VEHICLE_CLASSES = ("tram", "bus", "truck")
AUX_CLASSES = ("traffic",)
CLASS_NAMES = (BACKGROUND_CLASS, *VEHICLE_CLASSES, *AUX_CLASSES)


@dataclass(frozen=True)
class SiteConfig:
    """One recording site's distance-channel window (das_loader locations.csv row)."""

    name: str
    dist_start: float  # metres
    dist_end: float  # metres

    @property
    def n_channels(self) -> int:
        """Number of DAS channels spanned by this site at DX spacing."""
        return round((self.dist_end - self.dist_start) / DX)


# Mirrors labeling/locations.csv. Loaded dynamically at runtime via
# data/raster_labels.py::load_site_configs() (das_loader.load_locations());
# these are fallback/documentation defaults only.
DEFAULT_SITES: tuple[SiteConfig, ...] = (
    SiteConfig("most_mieszka", 920.0, 1000.0),
    SiteConfig("pcss", 400.0, 600.0),
    SiteConfig("estkowskiego", 1700.0, 2000.0),
    SiteConfig("garbary", 2200.0, 2500.0),
    SiteConfig("srodka", 400.0, 600.0),
)


@dataclass
class BaseConfig:
    """Paths and dataset-wide settings shared by every training mode."""

    das_path: str = das_loader.DAS_PATH
    locations_path: str = das_loader.LOCS_PATH
    label_studio_output: str = das_loader.LS_OUTPUT
    log_csv_paths: tuple[str, ...] = (
        "labeling/log2.csv",
        "labeling/log3.csv",
        "labeling/log4.csv",
    )
    footage_offsets_path: str = "labeling/footage_offsets.csv"

    # Clock-skew buffer applied around a footage's first/last event when
    # deriving its camera-coverage window from CSV-log events (no direct
    # video start/end timestamp is available for that source) — mirrors
    # FOOTAGE_BUFFER_SEC in co_training.ipynb.
    footage_buffer_sec: float = 30.0

    # Events shorter than this (seconds) are dropped as noise, mirrors
    # MIN_EVENT_DURATION in xgboost_baseline.ipynb / co_training.ipynb.
    min_event_duration: float = 0.05

    cache_dir: str = "cache/nn_segmentation"
    checkpoint_dir: str = "nn_segmentation/models/training"
    run_dir: str = "runs/nn_segmentation"

    class_names: tuple[str, ...] = CLASS_NAMES

    # Chronological train/test split fraction — single source of truth is
    # data/splits.py::get_temporal_split(), this is just the default value.
    train_frac: float = 0.70

    # Overlap fraction (of frame/event duration) required to rasterize a
    # timestep as belonging to an event's class — mirrors OVERLAP_THRESHOLD
    # in xgboost_baseline.ipynb / co_training.ipynb.
    overlap_threshold: float = 0.2

    seed: int = 42

    sites: tuple[SiteConfig, ...] = field(default_factory=lambda: DEFAULT_SITES)
