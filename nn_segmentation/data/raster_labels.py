"""
Rasterize event-span labels onto a continuous per-timestep timeline.

DESIGN.md section 2.2. Reuses das_loader.py's label parsing (load_annotations,
load_locations) rather than re-implementing it — this module's job is only
to turn those event spans into dense arrays aligned with a site's continuous
signal cache.

Event/footage-window handling mirrors co_training.ipynb exactly where that
notebook already established a convention:
  - CSV-log events get footage_offsets.csv clock-skew correction applied to
    both time_start and time_end (co_training.ipynb section "events_raw =
    pd.read_csv(...)").
  - A footage's camera-coverage window is [min(time_start), max(time_end)]
    +/- footage_buffer_sec, grouped by footage_id (co_training.ipynb's
    build_location_dataset "Per-footage coverage windows" cell) — there is
    no direct video start/end timestamp available for this source.
  - Label Studio-derived events already carry an explicit per-annotation
    coverage window (win_start/win_end, the exact bounds the waterfall image
    was rendered from) and need no offset correction (already applied when
    the export was generated).
  - Where multiple events overlap the same timestep, the later event (by
    time_start) wins — a simple last-write-wins rule, matching
    co_training.ipynb's `labels[ov_frac > OVERLAP_THRESHOLD] = ev["class"]`
    loop rather than an invented priority order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import das_loader
from nn_segmentation.config.base import BaseConfig, SiteConfig

_EVENT_COLUMNS = ["location", "class", "time_start", "time_end", "source", "footage_id", "win_start", "win_end"]


@dataclass
class SiteTimeline:
    """Per-timestep rasterized labels for one site, aligned to a signal cache."""

    time_index: pd.DatetimeIndex
    label: np.ndarray  # (T,) int, index into config.class_names
    labeled_mask: np.ndarray  # (T,) bool — True where camera/footage confirmed
    events: pd.DataFrame  # the source events used to build this timeline (this site only)


def load_site_configs(config: BaseConfig) -> list[SiteConfig]:
    """
    Load per-site distance-channel windows from labeling/locations.csv via
    das_loader.load_locations(), overriding config.sites' hardcoded defaults.
    """
    locs = das_loader.load_locations(config.locations_path)
    return [
        SiteConfig(name=str(name), dist_start=float(row["start"]), dist_end=float(row["end"]))
        for name, row in locs.iterrows()
    ]


def _load_footage_offsets(config: BaseConfig) -> dict[str, float]:
    path = Path(config.footage_offsets_path)
    if not path.exists():
        return {}
    offsets = pd.read_csv(path)
    return offsets.set_index("footage_id")["offset_sec"].to_dict()


def _load_labelstudio_events(config: BaseConfig) -> pd.DataFrame:
    """Label Studio events, already offset-corrected — no clock-skew adjustment needed."""
    output_dir = Path(config.label_studio_output)
    if not output_dir.exists():
        # Gitignored directory, may legitimately be absent locally — propagate
        # as "0 events" rather than letting das_loader's Path.iterdir() raise.
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    ann = das_loader.load_annotations(config.label_studio_output)
    if ann.empty:
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    ann = ann.copy()
    ann["source"] = "labelstudio"
    ann["footage_id"] = None
    return ann[_EVENT_COLUMNS]


def _load_csv_log_events(config: BaseConfig, das_t0: pd.Timestamp, das_t1: pd.Timestamp) -> pd.DataFrame:
    """CSV-log events, footage_offsets.csv clock-skew corrected, filtered to sane durations/bounds."""
    paths = [p for p in config.log_csv_paths if Path(p).exists()]
    if not paths:
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    raw = pd.concat(
        [pd.read_csv(p, parse_dates=["time_start", "time_end"]) for p in paths],
        ignore_index=True,
    )

    offsets = _load_footage_offsets(config)
    for fid, sec in offsets.items():
        mask = raw["footage_id"] == fid
        if mask.any():
            raw.loc[mask, "time_start"] += pd.Timedelta(seconds=sec)
            raw.loc[mask, "time_end"] += pd.Timedelta(seconds=sec)

    durations = (raw["time_end"] - raw["time_start"]).dt.total_seconds()
    raw = raw[
        (durations >= config.min_event_duration)
        & (raw["time_start"] >= das_t0)
        & (raw["time_end"] <= das_t1)
    ].copy()

    raw["source"] = "csv_log"
    raw["win_start"] = pd.NaT
    raw["win_end"] = pd.NaT
    return raw[_EVENT_COLUMNS]


def load_all_events(
    config: BaseConfig,
    das_t0: pd.Timestamp | None = None,
    das_t1: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Concatenate events from every label source: Label Studio exports
    (das_loader.load_annotations()) and the CSV logs
    (config.log_csv_paths). Tolerates both sources being entirely absent
    (labelStudio/output and the log CSVs are gitignored) — returns an empty,
    correctly-columned DataFrame rather than raising, so the unlabeled/pool-
    only path can still run.

    Parameters
    ----------
    das_t0, das_t1 : pd.Timestamp, optional
        DAS recording bounds, used only to bounds-filter and offset-correct
        CSV-log events (mirrors co_training.ipynb). If omitted, CSV-log
        events are not bounds-filtered (Label Studio events need no such
        filtering — they are already slices of the real recording).

    Returns
    -------
    pd.DataFrame
        Columns: location, class, time_start, time_end, source
        ("labelstudio" | "csv_log"), footage_id, win_start, win_end.
    """
    ls_events = _load_labelstudio_events(config)
    if das_t0 is not None and das_t1 is not None:
        csv_events = _load_csv_log_events(config, das_t0, das_t1)
    else:
        csv_events = pd.DataFrame(columns=_EVENT_COLUMNS)

    if ls_events.empty and csv_events.empty:
        return pd.DataFrame(columns=_EVENT_COLUMNS)
    return pd.concat([ls_events, csv_events], ignore_index=True)


def build_site_timeline(
    site: SiteConfig,
    time_index: pd.DatetimeIndex,
    events: pd.DataFrame,
    config: BaseConfig,
) -> SiteTimeline:
    """
    Rasterize ``events`` onto ``time_index`` for one site.

    Every timestep inside an event's [time_start, time_end] gets that
    event's class id (config.overlap_threshold is not needed here since
    rasterization is exact at full sample resolution, unlike the classical
    notebooks' fixed-frame overlap-fraction labeling). Every timestep
    defaults to "background", except where it falls outside all
    camera/footage-covered windows for this site — those get
    labeled_mask=False regardless of the default label (DESIGN.md
    section 2.2, generalizing co_training.ipynb's per-frame in_footage
    boolean to per-timestep resolution).
    """
    loc_events = events[events["location"] == site.name].sort_values("time_start").reset_index(drop=True)

    class_to_id = {name: i for i, name in enumerate(config.class_names)}
    label = np.zeros(len(time_index), dtype=np.int64)  # 0 == background

    for _, ev in loc_events.iterrows():
        cls_id = class_to_id.get(ev["class"])
        if cls_id is None:
            continue  # class not tracked by this segmentation task (e.g. "car")
        mask = (time_index >= ev["time_start"]) & (time_index <= ev["time_end"])
        label[mask] = cls_id  # last-write-wins, matches co_training.ipynb's loop order

    labeled_mask = np.zeros(len(time_index), dtype=bool)
    buf = pd.Timedelta(seconds=config.footage_buffer_sec)

    csv_subset = loc_events[loc_events["source"] == "csv_log"]
    for _, grp in csv_subset.groupby("footage_id"):
        win_s = grp["time_start"].min() - buf
        win_e = grp["time_end"].max() + buf
        labeled_mask |= (time_index >= win_s) & (time_index <= win_e)

    ls_subset = loc_events[loc_events["source"] == "labelstudio"]
    for _, ev in ls_subset.iterrows():
        labeled_mask |= (time_index >= ev["win_start"]) & (time_index <= ev["win_end"])

    return SiteTimeline(time_index=time_index, label=label, labeled_mask=labeled_mask, events=loc_events)
