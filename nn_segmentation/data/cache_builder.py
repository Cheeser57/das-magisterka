"""
Build the continuous per-site cache consumed by DASSegmentationDataset.

DESIGN.md section 2.2. Orchestrates: chunked loading via das_loader.open_das()
(FULL_CHUNK_MIN=20 minute chunks, matching co_training.ipynb), boundary
interpolation (data/anomaly.py), label rasterization (data/raster_labels.py),
and normalization-stat computation (data/normalization.py) restricted to the
training range (data/splits.py) — then persists everything to
``cache/nn_segmentation/<site>.npz``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import das_loader
from nn_segmentation.config.base import BaseConfig, SiteConfig
from nn_segmentation.data import anomaly, normalization, raster_labels, splits

FULL_CHUNK_MIN = 20  # minutes per xdas load chunk, matches co_training.ipynb


def _load_continuous_signal(
    das_data,
    site: SiteConfig,
    das_t0: pd.Timestamp,
    das_t1: pd.Timestamp,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load the full recording for ``site``'s distance window in
    FULL_CHUNK_MIN-minute chunks (tractable for a lazy xdas backend) and
    concatenate into one continuous (T, C) array + time index. Adjacent
    chunks are sliced with inclusive time bounds (mirrors co_training.ipynb),
    so the shared boundary sample is de-duplicated after concatenation.
    """
    chunk_dur = pd.Timedelta(minutes=FULL_CHUNK_MIN)
    total_chunks = int(np.ceil((das_t1 - das_t0).total_seconds() / (FULL_CHUNK_MIN * 60)))

    signal_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    t_cur = das_t0
    for _ in range(total_chunks):
        t_end = min(t_cur + chunk_dur, das_t1)
        da_chunk = das_data.sel(
            distance=slice(site.dist_start, site.dist_end),
            time=slice(t_cur.isoformat(), t_end.isoformat()),
        )
        chunk_time = pd.to_datetime(da_chunk.coords["time"].values)
        if len(chunk_time):
            signal_parts.append(np.asarray(da_chunk.values, dtype=np.float32))
            time_parts.append(chunk_time.values)
        t_cur = t_end

    if not signal_parts:
        raise ValueError(
            f"No DAS data found for site {site.name!r} in distance "
            f"[{site.dist_start}, {site.dist_end}]"
        )

    all_times = np.concatenate(time_parts)
    all_signal = np.concatenate(signal_parts, axis=0)

    # De-duplicate the sample shared by two adjacent chunks (inclusive slice
    # bounds on both ends): all_times is already non-decreasing, so the
    # first occurrence of each timestamp is the one to keep.
    unique_times, first_idx = np.unique(all_times, return_index=True)
    first_idx.sort()
    time_index = pd.DatetimeIndex(all_times[first_idx])
    signal = all_signal[first_idx]

    return signal, time_index


def build_site_cache(
    site: SiteConfig,
    config: BaseConfig,
    force_rebuild: bool = False,
) -> Path:
    """
    Build (or return the existing) cache file for one site.

    Pipeline
    --------
    1. Load das_loader.open_das(), slice to [site.dist_start, site.dist_end],
       in FULL_CHUNK_MIN-minute time chunks, concatenated into one
       continuous array (_load_continuous_signal).
    2. data.anomaly.get_second_boundaries() + interpolate_boundary_samples()
       once over the full continuous signal (not per-chunk, so a boundary
       sample landing on a chunk's first index is never missed).
    3. data.raster_labels.load_all_events() + build_site_timeline()
       -> label, labeled_mask.
    4. data.splits.get_temporal_split() -> train/test cutoff for this site.
    5. data.normalization.compute_channel_stats() on the train range only.
    6. Persist signal, time_index, label, labeled_mask, boundary_mask,
       norm_stats, and the split boundaries to
       ``{config.cache_dir}/{site.name}.npz``.

    Returns
    -------
    Path
        Location of the written (or pre-existing, if force_rebuild=False
        and the file already exists) cache file.
    """
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{site.name}.npz"
    if cache_path.exists() and not force_rebuild:
        return cache_path

    das_data = das_loader.open_das(config.das_path)
    all_times = pd.to_datetime(das_data.coords["time"].values)
    das_t0, das_t1 = all_times[0], all_times[-1]

    signal, time_index = _load_continuous_signal(das_data, site, das_t0, das_t1)

    boundary_idx = anomaly.get_second_boundaries(time_index)
    signal = anomaly.interpolate_boundary_samples(signal, boundary_idx)
    boundary_mask_arr = anomaly.boundary_mask(len(time_index), boundary_idx)

    events = raster_labels.load_all_events(config, das_t0=das_t0, das_t1=das_t1)
    timeline = raster_labels.build_site_timeline(site, time_index, events, config)

    event_times = timeline.events["time_start"] if not timeline.events.empty else pd.Series(dtype="datetime64[ns]")
    split = splits.get_temporal_split(time_index, event_times, train_frac=config.train_frac)

    train_mask = splits.in_train_range(time_index, split)
    norm_stats = normalization.compute_channel_stats(signal, train_mask)

    np.savez_compressed(
        cache_path,
        signal=signal,
        time_index_ns=time_index.values.astype("datetime64[ns]").astype(np.int64),
        label=timeline.label,
        labeled_mask=timeline.labeled_mask,
        boundary_mask=boundary_mask_arr,
        norm_median=norm_stats.median,
        norm_scale=norm_stats.scale,
        train_start_ns=np.int64(split.train_start.value),
        train_end_ns=np.int64(split.train_end.value),
        test_start_ns=np.int64(split.test_start.value),
        test_end_ns=np.int64(split.test_end.value),
        site_name=site.name,
        dist_start=site.dist_start,
        dist_end=site.dist_end,
    )
    return cache_path


def load_site_cache(site: SiteConfig, config: BaseConfig) -> dict[str, Any]:
    """Load a previously built ``<site>.npz`` cache. Raises if it doesn't exist."""
    cache_path = Path(config.cache_dir) / f"{site.name}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cache found for site {site.name!r} at {cache_path}; run `build-cache` first."
        )

    data = np.load(cache_path, allow_pickle=False)
    time_index = pd.DatetimeIndex(data["time_index_ns"].astype("datetime64[ns]"))
    split = splits.TemporalSplit(
        train_start=pd.Timestamp(int(data["train_start_ns"])),
        train_end=pd.Timestamp(int(data["train_end_ns"])),
        test_start=pd.Timestamp(int(data["test_start_ns"])),
        test_end=pd.Timestamp(int(data["test_end_ns"])),
    )
    return {
        "signal": data["signal"],
        "time_index": time_index,
        "label": data["label"],
        "labeled_mask": data["labeled_mask"],
        "boundary_mask": data["boundary_mask"],
        "norm_stats": normalization.ChannelNormStats(median=data["norm_median"], scale=data["norm_scale"]),
        "split": split,
        "site": SiteConfig(
            name=str(data["site_name"]),
            dist_start=float(data["dist_start"]),
            dist_end=float(data["dist_end"]),
        ),
    }


def load_and_normalize_sites(sites: list[SiteConfig], config: BaseConfig) -> dict[str, dict[str, Any]]:
    """
    Load each site's cache exactly ONCE and z-score-normalize its signal
    in-place, for sharing across every dataset object that needs it within
    one training run.

    Without this, DASSegmentationDataset(split="train"),
    DASSegmentationDataset(split="test"), and DASPoolDataset each
    independently call load_site_cache() + apply_zscore() per site — same
    site, same underlying .npz, loaded from disk and normalized 2-3x
    (train_supervised: 2x; train_fixmatch/train_cross_pseudo_supervision:
    3x), multiplying both load time and peak memory for wide/multi-site
    configurations (e.g. garbary alone is ~18 GiB per copy — this was
    observed to OOM in practice). Pass the result here into each dataset's
    ``preloaded_caches`` argument instead.

    Sites with no on-disk cache are silently omitted from the returned
    dict (matches DASSegmentationDataset/DASPoolDataset's existing
    FileNotFoundError-tolerant per-site behavior) — their absence is the
    signal to callers that no cache exists yet for that site.
    """
    caches: dict[str, dict[str, Any]] = {}
    for site in sites:
        try:
            cache = load_site_cache(site, config)
        except FileNotFoundError:
            continue
        cache["signal"] = normalization.apply_zscore(cache["signal"], cache["norm_stats"], in_place=True)
        caches[site.name] = cache
    return caches
