"""
das_scan_utils.py
==================
Parallel-scan worker for das_exploration.ipynb's full-fiber, full-recording
energy scan.

Defined in a real, importable module (not inline in the notebook) because
ProcessPoolExecutor on Windows uses the "spawn" start method: each worker
process re-imports the target function by module path, which only works if
that function lives in an actual .py file, not a Jupyter notebook cell.

Each worker call processes ONE chunk end-to-end (load, boundary-correct,
band-filter, bin) and returns only the small, per-bin REDUCED results --
never the raw (time, channel) chunk array itself, which would be expensive
to pickle back to the parent process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import das_loader
from das_features import bandpass_all_channels
from nn_segmentation.data import anomaly


def chunk_bounds(das_t0: pd.Timestamp, das_t1: pd.Timestamp, chunk_min: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Precompute (t_cur, t_end) pairs covering [das_t0, das_t1) in chunk_min-minute steps."""
    chunk_dur = pd.Timedelta(minutes=chunk_min)
    bounds = []
    t_cur = das_t0
    while t_cur < das_t1:
        t_end = min(t_cur + chunk_dur, das_t1)
        bounds.append((t_cur, t_end))
        t_cur = t_end
    return bounds


def scan_chunk_worker(
    das_path: str,
    dist_min: float,
    dist_max: float,
    t_cur: pd.Timestamp,
    t_end: pd.Timestamp,
    das_t0: pd.Timestamp,
    das_t1: pd.Timestamp,
    bands: dict,
    fs: float,
    pad_sec: float,
    min_chunk_samples: int,
    bin_edges_all: pd.DatetimeIndex,
    n_bins: int,
    waterfall_band: str,
) -> dict:
    """
    Process one chunk in a worker process. Opens its OWN das_loader handle --
    xdas's virtual source is not shared across process boundaries, and
    reopening it is cheap (lazy). Skips (returns skipped=True) chunks too
    short to bandpass-filter, same guard as the sequential version (a chunk
    can come back short if it happens to fall on a genuine recording
    gap/dropout).

    Returns
    -------
    dict:
      skipped : bool
      global  : {band: {bin_index: mean_rms_across_channels}} (only if not skipped)
      waterfall : {bin_index: rms_per_channel array} for waterfall_band only (only if not skipped)
    """
    das_data = das_loader.open_das(das_path)

    pad = pd.Timedelta(seconds=pad_sec)
    pad_start = max(t_cur - pad, das_t0)
    pad_end = min(t_end + pad, das_t1)

    da_chunk = das_data.sel(
        distance=slice(dist_min, dist_max),
        time=slice(pad_start.isoformat(), pad_end.isoformat()),
    )
    data_tc = np.asarray(da_chunk.values, dtype=np.float32)
    time_vals = pd.to_datetime(da_chunk.coords["time"].values)

    boundary_idx = anomaly.get_second_boundaries(time_vals)
    data_tc = anomaly.interpolate_boundary_samples(data_tc, boundary_idx)
    keep = (time_vals >= t_cur) & (time_vals < t_end)
    data_tc = data_tc[keep]
    time_vals = pd.DatetimeIndex(time_vals[keep])

    if data_tc.shape[0] < min_chunk_samples:
        return {"skipped": True}

    bin_idx = np.searchsorted(bin_edges_all, time_vals.values, side="right") - 1

    global_out: dict = {}
    waterfall_out: dict = {}
    for bname, (lo, hi) in bands.items():
        filt = bandpass_all_channels(data_tc, fs, lo, hi)
        band_global = {}
        for b in np.unique(bin_idx):
            if b < 0 or b >= n_bins:
                continue
            sel = bin_idx == b
            if sel.sum() < 2:
                continue
            rms_per_ch = np.sqrt(np.mean(filt[sel] ** 2, axis=0))
            band_global[int(b)] = float(np.mean(rms_per_ch))
            if bname == waterfall_band:
                waterfall_out[int(b)] = rms_per_ch
        global_out[bname] = band_global
        del filt
    del data_tc

    return {"skipped": False, "global": global_out, "waterfall": waterfall_out}
