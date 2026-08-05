"""
das_loader.py
=============
Shared utilities for loading DAS patches from Label Studio annotations.

Typical usage
-------------
    from das_loader import open_das, load_locations, load_annotations, load_patch

    das_data = open_das()
    locs     = load_locations()
    labels   = load_annotations()

    arr_u8, arr_raw, times, gt0, gt1 = load_patch(labels.iloc[0], das_data, locs)
"""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xdas
import xdas.signal as xs

# ── DAS physical parameters ───────────────────────────────────────────────────
FS       = 67.0   # Hz
DX       = 0.8    # m per channel
V_MIN_MS = 5.0    # m/s — slowest expected vehicle
V_MAX_MS = 100.0  # m/s — fastest expected vehicle

# ── Patch defaults ────────────────────────────────────────────────────────────
VPERCENTILE  = 99   # percentile used to clip raw array before uint8 conversion
WINDOW_SCALE = 1.0  # multiplier on annotation window duration (1.0 = use as-is)

# ── Default paths (relative to project root) ──────────────────────────────────
DAS_PATH  = "das/recorded.nc"
LOCS_PATH = "labeling/locations.csv"
LS_OUTPUT = "labelStudio/output"


def load_locations(path: str = LOCS_PATH) -> pd.DataFrame:
    """Return locations indexed by id with clean column names and numeric start/end."""
    locs = pd.read_csv(path, skipinitialspace=True)
    locs.columns = locs.columns.str.strip()
    locs["start"] = locs["start"].astype(str).str.strip().astype(float)
    locs["end"]   = locs["end"].astype(str).str.strip().astype(float)
    return locs.set_index("id")


def open_das(path: str = DAS_PATH) -> xdas.DataArray:
    """Open the DAS recording. Returns a virtual (lazy) DataArray."""
    return xdas.open_dataarray(path)


def load_annotations(output_dir: str = LS_OUTPUT) -> pd.DataFrame:
    """
    Parse all Label Studio JSON export files and return one row per bounding box.

    Columns: location, direction, class, origin, annotation_id,
             win_start, win_end, time_start, time_end, center
    """
    def _float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    records = []
    for fpath in sorted(Path(output_dir).iterdir()):
        if not fpath.is_file():
            continue
        try:
            ann = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [skip] {fpath.name}: {e}")
            continue

        data      = ann.get("task", {}).get("data", {})
        win_start = pd.to_datetime(data.get("win_start"), errors="coerce")
        win_end   = pd.to_datetime(data.get("win_end"),   errors="coerce")
        if pd.isna(win_start) or pd.isna(win_end) or win_end <= win_start:
            continue

        win_dur  = (win_end - win_start).total_seconds()
        location = data.get("location", "unknown")
        direction = data.get("direction", "unknown")

        for box in ann.get("result", []):
            if box.get("type") != "rectanglelabels":
                continue
            val   = box.get("value", {})
            cls   = (val.get("rectanglelabels") or ["unknown"])[0]
            y_pct = _float(val.get("y"))
            h_pct = _float(val.get("height"))
            if y_pct is None or h_pct is None:
                continue

            s = np.clip(y_pct / 100.0, 0.0, 1.0)
            e = np.clip((y_pct + h_pct) / 100.0, 0.0, 1.0)
            if e <= s:
                continue

            t0 = win_start + timedelta(seconds=s * win_dur)
            t1 = win_start + timedelta(seconds=e * win_dur)
            records.append({
                "location":      location,
                "direction":     direction,
                "class":         cls,
                "origin":        box.get("origin", "unknown"),
                "annotation_id": ann.get("id"),
                "win_start":     win_start,
                "win_end":       win_end,
                "time_start":    t0,
                "time_end":      t1,
                "center":        t0 + (t1 - t0) / 2,
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} annotations from {output_dir!r}")
    return df


def load_patch(
    row,
    das_data: xdas.DataArray,
    locs: pd.DataFrame,
    strain_rate: bool = True,
    window_scale: float = WINDOW_SCALE,
):
    """
    Load a DAS patch for one annotation row.

    Parameters
    ----------
    row          : pandas Series — one row from load_annotations()
    das_data     : xdas DataArray returned by open_das()
    locs         : DataFrame returned by load_locations()
    strain_rate  : if True, differentiate along time before returning
    window_scale : multiplier on the annotation window duration.
                   1.0 = use exactly the Label Studio window.
                   0.5 = halve the window, centred on the original.

    Returns
    -------
    arr_u8        : np.ndarray uint8  (n_time × n_dist) — normalised for display
    arr_raw       : np.ndarray float32 — strain rate (or raw) values
    times         : pd.DatetimeIndex
    gt_row_start  : float — pixel row corresponding to annotation start
    gt_row_end    : float — pixel row corresponding to annotation end
    """
    loc        = row["location"]
    dist_start = float(locs.loc[loc, "start"])
    dist_end   = float(locs.loc[loc, "end"])

    orig_start = pd.to_datetime(row["win_start"])
    orig_end   = pd.to_datetime(row["win_end"])
    dur        = orig_end - orig_start

    window_scale = max(0.1, float(window_scale))
    half_extra   = (window_scale - 1.0) * dur / 2.0
    sel_start    = orig_start - half_extra
    sel_end      = orig_end   + half_extra

    all_times = pd.to_datetime(das_data.coords["time"].values)
    sel_start = max(sel_start, all_times[0])
    sel_end   = min(sel_end,   all_times[-1])

    da = das_data.sel(
        time=slice(sel_start.isoformat(), sel_end.isoformat()),
        distance=slice(dist_start, dist_end),
    )
    if strain_rate:
        da = xs.differentiate(da, dim="time")

    arr   = da.values.astype(np.float32)
    times = pd.to_datetime(da.coords["time"].values)
    n     = len(times)
    total = (times[-1] - times[0]).total_seconds()
    if total == 0:
        raise ValueError(f"Zero-duration window for {loc} at {orig_start}")

    finite = arr[np.isfinite(arr)]
    vmax   = np.percentile(np.abs(finite), VPERCENTILE) if finite.size else 1.0
    arr_u8 = ((np.clip(arr, -vmax, vmax) + vmax) / (2 * vmax) * 255).astype(np.uint8)

    def _t2row(t):
        return float(np.clip(
            (pd.to_datetime(t) - times[0]).total_seconds() / total * n,
            0, n - 1,
        ))

    return arr_u8, arr, times, _t2row(row["time_start"]), _t2row(row["time_end"])


def expected_angle_range():
    """
    Return the expected (angle_min_deg, angle_max_deg) from horizontal
    for vehicle trajectories in a DAS waterfall (rows=time, cols=distance).
    """
    slope_max = (FS * DX) / V_MIN_MS   # slow vehicle → steep slope
    slope_min = (FS * DX) / V_MAX_MS   # fast vehicle → shallow slope
    ang_min   = np.degrees(np.arctan(slope_min))
    ang_max   = np.degrees(np.arctan(slope_max))
    return ang_min, ang_max
