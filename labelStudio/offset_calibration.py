"""
DAS timestamp offset calibration — Jupyter-friendly.

Quick start:
    %matplotlib widget
    from labelStudio.offset_calibration import calibrate, list_footage

    list_footage()                        # print available footage IDs
    calibrate()                           # same, then return
    calibrate(footage=0)                  # pick by index
    calibrate(footage="most_mieszka_06-08-12-50")
    calibrate(footage=0, n_samples=5, context_sec=45, offset_range=60)
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd

# ── Defaults anchored to project root (parent of this file's directory) ───────
_ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DAS_FILE      = os.path.join(_ROOT, "das", "recorded.nc")
_LOG_CSV       = os.path.join(_ROOT, "labeling", "log2.csv")
_LOCATIONS_CSV = os.path.join(_ROOT, "labeling", "locations.csv")
_OFFSETS_FILE  = os.path.join(_ROOT, "labeling", "footage_offsets.csv")


# ── Public helpers ────────────────────────────────────────────────────────────

def list_footage(log_path=_LOG_CSV):
    """Print available footage IDs and their event counts."""
    log = pd.read_csv(log_path, parse_dates=["time_start", "time_end"])
    if "footage_id" not in log.columns:
        print("No footage_id column in log — all events will be treated as one group.")
        return
    ids = log["footage_id"].dropna().unique()
    print(f"{'#':>3}  {'footage_id':<40}  events")
    print("─" * 55)
    for i, fid in enumerate(ids):
        n = (log["footage_id"] == fid).sum()
        print(f"{i:>3}  {fid:<40}  {n}")


def calibrate(
    footage       = None,        # footage_id string, int index, or None → list & return
    das_data       = None,        # xarray DataArray or path to recorded.nc (default _DAS_FILE)
    n_samples     = 3,           # number of waterfall panels
    context_sec   = 30,          # seconds shown each side of event center
    offset_range  = 30,          # slider range ±seconds
    log_path      = _LOG_CSV,
    das_file      = _DAS_FILE,
    locations_csv = _LOCATIONS_CSV,
    offsets_file  = _OFFSETS_FILE,
    colormap      = "RdBu_r",
    vpercentile   = 99,
):
    """
    Show an interactive offset-calibration figure for one footage ID.

    Parameters
    ----------
    footage       : str | int | None
        footage_id string, its integer index in list_footage(), or None to list and return.
    n_samples     : int   — waterfall panels to display (default 3)
    context_sec   : float — seconds of DAS shown each side of event center (default 30)
    offset_range  : float — slider range ±seconds (default 30)
    log_path      : path to events CSV
    das_file      : path to recorded.nc
    locations_csv : path to locations CSV
    offsets_file  : path where offsets are saved

    Returns
    -------
    The matplotlib Figure (useful in notebooks to keep it alive).
    """
    import matplotlib.pyplot as plt
    import matplotlib.widgets as mwidgets
    import xdas

    log  = pd.read_csv(log_path, parse_dates=["time_start", "time_end"])
    locs = pd.read_csv(locations_csv, skipinitialspace=True)
    locs.columns = locs.columns.str.strip()
    locs = locs.set_index("id")

    # ── Resolve footage_id ────────────────────────────────────────────────────
    has_footage_col = "footage_id" in log.columns
    if has_footage_col:
        all_ids = log["footage_id"].dropna().unique()
    else:
        all_ids = np.array(["all"])

    if footage is None:
        list_footage(log_path)
        return None

    if isinstance(footage, int):
        if footage >= len(all_ids):
            raise IndexError(f"Index {footage} out of range — {len(all_ids)} footage IDs available.")
        footage_id = all_ids[footage]
    else:
        footage_id = footage

    # ── Select events ─────────────────────────────────────────────────────────
    if has_footage_col and footage_id != "all":
        subset = log[log["footage_id"] == footage_id]
    else:
        subset = log
    if subset.empty:
        raise ValueError(f"No events found for footage_id='{footage_id}'")

    events = subset.sample(min(n_samples, len(subset)), random_state=42).reset_index(drop=True)

    # ── Load DAS patches ──────────────────────────────────────────────────────
    print("Loading DAS data …")
    if das_data is None:
        das_data = xdas.open_dataarray(das_file)

    patches = []
    for _, row in events.iterrows():
        loc = row["location"]
        if loc not in locs.index:
            print(f"  [skip] location '{loc}' not in locations.csv")
            continue
        dist_start = float(locs.loc[loc, "start"])
        dist_end   = float(locs.loc[loc, "end"])
        center     = row["time_start"] + (row["time_end"] - row["time_start"]) / 2
        win_start  = center - timedelta(seconds=context_sec)
        win_end    = center + timedelta(seconds=context_sec)
        try:
            da    = das_data.sel(
                time=slice(win_start.isoformat(), win_end.isoformat()),
                distance=slice(dist_start, dist_end),
            )
            arr   = da.values
            times = pd.to_datetime(da.coords["time"].values)
            patches.append((arr, times, center, row))
            print(f"  Loaded  {loc}  {center.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"  [skip] {e}")

    if not patches:
        raise RuntimeError("No patches loaded — check timestamps and DAS file.")

    # ── Build figure ──────────────────────────────────────────────────────────
    n = len(patches)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(bottom=0.14, top=0.90, wspace=0.08)

    hlines = []  # (Line2D, times, center)

    def _center_row(times, center, offset_sec=0.0):
        shifted  = center + timedelta(seconds=offset_sec)
        total    = (times[-1] - times[0]).total_seconds()
        frac     = (shifted - times[0]).total_seconds() / total
        return float(np.clip(frac * len(times), 0, len(times) - 1))

    for ax, (arr, times, center, row) in zip(axes, patches):
        finite = arr[np.isfinite(arr)]
        vmax   = np.percentile(np.abs(finite), vpercentile) if finite.size else 1.0
        ax.imshow(arr, aspect="auto", cmap=colormap, vmin=-vmax, vmax=vmax,
                  origin="upper", extent=[0, arr.shape[1], arr.shape[0], 0])

        row_idx = _center_row(times, center)
        line    = ax.axhline(y=row_idx, color="yellow", linewidth=1.8,
                             linestyle="--", alpha=0.9)
        hlines.append((line, times, center))

        total_sec  = (times[-1] - times[0]).total_seconds()
        center_sec = (center - times[0]).total_seconds()
        ticks      = np.linspace(0, arr.shape[0], 7)
        ax.set_yticks(ticks)
        ax.set_yticklabels(
            [f"{t / arr.shape[0] * total_sec - center_sec:+.0f}s" for t in ticks],
            fontsize=8,
        )
        ax.set_xticks([])
        cls = row.get("class", "tram") if "class" in row.index else "tram"
        ax.set_title(
            f"{row['location']}  ·  {cls}\n{center.strftime('%H:%M:%S')}",
            fontsize=9,
        )

    fig.suptitle(
        f"Offset calibration — {footage_id}\n"
        "Yellow line = event center  ·  drag slider to align with DAS signal",
        fontsize=10,
    )

    # ── Slider ────────────────────────────────────────────────────────────────
    ax_slider = fig.add_axes([0.12, 0.09, 0.76, 0.03])
    slider    = mwidgets.Slider(
        ax_slider, "Offset (s)", -offset_range, offset_range,
        valinit=0.0, valstep=0.05, color="steelblue",
    )

    def _on_slide(val):
        for line, times, center in hlines:
            line.set_ydata([_center_row(times, center, val)] * 2)
        fig.canvas.draw_idle()

    slider.on_changed(_on_slide)

    # ── Save button (ipywidgets — more reliable than mwidgets.Button in Jupyter) ─
    import ipywidgets as ipw
    from IPython.display import display as ipy_display

    save_btn = ipw.Button(
        description="Save offset",
        button_style="success",
        layout=ipw.Layout(width="160px"),
    )
    out = ipw.Output()

    def _on_save(_):
        with out:
            out.clear_output(wait=True)
            offset_sec = slider.val
            try:
                if os.path.exists(offsets_file):
                    df = pd.read_csv(offsets_file)
                else:
                    os.makedirs(os.path.dirname(offsets_file) or ".", exist_ok=True)
                    df = pd.DataFrame(columns=["footage_id", "offset_sec"])
                df = df[df["footage_id"] != footage_id]
                df = pd.concat(
                    [df, pd.DataFrame([{"footage_id": footage_id, "offset_sec": offset_sec}])],
                    ignore_index=True,
                )
                df.to_csv(offsets_file, index=False)
                save_btn.description = f"Saved {offset_sec:+.2f}s"
                save_btn.button_style = ""
                print(f"Saved: {footage_id} → {offset_sec:+.2f}s  →  {offsets_file}")
            except Exception as e:
                print(f"Error saving: {e}")

    save_btn.on_click(_on_save)

    # Keep slider and save_btn alive — without explicit references they get
    # garbage-collected after this function returns, silently breaking callbacks.
    fig._calibrate_slider  = slider
    fig._calibrate_save_btn = save_btn

    plt.show()
    ipy_display(ipw.HBox([save_btn, out]))

