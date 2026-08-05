"""
Generate Label Studio image tasks from DAS data and event timestamps.

Notebook usage:
    from labelStudio.generate_label_studio import generate_tasks
    generate_tasks(das_data=das_data)

CLI usage (from project root):
    python labelStudio/generate_label_studio.py

Offsets saved by offset_calibration.calibrate() are applied automatically.

Label Studio setup (after running):
    pip install label-studio
    set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:\\
    label-studio start
"""

import json
import os
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from labelStudio.filter_display_lab import DOMAINS, DISPLAY_MODES, chain_filters
except ImportError:
    from filter_display_lab import DOMAINS, DISPLAY_MODES, chain_filters

# ── Defaults anchored to project root ────────────────────────────────────────
_ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DAS_FILE      = os.path.join(_ROOT, "das", "recorded.nc")
_LOG_CSV       = os.path.join(_ROOT, "labeling", "log2.csv")
_LOCATIONS_CSV = os.path.join(_ROOT, "labeling", "locations.csv")
_OFFSETS_FILE  = os.path.join(_ROOT, "labeing", "footage_offsets.csv")
_INPUT_DIR     = os.path.join(_ROOT, "labelStudio", "input")

_LABEL_COLORS = ["#FF8800", "#0099FF", "#00CC66", "#FF3366", "#9933FF", "#FF6666"]


def _make_label_config(classes):
    labels = "\n".join(
        f'    <Label value="{cls}" background="{_LABEL_COLORS[i % len(_LABEL_COLORS)]}"/>'
        for i, cls in enumerate(classes)
    )
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        "</View>"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_tasks(
    das_data      = None,        # xarray DataArray; loaded from das_file if None
    das_file      = _DAS_FILE,
    log_path      = _LOG_CSV,
    locations_csv = _LOCATIONS_CSV,
    offsets_file  = _OFFSETS_FILE,
    output_dir    = _INPUT_DIR,
    clear_input   = False,       # delete images/ and tasks.json before regenerating
    time_margin   = 90,          # seconds of context around each event
    px_per_channel = 4,
    px_per_sample  = 0.1,
    max_img_dim    = 4000,
    colormap       = "RdBu_r",
    domain         = "strain",
    steps          = None,
    display_mode   = "global_p99",
    locations      = None,
):
    """
    Generate Label Studio PNG images and tasks.json for all events in log_path.

    Offsets from offsets_file (written by offset_calibration.calibrate) are
    applied automatically to time_start / time_end before slicing DAS data.

    Parameters
    ----------
    das_data      : xarray DataArray or None — pass the already-loaded array to
                    skip reloading (recommended from notebooks).
    das_file      : path to recorded.nc (used only when das_data is None).
    log_path      : path to events CSV with time_start, time_end columns.
    locations_csv : path to locations CSV.
    offsets_file  : path to footage_offsets.csv produced by calibrate().
    output_dir    : directory where images/ and tasks.json are written.
    time_margin   : seconds of DAS context shown before/after each event.
    domain        : "strain" (raw, default) or "strain_rate" (differentiated
                    first) — see labelStudio.filter_display_lab.DOMAINS.
    steps         : optional filter chain applied before rendering, e.g.
                    [("median", {"t_kernel": 9, "d_kernel": 3})] — same format
                    as labelStudio.filter_display_lab.chain_filters/pipeline_view.
    display_mode  : normalization applied before imshow — see
                    labelStudio.filter_display_lab.DISPLAY_MODES.
    locations     : optional list of location names — restrict to only these
                    rows of log_path (e.g. locations=["pcss"]). None (default)
                    processes every location present in the log.
    """
    import xdas

    images_dir = os.path.join(output_dir, "images")
    tasks_file = os.path.join(output_dir, "tasks.json")
    config_file = os.path.join(output_dir, "label_config.xml")

    if clear_input:
        import shutil
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
            print(f"Cleared {images_dir}")
        for f in (tasks_file, config_file):
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed {f}")

    Path(images_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if das_data is None:
        print("Loading DAS data …")
        das_data = xdas.open_dataarray(das_file)
        print(f"  {das_data}\n")

    log = pd.read_csv(log_path, parse_dates=["time_start", "time_end"])
    locs = pd.read_csv(locations_csv, skipinitialspace=True)
    locs.columns = locs.columns.str.strip()
    locs = locs.set_index("id")

    # Label set is always derived from the FULL log, even when `locations`
    # restricts which rows get (re)rendered below -- so label_config.xml
    # never narrows just because one location was refreshed in isolation.
    class_col = "class" if "class" in log.columns else None
    classes   = sorted(log[class_col].dropna().unique()) if class_col else ["object"]
    label_config_xml = _make_label_config(classes)
    print(f"Labels: {classes}")

    if locations is not None:
        log = log[log["location"].isin(locations)]
        print(f"Restricting to location(s) {list(locations)}: {len(log)} event(s)")

    # ── Load and index offsets ────────────────────────────────────────────────
    offsets = {}
    if os.path.exists(offsets_file):
        off_df = pd.read_csv(offsets_file)
        offsets = dict(zip(off_df["footage_id"], off_df["offset_sec"]))
        print(f"Loaded offsets for {len(offsets)} footage(s): {list(offsets.keys())}")
    else:
        print("No offsets file found — timestamps used as-is.")

    # ── Generate tasks ────────────────────────────────────────────────────────
    margin = timedelta(seconds=time_margin)
    tasks  = []

    for idx, row in log.iterrows():
        loc = row["location"]
        if loc not in locs.index:
            print(f"  [SKIP] row {idx}: location '{loc}' not in locations.csv")
            continue

        # Apply calibration offset for this footage
        offset_sec = 0.0
        if "footage_id" in row.index and pd.notna(row["footage_id"]):
            offset_sec = offsets.get(row["footage_id"], 0.0)
        shift = timedelta(seconds=offset_sec)
        t_start = row["time_start"] + shift
        t_end   = row["time_end"]   + shift

        dist_start = locs.loc[loc, "start"]
        dist_end   = locs.loc[loc, "end"]
        win_start  = t_start - margin
        win_end    = t_end   + margin

        img_name = f"event_{idx:03d}_{loc}_{row['time_start'].strftime('%H%M%S')}.png"
        img_path = os.path.join(images_dir, img_name)

        try:
            shape = _make_image(
                das_data, win_start, win_end, dist_start, dist_end, img_path,
                colormap, domain, steps, display_mode, px_per_channel, px_per_sample, max_img_dim,
            )
            offset_tag = f"  (offset {offset_sec:+.2f}s)" if offset_sec else ""
            print(f"  [{idx:02d}] {img_name}  ({shape[1]}×{shape[0]}){offset_tag}")
        except Exception as e:
            print(f"  [ERROR] row {idx}: {e}")
            continue

        cls = row[class_col] if class_col and pd.notna(row.get(class_col)) else "object"
        tasks.append(_build_task(idx, row, t_start, t_end, img_path, win_start, win_end, cls))

    # ── Merge into existing tasks.json when only some locations were (re)rendered ──
    # so refreshing e.g. locations=["pcss"] updates just those images/entries
    # without dropping other locations' tasks already present in the file.
    if locations is not None and os.path.exists(tasks_file):
        with open(tasks_file) as f:
            existing_tasks = json.load(f)
        refreshed_ids = {t["data"]["event_id"] for t in tasks}
        tasks = sorted(
            [t for t in existing_tasks if t["data"]["event_id"] not in refreshed_ids] + tasks,
            key=lambda t: t["data"]["event_id"],
        )

    # ── Write outputs ─────────────────────────────────────────────────────────
    with open(tasks_file, "w") as f:
        json.dump(tasks, f, indent=2)
    with open(config_file, "w") as f:
        f.write(label_config_xml)

    print(f"\n{'─'*60}")
    print(f"  {len(tasks)} tasks  →  {os.path.abspath(tasks_file)}")
    print(f"  Images         →  {os.path.abspath(images_dir)}")
    print(f"  Label config   →  {os.path.abspath(config_file)}")
    _print_setup_instructions(os.path.abspath(images_dir))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_image(das_data, win_start, win_end, dist_start, dist_end,
                img_path, colormap, domain, steps, display_mode,
                px_per_channel, px_per_sample, max_img_dim):
    da = das_data.sel(
        time=slice(win_start.isoformat(), win_end.isoformat()),
        distance=slice(float(dist_start), float(dist_end)),
    )
    base = DOMAINS[domain](da)
    filtered = chain_filters(base, steps) if steps else base
    arr = filtered.values if hasattr(filtered, "values") else np.asarray(filtered)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("No finite values in selected slice")
    shown, vmin, vmax = DISPLAY_MODES[display_mode](arr.astype(np.float64))

    n_time, n_dist = arr.shape
    img_w = min(max_img_dim, max(400, int(n_dist * px_per_channel)))
    img_h = min(max_img_dim, max(400, int(n_time * px_per_sample)))

    dpi = 100
    fig, ax = plt.subplots(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    ax.imshow(shown, aspect="auto", cmap=colormap, vmin=vmin, vmax=vmax,
              interpolation="nearest", origin="upper")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(img_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    return arr.shape


def _time_to_pct(t, win_start, win_end):
    total  = (win_end - win_start).total_seconds()
    offset = (t - win_start).total_seconds()
    return round(max(0.0, min(100.0, offset / total * 100)), 3)


def _build_task(idx, row, t_start, t_end, img_path, win_start, win_end, cls):
    y = _time_to_pct(t_start, win_start, win_end)
    h = round(_time_to_pct(t_end, win_start, win_end) - y, 3)
    abs_img = os.path.abspath(img_path).replace("\\", "/")
    return {
        "data": {
            "image":      f"/data/local-files/?d={abs_img}",
            "event_id":   idx,
            "location":   row["location"],
            "direction":  row["direction"],
            "time_start": str(t_start),
            "time_end":   str(t_end),
            "win_start":  win_start.isoformat(),
            "win_end":    win_end.isoformat(),
        },
        "predictions": [{
            "result": [{
                "value": {
                    "x": 0, "y": y, "width": 100, "height": h,
                    "rotation": 0, "rectanglelabels": [cls],
                },
                "from_name": "label",
                "to_name":   "image",
                "type":      "rectanglelabels",
            }]
        }],
    }


def _print_setup_instructions(images_dir):
    print(f"""{'─'*60}
Label Studio setup
──────────────────
1.  pip install label-studio

2.  $env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
    $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT   = "C:\\\\"

3.  label-studio start

4.  New project → Labeling Setup → Custom template
    Paste label_config.xml

5.  Settings → Cloud Storage → Local files → {images_dir}

6.  Import → tasks.json  (pre-drawn boxes appear as predictions)

7.  Export → JSON-MIN
""")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_tasks()
