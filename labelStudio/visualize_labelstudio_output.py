"""
Visualize Label Studio annotations over the loaded DAS signal.

This script loads Label Studio JSON exports, slices the corresponding DAS window,
and overlays the labeled rectangle(s) on top of the signal heatmap.

Examples:
    python labelStudio/visualize_labelstudio_output.py --start-index 0 --count 3
    python labelStudio/visualize_labelstudio_output.py --start-index 10 --count 5 --label tram

Notes:
    - The Label Studio export is expected to contain JSON files named by task id
      in labelStudio/output/.
    - Each JSON file should contain task.data.win_start / win_end and rectanglelabels.
    - The rectangle y/height values are interpreted as a fraction of the displayed
      window duration, matching the Label Studio image task layout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import xdas


ROOT = Path(__file__).resolve().parents[1]
DAS_FILE = ROOT / "das" / "recorded.nc"
LABELSTUDIO_OUTPUT_DIR = ROOT / "labelStudio" / "output"
LOCATIONS_CSV = ROOT / "labeling" / "locations.csv"
DEFAULT_VISUALIZATION_DIR = ROOT / "labelStudio" / "visualizations"


@dataclass
class AnnotationExample:
    file_path: Path
    task_id: int | str | None
    location: str
    direction: str | None
    win_start: pd.Timestamp
    win_end: pd.Timestamp
    rectangles: list[dict]


def load_locations(locations_csv: Path) -> pd.DataFrame:
    locations = pd.read_csv(locations_csv, skipinitialspace=True)
    locations.columns = locations.columns.str.strip()
    if "id" not in locations.columns:
        raise ValueError(f"Expected an 'id' column in {locations_csv}")
    return locations.set_index("id")


def as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_labelstudio_examples(output_dir: Path, label_filter: str | None = None) -> list[AnnotationExample]:
    examples: list[AnnotationExample] = []

    for file_path in sorted(p for p in output_dir.iterdir() if p.is_file()):
        try:
            annotation = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        task_data = annotation.get("task", {}).get("data", {})
        win_start = pd.to_datetime(task_data.get("win_start"), errors="coerce")
        win_end = pd.to_datetime(task_data.get("win_end"), errors="coerce")
        location = task_data.get("location")
        direction = task_data.get("direction")

        if pd.isna(win_start) or pd.isna(win_end) or win_end <= win_start or not location:
            continue

        rectangles: list[dict] = []
        for result in annotation.get("result", []):
            if result.get("type") != "rectanglelabels":
                continue
            value = result.get("value", {})
            labels = value.get("rectanglelabels", [])
            if not labels:
                continue
            raw_label = str(labels[0]).strip()
            if label_filter and raw_label.lower() != label_filter.lower():
                continue

            rectangles.append(
                {
                    "label": raw_label,
                    "y": as_float(value.get("y")),
                    "height": as_float(value.get("height")),
                    "origin": result.get("origin", "unknown"),
                    "original_width": as_int(value.get("original_width")),
                    "original_height": as_int(value.get("original_height")),
                }
            )

        if not rectangles:
            continue

        examples.append(
            AnnotationExample(
                file_path=file_path,
                task_id=task_data.get("event_id", annotation.get("id")),
                location=location,
                direction=direction,
                win_start=win_start,
                win_end=win_end,
                rectangles=rectangles,
            )
        )

    return examples


def load_das_window(das, location_bounds: tuple[float, float], win_start: pd.Timestamp, win_end: pd.Timestamp):
    dist_start, dist_end = location_bounds
    da = das.sel(
        time=slice(win_start.isoformat(), win_end.isoformat()),
        distance=slice(float(dist_start), float(dist_end)),
    )
    if da.size == 0:
        raise ValueError("Selected DAS slice is empty")
    return da


def rectangle_to_time_span(rect: dict, win_start: pd.Timestamp, win_end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    window_sec = (win_end - win_start).total_seconds()
    start_frac = np.clip(rect["y"] / 100.0, 0.0, 1.0)
    end_frac = np.clip((rect["y"] + rect["height"]) / 100.0, 0.0, 1.0)
    start = win_start + pd.to_timedelta(start_frac * window_sec, unit="s")
    end = win_start + pd.to_timedelta(end_frac * window_sec, unit="s")
    return start, end


def make_example_figure(da, example: AnnotationExample, label_bounds: tuple[float, float], scale_time: float = 3.0):
    arr = da.values
    finite = arr[np.isfinite(arr)]
    vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0

    times = pd.to_datetime(da.coords["time"].values)
    distances = da.coords["distance"].values
    if len(times) < 2:
        raise ValueError("Not enough time samples to plot")

    fig = px.imshow(
        arr,
        aspect="auto",
        labels=dict(x="Distance", y="Time", color="Strain"),
        x=distances,
        y=times,
        color_continuous_scale="Viridis",
        zmin=-vmax,
        zmax=vmax,
    )

    fig.update_layout(
        title=f"{example.location} | task {example.task_id} | {example.win_start.strftime('%H:%M:%S')}",
        autosize=True,
        height=1000,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis=dict(scaleanchor="x", scaleratio=scale_time),
    )

    dist_start, dist_end = label_bounds
    for rect in example.rectangles:
        t0, t1 = rectangle_to_time_span(rect, example.win_start, example.win_end)
        fig.add_shape(
            type="rect",
            x0=dist_start,
            x1=dist_end,
            y0=t0,
            y1=t1,
            line=dict(color="yellow", width=3),
            fillcolor="rgba(255, 255, 0, 0.12)",
        )
        fig.add_annotation(
            x=dist_start,
            y=t1,
            text=f"{rect['label']} ({rect['origin']})",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="yellow", size=12),
            bgcolor="rgba(0, 0, 0, 0.5)",
        )

    fig.update_yaxes(range=[times[0], times[-1]])
    fig.update_xaxes(range=[min(dist_start, dist_end), max(dist_start, dist_end)])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay Label Studio rectangles on DAS signal windows")
    parser.add_argument("--das", default=str(DAS_FILE), help="Path to recorded.nc")
    parser.add_argument("--output-dir", default=str(LABELSTUDIO_OUTPUT_DIR), help="Path to Label Studio JSON export folder")
    parser.add_argument("--locations", default=str(LOCATIONS_CSV), help="Path to locations.csv")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this example index")
    parser.add_argument("--count", type=int, default=5, help="Number of examples to display")
    parser.add_argument("--label", default=None, help="Optional label filter, e.g. tram")
    parser.add_argument("--time-margin-sec", type=float, default=0.0, help="Extra DAS context around the Label Studio window")
    parser.add_argument("--scale-time", type=float, default=3.0, help="Vertical scaling factor for the time axis")
    parser.add_argument("--save-dir", default=str(DEFAULT_VISUALIZATION_DIR), help="Directory where HTML plots will be saved")
    parser.add_argument("--show", action="store_true", help="Open each plot interactively after saving it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    locations_csv = Path(args.locations)
    das_path = Path(args.das)

    locations = load_locations(locations_csv)
    examples = load_labelstudio_examples(output_dir, label_filter=args.label)
    visualization_dir = Path(args.save_dir)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    if not examples:
        raise RuntimeError(f"No matching Label Studio examples found in {output_dir}")

    start = max(args.start_index, 0)
    end = min(start + max(args.count, 1), len(examples))
    subset = examples[start:end]
    if not subset:
        raise IndexError(f"Example range [{start}, {end}) is empty; {len(examples)} examples available")

    print(f"Loaded {len(examples)} examples; showing {len(subset)} starting at index {start}")

    das = xdas.open_dataarray(str(das_path))

    for example in subset:
        if example.location not in locations.index:
            print(f"Skipping {example.file_path.name}: {example.location} missing from locations.csv")
            continue

        row = locations.loc[example.location]
        label_bounds = (float(row["start"]), float(row["end"]))

        win_start = example.win_start - timedelta(seconds=args.time_margin_sec)
        win_end = example.win_end + timedelta(seconds=args.time_margin_sec)
        da = load_das_window(das, label_bounds, win_start, win_end)

        fig = make_example_figure(da, example, label_bounds, scale_time=args.scale_time)
        output_file = visualization_dir / f"{start:04d}_{example.file_path.stem}.html"
        fig.write_html(output_file, include_plotlyjs="cdn", full_html=True)
        print(f"Saved {output_file}")

        if args.show:
            fig.show()


if __name__ == "__main__":
    main()
