"""
Merge annotation classes across the labeling pipeline (default: bus -> truck).

Modifies, after taking a timestamped backup:
  - labelStudio/output/*.json          (rectanglelabels values)
  - labelStudio/input/label_config.xml (drops the merged-away <Label> entry)
  - labeling/log*.csv                  (the 'class' column, where present)

Usage
-----
    python labelStudio/merge_classes.py            # dry run: report only, no writes
    python labelStudio/merge_classes.py --apply     # back up, then write changes

This is a one-time fix for already-collected data. For future runs, the
class merge is also applied at the source in tram_detector/logger.py (see
CLASS_MERGE_MAP there), so newly logged events never produce a "bus" class
regardless of what track_classes/traffic_class_list a caller passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

MERGE_MAP = {"bus": "truck"}

REPO_ROOT = Path(__file__).resolve().parent.parent
LS_OUTPUT_DIR = REPO_ROOT / "labelStudio" / "output"
LABEL_CONFIG_PATH = REPO_ROOT / "labelStudio" / "input" / "label_config.xml"
LOG_CSV_GLOB = "labeling/log*.csv"


def _backup(path: Path, backup_root: Path) -> None:
    """Copy a file or directory into backup_root, preserving its path relative to REPO_ROOT."""
    rel = path.relative_to(REPO_ROOT)
    dest = backup_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, dest)
    else:
        shutil.copy2(path, dest)


def merge_labelstudio_output(apply: bool) -> int:
    """Remap rectanglelabels values in every labelStudio/output/*.json export. Returns boxes changed."""
    if not LS_OUTPUT_DIR.exists():
        print(f"  [skip] {LS_OUTPUT_DIR} does not exist")
        return 0

    n_changed = 0
    for fpath in sorted(LS_OUTPUT_DIR.iterdir()):
        if not fpath.is_file():
            continue
        try:
            ann = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [skip] {fpath.name}: {e}")
            continue

        file_changed = False
        for box in ann.get("result", []):
            if box.get("type") != "rectanglelabels":
                continue
            labels = box.get("value", {}).get("rectanglelabels", [])
            new_labels = [MERGE_MAP.get(lbl, lbl) for lbl in labels]
            if new_labels != labels:
                box["value"]["rectanglelabels"] = new_labels
                n_changed += 1
                file_changed = True

        if file_changed and apply:
            fpath.write_text(json.dumps(ann, indent=2, ensure_ascii=False), encoding="utf-8")

    return n_changed


def merge_label_config(apply: bool) -> bool:
    """
    Drop <Label value="..."/> entries for merged-away classes from
    label_config.xml. Line-based (not an XML round-trip) to avoid
    reformatting the rest of the file.
    """
    if not LABEL_CONFIG_PATH.exists():
        print(f"  [skip] {LABEL_CONFIG_PATH} does not exist")
        return False

    lines = LABEL_CONFIG_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    merged_away = set(MERGE_MAP.keys())

    kept_lines = []
    changed = False
    for line in lines:
        stripped = line.strip()
        is_merged_label = stripped.startswith("<Label ") and any(
            f'value="{value}"' in stripped for value in merged_away
        )
        if is_merged_label:
            changed = True
            continue
        kept_lines.append(line)

    if changed and apply:
        LABEL_CONFIG_PATH.write_text("".join(kept_lines), encoding="utf-8")

    return changed


def merge_log_csvs(apply: bool) -> dict[str, int]:
    """Remap the 'class' column in every labeling/log*.csv that has one. Returns {filename: n_rows_changed}."""
    results: dict[str, int] = {}
    for fpath in sorted(REPO_ROOT.glob(LOG_CSV_GLOB)):
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  [skip] {fpath.name}: {e}")
            continue
        if "class" not in df.columns:
            continue

        n_changed = int(df["class"].isin(MERGE_MAP.keys()).sum())
        if n_changed == 0:
            continue
        results[fpath.name] = n_changed
        if apply:
            df["class"] = df["class"].replace(MERGE_MAP)
            df.to_csv(fpath, index=False)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge annotation classes (default: bus -> truck)")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run / report only)")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup; only takes effect with --apply")
    args = parser.parse_args()

    print(f"Merge map: {MERGE_MAP}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN (pass --apply to write changes)'}\n")

    if args.apply and not args.no_backup:
        backup_root = REPO_ROOT / f"backup_class_merge_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"Backing up to {backup_root} ...")
        if LS_OUTPUT_DIR.exists():
            _backup(LS_OUTPUT_DIR, backup_root)
        if LABEL_CONFIG_PATH.exists():
            _backup(LABEL_CONFIG_PATH, backup_root)
        for fpath in REPO_ROOT.glob(LOG_CSV_GLOB):
            _backup(fpath, backup_root)
        print("Backup complete.\n")

    n_boxes = merge_labelstudio_output(args.apply)
    verb = "changed" if args.apply else "would change"
    print(f"labelStudio/output/*.json: {n_boxes} rectanglelabels box(es) {verb}")

    config_changed = merge_label_config(args.apply)
    if config_changed:
        print(f"labelStudio/input/label_config.xml: {'updated' if args.apply else 'would update'}")
    else:
        print("labelStudio/input/label_config.xml: no change needed")

    csv_results = merge_log_csvs(args.apply)
    if csv_results:
        for fname, n in csv_results.items():
            print(f"  {fname}: {n} row(s) {verb}")
    else:
        print("labeling/log*.csv: no matching rows found")

    if not args.apply:
        print("\nDry run only -- re-run with --apply to write changes (a backup will be made first).")


if __name__ == "__main__":
    main()
