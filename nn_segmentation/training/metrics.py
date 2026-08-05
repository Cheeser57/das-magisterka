"""
Evaluation metrics, chosen for comparability with the existing classical
baselines (xgboost_baseline.ipynb, co_training.ipynb).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def framewise_macro_f1(
    pred: np.ndarray,
    target: np.ndarray,
    labeled_mask: np.ndarray,
    n_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """
    Per-class and macro F1 over labeled_mask=True timesteps only.

    Parameters
    ----------
    pred, target : np.ndarray, shape (T,) int class ids
    labeled_mask : np.ndarray, shape (T,) bool
    class_names : sequence of str, optional
        Names for class ids 0..n_classes-1 (e.g. base_config.class_names —
        ("background", "tram", "bus", "truck", "traffic")), used to key the
        per-class results as "f1_<name>" instead of "f1_class_<id>" so W&B
        charts/tables read directly as label names. Falls back to
        "f1_class_<id>" if omitted.
    """
    pred = pred[labeled_mask]
    target = target[labeled_mask]

    def _key(c: int) -> str:
        return f"f1_{class_names[c]}" if class_names is not None else f"f1_class_{c}"

    result: dict[str, float] = {}
    if pred.size == 0:
        result["macro_f1"] = float("nan")
        return result

    per_class_f1 = np.empty(n_classes, dtype=np.float64)
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (target == c)))
        fp = int(np.sum((pred == c) & (target != c)))
        fn = int(np.sum((pred != c) & (target == c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_f1[c] = f1
        result[_key(c)] = f1

    result["macro_f1"] = float(per_class_f1.mean())
    return result


def event_level_f1(
    predicted_events: pd.DataFrame,
    ground_truth_events: pd.DataFrame,
    overlap_threshold: float = 0.2,
) -> dict[str, float]:
    """
    Event-level F1 using the same OVERLAP_THRESHOLD-style matching
    (fractional temporal overlap) as xgboost_baseline.ipynb /
    co_training.ipynb, so segmentation results are directly comparable to
    those baselines rather than only to a different, incompatible metric.

    Parameters
    ----------
    predicted_events, ground_truth_events : pd.DataFrame
        Columns: location, class, time_start, time_end. Predicted events are
        derived from the dense per-timestep output by run-length-encoding
        contiguous same-class regions (see utils/visualize.py for the
        matching decode-for-display logic).
    """
    raise NotImplementedError
