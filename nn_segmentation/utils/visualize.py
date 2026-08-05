"""
Prediction-vs-ground-truth timeline visualization.

Reuses the CLASS_COLORS convention already established in
xgboost_baseline.ipynb / co_training.ipynb / trajectory_detection.ipynb for
visual continuity with the existing classical-baseline plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_COLORS: dict[str, str] = {
    "background": "#888888",
    "tram": "yellow",
    "bus": "#FF8800",
    "truck": "#00CC66",
    "traffic": "#FF0000",
}


def plot_prediction_timeline(
    time_index: pd.DatetimeIndex,
    pred: np.ndarray,
    target: np.ndarray,
    labeled_mask: np.ndarray,
    class_names: tuple[str, ...],
    out_path: str | None = None,
):
    """
    Plot predicted vs. ground-truth class over time as two stacked colored
    strips (matplotlib), shading labeled_mask=False regions to indicate
    "no camera ground truth available" rather than "confirmed background".

    Parameters
    ----------
    time_index : pd.DatetimeIndex, length T
    pred, target : np.ndarray, shape (T,) int class ids
    labeled_mask : np.ndarray, shape (T,) bool
    out_path : str, optional
        If given, saves the figure to runs/nn_segmentation/<run>/... instead
        of only returning the Figure.
    """
    raise NotImplementedError
