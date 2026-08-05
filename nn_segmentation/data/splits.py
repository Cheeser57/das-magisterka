"""
Single source of truth for the chronological train/test split.

DESIGN.md section 4.4: every consumer of a per-site timeline (normalization
statistics, supervised crop sampling, and — critically — unlabeled pool
sampling for FixMatch/CPS) MUST derive its train/test boundary from
get_temporal_split() rather than recomputing it independently. This closes
the leakage gap that co_training.ipynb's pool construction only avoided
implicitly (by happening to slice train_df before sampling).

Split logic mirrors co_training.ipynb exactly:
    all_events_sorted = events.sort_values("time_start")
    n_test_events     = max(1, int(len(all_events_sorted) * (1 - train_frac)))
    split_time        = all_events_sorted.iloc[-n_test_events]["time_start"]
    train = time < split_time ; test = time >= split_time
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Half-open time ranges; test_range always follows train_range."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def get_temporal_split(
    time_index: pd.DatetimeIndex,
    event_times: pd.Series,
    train_frac: float = 0.70,
) -> TemporalSplit:
    """
    Compute the chronological train/test cutoff for one site.

    Mirrors xgboost_baseline.ipynb / co_training.ipynb's approach: sort
    known events by time_start, and let the last ``(1 - train_frac)``
    fraction of events (not of raw samples) define the test window, so a
    single physical pass-by is never split across train and test.

    If ``event_times`` is empty (no labels available locally — see
    DESIGN.md's "handling absent labels gracefully" requirement), falls
    back to splitting the raw time axis directly at ``train_frac`` rather
    than raising, so the pool-only/unsupervised path can still run.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Full continuous time axis for the site (defines train_start/test_end
        bounds).
    event_times : pd.Series
        Chronologically sortable event timestamps (e.g. time_start of every
        rasterized event for this site) used to place the cutoff.
    train_frac : float
        Fraction of (chronologically sorted) events assigned to train.

    Returns
    -------
    TemporalSplit
    """
    sorted_times = pd.Series(event_times).dropna().sort_values().reset_index(drop=True)

    if len(sorted_times) == 0:
        split_idx = int(len(time_index) * train_frac)
        split_idx = min(max(split_idx, 1), len(time_index) - 1)
        split_time = time_index[split_idx]
    else:
        n_test_events = max(1, int(len(sorted_times) * (1 - train_frac)))
        split_time = sorted_times.iloc[-n_test_events]

    return TemporalSplit(
        train_start=time_index[0],
        train_end=split_time,
        test_start=split_time,
        test_end=time_index[-1],
    )


def in_train_range(timestamps, split: TemporalSplit) -> np.ndarray:
    """Boolean mask, True where ``timestamps`` fall inside ``split``'s train range."""
    ts = pd.DatetimeIndex(timestamps)
    return np.asarray((ts >= split.train_start) & (ts < split.train_end))


def in_test_range(timestamps, split: TemporalSplit) -> np.ndarray:
    """Boolean mask, True where ``timestamps`` fall inside ``split``'s test range."""
    ts = pd.DatetimeIndex(timestamps)
    return np.asarray((ts >= split.test_start) & (ts <= split.test_end))
