"""
Per-second boundary anomaly detection and correction.

See DESIGN.md section 2.1: the raw DAS signal jumps at every UTC-second
boundary crossing (likely a Febus interrogator buffering artifact). Testing
in anomalies/check_statistics.ipynb / .py found the jump size is not
reliably predictable from context (best out-of-sample R^2 ~= 0.01-0.02), so
it must be removed structurally rather than modeled. This module is the
canonical, refactored version of that notebook's get_second_boundaries() /
get_second_crossings() and the interpolation strategy DESIGN.md prescribes;
downstream code should import from here rather than re-deriving it.

Unlike the classification notebooks (which align every frame start to a
second boundary and simply drop sample 0 of each frame), a fully-
convolutional model trained on arbitrary sliding crops cannot guarantee any
crop start is second-aligned, so the artifact must be fixed once, up front,
directly on the continuous per-site signal during cache-building
(data/cache_builder.py), not re-derived per training crop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_second_boundaries(time_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Return integer sample indices of the first sample after ``time_index``
    crosses into a new integer UTC second (i.e. where
    ``floor(t, "1s")`` differs from the previous sample's).

    This mirrors anomalies/check_statistics.ipynb's ``get_second_crossings``
    (detecting a change in the floored-to-second value between consecutive
    samples), not check_statistics.py's ``get_second_boundaries`` (exact
    equality ``t == floor(t, "1s")``) — the latter essentially never fires
    on real data, since FS=67 Hz doesn't divide 1 second evenly and no
    sample lands exactly on a whole second. Verified empirically against
    das/recorded.nc: the exact-equality variant finds 0 crossings in a
    45-minute window; this crossing-detection variant finds ~1/second as
    expected.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Timestamps of a (contiguous) DAS time axis, at the interrogator's
        native sampling rate (e.g. FS=67 Hz).

    Returns
    -------
    np.ndarray[int]
        Sorted sample indices at which the second-boundary artifact is
        expected to occur. Never includes index 0 (there is no preceding
        sample to detect a change against, or to interpolate from).
    """
    times = pd.DatetimeIndex(time_index)
    floored = times.floor("s")
    changed = np.asarray(floored[1:] != floored[:-1])
    return np.flatnonzero(changed) + 1


def interpolate_boundary_samples(
    signal: np.ndarray,
    boundary_idx: np.ndarray,
) -> np.ndarray:
    """
    Replace each detected second-boundary sample with a linear interpolation
    of its immediate time-neighbors, across all channels.

    Parameters
    ----------
    signal : np.ndarray, shape (T, C)
        Continuous per-site signal (time, channels), float32/float64.
    boundary_idx : np.ndarray[int]
        Indices from get_second_boundaries(), each expected to satisfy
        ``0 < idx < T - 1``.

    Returns
    -------
    np.ndarray, shape (T, C)
        Copy of ``signal`` with each boundary sample replaced by
        ``0.5 * (signal[idx - 1] + signal[idx + 1])``. Boundary indices that
        would fall on the first or last sample of the array are left
        untouched (no valid neighbor pair to interpolate from).
    """
    out = signal.copy()
    valid = boundary_idx[(boundary_idx > 0) & (boundary_idx < signal.shape[0] - 1)]
    out[valid, :] = 0.5 * (signal[valid - 1, :] + signal[valid + 1, :])
    return out


def boundary_mask(n_samples: int, boundary_idx: np.ndarray) -> np.ndarray:
    """
    Build a boolean mask of length ``n_samples``, True at every interpolated
    sample. Persisted alongside the cache for auditability (DESIGN.md
    section 2.1, step 3).
    """
    mask = np.zeros(n_samples, dtype=bool)
    mask[boundary_idx] = True
    return mask
