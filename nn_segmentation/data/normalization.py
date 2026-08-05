"""
Per-channel robust normalization statistics.

DESIGN.md section 2.3: per-channel robust z-score (median/MAD), computed on
the training range only (data/splits.py::get_temporal_split), NOT
das_loader.load_patch()'s percentile-clip-to-uint8 conversion (that path is
a lossy display normalization for Label Studio/OpenCV, unsuitable as model
input).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MAD -> std-equivalent scale factor for normally-distributed data.
_MAD_TO_STD = 1.4826


@dataclass
class ChannelNormStats:
    """Per-channel (median, scale) computed on the training range."""

    median: np.ndarray  # (C,)
    scale: np.ndarray  # (C,) robust scale, e.g. MAD * 1.4826


def compute_channel_stats(
    signal: np.ndarray,
    train_mask: np.ndarray,
) -> ChannelNormStats:
    """
    Compute per-channel median and robust scale from ``signal[train_mask]``.

    Parameters
    ----------
    signal : np.ndarray, shape (T, C)
        Continuous per-site signal, already boundary-interpolated
        (data/anomaly.py).
    train_mask : np.ndarray, shape (T,) bool
        True for samples inside the training range
        (data/splits.py::in_train_range) — statistics must never see test
        samples.
    """
    train_signal = signal[train_mask]
    if train_signal.shape[0] == 0:
        raise ValueError("train_mask selects zero samples; cannot compute normalization stats.")

    median = np.median(train_signal, axis=0)
    mad = np.median(np.abs(train_signal - median), axis=0)
    scale = mad * _MAD_TO_STD
    # Guard against a dead/constant channel producing a zero scale.
    scale = np.where(scale > 1e-8, scale, 1.0)
    return ChannelNormStats(median=median.astype(np.float32), scale=scale.astype(np.float32))


def apply_zscore(
    signal: np.ndarray,
    stats: ChannelNormStats,
    clip: float | None = 8.0,
    in_place: bool = False,
) -> np.ndarray:
    """
    Apply ``(signal - median) / scale`` per channel, optionally clipping to
    ``[-clip, clip]`` to bound extreme-outlier influence (not the primary
    normalization mechanism, see DESIGN.md section 2.3).

    ``in_place=True`` mutates ``signal`` directly (requires ``signal`` to
    already be float32) instead of allocating a full second array — halves
    peak memory for a whole-recording per-site signal (tens of GB for wide
    sites, see cache_builder.py), at the cost of the caller's array being
    overwritten. Only safe when the caller owns ``signal`` outright and
    doesn't need the pre-normalization values afterward (true for every
    current caller — see data/cache_builder.py::load_and_normalize_sites()).
    """
    if in_place:
        if signal.dtype != np.float32:
            raise ValueError(f"in_place=True requires float32 signal, got {signal.dtype}")
        signal -= stats.median
        signal /= stats.scale
        if clip is not None:
            np.clip(signal, -clip, clip, out=signal)
        return signal

    z = (signal - stats.median) / stats.scale
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z.astype(np.float32)
