"""
Weak/strong augmentation pair for FixMatch, shared building blocks for CPS.

DESIGN.md section 4.1. Strong augmentation deliberately includes synthetic
per-second boundary-jump injection, motivated directly by the anomaly
finding in anomalies/check_statistics.ipynb (DESIGN.md section 2.1) — since
the real artifact can't be regressed out, training the model to be
invariant to its presence/phase is the practical mitigation.

Operates on individual (C, T) numpy crops (applied per-sample by the
training loops in training/fixmatch.py and training/cotraining.py) rather
than inside the Dataset classes, since FixMatch and CPS each need a
different augmentation pairing of the same underlying pool crop.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from nn_segmentation.config.base import FS
from nn_segmentation.config.cotraining import CoTrainingConfig
from nn_segmentation.config.fixmatch import FixMatchConfig

# weak_augment() is shared by both SSL modes (training/fixmatch.py's
# weak/strong pair and training/cotraining.py's per-branch augmentation
# stream); both configs declare the same weak_* fields (duck-typed, not a
# shared base class — see config/cotraining.py's comment).
WeakAugmentConfig = Union[FixMatchConfig, CoTrainingConfig]

# Default synthetic jump-amplitude pool used by strong_augment() when no
# empirical anomaly_jump sample array is supplied. Crops reaching this
# function are already z-scored (data/normalization.py), so amplitudes are
# expressed in robust-z units; anomalies/check_statistics.ipynb found real
# jump magnitudes comparable to the signal's own local std, so a few sigma
# is a reasonable default range for this augmentation's purpose (training
# invariance to the artifact's presence/phase, not exact amplitude fidelity).
DEFAULT_JUMP_AMPLITUDE_DIST = np.linspace(-3.0, 3.0, 61)


def weak_augment(crop: np.ndarray, config: WeakAugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Apply the weak augmentation view: small time shift
    (config.weak_time_shift_sec), small channel-window shift
    (config.weak_channel_shift), mild Gaussian noise
    (config.weak_noise_std_frac of per-channel std).

    Parameters
    ----------
    crop : np.ndarray, shape (C, T)
        A single normalized training crop.
    """
    out = crop.astype(np.float32, copy=True)

    max_time_shift = int(round(config.weak_time_shift_sec * FS))
    if max_time_shift > 0:
        shift = int(rng.integers(-max_time_shift, max_time_shift + 1))
        if shift != 0:
            out = np.roll(out, shift, axis=1)
            # Overwrite the wrapped-around region with the nearest real edge
            # value instead of leaving a spurious wraparound discontinuity.
            if shift > 0:
                out[:, :shift] = out[:, shift : shift + 1]
            else:
                out[:, shift:] = out[:, shift - 1 : shift]

    max_channel_shift = config.weak_channel_shift
    if max_channel_shift > 0:
        shift = int(rng.integers(-max_channel_shift, max_channel_shift + 1))
        if shift != 0:
            out = np.roll(out, shift, axis=0)
            if shift > 0:
                out[:shift, :] = out[shift : shift + 1, :]
            else:
                out[shift:, :] = out[shift - 1 : shift, :]

    if config.weak_noise_std_frac > 0:
        noise_std = config.weak_noise_std_frac * float(np.std(out))
        if noise_std > 0:
            out = out + rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    return out


def strong_augment(crop: np.ndarray, config: FixMatchConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Apply the strong augmentation view: channel-block masking, time-block
    masking (SpecAugment-style, capped fractions per config), amplitude
    scaling, and (if config.strong_inject_boundary_jump) synthetic
    boundary-jump injection via inject_boundary_jump().
    """
    out = crop.astype(np.float32, copy=True)
    n_channels, n_samples = out.shape

    channel_mask = spec_augment_mask(n_channels, config.strong_channel_mask_frac, rng)
    out[channel_mask, :] = 0.0

    time_mask = spec_augment_mask(n_samples, config.strong_time_mask_frac, rng)
    out[:, time_mask] = 0.0

    lo, hi = config.strong_amplitude_scale_range
    out = out * float(rng.uniform(lo, hi))

    if config.strong_inject_boundary_jump:
        out = inject_boundary_jump(out, FS, DEFAULT_JUMP_AMPLITUDE_DIST, rng)

    return out


def inject_boundary_jump(
    crop: np.ndarray,
    fs: float,
    jump_amplitude_dist: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Inject a synthetic per-second-boundary jump at a randomly phase-shifted
    grid, sampling jump amplitudes from ``jump_amplitude_dist`` (the
    empirical anomaly_jump distribution characterized in
    anomalies/check_statistics.ipynb — see DESIGN.md section 2.1's
    belt-and-suspenders note).

    Parameters
    ----------
    crop : np.ndarray, shape (C, T)
    fs : float
        Sampling rate, used to derive the ~1-second injection cadence.
    jump_amplitude_dist : np.ndarray
        Empirical samples of anomaly_jump to draw injected amplitudes from.
    """
    out = crop.astype(np.float32, copy=True)
    n_samples = out.shape[1]

    period = int(round(fs))
    if period <= 0 or period >= n_samples:
        return out

    phase = int(rng.integers(0, period))
    boundary_indices = np.arange(phase, n_samples, period)
    if boundary_indices.size == 0:
        return out

    # One shared amplitude per crossing, applied across all channels — the
    # real artifact is a timing/buffering effect common to the whole
    # distance window at that instant, not an independent per-channel draw.
    amplitudes = rng.choice(jump_amplitude_dist, size=boundary_indices.size).astype(np.float32)
    out[:, boundary_indices] += amplitudes[np.newaxis, :]
    return out


def spec_augment_mask(
    length: int,
    max_mask_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return a boolean mask over a single axis (time or channel) of length
    ``length``, masking one contiguous block sized up to
    ``max_mask_fraction`` of the axis. Shared helper for both the
    channel-block and time-block masks in strong_augment().
    """
    mask = np.zeros(length, dtype=bool)
    if max_mask_fraction <= 0 or length == 0:
        return mask

    max_block_len = max(1, int(round(length * max_mask_fraction)))
    block_len = int(rng.integers(1, max_block_len + 1))
    start = int(rng.integers(0, length - block_len + 1))
    mask[start : start + block_len] = True
    return mask
