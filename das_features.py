"""
das_features.py
================
Shared classical feature-extraction and co-training utilities for DAS
vehicle-detection frame classification.

Consolidates logic that was previously pasted independently into
xgboost_baseline.ipynb and co_training.ipynb -- that duplication is what let
each notebook drift onto its own stale, hardcoded distance-channel ranges
(neither matching labeling/locations.csv) instead of a single source of
truth. classical_baseline.ipynb imports from here and from das_loader.py /
nn_segmentation.data.raster_labels for the unified 3-class
(background/tram/truck) pipeline across all 5 sites.

Feature set expanded from the original 26 (4 bands x 5 stats + 4 ratios +
tilt + spatial_coherence) based on a literature review of papers/ (35 DAS
papers) -- see classical_baseline.ipynb's discussion section for citations.
New: a quasi-static band (weight proxy), multi-lag spatial coherence,
apparent-velocity/moveout across channels, spatial footprint, time-domain
shape factors, and frequency-domain spectral-shape statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.stats import kurtosis, skew
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

import das_loader

FS = das_loader.FS  # 67.0 Hz
DX = das_loader.DX  # 0.8 m per channel
V_MIN_MS = das_loader.V_MIN_MS  # 5.0 -- slowest expected vehicle
V_MAX_MS = das_loader.V_MAX_MS  # 100.0 -- fastest expected vehicle

# bus was merged into truck repo-wide (labelStudio/merge_classes.py,
# tram_detector/logger.py CLASS_MERGE_MAP), but at least one straggler
# ("bus" in labelStudio/output/208) escaped the retroactive merge -- applied
# defensively here rather than trusting every data source is already clean.
CLASS_MERGE_MAP = {"bus": "truck"}

BACKGROUND_CLASS = "nothing"
VEHICLE_CLASSES = ("tram", "truck")

# 5 bands (was 4): "qs" (quasi-static, <1Hz) added as a vehicle-weight proxy
# -- Liu et al. (TelecomTM) and Cohen et al. 2025's Flamant-Boussinesq
# derivation both point to sub-1Hz strain as the cleanest physical weight
# signal, which the original 4 bands (starting at 1Hz) missed entirely.
BANDS = {
    "qs":   (0.1, 1.0),
    "lf":   (1.0, 5.0),
    "mid1": (5.0, 15.0),
    "mid2": (15.0, 28.0),
    "hf":   (28.0, 33.0),  # capped below Nyquist (33.5 Hz at FS=67)
}

# Multi-lag spatial coherence (generalizes the original single adjacent-pair
# Pearson correlation) -- Shi & Zong 2025's cross-spectral density matrix
# and Ye et al. 2023's semblance both motivate looking beyond lag=1.
COHERENCE_LAGS = (1, 2, 4)

OVERLAP_THRESHOLD = 0.2  # unchanged from both original notebooks

# ── Co-training pseudo-label thresholds -- unchanged from co_training.ipynb ──
CT_CONF_VEH = 0.60
CT_CONF_NEG = 0.97
CT_CONF_MARGIN = 0.15
CT_MAX_ITER = 10
CT_MAX_VEH_PER_ITER = 30
CT_MAX_NEG_PER_ITER = 100
MAX_POOL_SIZE = 20_000

# New: physically-motivated pseudo-label acceptance gate (see select_vehicles
# docstring). No literature-given number exists for this repo's channel
# spacing/geometry -- classical_baseline.ipynb includes a small sensitivity
# sweep rather than asserting this value blind.
CT_MOVEOUT_R2_MIN = 0.5


def remap_classes(df: pd.DataFrame, col: str = "class") -> pd.DataFrame:
    """Apply CLASS_MERGE_MAP defensively to a class-label column."""
    df = df.copy()
    df[col] = df[col].replace(CLASS_MERGE_MAP)
    return df


# Schema-compatible with nn_segmentation.data.raster_labels' internal event
# columns, so the result can be passed directly to raster_labels.build_site_timeline().
EVENT_COLUMNS = ["location", "class", "time_start", "time_end", "source", "footage_id", "win_start", "win_end"]


def load_labelstudio_events(label_studio_output: str = das_loader.LS_OUTPUT) -> pd.DataFrame:
    """
    Label Studio annotations ONLY (das_loader.load_annotations()) -- does
    NOT include labeling/log*.csv events.

    labeling/log*.csv rows are auto-detected (YOLO) candidates whose only
    intended purpose is seeding Label Studio task-image generation
    (labelStudio/generate_label_studio.py); they are not verified ground
    truth. Per-site counts have diverged substantially from the reviewed
    Label Studio output since (e.g. pcss: 372 log rows vs. 195 reviewed
    boxes -- many still unreviewed or rejected as false positives;
    most_mieszka: 62 vs. 134 -- reviewers added events the auto-detector
    missed). Treating both as equally valid labels (as
    nn_segmentation.data.raster_labels.load_all_events does, for that
    package's different full-timeline-coverage tradeoff) would risk
    double-counting reviewed events and treating unreviewed/rejected
    candidates as confirmed. Use this function instead of
    raster_labels.load_all_events() wherever CSV-log rows should not be
    treated as ground truth.

    The returned "source" column is always "labelstudio" and "footage_id"
    is always None -- when passed to raster_labels.build_site_timeline(),
    labeled_mask is therefore derived purely from each annotation's own
    win_start/win_end (the exact reviewed image span), never from a
    CSV-log footage-coverage window.
    """
    ann = das_loader.load_annotations(label_studio_output)
    if ann.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    ann = ann.copy()
    ann["source"] = "labelstudio"
    ann["footage_id"] = None
    return ann[EVENT_COLUMNS]


# ── Bandpass / framing (unchanged from co_training.ipynb) ───────────────────

def bandpass_all_channels(data_tc: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """data_tc: (time, channels) -> bandpass filtered (time, channels)."""
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.999)  # guard: never reach or exceed Nyquist
    if lo >= hi:
        return np.zeros_like(data_tc)
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, data_tc, axis=0).astype(np.float32)


def get_frame_starts(time_values, frame_samples: int, fs: float = FS):
    """Returns (start_indices, timestamps) aligned to frame-length boundaries."""
    frame_sec = frame_samples // fs
    times = pd.to_datetime(time_values)
    floored = times.floor(f"{int(frame_sec)}s")
    starts, timestamps = [], []
    prev = None
    for i, (t, tf) in enumerate(zip(times, floored)):
        if tf != prev:
            if i + frame_samples <= len(times):
                starts.append(i)
                timestamps.append(tf)
            prev = tf
    return np.array(starts), timestamps


def rasterize_labels_to_frames(
    label_ids: np.ndarray,
    labeled_mask: np.ndarray,
    starts: np.ndarray,
    frame_samples: int,
    class_names,
    overlap_threshold: float = OVERLAP_THRESHOLD,
):
    """
    Convert nn_segmentation.data.raster_labels.build_site_timeline()'s
    per-SAMPLE label/labeled_mask arrays into per-FRAME labels, using the
    same overlap-threshold convention the original notebooks applied
    per-event: a frame is assigned a vehicle class if that class covers
    more than `overlap_threshold` of the frame's samples (the
    highest-coverage vehicle class wins ties), else the frame gets
    class_names[0] (background/"nothing"). A frame's labeled_mask is True
    if a majority of its samples are footage-confirmed.

    Parameters
    ----------
    label_ids : np.ndarray[int], shape (n_samples,) -- index into class_names
    labeled_mask : np.ndarray[bool], shape (n_samples,)
    starts : np.ndarray[int] -- frame start sample indices (get_frame_starts)
    class_names : sequence, class_names[0] MUST be the background class

    Returns
    -------
    frame_labels : np.ndarray[object], shape (len(starts),)
    frame_labeled_mask : np.ndarray[bool], shape (len(starts),)
    """
    n_classes = len(class_names)
    frame_labels = np.empty(len(starts), dtype=object)
    frame_labeled_mask = np.zeros(len(starts), dtype=bool)

    for i, s in enumerate(starts):
        window = label_ids[s:s + frame_samples]
        mask_window = labeled_mask[s:s + frame_samples]
        frame_labeled_mask[i] = mask_window.mean() > 0.5

        counts = np.bincount(window, minlength=n_classes)
        fracs = counts / len(window)
        if n_classes > 1:
            best_cls = int(np.argmax(fracs[1:])) + 1
            frame_labels[i] = class_names[best_cls] if fracs[best_cls] > overlap_threshold else class_names[0]
        else:
            frame_labels[i] = class_names[0]

    return frame_labels, frame_labeled_mask


# ── New feature functions ────────────────────────────────────────────────────

def estimate_moveout(all_bands: np.ndarray, dx: float = DX, fs: float = FS, min_channels: int = 4):
    """
    Apparent velocity ("moveout") across channels within each frame --
    the most-repeated multi-channel technique across the literature review
    (Chambers 2020, Litzenberger 2021, Ye et al. 2023, Xie et al. 2025,
    Yi et al. 2026): a real vehicle produces a peak-amplitude ridge that
    moves linearly across channels over time; its slope is the vehicle's
    speed.

    For each frame: find the peak-|amplitude| time index per channel, keep
    channels whose peak amplitude exceeds the frame's own median (self-
    normalizing -- no absolute noise floor needed), and fit a line
    (channel position [m] vs. peak time [s]) via least squares. The fitted
    slope is the apparent velocity [m/s]; R^2 measures how well the peaks
    line up on one moving trajectory (a real vehicle) vs. scattered noise --
    also used as a pseudo-label acceptance gate, see select_vehicles().

    Parameters
    ----------
    all_bands : np.ndarray, shape (n_frames, n_samples, n_channels)

    Returns
    -------
    velocity, r2 : np.ndarray, shape (n_frames,)
        Frames with fewer than `min_channels` above-median channels get
        velocity=0, r2=0 (no coherent moving object detected).
    """
    n_frames, n_samples, n_channels = all_bands.shape
    velocity = np.zeros(n_frames, dtype=np.float32)
    r2 = np.zeros(n_frames, dtype=np.float32)

    abs_sig = np.abs(all_bands)
    peak_t_idx = np.argmax(abs_sig, axis=1)  # (n_frames, n_channels)
    peak_amp = np.max(abs_sig, axis=1)  # (n_frames, n_channels)
    channel_pos = np.arange(n_channels) * dx

    for f in range(n_frames):
        amp = peak_amp[f]
        sel = amp > np.median(amp)
        if sel.sum() < min_channels:
            continue
        t_sel = peak_t_idx[f, sel] / fs
        x_sel = channel_pos[sel]
        A = np.vstack([t_sel, np.ones_like(t_sel)]).T
        coef, *_ = np.linalg.lstsq(A, x_sel, rcond=None)
        pred = A @ coef
        ss_res = np.sum((x_sel - pred) ** 2)
        ss_tot = np.sum((x_sel - x_sel.mean()) ** 2)
        velocity[f] = coef[0]
        r2[f] = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return velocity, r2


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int((ends - starts).max())


def spatial_footprint(all_bands: np.ndarray, threshold_frac: float = 0.5) -> np.ndarray:
    """
    Count of contiguous channels exceeding threshold_frac of the frame's
    peak channel energy, at that frame's own peak-energy time index -- a
    cheap proxy for disturbance length along the fiber (vehicle length/mass
    proxy; Chiang et al. 2022, Chambers 2020).
    """
    n_frames = all_bands.shape[0]
    energy = all_bands ** 2
    frame_energy_profile = energy.sum(axis=2)  # (n_frames, n_samples)
    peak_t = np.argmax(frame_energy_profile, axis=1)

    footprint = np.zeros(n_frames, dtype=np.float32)
    for f in range(n_frames):
        row = energy[f, peak_t[f], :]
        row_max = row.max()
        if row_max <= 0:
            continue
        active = row > threshold_frac * row_max
        footprint[f] = _longest_run(active)
    return footprint


def spatial_coherence_multilag(all_bands: np.ndarray, lags=COHERENCE_LAGS) -> dict:
    """
    Multi-lag channel coherence: mean Pearson correlation between channels
    offset by `lag`, for each lag. Generalizes the original single
    adjacent-channel (lag=1) spatial_coherence feature -- Shi & Zong 2025's
    cross-spectral density matrix and Ye et al. 2023's semblance-with-QC
    both motivate looking beyond immediate neighbors.
    """
    centered = all_bands - np.mean(all_bands, axis=1, keepdims=True)
    out = {}
    for lag in lags:
        a = centered[:, :, :-lag]
        b = centered[:, :, lag:]
        num = np.sum(a * b, axis=1)
        d1 = np.sqrt(np.sum(a ** 2, axis=1)) + 1e-30
        d2 = np.sqrt(np.sum(b ** 2, axis=1)) + 1e-30
        out[f"spatial_coherence_lag{lag}"] = np.mean(num / (d1 * d2), axis=1)
    return out


def shape_factors(all_bands: np.ndarray) -> dict:
    """
    Classic vibration-analysis shape factors (Qin et al. 2025), computed per
    frame on the band-averaged signal, averaged across channels. Derived
    for near-zero marginal cost from stats already computed elsewhere
    (RMS, peak, mean).
    """
    eps = 1e-30
    rms = np.sqrt(np.mean(all_bands ** 2, axis=1))
    peak = np.max(np.abs(all_bands), axis=1)
    mean_abs = np.mean(np.abs(all_bands), axis=1)
    clearance_denom = np.mean(np.sqrt(np.abs(all_bands)), axis=1) ** 2

    return {
        "crest_factor": (peak / (rms + eps)).mean(axis=1),
        "shape_factor": (rms / (mean_abs + eps)).mean(axis=1),
        "impulse_factor": (peak / (mean_abs + eps)).mean(axis=1),
        "clearance_factor": (peak / (clearance_denom + eps)).mean(axis=1),
    }


def spectral_shape_stats(all_bands: np.ndarray, fs: float = FS) -> dict:
    """
    Spectral centroid / bandwidth / entropy / roll-off (85%) / kurtosis via
    Welch PSD, computed per-frame on the channel-mean of the band-averaged
    signal (Qin et al. 2025, Fakhruzi et al. 2025, Deng et al. 2025 review)
    -- richer than the fixed-band energy ratios alone.

    Note: computed on the band-averaged reconstruction (sum of the 5
    bandpassed copies used elsewhere in this module), not the original raw
    wideband signal, for consistency/code-reuse with the other spatial
    features -- a minor simplification (see classical_baseline.ipynb
    discussion section).
    """
    n_frames, n_samples, _ = all_bands.shape
    sig = all_bands.mean(axis=2)  # (n_frames, n_samples) -- channel-mean

    centroid = np.zeros(n_frames, dtype=np.float32)
    bandwidth = np.zeros(n_frames, dtype=np.float32)
    entropy = np.zeros(n_frames, dtype=np.float32)
    rolloff = np.zeros(n_frames, dtype=np.float32)
    spec_kurt = np.zeros(n_frames, dtype=np.float32)

    nperseg = min(n_samples, 128)
    for f in range(n_frames):
        freqs, psd = sps.welch(sig[f], fs=fs, nperseg=nperseg)
        psd = psd + 1e-30
        p = psd / psd.sum()
        centroid[f] = np.sum(freqs * p)
        bandwidth[f] = np.sqrt(np.sum(((freqs - centroid[f]) ** 2) * p))
        entropy[f] = -np.sum(p * np.log(p))
        rolloff[f] = freqs[np.searchsorted(np.cumsum(p), 0.85)]
        spec_kurt[f] = kurtosis(psd)

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_entropy": entropy,
        "spectral_rolloff": rolloff,
        "spectral_kurtosis": spec_kurt,
    }


# ── Full feature extraction ──────────────────────────────────────────────────

def extract_features_vectorized(frames_ftc: dict, band_names, dx: float = DX, fs: float = FS) -> pd.DataFrame:
    """
    frames_ftc: dict band -> (n_frames, frame_samples, n_channels)
    Returns a DataFrame with one row per frame, ~45-55 feature columns
    (expanded from the original 26 -- see module docstring).
    """
    feat_dict = {}
    band_rms_mean = {}

    for bname in band_names:
        w = frames_ftc[bname][:, 1:, :]  # skip sample-0 boundary artifact
        rms_per_ch = np.sqrt(np.mean(w ** 2, axis=1))
        rms_mean = np.mean(rms_per_ch, axis=1)
        band_rms_mean[bname] = rms_mean

        feat_dict[f"{bname}_rms_mean"] = rms_mean
        feat_dict[f"{bname}_rms_max"] = np.max(rms_per_ch, axis=1)
        feat_dict[f"{bname}_rms_std"] = np.std(rms_per_ch, axis=1)
        feat_dict[f"{bname}_kurt"] = np.mean(kurtosis(w, axis=1), axis=1)
        feat_dict[f"{bname}_skew"] = np.mean(skew(w, axis=1), axis=1)

    total_energy = sum(band_rms_mean.values()) + 1e-30
    for bname, e in band_rms_mean.items():
        feat_dict[f"{bname}_ratio"] = e / total_energy

    hf_name = band_names[-1]
    if "lf" in band_rms_mean:
        feat_dict["lf_hf_ratio"] = np.log1p(band_rms_mean["lf"]) - np.log1p(band_rms_mean[hf_name])
    if "qs" in band_rms_mean:
        feat_dict["qs_hf_ratio"] = np.log1p(band_rms_mean["qs"]) - np.log1p(band_rms_mean[hf_name])

    all_bands = np.stack([frames_ftc[b][:, 1:, :] for b in band_names], axis=0).mean(axis=0)

    feat_dict.update(spatial_coherence_multilag(all_bands))

    velocity, r2 = estimate_moveout(all_bands, dx=dx, fs=fs)
    feat_dict["moveout_velocity"] = velocity
    feat_dict["moveout_r2"] = r2

    feat_dict["spatial_footprint"] = spatial_footprint(all_bands)
    feat_dict.update(shape_factors(all_bands))
    feat_dict.update(spectral_shape_stats(all_bands, fs=fs))

    return pd.DataFrame(feat_dict)


# ── Feature-view split (Blum & Mitchell two-view co-training) ───────────────

VIEW1_SUFFIXES = ("_rms_mean", "_rms_max", "_rms_std")
VIEW1_EXTRA = {"moveout_velocity", "moveout_r2", "spatial_footprint"} | {
    f"spatial_coherence_lag{lag}" for lag in COHERENCE_LAGS
}
VIEW2_SUFFIXES = ("_kurt", "_skew", "_ratio")
VIEW2_EXTRA = {
    "lf_hf_ratio", "qs_hf_ratio",
    "crest_factor", "shape_factor", "impulse_factor", "clearance_factor",
    "spectral_centroid", "spectral_bandwidth", "spectral_entropy",
    "spectral_rolloff", "spectral_kurtosis",
}


def split_views(feature_cols):
    v1 = [f for f in feature_cols if any(f.endswith(s) for s in VIEW1_SUFFIXES) or f in VIEW1_EXTRA]
    v2 = [f for f in feature_cols if any(f.endswith(s) for s in VIEW2_SUFFIXES) or f in VIEW2_EXTRA]
    missing = set(feature_cols) - set(v1) - set(v2)
    if missing:
        raise ValueError(f"Un-assigned features: {missing}")
    overlap = set(v1) & set(v2)
    if overlap:
        raise ValueError(f"Features assigned to both views: {overlap}")
    return v1, v2


# ── Co-training (Blum & Mitchell 1998) ───────────────────────────────────────

def _feature_jitter(X: np.ndarray, rng: np.random.Generator, noise_frac: float = 0.05) -> np.ndarray:
    """
    Small Gaussian jitter scaled to each feature's std within X -- a cheap,
    tractable stand-in for FixMatch-style weak/strong raw-signal
    augmentation (Sohn et al. 2020). Re-running the full bandpass +
    feature-extraction pipeline per pseudo-label candidate at co-training
    time would be far more expensive; this jitters the already-extracted
    feature vector instead, as an approximation, used only by the optional
    consistency-check gate below.
    """
    std = X.std(axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return X + rng.normal(scale=noise_frac * std, size=X.shape).astype(X.dtype)


def select_vehicles(
    probs: np.ndarray,
    moveout_r2: np.ndarray,
    moveout_velocity: np.ndarray,
    nothing_enc: int,
    conf_veh: float = CT_CONF_VEH,
    margin: float = CT_CONF_MARGIN,
    max_veh: int = CT_MAX_VEH_PER_ITER,
    moveout_r2_min: float = CT_MOVEOUT_R2_MIN,
    v_min: float = V_MIN_MS,
    v_max: float = V_MAX_MS,
    require_moveout_gate: bool = True,
    rng: np.random.Generator | None = None,
):
    """
    Select high-confidence vehicle frames from an unlabeled pool slice.

    Base criterion (unchanged from co_training.ipynb):
        P(best vehicle class) >= conf_veh
        AND P(best) - P(2nd best) >= margin   (decisiveness gate)
        AND count < max_veh per call

    New: an additional, independent physically-motivated gate --
    require_moveout_gate=True also requires the candidate frame to show a
    spatially coherent, physically plausible moving-object signature
    (moveout_r2 >= moveout_r2_min AND a velocity within
    [v_min, v_max], reusing das_loader's existing V_MIN_MS/V_MAX_MS rather
    than inventing new bounds) before its pseudo-label is accepted. This
    directly targets confirmation bias (Shao et al. 2025) -- classifier
    confidence alone is never sufficient, the candidate must also look like
    a real moving object crossing the array.

    Returns
    -------
    sel : np.ndarray[bool], shape (n,) -- selection mask
    count : int -- number selected
    """
    rng = rng or np.random.default_rng()
    n = len(probs)
    sel = np.zeros(n, dtype=bool)
    v_cnt = 0

    best_cls = probs.argmax(axis=1)
    best_prob = probs.max(axis=1)
    second_prob = np.partition(probs, -2, axis=1)[:, -2]
    gap = best_prob - second_prob

    if require_moveout_gate:
        moveout_ok = (
            (moveout_r2 >= moveout_r2_min)
            & (np.abs(moveout_velocity) >= v_min)
            & (np.abs(moveout_velocity) <= v_max)
        )
    else:
        moveout_ok = np.ones(n, dtype=bool)

    for i in rng.permutation(n):
        cls = best_cls[i]
        if cls == nothing_enc or not moveout_ok[i]:
            continue
        if best_prob[i] >= conf_veh and gap[i] >= margin and v_cnt < max_veh:
            sel[i] = True
            v_cnt += 1

    return sel, v_cnt


def co_train(
    X1_lab, X2_lab, y_lab,
    X1_pool, X2_pool,
    moveout_r2_pool, moveout_velocity_pool,
    le,
    n_est_xgb: int = 500,
    n_est_rf: int = 1000,
    max_iter: int = CT_MAX_ITER,
    conf_veh: float = CT_CONF_VEH,
    margin: float = CT_CONF_MARGIN,
    max_veh: int = CT_MAX_VEH_PER_ITER,
    require_moveout_gate: bool = True,
    moveout_r2_min: float = CT_MOVEOUT_R2_MIN,
    use_consistency_check: bool = False,
    n_consistency_checks: int = 2,
    consistency_noise_frac: float = 0.05,
    seed: int = 42,
):
    """
    Two-view co-training (Blum & Mitchell 1998) -- vehicle pseudo-labels
    only, extended with (a) a physically-motivated moveout acceptance gate
    (select_vehicles) and (b) an optional feature-space consistency check
    (use_consistency_check=True) approximating FixMatch-style
    weak/strong-augmentation agreement (Sohn et al. 2020) -- see
    classical_baseline.ipynb for the with/without ablation of both.

    moveout_r2_pool, moveout_velocity_pool must be row-aligned with
    X1_pool/X2_pool (i.e. computed from the same feature DataFrame).

    clf1 (XGBoost / View 1): confident vehicle picks -> clf2's training set.
    clf2 (RF / View 2): confident vehicle picks -> clf1's training set.
    """
    rng = np.random.default_rng(seed)
    nothing_enc = list(le.classes_).index(BACKGROUND_CLASS)

    X1_tr, X2_tr = X1_lab.copy(), X2_lab.copy()
    y1_tr, y2_tr = y_lab.copy(), y_lab.copy()
    pool_alive = np.ones(len(X1_pool), dtype=bool)
    history = []

    clf1 = clf2 = None
    for it in range(1, max_iter + 1):
        sw1 = compute_sample_weight("balanced", y1_tr)
        clf1 = xgb.XGBClassifier(
            n_estimators=n_est_xgb,
            max_depth=5, learning_rate=0.02, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5,
            eval_metric="mlogloss", n_jobs=-1, verbosity=0, random_state=seed,
        )
        clf1.fit(X1_tr, y1_tr, sample_weight=sw1)

        clf2 = RandomForestClassifier(
            n_estimators=n_est_rf,
            max_depth=7, min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        )
        clf2.fit(X2_tr, y2_tr)

        alive = np.where(pool_alive)[0]
        if len(alive) == 0:
            print(f"  Iter {it}: pool exhausted")
            break

        p1 = clf1.predict_proba(X1_pool[alive])
        p2 = clf2.predict_proba(X2_pool[alive])
        r2_alive = moveout_r2_pool[alive]
        v_alive = moveout_velocity_pool[alive]

        sel1, v1 = select_vehicles(
            p1, r2_alive, v_alive, nothing_enc, conf_veh, margin, max_veh,
            moveout_r2_min, require_moveout_gate=require_moveout_gate, rng=rng,
        )
        sel2, v2 = select_vehicles(
            p2, r2_alive, v_alive, nothing_enc, conf_veh, margin, max_veh,
            moveout_r2_min, require_moveout_gate=require_moveout_gate, rng=rng,
        )

        if use_consistency_check:
            sel1, v1 = _apply_consistency_check(
                sel1, clf1, X1_pool[alive], p1, rng, n_consistency_checks, consistency_noise_frac,
            )
            sel2, v2 = _apply_consistency_check(
                sel2, clf2, X2_pool[alive], p2, rng, n_consistency_checks, consistency_noise_frac,
            )

        p1_veh_max = p1[:, [c for c in range(p1.shape[1]) if c != nothing_enc]].max() if p1.shape[1] > 1 else 0.0
        p2_veh_max = p2[:, [c for c in range(p2.shape[1]) if c != nothing_enc]].max() if p2.shape[1] > 1 else 0.0
        p1_neg_med = float(np.median(p1[:, nothing_enc]))
        p2_neg_med = float(np.median(p2[:, nothing_enc]))
        total_veh = v1 + v2

        if sel2.any():
            idx2 = np.where(sel2)[0]
            X1_tr = np.vstack([X1_tr, X1_pool[alive[idx2]]])
            y1_tr = np.concatenate([y1_tr, clf2.classes_[p2[idx2].argmax(axis=1)]])

        if sel1.any():
            idx1 = np.where(sel1)[0]
            X2_tr = np.vstack([X2_tr, X2_pool[alive[idx1]]])
            y2_tr = np.concatenate([y2_tr, clf1.classes_[p1[idx1].argmax(axis=1)]])

        pool_alive[alive[sel1 | sel2]] = False

        history.append({
            "iter": it,
            "new_veh": total_veh,
            "train1_size": len(X1_tr),
            "train2_size": len(X2_tr),
            "pool_left": int(pool_alive.sum()),
            "p1_veh_max": round(float(p1_veh_max), 3),
            "p2_veh_max": round(float(p2_veh_max), 3),
            "p1_neg_med": round(p1_neg_med, 3),
            "p2_neg_med": round(p2_neg_med, 3),
        })
        print(f"  Iter {it:2d}: +{total_veh:3d} vehicle labels (clf1={v1} clf2={v2}) "
              f"| pool={pool_alive.sum():,} "
              f"| veh_conf max: clf1={p1_veh_max:.2f} clf2={p2_veh_max:.2f} "
              f"| neg_conf med: clf1={p1_neg_med:.2f} clf2={p2_neg_med:.2f}")

        if total_veh == 0:
            print(f"  Converged at iter {it}")
            break

    clf1.fit(X1_tr, y1_tr, sample_weight=compute_sample_weight("balanced", y1_tr))
    clf2.fit(X2_tr, y2_tr)

    return clf1, clf2, history, X1_tr, X2_tr, y1_tr, y2_tr


def _apply_consistency_check(sel, clf, X_pool_alive, probs, rng, n_checks, noise_frac):
    """
    Post-filter: for each already-selected candidate, require its predicted
    class to survive n_checks independent feature-space jitters (see
    _feature_jitter docstring for why this approximates, rather than
    replicates, raw-signal weak/strong augmentation consistency).
    """
    if not sel.any():
        return sel, 0
    idx = np.where(sel)[0]
    target_cls = clf.classes_[probs[idx].argmax(axis=1)]
    keep = np.ones(len(idx), dtype=bool)
    for _ in range(n_checks):
        X_jit = _feature_jitter(X_pool_alive[idx], rng, noise_frac)
        keep &= (clf.predict(X_jit) == target_cls)
    sel_out = sel.copy()
    sel_out[idx[~keep]] = False
    return sel_out, int(sel_out.sum())
