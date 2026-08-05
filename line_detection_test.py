"""
line_detection_test.py
======================
Test Hough / LSD / RANSAC line detection on DAS patches loaded from
Label Studio annotations (labelStudio/output) and recorded.nc.

Usage
-----
    python line_detection_test.py [--n N] [--class CLASS] [--seed SEED] [--compare]

    --n        number of patches to show (default: 4)
    --class    filter by vehicle class, e.g. tram (default: all)
    --seed     random seed for patch sampling (default: 0)
    --compare  show all preprocessing modes side-by-side on the first patch
"""

import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from das_loader import (
    open_das, load_locations, load_annotations, load_patch,
    expected_angle_range, FS, DX, V_MIN_MS, V_MAX_MS,
)

# ── Preprocessing mode ────────────────────────────────────────────────────────
# Options: standard | rowmed | rowmed_colmean | fk | svd
PREPROCESS_MODE = "rowmed"  # standard | rowmed | rowmed_colmed | fk | svd

SVD_REMOVE_K = 3     # how many top singular vectors to suppress (svd mode)

# ── Shared filter parameters ──────────────────────────────────────────────────
CLAHE_CLIP   = 2.0
CLAHE_TILE   = (8, 8)

BILATERAL_D  = 9
BILATERAL_SC = 30
BILATERAL_SS = 30

SOBEL_KSIZE  = 5

CANNY_LO     = 10
CANNY_HI     = 40

# ── Detector parameters ───────────────────────────────────────────────────────
HOUGH_THRESHOLD = 80
HOUGH_MIN_LEN   = 0.05   # fraction of image HEIGHT
HOUGH_MAX_GAP   = 0.005  # fraction of image HEIGHT

LSD_SCALE       = 0.8
LSD_SIGMA_SCALE = 0.6
LSD_QUANT       = 2.0
LSD_ANG_TH      = 22.5
LSD_LOG_EPS     = 0
LSD_DENSITY_TH  = 0.6
LSD_N_BINS      = 1024
LSD_MIN_LEN     = 0.05   # fraction of image HEIGHT

RANSAC_RESIDUAL = 2.0
RANSAC_MIN_SAMP = 50
RANSAC_MAX_ITER = 500

# Angle gate: DAS tram trajectories span ~5–85° from horizontal depending on speed
ANGLE_MIN_DEG = 5.0
ANGLE_MAX_DEG = 85.0


# ── Preprocessing modes ───────────────────────────────────────────────────────

def _clahe(img):
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(img)

def _bilateral(img):
    return cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SC, BILATERAL_SS)

def _sobel_canny(img):
    sob = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    sob_u8 = cv2.normalize(np.abs(sob), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(sob_u8, CANNY_LO, CANNY_HI)
    return sob_u8, edges

def _norm_u8(f):
    return cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def _row_median_subtract(img):
    f = img.astype(np.float32)
    return _norm_u8(f - np.median(f, axis=1, keepdims=True))

def _col_median_subtract(img):
    f = img.astype(np.float32)
    return _norm_u8(f - np.median(f, axis=0, keepdims=True))

def _fk_filter(img):
    """Keep only energy in the expected vehicle velocity cone (f-k domain)."""
    h, w  = img.shape
    f     = img.astype(np.float32) - img.mean()
    F     = np.fft.fftshift(np.fft.fft2(f))
    ft    = np.fft.fftshift(np.fft.fftfreq(h, d=1.0 / FS))   # Hz
    fk    = np.fft.fftshift(np.fft.fftfreq(w, d=DX))          # cycles/m
    FT, FK = np.meshgrid(ft, fk, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(np.abs(FK) > 1e-9, np.abs(FT / FK), np.inf)
    mask  = (v >= V_MIN_MS) & (v <= V_MAX_MS)
    # soft taper: Gaussian blur on the mask to avoid Gibbs ringing
    mask_soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.0)
    result = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask_soft)))
    return _norm_u8(result)

def _svd_denoise(img):
    """Remove top-k singular vectors (dominant horizontal noise patterns)."""
    f  = img.astype(np.float32) - img.mean()
    U, S, Vt = np.linalg.svd(f, full_matrices=False)
    S_clean  = S.copy()
    S_clean[:SVD_REMOVE_K] = 0
    clean = U @ np.diag(S_clean) @ Vt
    return _norm_u8(clean)



def preprocess(img):
    """
    Dispatch to the selected PREPROCESS_MODE.

    Returns
    -------
    stages  : list of (name, image) pairs for visualisation
    denoised: uint8 image fed to LSD
    edges   : uint8 binary edge image fed to Hough / RANSAC
    """
    eq = _clahe(img)

    if PREPROCESS_MODE == "standard":
        bilat       = _bilateral(eq)
        sob_u8, edges = _sobel_canny(bilat)
        stages = [("CLAHE", eq), ("Bilateral", bilat), ("SobelY", sob_u8), ("Canny", edges)]
        return stages, bilat, edges

    elif PREPROCESS_MODE == "rowmed":
        rm          = _row_median_subtract(eq)
        bilat       = _bilateral(rm)
        sob_u8, edges = _sobel_canny(bilat)
        stages = [("CLAHE", eq), ("RowMedian", rm), ("Bilateral", bilat), ("SobelY", sob_u8), ("Canny", edges)]
        return stages, bilat, edges

    elif PREPROCESS_MODE == "rowmed_colmed":
        rm          = _row_median_subtract(eq)
        cm          = _col_median_subtract(rm)
        bilat       = _bilateral(cm)
        sob_u8, edges = _sobel_canny(bilat)
        stages = [("CLAHE", eq), ("RowMedian", rm), ("ColMedian", cm), ("Bilateral", bilat), ("Canny", edges)]
        return stages, bilat, edges

    elif PREPROCESS_MODE == "fk":
        fk_img      = _fk_filter(eq)
        bilat       = _bilateral(fk_img)
        edges       = cv2.Canny(bilat, CANNY_LO, CANNY_HI)
        stages = [("CLAHE", eq), ("f-k filter", fk_img), ("Bilateral", bilat), ("Canny", edges)]
        return stages, bilat, edges

    elif PREPROCESS_MODE == "svd":
        svd_img     = _svd_denoise(eq)
        bilat       = _bilateral(svd_img)
        sob_u8, edges = _sobel_canny(bilat)
        stages = [("CLAHE", eq), (f"SVD -top{SVD_REMOVE_K}", svd_img), ("Bilateral", bilat), ("SobelY", sob_u8), ("Canny", edges)]
        return stages, bilat, edges

    else:
        raise ValueError(f"Unknown PREPROCESS_MODE: {PREPROCESS_MODE!r}")


ALL_MODES = ["standard", "rowmed", "rowmed_colmed", "fk", "svd"]


def preprocess_mode(img, mode):
    """Run a specific mode (used by compare plot)."""
    global PREPROCESS_MODE
    original = PREPROCESS_MODE
    PREPROCESS_MODE = mode
    try:
        return preprocess(img)
    finally:
        PREPROCESS_MODE = original


# ── Angle gate ────────────────────────────────────────────────────────────────

def _angle_ok(x1, y1, x2, y2):
    dx = x2 - x1
    if dx == 0:
        return False
    angle = abs(np.degrees(np.arctan2(abs(y2 - y1), abs(dx))))
    return ANGLE_MIN_DEG <= angle <= ANGLE_MAX_DEG


# ── Detectors ─────────────────────────────────────────────────────────────────

def detect_hough(edges):
    h = edges.shape[0]
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, HOUGH_THRESHOLD,
        minLineLength=max(10, int(h * HOUGH_MIN_LEN)),
        maxLineGap=max(5,  int(h * HOUGH_MAX_GAP)),
    )
    if lines is None:
        return []
    return [(x1, y1, x2, y2) for (x1, y1, x2, y2) in lines[:, 0] if _angle_ok(x1, y1, x2, y2)]


def detect_lsd(denoised):
    lsd = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD,
        LSD_SCALE, LSD_SIGMA_SCALE, LSD_QUANT,
        LSD_ANG_TH, LSD_LOG_EPS, LSD_DENSITY_TH, LSD_N_BINS,
    )
    segs, _, _, _ = lsd.detect(denoised)
    if segs is None:
        return []
    min_len = denoised.shape[0] * LSD_MIN_LEN
    return [
        (int(x1), int(y1), int(x2), int(y2))
        for (x1, y1, x2, y2) in segs[:, 0]
        if _angle_ok(x1, y1, x2, y2) and np.hypot(x2 - x1, y2 - y1) >= min_len
    ]


def detect_ransac(edges):
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    _, w   = edges.shape
    ys, xs = np.where(edges > 0)
    if len(xs) < RANSAC_MIN_SAMP:
        return None
    try:
        model = RANSACRegressor(
            LinearRegression(),
            residual_threshold=RANSAC_RESIDUAL,
            min_samples=RANSAC_MIN_SAMP,
            max_trials=RANSAC_MAX_ITER,
            random_state=42,
        )
        model.fit(xs.reshape(-1, 1).astype(np.float64), ys.astype(np.float64))
    except Exception:
        return None
    slope = model.estimator_.coef_[0]
    icept = model.estimator_.intercept_
    if not _angle_ok(0, int(icept), w - 1, int(slope * (w - 1) + icept)):
        return None
    return (0, int(icept), w - 1, int(slope * (w - 1) + icept), slope, icept, model.inlier_mask_.mean())


# ── Hough parameter sweep ─────────────────────────────────────────────────────

def hough_sweep(edges):
    thresholds = [40, 80, 120, 160]
    min_fracs  = [0.02, 0.05, 0.10, 0.20]
    h = edges.shape[0]
    grid = np.zeros((len(thresholds), len(min_fracs)), dtype=int)
    for i, thr in enumerate(thresholds):
        for j, frac in enumerate(min_fracs):
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, thr,
                minLineLength=max(10, int(h * frac)),
                maxLineGap=max(5, int(h * HOUGH_MAX_GAP)),
            )
            if lines is not None:
                grid[i, j] = sum(1 for (x1, y1, x2, y2) in lines[:, 0] if _angle_ok(x1, y1, x2, y2))
    return grid, thresholds, min_fracs


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _draw_segs(ax, segs, color, lw=1.5):
    for x1, y1, x2, y2 in segs:
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw)

def _draw_gt(ax, gt0, gt1, cls):
    ax.axhspan(gt0, gt1, alpha=0.25, color="yellow", label=f"GT [{cls}]")
    ax.axhline((gt0 + gt1) / 2, color="yellow", lw=1, ls="--")

def _show(ax, data, title, gt0=None, gt1=None, cls=""):
    ax.imshow(data, cmap="gray", aspect="auto", origin="upper")
    if gt0 is not None:
        _draw_gt(ax, gt0, gt1, cls)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


# ── Per-patch figures ─────────────────────────────────────────────────────────

def plot_patch(img, stages, segs_h, segs_l, ransac, gt0, gt1, cls, label, sweep=False):
    title = f"{label}  |  mode={PREPROCESS_MODE}  |  GT class: {cls}"
    edges = stages[-1][1]  # last stage is always Canny edges

    # ── Figure 1: pipeline stages ──────────────────────────────────────
    ncols = 1 + len(stages)   # raw + all stages
    fig1, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    fig1.suptitle(f"{title}  —  pipeline", fontsize=9)
    _show(axes[0], img, "raw arr_u8", gt0, gt1, cls)
    for ax, (name, data) in zip(axes[1:], stages):
        _show(ax, data, name, gt0, gt1, cls)
    fig1.tight_layout()

    # ── Figure 2: detectors ────────────────────────────────────────────
    n_rows = 2 if sweep else 1
    fig2 = plt.figure(figsize=(24, 5 * n_rows))
    gs   = gridspec.GridSpec(n_rows, 4, figure=fig2, hspace=0.4, wspace=0.06)
    fig2.suptitle(f"{title}  —  detectors", fontsize=9)

    panels = [fig2.add_subplot(gs[0, c]) for c in range(4)]
    for ax in panels:
        _show(ax, img, "", gt0, gt1, cls)

    _draw_segs(panels[0], segs_h, "red")
    panels[0].set_title(f"Hough ({len(segs_h)})", fontsize=9, fontweight="bold")

    _draw_segs(panels[1], segs_l, "lime")
    panels[1].set_title(f"LSD ({len(segs_l)})", fontsize=9, fontweight="bold")

    if ransac:
        x0r, y0r, x1r, y1r, slope, _, inlier_rate = ransac
        panels[2].plot([x0r, x1r], [y0r, y1r], color="cyan", lw=2,
                       label=f"inliers {inlier_rate:.0%}")
        panels[2].legend(fontsize=8, loc="upper right")
        panels[2].set_title(f"RANSAC slope={slope:.4f}", fontsize=9, fontweight="bold")
    else:
        panels[2].text(0.5, 0.5, "not found", transform=panels[2].transAxes,
                       ha="center", va="center", color="red", fontsize=9)

    panels[3].imshow(edges, cmap="gray", aspect="auto", origin="upper")
    panels[3].set_xticks([]); panels[3].set_yticks([])
    panels[3].set_title("Canny (RANSAC / Hough input)", fontsize=9, fontweight="bold")
    if ransac:
        panels[3].plot([x0r, x1r], [y0r, y1r], color="cyan", lw=1.5)

    if sweep:
        sg, s_thr, s_frac = hough_sweep(edges)
        ax_sw = fig2.add_subplot(gs[1, :2])
        im = ax_sw.imshow(sg, cmap="YlOrRd", aspect="auto")
        ax_sw.set_xticks(range(len(s_frac)))
        ax_sw.set_xticklabels([f"{f:.0%}" for f in s_frac])
        ax_sw.set_yticks(range(len(s_thr)))
        ax_sw.set_yticklabels(s_thr)
        ax_sw.set_xlabel("minLineLength (fraction of height)")
        ax_sw.set_ylabel("vote threshold")
        ax_sw.set_title("Hough sweep — n lines in angle gate", fontsize=9)
        for ii in range(len(s_thr)):
            for jj in range(len(s_frac)):
                ax_sw.text(jj, ii, str(sg[ii, jj]), ha="center", va="center", fontsize=10,
                           color="white" if sg.max() and sg[ii, jj] > sg.max() * 0.6 else "black")
        fig2.colorbar(im, ax=ax_sw, fraction=0.03)

        ax_info = fig2.add_subplot(gs[1, 2:])
        ax_info.axis("off")
        h_img, w_img = img.shape
        ang_min, ang_max = expected_angle_range()
        info = [
            f"Mode:            {PREPROCESS_MODE}",
            f"Shape:           {w_img} × {h_img} px",
            f"CLAHE:           clip={CLAHE_CLIP}  tile={CLAHE_TILE}",
            f"Bilateral:       d={BILATERAL_D}  σc={BILATERAL_SC}  σs={BILATERAL_SS}",
            f"SobelY ksize:    {SOBEL_KSIZE}",
            f"Canny:           lo={CANNY_LO}  hi={CANNY_HI}",
            f"Hough thr:       {HOUGH_THRESHOLD}  minLen={HOUGH_MIN_LEN:.0%} of H",
            f"LSD density:     {LSD_DENSITY_TH}  minLen={LSD_MIN_LEN:.0%} of H",
            f"RANSAC residual: {RANSAC_RESIDUAL} px  min_samp={RANSAC_MIN_SAMP}",
            f"Angle gate:      {ANGLE_MIN_DEG}°–{ANGLE_MAX_DEG}° from horizontal",
            f"Expected DAS:    {ang_min:.1f}°–{ang_max:.1f}° (from speed bounds)",
            "",
            f"Hough:  {len(segs_h)} line(s)",
            f"LSD:    {len(segs_l)} segment(s)",
            f"RANSAC: {'slope=' + f'{ransac[4]:.5f}  inliers={ransac[6]:.0%}' if ransac else 'not found'}",
        ]
        ax_info.text(0.04, 0.97, "\n".join(info), transform=ax_info.transAxes,
                     va="top", fontsize=9, fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax_info.set_title("Parameters & results", fontsize=9)

    fig2.tight_layout()
    return fig1, fig2


def plot_preprocess_compare(img, gt0, gt1, cls, label):
    """
    For one patch, show denoised image + Canny edges for every preprocessing mode,
    then run all three detectors and report counts.
    """
    n = len(ALL_MODES)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    fig.suptitle(f"Preprocessing comparison  |  {label}", fontsize=10)

    results = []
    for j, mode in enumerate(ALL_MODES):
        _, denoised, edges = preprocess_mode(img, mode)

        segs_h = detect_hough(edges)
        segs_l = detect_lsd(denoised)
        ransac = detect_ransac(edges)
        results.append((mode, len(segs_h), len(segs_l), ransac))

        # row 0: denoised image (last before Canny)
        _show(axes[0, j], denoised, f"{mode}\n(denoised)", gt0, gt1, cls)

        # row 1: Canny edges
        _show(axes[1, j], edges, "Canny edges", gt0, gt1, cls)

        # row 2: detections on raw image
        _show(axes[2, j], img, "detections", gt0, gt1, cls)
        _draw_segs(axes[2, j], segs_h, "red")
        _draw_segs(axes[2, j], segs_l, "lime")
        if ransac:
            x0r, y0r, x1r, y1r = ransac[:4]
            axes[2, j].plot([x0r, x1r], [y0r, y1r], color="cyan", lw=2)
        axes[2, j].set_title(
            f"H={len(segs_h)}  L={len(segs_l)}  R={'✓' if ransac else '✗'}",
            fontsize=8, fontweight="bold",
        )

    fig.tight_layout()

    print("\nPreprocessing comparison:")
    print(f"  {'mode':<20}  {'Hough':>6}  {'LSD':>6}  {'RANSAC':>8}")
    for mode, nh, nl, r in results:
        print(f"  {mode:<20}  {nh:>6}  {nl:>6}  {'✓ ' + f'{r[6]:.0%}' if r else '✗':>8}")

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int,   default=4)
    parser.add_argument("--class",   dest="cls", default=None)
    parser.add_argument("--seed",    type=int,   default=0)
    parser.add_argument("--compare", action="store_true",
                        help="show all preprocessing modes side-by-side on the first patch")
    args = parser.parse_args()

    ang_min, ang_max = expected_angle_range()
    print(f"Expected tram angle: {ang_min:.1f}°–{ang_max:.1f}° from horizontal")
    print(f"Active mode: {PREPROCESS_MODE}   angle gate: {ANGLE_MIN_DEG}°–{ANGLE_MAX_DEG}°")

    print("Opening DAS data…")
    das_data = open_das()
    locs     = load_locations()
    labels   = load_annotations()

    if args.cls:
        labels = labels[labels["class"] == args.cls].reset_index(drop=True)
        if labels.empty:
            print(f"No annotations with class '{args.cls}'")
            return

    n      = min(args.n, len(labels))
    sample = labels.sample(n, random_state=args.seed).reset_index(drop=True)

    for i, (_, row) in enumerate(sample.iterrows()):
        label = (f"{row['location']} | {row['direction']} | "
                 f"{row['time_start'].strftime('%H:%M:%S')}–{row['time_end'].strftime('%H:%M:%S')}")
        print(f"\n[{i+1}/{n}] {label}")

        try:
            arr_u8, _, _, gt0, gt1 = load_patch(row, das_data, locs)
        except Exception as e:
            print(f"  load_patch failed: {e}")
            continue

        h, w = arr_u8.shape
        print(f"  shape {w}×{h}  GT rows {gt0:.0f}–{gt1:.0f}")

        if args.compare and i == 0:
            plot_preprocess_compare(arr_u8, gt0, gt1, row["class"], label)
            continue

        stages, denoised, edges = preprocess(arr_u8)
        segs_h = detect_hough(edges)
        segs_l = detect_lsd(denoised)
        ransac = detect_ransac(edges)
        print(f"  Hough {len(segs_h)}  LSD {len(segs_l)}  RANSAC {'✓' if ransac else '✗'}")

        plot_patch(
            arr_u8, stages, segs_h, segs_l, ransac,
            gt0, gt1, row["class"], label,
            sweep=(i == 0),
        )

    plt.show()


if __name__ == "__main__":
    main()
