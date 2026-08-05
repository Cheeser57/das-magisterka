"""
Filter & display comparison lab — pick the signal processing / colormap
combination that makes events easiest to see before running
generate_label_studio.generate_tasks().

QUICK START — just want the best-known view for human labeling? Skip the
comparison grids and use the "recommended" feature directly:

    Notebook:
        from labelStudio.filter_display_lab import recommended_view
        das_data = xdas.open_dataarray("das/recorded.nc")
        recommended_view(das_data, event_index=5)   # or random=True

    CLI (from project root):
        python labelStudio/filter_display_lab.py recommended --event-index 5
        python labelStudio/filter_display_lab.py recommended --random --count 3

    Either way this renders 3 panels — raw strain-rate, the recommended
    signed view (filter "recommended" + colormap RdBu_r + display
    symlog_p99), and an envelope quick-scan (filter "envelope" + colormap
    viridis + display sequential_p99) — and saves the PNG to
    labelStudio/filter_lab/recommended_<event>.png.

    To use the recommended filter/colormap/display individually (e.g. inside
    compare_filters/compare_display or your own plotting code), the presets
    are just dicts:
        from labelStudio.filter_display_lab import FILTERS, RECOMMENDED
        filtered = FILTERS[RECOMMENDED["filter"]](da)   # -> feed to imshow
                                                          #    with RECOMMENDED["colormap"]
                                                          #    via DISPLAY_MODES[RECOMMENDED["display_mode"]]

    Want to build your OWN combo instead (chain multiple filters together,
    then pick a colormap/display)? Use pipeline_view -- e.g. RdBu_r +
    global_p99 with a slight median filter:
        from labelStudio.filter_display_lab import pipeline_view
        pipeline_view(
            das_data, event_index=5,
            steps=[("median", {"t_kernel": 3, "d_kernel": 1})],
            colormap="RdBu_r", display_mode="global_p99",
        )
    `steps` is an ordered list of FILTERS keys (or (name, kwargs) tuples for
    custom parameters) -- each is applied in sequence, e.g.
    ["bandpass", "median"] or [("median", {"t_kernel": 3})]. CLI equivalent:
        python labelStudio/filter_display_lab.py pipeline --steps median \
            --colormap RdBu_r --display-mode global_p99 --event-index 5

Every filter in FILTERS is domain-agnostic (no differentiation baked in) --
DOMAINS picks the base quantity ("strain" = raw, undifferentiated; "strain_rate"
= differentiated first), then a filter (or chain of filters, via
pipeline_view/chain_filters) runs on top of that. compare_filters and
compare_display both take a `domain` argument ("strain" / "strain_rate" /
"both"); "both" (the CLI default) renders ONE combined image with two stacked
grids for the same event -- one with every method applied to raw strain, one
with every method applied to strain-rate -- via compare_filters_both_domains /
compare_display_both_domains.

Full notebook usage (exploration + recommended + custom pipelines):
    from labelStudio.filter_display_lab import (
        compare_filters, compare_filters_both_domains,
        compare_display, compare_display_both_domains,
        recommended_view, pipeline_view, chain_filters,
    )

    das_data = xdas.open_dataarray("das/recorded.nc")
    compare_filters_both_domains(das_data, event_index=5)      # 1 image, 2 grids: strain vs strain_rate, all filters
    compare_display_both_domains(das_data, event_index=5)      # 1 image, 2 grids: colormap x display, strain vs strain_rate
    compare_filters(das_data, event_index=5, domain="strain_rate")            # single-domain grid
    compare_display(das_data, event_index=5, filter_name="raw", domain="strain_rate")  # single-domain grid
    recommended_view(das_data, event_index=5)                  # best-found combo vs. raw
    pipeline_view(das_data, event_index=5,                     # your own filter chain vs. raw
                  steps=[("median", {"t_kernel": 3, "d_kernel": 1})],
                  colormap="RdBu_r", display_mode="global_p99")

Full CLI usage (from project root):
    python labelStudio/filter_display_lab.py filters --event-index 5                     # both domains (default)
    python labelStudio/filter_display_lab.py filters --event-index 5 --domain strain_rate
    python labelStudio/filter_display_lab.py filters --random --count 3
    python labelStudio/filter_display_lab.py display --event-index 5 --filter raw         # both domains (default)
    python labelStudio/filter_display_lab.py recommended --event-index 5
    python labelStudio/filter_display_lab.py pipeline --steps median --event-index 5 \
        --colormap RdBu_r --display-mode global_p99

All commands save a PNG (grid or combined image) to labelStudio/filter_lab/ and print the path.

Recommended combo (RECOMMENDED / RECOMMENDED_ENVELOPE below) is not a guess —
it distills a literature review of ~35 DAS papers (see papers/) on how traffic-
monitoring and seismic DAS work displays and filters data for human review:
  - bandpass(1-20 Hz) + common-mode removal + light median = a fast, hybrid
    denoiser matching the traffic-monitoring literature's preference for
    lightweight, near-real-time pipelines over heavy seismic-grade denoisers
    (f-k cascades, curvelet, DAS-N2N), which exist mainly for much lower-SNR
    seismic/microseismic data (Deng et al. 2025).
  - Signed strain-rate (no envelope) rendered with a zero-centered diverging
    colormap (RdBu_r) is the field convention for human-facing signed-strain
    figures, letting an annotator read compression vs. extension directly
    (Jousset et al. 2018; Wang et al. 2024).
  - symlog normalization log-compresses strong vehicle passes without
    saturating them while lifting faint/distant vehicles out of the noise
    floor, mirroring the log-scaled colorbars in Chambers 2020.
  - As an alternative "at a glance" scan, the envelope (Hilbert transform of
    strain-rate) rendered on a sequential colormap matches how traffic papers
    highlight vehicle trajectories as bright unsigned tracks (Litzenberger et
    al. 2021; Xie et al. 2025) — useful for quickly spotting events, though
    the signed RdBu_r view keeps more information for labeling. Viridis is
    used here rather than a rainbow/jet-style map (turbo, jet) since those
    are notoriously hard to read: their non-monotonic lightness and abrupt
    hue transitions create false edges and make it hard to judge relative
    magnitude at a glance, whereas viridis is perceptually uniform and
    monotonically brightens with signal strength.
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Defaults anchored to project root ────────────────────────────────────────
_ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DAS_FILE      = os.path.join(_ROOT, "das", "recorded.nc")
_LOG_CSV       = os.path.join(_ROOT, "labeling", "log4.csv")
_LOCATIONS_CSV = os.path.join(_ROOT, "labeling", "locations.csv")
_OFFSETS_FILE  = os.path.join(_ROOT, "labeling", "footage_offsets.csv")
_OUTPUT_DIR    = os.path.join(_ROOT, "labelStudio", "filter_lab")


# ── Domains: the base physical quantity filters are applied to ──────────────
# Every filter below is domain-agnostic (no differentiation baked in) so it
# can run on either raw strain or strain-rate -- pick the domain, then pick
# the filter.

def _domain_strain(da):
    return da


def _domain_strain_rate(da):
    import xdas.signal as xs
    return xs.differentiate(da, dim="time")


DOMAINS = {
    "strain":      _domain_strain,
    "strain_rate": _domain_strain_rate,
}


# ── Filter methods (xdas DataArray -> xdas DataArray or ndarray) ─────────────

def _f_raw(da):
    return da


def _f_bandpass(da, freq=(1.0, 20.0)):
    import xdas.signal as xs
    return xs.filter(da, list(freq), btype="bandpass", dim="time", zerophase=True)


def _f_highpass(da, freq=1.0):
    import xdas.signal as xs
    return xs.filter(da, freq, btype="highpass", dim="time", zerophase=True)


def _f_lowpass(da, freq=15.0):
    import xdas.signal as xs
    return xs.filter(da, freq, btype="lowpass", dim="time", zerophase=True)


def _f_median(da, t_kernel=5, d_kernel=3):
    import xdas.signal as xs
    return xs.medfilt(da, {"time": t_kernel, "distance": d_kernel})


def _f_common_mode(da, wlen=200.0):
    """Remove the sliding spatial mean — suppresses banding common to all channels."""
    import xdas.signal as xs
    return xs.sliding_mean_removal(da, wlen, dim="distance")


def _f_envelope(da):
    """Amplitude envelope via the Hilbert transform."""
    import xdas.signal as xs
    analytic = xs.hilbert(da, dim="time")
    return np.abs(_values(analytic))


def _f_recommended(da, freq=(1.0, 20.0), wlen=200.0, t_kernel=3, d_kernel=3):
    """Best-found pipeline for human review of vehicle events: bandpass
    isolates the vehicle vibration band, common-mode removal suppresses
    cross-channel banding/coherent noise, and a light median pass removes
    speckle -- while keeping the signal signed so a diverging colormap can
    still show compression vs. extension. See module docstring for the
    literature this is based on."""
    import xdas.signal as xs
    bp = xs.filter(da, list(freq), btype="bandpass", dim="time", zerophase=True)
    decommed = xs.sliding_mean_removal(bp, wlen, dim="distance")
    return xs.medfilt(decommed, {"time": t_kernel, "distance": d_kernel})


FILTERS = {
    "raw":          _f_raw,
    "bandpass":     _f_bandpass,
    "highpass":     _f_highpass,
    "lowpass":      _f_lowpass,
    "median":       _f_median,
    "common_mode":  _f_common_mode,
    "envelope":     _f_envelope,
    "recommended":  _f_recommended,
}

# Descriptive chart-title labels for each FILTERS key -- the dict key stays
# stable (used in code, CLI, filenames), this is purely cosmetic for figures.
FILTER_LABELS = {
    "raw":          "raw",
    "bandpass":     "bandpass (1-20 Hz)",
    "highpass":     "highpass",
    "lowpass":      "lowpass",
    "median":       "median",
    "common_mode":  "common-mode removal",
    "envelope":     "envelope (Hilbert)",
    "recommended":  "bandpass + common-mode + median (recommended)",
}


def _filter_label(name):
    return FILTER_LABELS.get(name, name)


def _values(x):
    return x.values if hasattr(x, "values") else np.asarray(x)


# ── Display / normalization methods (ndarray -> ndarray, vmin, vmax) ─────────

def _display_global(arr, vpercentile=99):
    finite = arr[np.isfinite(arr)]
    vmax = float(np.percentile(np.abs(finite), vpercentile)) if finite.size else 1.0
    vmax = vmax or 1.0
    return np.clip(arr, -vmax, vmax), -vmax, vmax


def _display_per_channel(arr, vpercentile=99):
    """Normalize each distance column by its own percentile — evens out channels
    with very different noise floors, at the cost of true relative amplitude."""
    out = np.array(arr, dtype=np.float64, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = col[np.isfinite(col)]
        vmax = float(np.percentile(np.abs(finite), vpercentile)) if finite.size else 1.0
        out[:, j] = col / (vmax or 1.0)
    return out, -1.0, 1.0


def _display_symlog(arr, vpercentile=99, linthresh_frac=0.05):
    """Log-compress amplitude beyond a linear threshold — boosts weak signal
    without saturating strong ones."""
    clipped, _, vmax = _display_global(arr, vpercentile)
    linthresh = max(linthresh_frac * vmax, 1e-12)

    def _symlog(x):
        return np.sign(x) * np.log1p(np.abs(x) / linthresh)

    out = _symlog(clipped)
    bound = float(_symlog(np.array([vmax]))[0]) or 1.0
    return out, -bound, bound


def _display_sequential(arr, vpercentile=99):
    """0..vmax range for positive-valued data (e.g. an envelope) -- avoids
    wasting half a symmetric colormap on values that never go negative."""
    finite = arr[np.isfinite(arr)]
    finite = finite[finite >= 0] if finite.size else finite
    vmax = float(np.percentile(finite, vpercentile)) if finite.size else 1.0
    vmax = vmax or 1.0
    return np.clip(arr, 0, vmax), 0.0, vmax


DISPLAY_MODES = {
    "global_p99":      lambda arr: _display_global(arr, 99),
    "global_p95":      lambda arr: _display_global(arr, 95),
    "per_channel_p99": lambda arr: _display_per_channel(arr, 99),
    "symlog_p99":      lambda arr: _display_symlog(arr, 99),
    "sequential_p99":  lambda arr: _display_sequential(arr, 99),
}

COLORMAPS = ["RdBu_r", "seismic", "coolwarm", "gray", "viridis"]

# Best-found combos for human-facing review (see module docstring for the
# literature review backing these choices).
RECOMMENDED = {
    "domain": "strain_rate",
    "filter": "recommended",
    "colormap": "RdBu_r",
    "display_mode": "symlog_p99",
}
RECOMMENDED_ENVELOPE = {
    "domain": "strain_rate",
    "filter": "envelope",
    "colormap": "viridis",
    "display_mode": "sequential_p99",
}


# ── Event loading (shared with generate_label_studio's offset logic) ────────

def _load_offsets(offsets_file):
    if not os.path.exists(offsets_file):
        return {}
    off_df = pd.read_csv(offsets_file)
    return dict(zip(off_df["footage_id"], off_df["offset_sec"]))


def pick_event(
    event_index=None,
    random=False,
    log_path=_LOG_CSV,
    seed=None,
):
    """Return one row from the events log, by position or at random."""
    log = pd.read_csv(log_path, parse_dates=["time_start", "time_end"])
    if random:
        return log.sample(1, random_state=seed).iloc[0]
    if event_index is None:
        event_index = 0
    return log.iloc[event_index]


def load_event_window(
    das_data,
    row,
    locations_csv=_LOCATIONS_CSV,
    offsets_file=_OFFSETS_FILE,
    time_margin=15,
):
    """Slice das_data around one event row, applying the footage offset."""
    locs = pd.read_csv(locations_csv, skipinitialspace=True)
    locs.columns = locs.columns.str.strip()
    locs = locs.set_index("id")

    loc = row["location"]
    if loc not in locs.index:
        raise KeyError(f"location '{loc}' not in {locations_csv}")

    offsets = _load_offsets(offsets_file)
    offset_sec = 0.0
    if "footage_id" in row.index and pd.notna(row.get("footage_id")):
        offset_sec = offsets.get(row["footage_id"], 0.0)
    shift = timedelta(seconds=offset_sec)

    t_start = row["time_start"] + shift
    t_end   = row["time_end"] + shift
    margin  = timedelta(seconds=time_margin)
    win_start = t_start - margin
    win_end   = t_end + margin

    dist_start = float(locs.loc[loc, "start"])
    dist_end   = float(locs.loc[loc, "end"])

    da = das_data.sel(
        time=slice(win_start.isoformat(), win_end.isoformat()),
        distance=slice(dist_start, dist_end),
    )
    if da.sizes.get("time", 0) == 0 or da.sizes.get("distance", 0) == 0:
        raise ValueError(f"Empty slice for {loc} at {win_start} .. {win_end}")
    return da, t_start, t_end


# ── Panel rendering ───────────────────────────────────────────────────────────

def _draw_panel(ax, arr, colormap, display_mode, title):
    try:
        display_fn = DISPLAY_MODES[display_mode]
        shown, vmin, vmax = display_fn(np.asarray(arr, dtype=np.float64))
        ax.imshow(shown, aspect="auto", cmap=colormap, vmin=vmin, vmax=vmax,
                  interpolation="nearest", origin="upper")
    except Exception as e:
        ax.text(0.5, 0.5, f"error:\n{e}", ha="center", va="center",
                 fontsize=8, color="red", transform=ax.transAxes, wrap=True)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _event_label(row, t_start, t_end):
    cls = row.get("class", "?") if hasattr(row, "get") else "?"
    return f"{row['location']} · {cls} · {t_start.strftime('%H:%M:%S')}"


def _annotate_block(fig, axes_block, label):
    """Vertical bold label to the left of a row-block of axes -- call after
    fig.tight_layout()+fig.canvas.draw() so ax positions are final."""
    tops = [ax.get_position().y1 for row in axes_block for ax in row]
    bots = [ax.get_position().y0 for row in axes_block for ax in row]
    y_center = (max(tops) + min(bots)) / 2
    fig.text(0.008, y_center, label, rotation=90, va="center", ha="center",
              fontsize=13, fontweight="bold")


# ── Public API ────────────────────────────────────────────────────────────────

def compare_filters(
    das_data,
    event_index=None,
    random=False,
    seed=None,
    filters=None,
    domain="strain_rate",
    colormap="RdBu_r",
    display_mode="per_channel_p99",
    log_path=_LOG_CSV,
    locations_csv=_LOCATIONS_CSV,
    offsets_file=_OFFSETS_FILE,
    time_margin=15,
    output_dir=_OUTPUT_DIR,
    save=True,
):
    """Grid of one event rendered through every filter method (fixed display),
    all applied within a single domain -- "strain" (raw, undifferentiated) or
    "strain_rate" (differentiated first). Use compare_filters_both_domains()
    to get one grid per domain in a single call."""
    filters = filters or list(FILTERS)
    row = pick_event(event_index, random, log_path, seed)
    da, t_start, t_end = load_event_window(das_data, row, locations_csv, offsets_file, time_margin)
    base = DOMAINS[domain](da)

    n = len(filters)
    n_cols = min(4, n)
    n_rows = -(-n // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), squeeze=False)

    for i, name in enumerate(filters):
        ax = axes[i // n_cols][i % n_cols]
        try:
            filtered = FILTERS[name](base)
            arr = _values(filtered)
        except Exception as e:
            ax.text(0.5, 0.5, f"filter failed:\n{e}", ha="center", va="center",
                     fontsize=8, color="red", transform=ax.transAxes, wrap=True)
            ax.set_title(_filter_label(name), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        _draw_panel(ax, arr, colormap, display_mode, _filter_label(name))

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")

    fig.suptitle(
        f"Filter comparison [{domain}] — {_event_label(row, t_start, t_end)}\n"
        f"colormap={colormap}  display={display_mode}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        _save(fig, output_dir, f"filters_{domain}_{row.name}")
    return fig


def compare_filters_both_domains(
    das_data,
    event_index=None,
    random=False,
    seed=None,
    filters=None,
    colormap="RdBu_r",
    display_mode="per_channel_p99",
    log_path=_LOG_CSV,
    locations_csv=_LOCATIONS_CSV,
    offsets_file=_OFFSETS_FILE,
    time_margin=15,
    output_dir=_OUTPUT_DIR,
    save=True,
):
    """One combined image, two stacked grids for the SAME event: every filter
    applied to raw strain (top block) and every filter applied to strain-rate
    (bottom block)."""
    filters = filters or list(FILTERS)
    row = pick_event(event_index, random, log_path, seed)
    da, t_start, t_end = load_event_window(das_data, row, locations_csv, offsets_file, time_margin)

    n = len(filters)
    n_cols = min(4, n)
    n_rows_block = -(-n // n_cols)
    gap = 1
    total_rows = n_rows_block * 2 + gap
    height_ratios = [1] * n_rows_block + [0.15] + [1] * n_rows_block
    fig, axes = plt.subplots(
        total_rows, n_cols, figsize=(4 * n_cols, 3.4 * n_rows_block * 2 + 0.6),
        gridspec_kw={"height_ratios": height_ratios}, squeeze=False,
    )

    for c in range(n_cols):
        axes[n_rows_block][c].axis("off")

    for domain, row_offset in (("strain", 0), ("strain_rate", n_rows_block + gap)):
        base = DOMAINS[domain](da)
        for i, name in enumerate(filters):
            ax = axes[row_offset + i // n_cols][i % n_cols]
            try:
                arr = _values(FILTERS[name](base))
            except Exception as e:
                ax.text(0.5, 0.5, f"filter failed:\n{e}", ha="center", va="center",
                         fontsize=8, color="red", transform=ax.transAxes, wrap=True)
                ax.set_title(_filter_label(name), fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            _draw_panel(ax, arr, colormap, display_mode, _filter_label(name))
        for i in range(n, n_rows_block * n_cols):
            axes[row_offset + i // n_cols][i % n_cols].axis("off")

    fig.suptitle(
        f"Filter comparison: strain vs strain_rate — {_event_label(row, t_start, t_end)}\n"
        f"colormap={colormap}  display={display_mode}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.035, 0, 1, 0.94])
    fig.canvas.draw()
    _annotate_block(fig, axes[0:n_rows_block], "strain")
    _annotate_block(fig, axes[n_rows_block + gap:total_rows], "strain_rate")

    if save:
        _save(fig, output_dir, f"filters_combined_{row.name}")
    return fig


def compare_display(
    das_data,
    event_index=None,
    random=False,
    seed=None,
    filter_name="raw",
    domain="strain_rate",
    colormaps=None,
    display_modes=None,
    log_path=_LOG_CSV,
    locations_csv=_LOCATIONS_CSV,
    offsets_file=_OFFSETS_FILE,
    time_margin=15,
    output_dir=_OUTPUT_DIR,
    save=True,
):
    """Grid of one (fixed-filter) event across colormap × normalization combos,
    with the filter applied within a single domain -- "strain" (raw) or
    "strain_rate" (differentiated first). Use compare_display_both_domains()
    to get one grid per domain in a single call."""
    colormaps = colormaps or COLORMAPS
    display_modes = display_modes or list(DISPLAY_MODES)

    row = pick_event(event_index, random, log_path, seed)
    da, t_start, t_end = load_event_window(das_data, row, locations_csv, offsets_file, time_margin)
    arr = _values(FILTERS[filter_name](DOMAINS[domain](da)))

    n_rows, n_cols = len(display_modes), len(colormaps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), squeeze=False)

    for r, dmode in enumerate(display_modes):
        for c, cmap in enumerate(colormaps):
            _draw_panel(axes[r][c], arr, cmap, dmode, f"{cmap} · {dmode}")

    fig.suptitle(
        f"Display comparison [{domain}] — {_event_label(row, t_start, t_end)}\n"
        f"filter={filter_name}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        _save(fig, output_dir, f"display_{domain}_{filter_name}_{row.name}")
    return fig


def compare_display_both_domains(
    das_data,
    event_index=None,
    random=False,
    seed=None,
    filter_name="raw",
    colormaps=None,
    display_modes=None,
    log_path=_LOG_CSV,
    locations_csv=_LOCATIONS_CSV,
    offsets_file=_OFFSETS_FILE,
    time_margin=15,
    output_dir=_OUTPUT_DIR,
    save=True,
):
    """One combined image, two stacked grids for the SAME event: colormap ×
    normalization combos with filter_name applied to raw strain (top block)
    and to strain-rate (bottom block)."""
    colormaps = colormaps or COLORMAPS
    display_modes = display_modes or list(DISPLAY_MODES)

    row = pick_event(event_index, random, log_path, seed)
    da, t_start, t_end = load_event_window(das_data, row, locations_csv, offsets_file, time_margin)

    n_rows_block, n_cols = len(display_modes), len(colormaps)
    gap = 1
    total_rows = n_rows_block * 2 + gap
    height_ratios = [1] * n_rows_block + [0.15] + [1] * n_rows_block
    fig, axes = plt.subplots(
        total_rows, n_cols, figsize=(4 * n_cols, 3.4 * n_rows_block * 2 + 0.6),
        gridspec_kw={"height_ratios": height_ratios}, squeeze=False,
    )

    for c in range(n_cols):
        axes[n_rows_block][c].axis("off")

    for domain, row_offset in (("strain", 0), ("strain_rate", n_rows_block + gap)):
        arr = _values(FILTERS[filter_name](DOMAINS[domain](da)))
        for r, dmode in enumerate(display_modes):
            for c, cmap in enumerate(colormaps):
                _draw_panel(axes[row_offset + r][c], arr, cmap, dmode, f"{cmap} · {dmode}")

    fig.suptitle(
        f"Display comparison: strain vs strain_rate — {_event_label(row, t_start, t_end)}\n"
        f"filter={filter_name}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.035, 0, 1, 0.94])
    fig.canvas.draw()
    _annotate_block(fig, axes[0:n_rows_block], "strain")
    _annotate_block(fig, axes[n_rows_block + gap:total_rows], "strain_rate")

    if save:
        _save(fig, output_dir, f"display_combined_{filter_name}_{row.name}")
    return fig


def chain_filters(da, steps):
    """Apply a sequence of named filters in order. Each step is either a
    filter name (default params) or (name, kwargs) for custom parameters,
    e.g. [("median", {"t_kernel": 3, "d_kernel": 1})] for a slight median
    filter. All FILTERS keys are valid step names."""
    result = da
    for step in steps:
        name, kwargs = (step, {}) if isinstance(step, str) else step
        result = FILTERS[name](result, **kwargs)
    return result


def _pipeline_label(steps):
    parts = []
    for step in steps:
        name, kwargs = (step, {}) if isinstance(step, str) else step
        label = _filter_label(name)
        if kwargs:
            label += " (" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")"
        parts.append(label)
    return " → ".join(parts) if parts else "identity"


def pipeline_view(
    das_data,
    steps,
    event_index=None,
    random=False,
    seed=None,
    domain="strain_rate",
    colormap="RdBu_r",
    display_mode="global_p99",
    log_path=_LOG_CSV,
    locations_csv=_LOCATIONS_CSV,
    offsets_file=_OFFSETS_FILE,
    time_margin=15,
    output_dir=_OUTPUT_DIR,
    save=True,
):
    """Build-your-own-combo tool: render one event through a CUSTOM filter
    pipeline (a chain of FILTERS applied in sequence, each optionally with
    its own parameters), shown with a chosen colormap/display, next to the
    raw signal for reference.

    Example -- RdBu_r + global_p99 with a slight median filter:
        pipeline_view(
            das_data, event_index=5,
            steps=[("median", {"t_kernel": 3, "d_kernel": 1})],
            colormap="RdBu_r", display_mode="global_p99",
        )

    `steps` is a list of filter names (FILTERS keys, default params) or
    (name, kwargs) tuples for custom parameters -- applied in order, within
    `domain` ("strain" or "strain_rate")."""
    row = pick_event(event_index, random, log_path, seed)
    da, t_start, t_end = load_event_window(das_data, row, locations_csv, offsets_file, time_margin)
    base = DOMAINS[domain](da)

    raw_arr      = _values(base)
    filtered_arr = _values(chain_filters(base, steps))
    label        = _pipeline_label(steps)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _draw_panel(axes[0], raw_arr, "gray", "per_channel_p99", f"raw {domain} (reference)")
    _draw_panel(axes[1], filtered_arr, colormap, display_mode, f"{label}\n{colormap} · {display_mode}")

    fig.suptitle(f"Custom pipeline [{domain}] — {_event_label(row, t_start, t_end)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    if save:
        stem = "-".join(s if isinstance(s, str) else s[0] for s in steps) or "identity"
        _save(fig, output_dir, f"pipeline_{domain}_{stem}_{row.name}")
    return fig


def recommended_view(
    das_data,
    event_index=None,
    random=False,
    seed=None,
    log_path=_LOG_CSV,
    locations_csv=_LOCATIONS_CSV,
    offsets_file=_OFFSETS_FILE,
    time_margin=15,
    output_dir=_OUTPUT_DIR,
    save=True,
):
    """Render one event through the best-found filter/colormap/display combos
    (RECOMMENDED and RECOMMENDED_ENVELOPE), next to the raw signal for
    comparison. Use this instead of compare_filters/compare_display once
    you're past exploration and just want the best-known view for labeling."""
    row = pick_event(event_index, random, log_path, seed)
    da, t_start, t_end = load_event_window(das_data, row, locations_csv, offsets_file, time_margin)

    raw_arr       = _values(DOMAINS["strain_rate"](da))
    signed_arr    = _values(FILTERS[RECOMMENDED["filter"]](DOMAINS[RECOMMENDED["domain"]](da)))
    envelope_arr  = _values(FILTERS[RECOMMENDED_ENVELOPE["filter"]](DOMAINS[RECOMMENDED_ENVELOPE["domain"]](da)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _draw_panel(axes[0], raw_arr, "gray", "per_channel_p99", "raw strain-rate (reference)")
    _draw_panel(
        axes[1], signed_arr, RECOMMENDED["colormap"], RECOMMENDED["display_mode"],
        f"{_filter_label(RECOMMENDED['filter'])}\n{RECOMMENDED['colormap']} · {RECOMMENDED['display_mode']}",
    )
    _draw_panel(
        axes[2], envelope_arr, RECOMMENDED_ENVELOPE["colormap"], RECOMMENDED_ENVELOPE["display_mode"],
        f"{_filter_label(RECOMMENDED_ENVELOPE['filter'])} (quick scan)\n"
        f"{RECOMMENDED_ENVELOPE['colormap']} · {RECOMMENDED_ENVELOPE['display_mode']}",
    )

    fig.suptitle(f"Recommended view — {_event_label(row, t_start, t_end)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        _save(fig, output_dir, f"recommended_{row.name}")
    return fig


def _save(fig, output_dir, stem):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, f"{stem}.png")
    fig.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli():
    import xdas

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = dict(add_help=False)
    p_filters = sub.add_parser("filters", help="scan filter methods for one or more events")
    p_display = sub.add_parser("display", help="scan colormap/normalization for one filter")
    p_recommended = sub.add_parser("recommended", help="best-found filter/colormap/display combo vs. raw")
    p_pipeline = sub.add_parser("pipeline", help="custom filter chain with a chosen colormap/display")

    for p in (p_filters, p_display, p_recommended, p_pipeline):
        p.add_argument("--das-file", default=_DAS_FILE)
        p.add_argument("--log-csv", default=_LOG_CSV)
        p.add_argument("--locations-csv", default=_LOCATIONS_CSV)
        p.add_argument("--offsets-file", default=_OFFSETS_FILE)
        p.add_argument("--output-dir", default=_OUTPUT_DIR)
        p.add_argument("--event-index", type=int, default=None)
        p.add_argument("--random", action="store_true")
        p.add_argument("--count", type=int, default=1, help="number of (random) events to render")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--time-margin", type=float, default=15)

    p_filters.add_argument("--filters", default=None,
                            help=f"comma-separated subset of: {','.join(FILTERS)}")
    p_filters.add_argument("--domain", default="both", choices=["strain", "strain_rate", "both"],
                            help="'both' (default) renders two grids: one per domain")
    p_filters.add_argument("--colormap", default="RdBu_r")
    p_filters.add_argument("--display-mode", default="per_channel_p99", choices=list(DISPLAY_MODES))

    p_display.add_argument("--filter", dest="filter_name", default="raw", choices=list(FILTERS))
    p_display.add_argument("--domain", default="both", choices=["strain", "strain_rate", "both"],
                            help="'both' (default) renders two grids: one per domain")
    p_display.add_argument("--colormaps", default=None,
                            help=f"comma-separated subset of: {','.join(COLORMAPS)}")
    p_display.add_argument("--display-modes", default=None,
                            help=f"comma-separated subset of: {','.join(DISPLAY_MODES)}")

    p_pipeline.add_argument("--steps", required=True,
                             help=f"comma-separated chain of filters (default params only), e.g. "
                                  f"'median,common_mode'. From: {','.join(FILTERS)}")
    p_pipeline.add_argument("--domain", default="strain_rate", choices=["strain", "strain_rate"])
    p_pipeline.add_argument("--colormap", default="RdBu_r")
    p_pipeline.add_argument("--display-mode", default="global_p99", choices=list(DISPLAY_MODES))

    args = parser.parse_args()

    print("Loading DAS data …")
    das_data = xdas.open_dataarray(args.das_file)

    for i in range(args.count):
        seed = args.seed if args.seed is None else args.seed + i
        if args.mode == "filters":
            fn = compare_filters_both_domains if args.domain == "both" else compare_filters
            extra = {} if args.domain == "both" else {"domain": args.domain}
            fn(
                das_data,
                event_index=args.event_index,
                random=args.random or args.count > 1,
                seed=seed,
                filters=args.filters.split(",") if args.filters else None,
                colormap=args.colormap,
                display_mode=args.display_mode,
                log_path=args.log_csv,
                locations_csv=args.locations_csv,
                offsets_file=args.offsets_file,
                time_margin=args.time_margin,
                output_dir=args.output_dir,
                **extra,
            )
        elif args.mode == "display":
            fn = compare_display_both_domains if args.domain == "both" else compare_display
            extra = {} if args.domain == "both" else {"domain": args.domain}
            fn(
                das_data,
                event_index=args.event_index,
                random=args.random or args.count > 1,
                seed=seed,
                filter_name=args.filter_name,
                colormaps=args.colormaps.split(",") if args.colormaps else None,
                display_modes=args.display_modes.split(",") if args.display_modes else None,
                log_path=args.log_csv,
                locations_csv=args.locations_csv,
                offsets_file=args.offsets_file,
                time_margin=args.time_margin,
                output_dir=args.output_dir,
                **extra,
            )
        elif args.mode == "pipeline":
            pipeline_view(
                das_data,
                steps=args.steps.split(","),
                event_index=args.event_index,
                random=args.random or args.count > 1,
                seed=seed,
                domain=args.domain,
                colormap=args.colormap,
                display_mode=args.display_mode,
                log_path=args.log_csv,
                locations_csv=args.locations_csv,
                offsets_file=args.offsets_file,
                time_margin=args.time_margin,
                output_dir=args.output_dir,
            )
        else:
            recommended_view(
                das_data,
                event_index=args.event_index,
                random=args.random or args.count > 1,
                seed=seed,
                log_path=args.log_csv,
                locations_csv=args.locations_csv,
                offsets_file=args.offsets_file,
                time_margin=args.time_margin,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    _cli()
