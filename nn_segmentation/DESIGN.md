# Temporal CNN Segmentation for DAS Vehicle Events — Design

This document specifies the architecture, preprocessing, and semi-supervised
training design for `nn_segmentation/`. It complements (does not replace) the
existing classical baselines in the repo:

- `xgboost_baseline.ipynb` — binary tram/nothing classifier, 1 s frames, 26
  handcrafted features.
- `co_training.ipynb` — multi-class (tram/bus/truck/nothing) two-view
  XGBoost + RandomForest co-training, 5 s frames, same 26 features split into
  a "spectral" view and a "shape/coherence" view.

Both classify **isolated fixed-length frames** independently. The goal here is
a genuinely **temporal, fully-convolutional** model that produces a dense
label over the whole recording, and to compare two semi-supervised training
strategies for exploiting the large amount of unlabeled DAS data.

## 1. Task framing

Ground truth comes from two sources, both parsed relative to the physical
constants in `das_loader.py` (`FS=67 Hz`, `DX=0.8 m/channel`):

- Label Studio exports (`labelStudio/output/*.json`, parsed by
  `das_loader.load_annotations()`) — a rectangle whose vertical extent
  (`y%`/`height%`) encodes a **time span** only. The box always spans the
  full distance window of the site (`x=0, width=100`); there is no
  per-channel/distance annotation anywhere in the repo.
- CSV event logs (`labeling/log2.csv`, `log3.csv`, `log4.csv`,
  `log-test.csv`), schema `event_id,location,time_start,time_end,direction,
  class,footage_id`, produced by `tram_detector/logger.py`'s video-based
  auto-labeler.

Because labels never carry distance-axis information, the only defensible
dense-prediction target is **framewise temporal segmentation** — analogous to
action segmentation (MS-TCN) rather than 2D image segmentation:

```
For a fixed per-site distance-channel window [dist_start, dist_end]:
    y_hat[t] ∈ {background, tram, bus, truck, traffic}   for every output timestep t
```

using all channels in that window as input to a single per-timestep
prediction.

**Future option, not implemented now — trajectory pseudo-masks.**
`trajectory_detection.ipynb` fits diagonal lines (Hough / LSD / RANSAC) to
the time-distance waterfall within an expected-angle cone derived from
vehicle-speed physics (`das_loader.expected_angle_range()`,
`V_MIN_MS=5`, `V_MAX_MS=100`). In principle this could turn a crude
"whole-distance-range, this time span" label into a diagonal per-channel
soft mask, letting a future 2D model actually use the distance axis.
Deliberately left as a documented option rather than built now because:

- The three detectors were tuned ad hoc per-event in that notebook (varying
  `sobel_pct`, `canny_lo/hi`, `vote_thresh`) — they only work reliably on
  high-SNR, single-vehicle, low-occlusion events, and will silently fail or
  hallucinate on `traffic` (multi-vehicle aggregate) or low-SNR truck events.
- Using a classical CV detector's output as *both* the training target and
  the evaluation reference risks circularity — the model would partly learn
  to reproduce that detector's biases, not ground truth.
- If pursued later, it should enter only as a **weak auxiliary loss** on a
  small subset of high-confidence single-vehicle events (all three detectors
  agreeing within an angular tolerance), reported as an ablation — never as
  primary supervision.

## 2. Preprocessing pipeline

### 2.1 Boundary-anomaly handling (critical)

`anomalies/check_statistics.ipynb` / `anomalies/check_statistics.py` found
that the raw DAS signal jumps at every UTC-second boundary crossing (likely a
Febus interrogator buffering artifact). Statistical testing across three
subsets (single-channel, multi-channel, 5/10/15-minute windows) found:

- Best Random Forest regressor predicting the jump size from 14 context
  features: **R² ≈ 0.01–0.02** (out-of-sample) — essentially unpredictable
  from surrounding signal shape.
- Best single feature by mutual information: `boundary_offset` (MI ≈ 0.07–0.09),
  still far too weak to regress out reliably.

**Conclusion: do not try to predict/correct the jump value.** Instead, treat
it as a fixed-cadence hardware artifact and remove it structurally:

1. `data/anomaly.py::get_second_boundaries(time_index)` — refactor of
   `check_statistics.py`'s `get_second_boundaries()` /
   `check_statistics.ipynb`'s `get_second_crossings()`
   (`pd.to_datetime(time_index).floor("s")` crossing detection). Runs once
   per loaded time chunk — the boundary index set is a property of the time
   axis only, identical across all distance channels, so it is **not**
   recomputed per channel.
2. `data/anomaly.py::interpolate_boundary_samples(signal, boundary_idx)` —
   for every detected boundary index `idx > 0`, replace `signal[idx, :]`
   (all channels at that one time-sample) with a linear interpolation
   between `signal[idx-1, :]` and `signal[idx+1, :]`. Applied once, during
   offline cache-building, immediately after loading a chunk from `xdas` and
   before any windowing/normalization/caching.
3. Persist a companion `boundary_mask` array (same length as the time axis)
   in the cache, `True` at interpolated samples, for auditability.

**Why interpolate rather than reuse the existing "skip sample 0" trick**
(`co_training.ipynb`/`xgboost_baseline.ipynb` deliberately align frame starts
to `floor("1s")` so the bad sample always lands at index 0 and can be
dropped): that only works because those pipelines classify fixed,
second-aligned frames. A fully-convolutional TCN trained on **arbitrary
sliding crops** cannot guarantee any crop boundary is second-aligned — the
artifact would appear at an unpredictable *phase* inside the receptive field
of an arbitrary window. Removing it from the raw signal once, up front,
removes the dependency on alignment entirely.

**Belt-and-suspenders:** also include *synthetic* boundary-jump injection as
a FixMatch strong augmentation (Section 4), sampling jump amplitudes from the
empirical `anomaly_jump` distribution the anomaly notebook already
characterizes, at a randomly phase-shifted second-boundary grid. This trains
the model to be robust to any residual/imperfectly-detected artifact, rather
than relying solely on the offline cleanup being perfect.

### 2.2 Windowing / chunking

Not the classification-notebook pattern of isolated fixed 1 s/5 s frames.
Instead:

- **Offline (cache-build time):** load each site in `FULL_CHUNK_MIN = 20`
  minute chunks (same chunk size as `co_training.ipynb`), boundary-interpolate,
  concatenate into one **continuous per-site array** for the full recording.
  Save to `cache/nn_segmentation/<site>.npz` (or `.zarr` for memory-mapping)
  with: `signal (T, C)` float32, `time_index (T,)`, `label (T,)` int class id,
  `labeled_mask (T,)` bool, `boundary_mask (T,)` bool.
- **Label rasterization:** rasterize every Label Studio box + CSV-log event
  onto the continuous timeline — timesteps inside `[time_start, time_end]`
  get that event's class id; everything else defaults to `background`,
  **except** timesteps outside all camera-covered ("footage") windows, which
  get `labeled_mask = False` regardless of the default label. This
  generalizes `co_training.ipynb`'s per-**frame** `in_footage` boolean to
  per-**timestep** resolution — necessary because a 30 s training crop can
  straddle a footage-covered region and an uncovered region within the same
  window, so loss masking must operate at the same resolution as the output.
- **Online (training time):** draw random contiguous crops of length
  `CHUNK_LEN` from the cached array — filtered to sufficient `labeled_mask`
  coverage for supervised crops, or anywhere for the unlabeled SSL stream.

**`CHUNK_LEN = 30 s`** (2010 samples @ FS=67 Hz). Justification from
`labeling/log4.csv` (513 events, tram/bus/truck): median duration ≈1.0 s,
90th pct ≈5.7 s, 95th pct ≈8.4 s. 30 s comfortably covers ≥95% of events with
margin for temporal context, while keeping per-batch-item compute bounded.

**Inference:** run the fully-convolutional model over the entire continuous
recording in overlapping chunks (e.g. 30 s stride, 25 s step, 5 s overlap),
stitching by averaging logits in the overlap region — standard practice for
fully-convolutional temporal models, avoids edge artifacts at chunk
boundaries.

### 2.3 Normalization

**Per-channel robust z-score** — median + MAD (or 1st/99th-percentile-based
scale), computed once over the **training range only** (see Section 4.4 on
leakage) — not `das_loader.load_patch()`'s percentile-clip-to-`uint8`
conversion, which is a lossy 8-bit **display** path built for Label
Studio/OpenCV visualization, not model input. Cache per-site, per-channel
`(median, scale)` as `norm_stats.npz`, applied identically at train and
inference. Optionally clip at ±8 robust-z post-normalization to bound
extreme-outlier influence on batch statistics.

### 2.4 No decimation

The existing spectral band split (`lf 1–5 Hz, mid1 5–15 Hz, mid2 15–28 Hz,
hf 28–33 Hz`, against Nyquist = 33.5 Hz at FS=67 Hz) already spans nearly the
full available bandwidth — the `hf` band sits at the edge. Decimating the
input would directly discard content the existing feature engineering
treats as informative. Temporal resolution reduction happens only at the
**model's output stride** (Section 3), a separate, later-stage decision.

### 2.5 Distance dimension as CNN input channels

`signal (T, C_site)` is transposed to `(C_site, T)` and fed as the
`Conv1d` **input-channel** dimension (distance is not a second spatial-conv
axis in the primary design). Sites have very different channel counts
(`most_mieszka` ~100 m → ~125 ch; `pcss`/`srodka` ~200 m → ~250 ch;
`estkowskiego`/`garbary` ~300 m → ~375 ch, from `locations.csv` ÷
`DX=0.8 m/channel`), so each site gets a thin **input adapter**
`nn.Conv1d(C_site → 64, kernel_size=1)` feeding one **shared** backbone. This
lets one backbone train on pooled data from all sites — important given the
total labeled event count is small (158 tram / 76 bus / 278 truck across all
sites combined, per `log4.csv`) — while each site keeps its own
channel-count-specific adapter.

Optional ablation (not primary): a small distance-local `Conv2d` front-end
(kernel `(1, 5)`, local across distance, none across time) before the
adapter, to explicitly capture the "spatial coherence across channels" prior
that View2 in `co_training.ipynb` encoded by hand. Not required since a
learned 1×1 adapter can already mix channels arbitrarily.

## 3. Model architecture

**Primary: multi-stage dilated Temporal Convolutional Network (MS-TCN-style).**

```
per-site adapter: Conv1d(C_site -> 64, k=1)
                -> AvgPool1d(kernel=stride=OUT_HOP)      # temporal downsample
                -> Stage 1   (L1 dilated residual TemporalBlocks)
                -> softmax -> Stage 2..N (refinement, each consumes softmax(prev))
                -> final per-stage logits (B, n_classes, T')
```

- `TemporalBlock`: `Conv1d(k=3, dilation=d, padding=d)` → `ReLU` →
  `Dropout(0.3)` → `Conv1d(1x1)` → residual add.
- `TCNStage`: input 1x1 conv → stack of `L` `TemporalBlock`s, dilation
  `2^i` for `i = 0..L-1` → output 1x1 conv to `n_classes` logits (mirrors
  MS-TCN's `SingleStageModel`).
- `DASMultiStageTCN`: per-site `nn.ModuleDict` of adapters, one `stage1`, a
  list of refinement stages. `forward()` returns **all** per-stage logits
  (multi-stage supervision); inference uses the last stage.

**Sizing, with receptive-field justification:**

- Hidden width **64** (MS-TCN default; appropriate for this dataset size).
- `n_classes = 5`: `background, tram, bus, truck, traffic`. `traffic` is an
  aggregate "high-vehicle-count period" label, not one physically coherent
  signal the way tram/bus/truck are — **masked out of the supervised loss by
  default** (same masking mechanism as `labeled_mask`), kept as a
  configurable class-set for ablation.
- Kernel size `k=3` throughout.
- **Output stride `OUT_HOP ≈ 17` samples** (≈0.25 s @ FS=67 Hz, ~4
  predictions/sec), applied via `AvgPool1d` right after the site adapter.
  Justified because: (a) Label Studio box boundaries are only
  pixel-resolution accurate — far coarser than 67 Hz — so per-sample output
  buys nothing in label fidelity; (b) it cuts sequence length ~17×, a large,
  practically important compute reduction; (c) it lets a modest number of
  dilated layers reach a large real-world receptive field cheaply.
- **Receptive field** (`RF = 1 + 2·(2^L − 1)` frames at reduced resolution):
  Stage 1 `L1 = 6` → `RF = 127` reduced-frames ≈ **31.8 s** real-world
  context (covers essentially all events including the rare long tail up to
  the 95th-pct 8.4 s duration, with wide margin). Refinement stages
  `L_r = 4` → `RF ≈ 7.75 s`, `n_refine_stages = 3` (4 stages total,
  MS-TCN-standard count).
- Dropout `0.3` inside each `TemporalBlock`.

**Loss:**

- Per-stage masked weighted cross-entropy — masked by `labeled_mask` pooled
  to the same `OUT_HOP` resolution (a reduced frame counts as labeled only if
  fully covered) — summed over all stages (standard MS-TCN multi-stage
  supervision).
- Class weights: inverse frequency computed from confirmed
  (`labeled_mask=True`) timesteps in the training range only.
- **T-MSE smoothing loss** (truncated MSE between adjacent-frame
  log-probabilities, clip=4, weight `λ=0.15`) to discourage flicker /
  over-segmentation — useful here since residual boundary-artifact leakage
  or raw single-timestep noise could otherwise cause spurious class flips.
- Optional ablation: focal loss (`γ=2`) or a Dice-style term on vehicle
  classes, given severe background dominance.

**Documented, not built — 2D (time × distance) U-Net ablation**
(`models/unet2d.py` is a stub only). Given labels carry **zero** distance-axis
signal (every box spans the full site width), a 2D U-Net's target would just
be the same time-only label broadcast as a row-constant image. It can only
meaningfully outperform the 1D TCN if combined with the trajectory
pseudo-masks (Section 1) as weak per-pixel targets — absent that, it is
strictly more parameter/compute-heavy for no additional supervisory signal.
Build only if the pseudo-mask stretch goal is pursued.

## 4. Semi-supervised learning: FixMatch vs. deep co-training

Both get equal-quality skeletons so they can be compared side-by-side against
the existing classical co-training baseline.

### 4.1 FixMatch (primary recommendation)

- **Weak augmentation:** small time shift (±0.1 s), small distance-window
  crop/shift (±5–10 channels, simulating site-boundary uncertainty), mild
  Gaussian noise (σ ≈ 0.01–0.05× per-channel std).
- **Strong augmentation:** SpecAugment-style random contiguous
  **channel-block masking** (distance-axis analogue of frequency masking)
  and random contiguous **time-block masking**, capped so total masked
  fraction stays ≤15% (must not fully erase a short vehicle event); random
  per-channel amplitude scaling (0.7–1.3×); and — directly motivated by
  Section 2.1 — **synthetic boundary-jump injection** sampling from the
  empirical `anomaly_jump` distribution at a randomly phase-shifted
  second-boundary grid, applied to the strong view only. Mixup is
  lower-priority/optional: natural for whole-window classification but
  awkward for dense per-timestep targets unless crops are perfectly aligned.
- **Confidence thresholding, per output-frame** (not per-window, since the
  output is dense): the weak view's softmax at each reduced-resolution
  timestep supplies a pseudo-label target for the strong view's prediction
  at the same timestep, only where thresholds are met. **Asymmetric
  per-class thresholds, reusing the tuned values from `co_training.ipynb`:**
  `tau_vehicle ≈ 0.60` (same as `CT_CONF_VEH`) for any vehicle class plus a
  decisiveness margin `top1 − top2 ≥ 0.15` (same as `CT_CONF_MARGIN`); a
  separate, higher `tau_background ≈ 0.95` (background is trivially easy and
  dominant early in training, so it must be gated harder to avoid flooding —
  same reasoning already in the existing notebook).
- **Class imbalance handling:** mirror the existing, already-validated
  design choice — **never** generate unsupervised loss from
  predicted-background frames (only confirmed `labeled_mask=True` background
  contributes to the supervised CE term); the unsupervised loss only fires
  on frames confidently predicted as a vehicle class. `co_training.ipynb`'s
  own comment states the reasoning directly: co-training's only useful
  contribution is discovering vehicle events in the unlabeled period — this
  design carries that forward unchanged into the dense/framewise setting.
  Also cap pseudo-labeled vehicle-frames per batch/epoch (continuous
  analogue of `CT_MAX_VEH_PER_ITER=30/iter`) to avoid early-training
  confirmation-bias runaway.

### 4.2 Deep co-training (Cross-Pseudo Supervision, CPS)

Two identically-structured `DASMultiStageTCN`s, different random init and
different augmentation streams, cross-supervising each other's confident
predictions on unlabeled crops **every batch** (not in discrete
alternating-retrain rounds like the classical `co_train()` loop in
`co_training.ipynb`).

Two adaptation options were considered:

- **Option A — literal two-view CNN branches**, mirroring the classical
  split: a "spectral" branch fed 4-band-filtered multi-channel signal
  (`lf/mid1/mid2/hf`, keeping full temporal resolution rather than collapsing
  to RMS scalars) and a "shape/coherence" branch fed raw signal plus rolling
  kurtosis/skew/cross-channel-coherence as extra input channels.
- **Option B — CPS** (recommended if pursuing this family): avoids having to
  justify that the spectral/shape split gives genuinely conditionally
  independent views for raw waveforms (a heuristic, not proven, split) — CPS
  gets diversity from init + augmentation instead, which is simpler and more
  robust in practice.

**`training/cotraining.py` implements Option B.**

Same asymmetric-threshold, vehicle-only pseudo-labeling design as FixMatch
(Section 4.1) applies to which cross-predictions are trusted.

### 4.3 Recommendation

**Primary: FixMatch.** Reasoning:

1. Single-model training loop — simpler to integrate with a from-scratch
   dense-segmentation TCN than two co-evolving networks.
2. The classical co-training loop was built around whole-frame labels;
   extending Blum & Mitchell's alternating-retrain framework to per-timestep
   dense outputs adds bookkeeping FixMatch's elementwise-threshold framework
   avoids.
3. It reuses the two most load-bearing numeric decisions already validated
   in this repo's classical work (asymmetric per-class threshold + margin
   gate; vehicle-only pseudo-labeling) without inheriting the two-view
   engineering's complexity.
4. Better-benchmarked in the general semi-supervised deep learning
   literature, easier to situate against related work.

**Keep CPS as the documented alternative/ablation** — a clean three-way
comparison: classical co-training (existing baseline) → CPS (deep analogue of
this repo's own method) → FixMatch (primary proposed contribution), all on
the same temporal splits and events.

### 4.4 Unlabeled-pool leakage guardrail

`co_training.ipynb`'s pool construction
(`unlabeled_nothing = train_df[(label=="nothing") & (~in_footage)]`, sampled
up to `MAX_POOL_SIZE=20000`, `TRAIN_FRAC=0.70` chronological split) already
avoids leakage only because it slices `train_df`, not the full dataset. This
must be an **explicit, single source of truth** here:
`data/splits.py::get_temporal_split(site_timeline, train_frac=0.70)`,
computed once per site, before any sampling, used identically by:

1. normalization-stat computation (Section 2.3),
2. supervised crop sampling,
3. unlabeled-pool crop sampling (FixMatch weak/strong pair, and CPS).

No SSL pool crop may ever be drawn from the held-out test range, regardless
of its `labeled_mask` / `in_footage` status.

## 5. Artifact locations

- `cache/nn_segmentation/<site>.npz` — preprocessed continuous per-site
  arrays + norm stats. Gitignored (mirrors the existing
  `cache/dataset_all.parquet` convention), expensive to rebuild.
- `nn_segmentation/models/training/<run_name>/` — model **weight
  checkpoints only** (`checkpoint_best.pt`, `checkpoint_last.pt`, optimizer
  state). This is the pre-existing empty directory, now given a concrete,
  narrow purpose. Gitignored.
- `runs/nn_segmentation/<run_name>/` — run **logs/artifacts**: config
  snapshot, W&B run id, prediction visualizations. Follows the existing
  repo-root `runs/` convention (currently used by `tram_detector/runs/train`),
  scoped under a `nn_segmentation/` subfolder to avoid colliding with YOLO's
  usage.
- `nn_segmentation/training/` — training-loop **source code** (versioned).
  Deliberately a distinct top-level folder from `models/training/`
  (checkpoints) to resolve the naming collision the pre-existing empty
  scaffold created.

## 6. Reuse

`data/raster_labels.py` and `data/cache_builder.py` import directly from the
existing `das_loader.py` (`load_locations`, `load_annotations`, `open_das`,
`FS`, `DX`) rather than re-implementing label parsing. `load_annotations()`
already handles missing/unparseable files gracefully (labels are gitignored,
may be absent locally) — the segmentation pipeline must propagate that as
"0 labeled events, all `labeled_mask=False`" rather than erroring, so the
unsupervised/pool-only path still runs without local label data.

Config format: plain Python dataclasses (`nn_segmentation/config/*.py`) — the
repo has no YAML/`requirements.txt` convention yet, so no new dependency is
introduced for configuration.
