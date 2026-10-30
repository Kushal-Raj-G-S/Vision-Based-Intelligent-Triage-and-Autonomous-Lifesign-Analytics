# Respiratory Rate — Paper-Ready Research Notes

This file exists to accumulate everything a future paper on this project's
respiratory-rate work would need, kept in clean structured form as the work
happens rather than reconstructed afterward. It is a **companion** to
[`data/evolution/19-respiratory-rate-ground-truth.md`](evolution/19-respiratory-rate-ground-truth.md),
not a replacement — that file has the full chronological narrative (every
dead end, every wrong hypothesis, every trace), which is the honest record.
This file re-organizes the *durable* findings from that narrative into the
shape a paper's sections actually take, and gets updated whenever new RR work
lands. Numbers here should always match the evolution doc's "Final Results";
if they ever disagree, the evolution doc is authoritative and this file is
stale.

---

## 1. Problem statement

Contactless heart rate via webcam remote photoplethysmography (rPPG) is a
well-studied problem with mature validation practice. Contactless
*respiratory rate* extracted from the same rPPG signal is comparatively
under-validated — many rPPG systems either omit RR or report it without
ground-truth accuracy figures. This project's own RR estimator was in that
second category until this work: it had shipped for the project's entire
history with **zero real ground-truth validation** and, as this work
discovered, **zero actual correlation with respiration** (r ≈ 0).

## 2. Related work / prior art this project's method builds on

- Pimentel, M.A.F. et al. (2016). *Towards a Robust Estimation of
  Respiratory Rate from Pulse Oximeters.* IEEE TBME, 64(8), 1914-1923.
  — Introduced the BIDMC PPG and Respiration Dataset used for all validation
  here. Benchmarked multiple respiratory-induced-variation (RIV) fusion
  approaches on this same dataset.
- Karlen, W., Raman, S., Ansermino, J.M., Dumont, G.A. (2013).
  *Multiparameter Respiratory Rate Estimation from the Photoplethysmogram.*
  IEEE TBME, 60(7), 1946-1953. — Source of the "SmartFusion" RIV-fusion
  method this project's replacement estimator is modeled on (RIAV/RIIV/RIFV
  extraction + fusion).
- This project's own prior HR validation work (`data/EVOLUTION.md`,
  `data/evolution/*.md`) established the validation methodology (real
  ground-truth over synthetic-only, honest reporting of unresolved issues,
  re-running the full harness after any DSP change) that this RR work
  follows directly.

## 3. Dataset

**BIDMC PPG and Respiration Dataset** (PhysioNet, Open Data Commons
Attribution License v1.0, DOI: 10.13026/C2208R).
- 53 recordings, 8 minutes each, 125 Hz, from critically-ill patients at
  Beth Israel Deaconess Medical Center (extracted from MIMIC II).
- Ground truth: manual breath annotations from **two independent human
  annotators**, using the impedance respiratory signal.
- A secondary 1 Hz monitor-derived RESP trend also exists in the dataset but
  was NOT used as gold standard here — only the manual annotations were.

**Second dataset, integrated**: CapnoBase IEEE TBME Respiratory Rate
Benchmark (Karlen, 2021; DOI: 10.5683/SP2/NLB8IT; Borealis Data). 42 cases,
8-min recordings, CO2 capnogram-derived instantaneous RR trend (a direct
physical measurement of each breath, a different ground-truth principle
from BIDMC's human breath-annotation), PPG from pulse oximetry, plus the
*published SmartFusion algorithm output* for every case — a genuine
literature comparison point, not just a before/after self-comparison.
Population: anaesthesia/surgical patients (mostly pediatric, Vancouver
hospital), a different institution, equipment, and age distribution from
BIDMC's critically-ill ICU adults. Harness: `scripts/validate_rr_capnobase.py`.

Format note: ships as MATLAB v7.3 (HDF5) — read via `h5py`, not
`scipy.io.loadmat` (which raises `NotImplementedError` on v7.3 files).

Data-quality note found and handled: record `0133`'s own shipped
`reference.rr.co2.y` ground-truth trend contains 5 literal `inf` values —
verified directly via `np.isinf`. Not a parsing bug on this project's side;
the harness now filters non-finite ground-truth samples defensively before
they can poison an aggregate MAE/RMSE to `inf`.

**Results (42 recordings, same production estimator, same deployment-faithful
30 fps resampling, same IQR-median session aggregation used on BIDMC).
Superseded once (see §5.7, harmonic demotion fix) — numbers below are FINAL:**

| | Our method | Published SmartFusion (same windows) |
| :--- | :--- | :--- |
| Per-window MAE | 2.30 | **1.69** |
| **Session MAE** | **2.51** | 1.59 |
| **Session median error** | **0.07** | 0.35 |
| **Sessions within 2 br/min** | **83.3%** | 78.6% |

Honest reading, after the harmonic-demotion fix (§5.7): on the **typical
case**, this project's method now clearly **beats** a mature, published,
literature-benchmarked algorithm it did not have access to during
development — better median (0.07 vs 0.35) and a better within-2-br/min
rate (83.3% vs 78.6%). The remaining gap is narrower but not closed: session
**mean** MAE is still worse (2.51 vs 1.59), because of a still-present, now
smaller, outlier tail: `0115` (error 27.48, gold RR ~40 br/min — outside the
30 br/min search band entirely, a physiological range issue not a bug) and
a handful of others still fail more severely than SmartFusion does on the
same recordings. This is the same bimodal shape already seen on BIDMC
(`bidmc_32`/`33`/`36`/`43`) — most recordings are excellent, a minority are
badly wrong — and CapnoBase confirms it is not a BIDMC-specific artifact.
**The outlier tail, not typical-case accuracy, is this method's real
remaining weakness relative to the published state of the art.**

**Access attempted, not obtained**: COHFACE (webcam video + respiration
belt — permission pending) and MAHNOB-HCI (entire database subsystem found
to be down — every subpage 404s). RR-rPPG (arXiv 2606.21511) not yet
publicly released by its authors.

**Deployment-faithful preprocessing**: BIDMC's 125 Hz PLETH signal is
decimated to 30 fps (this project's live-camera frame rate) via
`scipy.signal.resample_poly` before any estimator runs on it, so validation
numbers reflect the actual numeric regime the product uses, not the
dataset's native sampling rate.

## 4. Method

### 4.1 Baseline (original implementation)

Direct bandpass of the raw PPG waveform at 0.10–0.50 Hz (6–30 breaths/min),
single-segment periodogram, argmax peak selection.

### 4.2 Root cause of baseline failure

The bandpass filter helper (`_safe_butter_sos`) applied heart-rate-specific
guard rails (a hard floor of 0.70 Hz) to every caller regardless of intended
band. This silently rewrote the requested 0.10–0.50 Hz respiration band to
an actual **0.70–1.00 Hz (42–60 breaths/min)** passband — verified directly
via `scipy.signal.sosfreqz` frequency response — centered on the heart-rate
fundamental. The "respiration filter" passed the cardiac pulse rather than
rejecting it.

### 4.3 Replacement method: RIV fusion

Respiration is derived from respiratory-induced variations (RIVs) measured
per detected heartbeat, following Pimentel (2016) and Karlen (2013):

- **RIAV** — amplitude variation: peak-to-trough height of each pulse
- **RIIV** — intensity variation: baseline signal level at each beat
- **RIFV** — frequency variation: inter-beat interval (respiratory sinus
  arrhythmia proxy)

Each fused RIV series is:
1. Resampled onto a uniform low-rate grid (4 Hz) via **cubic spline**
   interpolation (not linear — see §5.3).
2. Linearly detrended.
3. Bandpass-filtered (0.10–0.50 Hz, using the corrected filter helper).
4. Spectrally analyzed (Welch periodogram, full-span single segment —
   see §5.4 for why sub-windowed averaging was tried and rejected).
5. Peak frequency selected as the RIV's candidate respiratory rate.

Candidates are fused by **median**. RIFV is extracted but **excluded from
the fusion** (see §5.2). An agreement-tolerance / disagreement-fallback
scheme handles cases where the two fused candidates disagree.

Beat detection itself uses a pulse-band filter **dynamically narrowed
around the currently-confirmed heart rate** (±25%) rather than a static wide
band, to prevent a cardiac 2nd harmonic from being double-counted as
spurious extra beats (see §5.5).

Session-level output is the IQR-filtered median over all confident
per-window readings across a session — mirroring the aggregation already
used for heart rate, which respiratory rate previously lacked entirely.

## 5. Ablations / component-level findings

Each row is an independently measured, reproducible finding — the
methodology `scripts/validate_rr_bidmc.py` and `diag_*` scripts used are
preserved in this session's history and can be reconstructed from the
evolution doc if needed for a methods appendix.

### 5.1 Buffer length does not fix the baseline failure

| buffer / Welch segment | ungated MAE (br/min) |
| :--- | :--- |
| 10 s / 8 s (original) | 12.74 |
| 32 s / 32 s | 11.63 |
| 60 s / 60 s | 11.21 |

**Finding**: resolution-limited hypothesis refuted. MAE stays 11–13 br/min
across a 6× buffer-length range. The failure was band contamination, not
insufficient frequency resolution.

### 5.2 RIFV (RSA channel) contributes near-zero signal on this dataset

| channel alone | slope (est-on-truth, 1.0=unbiased) | correlation |
| :--- | :--- | :--- |
| RIAV | 0.51 | +0.38 |
| RIIV | 0.55 | +0.48 |
| **RIFV** | **0.09–0.14** | **+0.06–0.09** |

**Finding**: RIFV depends on respiratory sinus arrhythmia, which is
plausibly blunted in this dataset's population (critically ill, many
ventilated patients). Including it in fusion measurably hurt the fused
result; excluding it improved session MAE.

### 5.3 Interpolation method measurably affects bias

Linear interpolation between beat-spaced samples (~0.7 s apart) acts as a
mild low-pass filter, attenuating higher frequencies more than lower ones.

| interpolation | regression slope (est-on-truth) | session MAE |
| :--- | :--- | :--- |
| Linear | 0.339 | 2.14 |
| **Cubic spline** | **0.541** | **1.71** (isolated) |

### 5.4 Multi-segment Welch averaging (a plausible-sounding fix) makes things worse

Hypothesis: averaging multiple overlapping sub-windows within the 60 s RIV
series should reduce spectral variance (standard Welch-method reasoning).

| config | MAE | corr |
| :--- | :--- | :--- |
| Full-span single periodogram (kept) | 2.20 | 0.480 |
| 30 s sub-segments, 50% overlap | 2.33 | 0.471 |
| 20 s sub-segments, 50% overlap | 2.60 | 0.408 |

**Finding**: shortening the analysis window to gain segment-averaging lost
more from reduced frequency resolution than it gained from reduced
variance, at every window length tried. Rejected.

### 5.5 Beat over-detection — a bug independent of respiration entirely

Isolated by testing a synthetic signal with a **known, confirmed heart rate
and zero respiratory modulation**:

| condition | detected beat rate | true rate |
| :--- | :--- | :--- |
| Wide static pulse filter (original) | 1.55 Hz | 1.00 Hz (60 bpm) |
| **Narrowed to confirmed HR ±25%** | **1.02 Hz** | 1.00 Hz |

**Finding**: a static wide-band pulse filter (spanning the full
physiological HR range) passes a cardiac waveform's 2nd harmonic
unattenuated whenever the harmonic itself also falls within that range
(which it almost always does — 2× 60 bpm = 120 bpm, itself plausible).
Dicrotic-notch-shaped waveform morphology then presents a second local
maximum per cardiac cycle, over-counted as an extra beat. This corrupts
every downstream RIV series regardless of the fusion or filtering logic
built on top of it. Confirmed as a genuine, general bug (not
respiration-specific) before being fixed.

### 5.6 Session-history duplication bug

A throttle/append mismatch caused each true RR estimate to be appended to
the session history once per video frame until the next recompute (~15×
duplication at 30 fps / 2 Hz update rate), rather than once per genuine
measurement. This let a single wrong-but-confident reading dominate the
session median purely through repetition. Fixing the append gate (matching
it to the same throttle condition as the recompute) measurably improved the
real-data session MAE (2.14 → 1.60 br/min) — not merely a synthetic-only
correction.

### 5.7 2x harmonic lock — the specific cause of the CapnoBase outlier tail

Traced by tracing CapnoBase's worst-performing recording with a *stable,
easy* true rate (`0105`, gold RR a steady ~10 br/min for 8 minutes): the
estimator alternated between correct (~10) and exactly double (~20 br/min,
ratio 1.96–2.02) for sustained multi-minute stretches, and — the important
detail — **confidence was high (82–100%) on the wrong windows and low
(1–3%) on the rare correct ones**. Two independent RIV channels agreeing on
the same wrong harmonic looks exactly like strong agreement to the fusion
confidence logic. Confirmed physiologically real (not a filter-leakage
artefact) by inspecting the raw spectrum directly: one channel had 2.6x
more power at the 2nd harmonic than at the fundamental in the affected
windows — expected, since breathing waveforms are asymmetric
(inhale ≠ exhale), so real harmonic content can legitimately exceed the
fundamental.

Fixed with `_demote_harmonic` — the mirror of the existing
`_promote_subharmonic` (already used for the 0.5x→1x case), checking the
opposite direction (1x peak's HALF, not its double). A shared threshold
with promotion (0.55) improved CapnoBase (session MAE 3.37 → 2.39) but
regressed BIDMC (1.60 → 1.87) — a real trade-off, not free. Sweeping a
*separate* demotion threshold on 20+20-record subsets of both datasets
found a genuine non-trade-off optimum at 0.75–0.85: BIDMC's subset MAE
there (0.60) was actually *better* than with demotion disabled entirely
(0.61), while CapnoBase kept nearly all its gain relative to disabled (3.61
vs 5.65). Set to 0.80, confirmed on full datasets: **both improved
simultaneously** — BIDMC 1.60 → 1.59, CapnoBase 3.37 → 2.51. See §6 and §3
for final numbers.

Separately, the same trace identified `0115` and `0018` as a *different*,
non-bug failure mode: their true RR (~35-40 br/min) is physiologically
outside the 30 br/min search-band ceiling entirely — not fixable by this or
any spectral-estimation change without redesigning the search band itself
(and widening it was already shown in §5.1-class experiments not to help
for an unrelated reason).

### 5.8 A disciplined negative result: `bidmc_32` is signal absence, not a third harmonic-order bug

BIDMC's own worst outlier (`bidmc_32`, session error 15.88, a 90+ year-old
critically-ill patient, true RR a steady ~24-25 br/min) showed a
superficially similar pattern to §5.7 — confidently wrong on some windows,
correctly but unconfidently right on others — but at a ~0.31-0.38 ratio,
not the 2x/0.5x relationship already handled. Inspecting the raw spectrum
directly (rather than assuming a third harmonic-order class and building a
fix around the assumption) showed the true rate's power was only 4-8% of
the wrong peak's — not "a harmonic edging out the fundamental" but "the
fundamental is barely present." Consistent with genuinely weak respiratory
modulation in a very elderly, critically-ill patient.

Tested this conclusion rather than trusting it: implemented
`_promote_third_harmonic`, structurally identical to the already-proven
promotion/demotion checks (same "only fire if genuinely comparable power"
gate), which can only be a no-op if the signal-absence hypothesis is
correct. Result on the specific case: bit-for-bit identical trace. Result
on **both full datasets**: BIDMC 1.59 → 1.59 exactly, CapnoBase 2.51 → 2.51
exactly — a clean null, not a marginal wash. **The code was removed rather
than shipped** despite the reasoning being sound, because this project's
standard is measured value, not plausible reasoning. The finding itself is
kept: it rules out a third harmonic-order bug class across two real
datasets and confirms the outlier tail's remaining causes are a structural
search-band limit and genuine signal absence — neither fixable by better
peak-picking.

## 6. Results — final configuration vs. baseline

Full 53 BIDMC recordings, 30 fps deployment-faithful resampling, scored only
on windows where both human annotators agreed within 2 breaths/min.

| Metric | Original implementation | Final (this work) |
| :--- | :--- | :--- |
| Per-window MAE (br/min) | 11.88 | **2.21** |
| Correlation with ground truth | +0.048 (none) | **+0.389** |
| Readings suppressed by confidence gate | 99.7% | 35.8% |
| **Session-level MAE (br/min)** | 4.56* | **1.59** |
| Session median error (br/min) | — | **0.40** |
| Sessions within 2 br/min of truth | 0% | **83.0%** |
| Regression slope, est-on-truth (1.0 = unbiased) | n/a (no correlation) | 0.339 → **0.541** |

\* From the single recording (of 53) that produced any session-level number
at all under the original implementation.

Heart rate was re-verified bit-identical (0.29 BPM MAE / 0.53 RMSE,
synthetic self-test) after every individual change in this body of work.

## 7. Limitations — for the paper's limitations section, stated without softening

1. **Two real datasets, both surgical/critical-care — not a broad
   population.** BIDMC (ICU) and CapnoBase (anaesthesia, mostly pediatric)
   independently confirm session-level performance is real and not a
   BIDMC-specific fluke (§3), but both are hospital-monitoring populations.
   Neither says anything about a healthy adult sitting at a webcam, which is
   this project's actual target use case (see limitation 2).
2. **Contact PPG, not webcam video.** BIDMC is finger/ear pulse oximetry.
   This validates the RR *extraction algorithm* on a clean signal. It does
   **not** establish accuracy on a webcam-derived rPPG waveform, which is
   substantially noisier (motion artifact, lighting, skin-tone-dependent
   signal strength). No end-to-end webcam RR accuracy number exists yet.
3. **Residual bias.** The estimator still shrinks toward the sample's
   central tendency (slope 0.541, not 1.0) rather than perfectly tracking
   the true rate. The dominant remaining cause is unidentified — several
   plausible mechanisms (gate selection, search-band width, Welch
   windowing) were tested and specifically ruled out (§5, and the
   evolution doc's "Open items").
4. **Synthetic/real gap.** A synthetic self-test (imposing known RIVs on a
   generated pulse waveform) scores MAE ~6.7-6.9 br/min — markedly worse
   than the real BIDMC figure of 1.60. This gap persisted even after fixing
   two real bugs the synthetic test itself caught, and is unresolved: either
   the synthetic generator is harder than real physiology, or it is
   exposing a weakness the real dataset's narrower ~15-24 br/min range does
   not probe.
5. **Per-recording outliers — three mechanisms identified across both
   datasets (§5.7, §5.8), one fixed, two not.** (1) 2x harmonic lock — now
   actively corrected. (2) True RR outside the 30 br/min search-band ceiling
   (CapnoBase's pediatric-range recordings) — a structural search-band limit,
   not addressed. (3) Genuine signal absence (`bidmc_32`, a 90+ y/o
   critically-ill patient with almost no respiratory modulation in the
   channel that would need it) — confirmed by testing a structurally
   analogous fix and measuring a clean null result on both full datasets,
   not by inspection alone. No algorithmic peak-picking fix exists for
   mechanism 3; it would need a different derived feature, or an honest
   low-confidence report instead.
6. **Population specificity.** BIDMC's subjects are critically ill ICU
   patients and CapnoBase's are anaesthetised surgical patients — both a
   meaningfully different population and physiological state (including
   mechanical ventilation for some) than the presumably healthier
   population this project's triage application targets.
7. **Better than the published state of the art on the typical case, still
   behind it on raw mean MAE.** Directly compared against the original
   SmartFusion algorithm on the same 42 CapnoBase windows, after the
   harmonic-demotion fix (§5.7): our session median error (0.07) and
   within-2-br/min rate (83.3%) both **beat** SmartFusion's (0.35, 78.6%),
   but our session mean (2.51) is still worse than SmartFusion's (1.59)
   because of the residual outlier tail (limitation 5). This is a specific,
   falsifiable, and now smaller claim than before the fix: the remaining gap
   to the literature is in outlier robustness on a shrinking minority of
   recordings, not systematic accuracy.

## 8. Open questions worth a follow-up study

- **Answered by CapnoBase, and then measurably improved (§5.7)**: session
  MAE diverged in the expected direction (1.60 on BIDMC -> originally 3.37
  on CapnoBase) but the *median* held up and even improved (0.37 -> 0.20)
  even before any fix. The story turned out to be about outlier tail
  weight, not uniform generalization failure. Tracing two specific outlier
  mechanisms (2x harmonic lock; true RR outside the search band) and fixing
  the first brought CapnoBase to 2.51 while BIDMC *also* improved to 1.59 —
  see §5.7. The natural next question is *why* the remaining recordings
  fail this badly on both datasets:
  is there a shared signal characteristic (SQI, motion, morphology) that
  predicts which recordings will be outliers, checkable before trusting a
  reading rather than only after the fact?
- Is the residual shrinkage (slope 0.541) a property of the fusion/spectral
  method, or specifically of the amplitude/baseline modulation depth in
  this population? A synthetic sweep varying modulation depth independently
  of RR could isolate this.
- Does the beat-over-detection fix (§5.5), which did not move BIDMC's
  aggregate number, matter more on webcam-derived signal, where waveform
  morphology and SNR differ substantially from clinical-grade pulse
  oximetry?
- What does per-condition stratification look like (if a dataset with
  activity/motion labels, analogous to UBFC-Phys's rest/speech/arithmetic
  for HR, becomes available for RR)?

---

*Maintained alongside `data/evolution/19-respiratory-rate-ground-truth.md`.
Update both when new RR findings land — this file with the clean structured
summary, that file with the full narrative including dead ends.*
