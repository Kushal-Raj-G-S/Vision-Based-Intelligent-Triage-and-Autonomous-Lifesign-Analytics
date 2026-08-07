# Era 6 — Grounding Against the Literature (corrected)

**This chapter was wrong for a long time and has been rewritten with real
numbers, verified directly against the actual paper PDF.** The original
version of this file cited a "POS 8.35 / CHROM 8.20 / ChPOS 5.71 BPM"
comparison table that does not exist in the cited source. There is no
method called "ChPOS" anywhere in that paper, and neither the 8.35 nor
8.20 figures appear in it either. This was caught only because a much
later request (comparing this project against the same paper for a
different reason) led to actually reading the full PDF rather than
trusting a web-search snippet a second time. It is left documented here,
not deleted, for the same reason every other mistake in this file is kept:
a wrong number that gets caught and fixed is more useful to a reader than
a silently corrected one.

## The real paper

Qaadan, Al Jayyousi, and Alkhalaileh, ["A Unified Benchmarking Framework
for Classical Machine Learning Based Heart Rate Estimation from RGB and
NIR rPPG"](https://doi.org/10.3390/electronics15010218) (MDPI *Electronics*
15(1):218, January 2026). It benchmarks five rPPG extraction methods
(Green, POS, CHROM, PBV, PCA/ICA) combined with four classical ML
regressors (Random Forest, XGBoost, SVR, Linear Regression) across four
datasets, using a single 80/20 train/test split held out by recording (not
cross-validated) and 15 handcrafted per-segment features (time-domain,
frequency-domain, and nonlinear/entropy descriptors).

**UBFC-rPPG Part 2** (42 subjects, math-stress task — the same public
dataset as this project's `dataset_2`, 40 of its 42 subjects) is the
directly comparable case:

| Method | MAE | RMSE | R² |
| :--- | :--- | :--- | :--- |
| CHROM + Random Forest (all features) | 4.73 | 6.02 | 0.596 |
| **CHROM + XGBoost (PFI-reduced features)** | **4.60** | **5.57** | **0.654** |
| CHROM + SVR (PFI-reduced) | 5.14 | 6.47 | 0.533 |
| Green + SVR (PFI-reduced) | 5.40 | 6.77 | 0.488 |

**UBFC-rPPG Part 1** (6 subjects, the smaller original release — this
project's `dataset_1`): their best is CHROM + Random Forest (PFI-reduced),
**3.84 MAE**, 4.74 RMSE, R²=0.638.

## The honest comparison, this time verified

| | MAE (UBFC-rPPG subset) |
| :--- | :--- |
| This project — classical DSP only | 7.42 |
| This project — hybrid (DSP + Huber-loss ML correction) | 6.12 |
| **Qaadan et al. — CHROM + XGBoost (their best)** | **4.60** |

**Their approach currently beats both of ours on the directly comparable
dataset.** Not a rounding difference — a real gap. Their ground truth is
actually *coarser* than ours (segment-level peak-counting with an
acknowledged ±6 BPM quantization error on 10s windows, versus this
project's dense per-sample/per-frame ground truth), which makes the gap
more uncomfortable, not less: their method is extracting more real signal
from noisier labels than this project's correction layer currently does
from cleaner ones.

## Why, most likely — a concrete, actionable gap, not a vague one

Their permutation-feature-importance analysis repeatedly ranks **skewness,
lag-1 autocorrelation, sample entropy, and spectral centroid** among the
top features for CHROM. None of these are in this project's correction
model's feature set (`scripts/build_ml_dataset.py::FEATURE_COLUMNS`) at
all — it uses SQI/SNR/stability/harmonic-ratio/motion/POS-based features,
which are a different family of engineered signals, not the classical
statistical/entropy descriptors this paper found mattered most. Their
per-segment (10s window) feature extraction also gives their regressor a
whole window's statistics to work with per prediction, versus this
project's per-cycle streaming approach.

## Why this project is not just "the same thing, but worse"

The same paper's own conclusion section explicitly names **"hybrid
algorithmic–learned models"** — an existing DSP-extracted signal (e.g.
CHROM's output) fed into a learned correction — as **future work they
did not build**: *"algorithmically extracted rPPG signals (e.g., CHROM or
PCA outputs) may serve as structured inputs to learning-based models."*
That is this project's actual architecture (Era 9 onward). This project
also has no counterpart in that paper: a live real-time camera pipeline
with adaptive filtering and documented, fixed failure modes (harmonic
lock-on, FPS mismatch, history-freeze), a downstream deterministic
clinical safety layer, and — the strongest single finding in this whole
project — an empirically measured LLM-agent narration-faithfulness result
that has no equivalent in a pure signal-benchmarking paper. The honest
position is not "we beat them," it is "they identified the right next
step and didn't take it; we took it, on top of a more complete system, and
have a real, currently unclosed accuracy gap on the piece they *did*
build well — plus a concrete, named list of features worth trying to
close it."

**Next: [07 — Going Public](07-going-public.md)**
