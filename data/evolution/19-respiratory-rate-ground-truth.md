# Era 19 — Respiratory rate gets real ground truth, and turns out to have been measuring nothing

## Why this era happened

Heart rate in this project has trustworthy numbers because it was validated
against real ground truth. Respiratory rate never was: none of the three
datasets in `test/` (UBFC-rPPG, UBFC-Phys, rPPG-10) ship a respiration
reference, so `_estimate_rr`/`_smooth_rr` had only ever been sanity-checked
against a synthetic signal for "does it converge instead of edge-locking."
CLAUDE.md said so explicitly: *"Never validated against real ground truth."*

The goal for this era was to make RR work as well as HR. The first step was
getting a ground truth at all.

## The dataset

**BIDMC PPG and Respiration Dataset** (PhysioNet, Open Data Commons
Attribution v1.0) — downloaded to `dataset/`. 53 recordings, 8 minutes each,
from critically-ill patients at Beth Israel Deaconess Medical Center:

* `PLETH` — photoplethysmogram at 125 Hz (the estimator's input)
* Breath annotations from **two independent human annotators** (the gold standard)
* A monitor-derived 1 Hz `RESP` trend (secondary, not used as gold)

Chosen because it is instantly downloadable with no access application.
COHFACE (webcam video + respiration belt — a much closer match to this
project's actual input) requires permission that had not come through, and
MAHNOB-HCI's entire database subsystem is **down**: `mahnob-db.eu` serves a
landing page but every subsection 404s, including from its own internal
links. RR-rPPG (arXiv 2606.21511) states "dataset will be made publicly
available upon paper acceptance" — it does not exist publicly yet.

**What BIDMC validates and what it does not.** It is contact pulse oximetry,
not webcam video. So it validates the *RR extraction DSP* on a clean input.
It does NOT validate the camera/ROI/face-detection layer, and does NOT prove
the same MAE holds for a webcam-derived rPPG waveform, which is far noisier.
Same distinction CLAUDE.md already draws for `validate_cropped_roi.py`.
Harness: `scripts/validate_rr_bidmc.py`.

Protocol: PLETH is resampled 125 Hz -> 30 fps first, so the test exercises
the numeric regime the product actually runs in. A window is scored only when
BOTH annotators saw >=3 breaths AND agreed within 2 br/min — otherwise the
"ground truth" is itself uncertain.

## Finding 1 — the old estimator had zero correlation with respiration

Measured against human-annotated breaths, the shipping estimator scored
**r = +0.048 across all 53 recordings** (and -0.088 on the first 8-record
probe — it wanders around zero depending on subset, which is itself the
point). Not weak — absent. Where its answers landed:

| | 10 s buffer / 8 s segment (shipped) | 32 s / 32 s |
| :--- | :--- | :--- |
| pinned to the 6 br/min floor | 71.8% | 0% |
| pinned to the 30 br/min ceiling | 25.0% | 78.6% |
| anywhere in between | ~3% | ~21% |

Gold standard sat at 15-24 br/min. **97% of readings were on a band edge.**
The `_estimate_rr` docstring claimed the multi-segment Welch rewrite had
fixed the "bounces between ~6 and ~30" symptom. It had not.

This also explains why users rarely saw a wrong RR: the edge penalty and the
`rr_conf < 30` gate suppressed **99.7%** of readings. Across all 53
recordings only ONE produced a session-level number at all. The confidence
gate was the only thing keeping uncorrelated noise out of the Compensated
Shock rule — it was doing real safety work.

## Finding 2 — the buffer length was a red herring (my first hypothesis, refuted)

The initial hypothesis was information-theoretic: a 10 s buffer has 1/10 s =
0.1 Hz resolution, the whole RR search band is only 0.4 Hz wide, so one
resolution cell spans ~6 br/min. At 12 br/min a 10 s window holds just 2
breath cycles. `nfft=4096` buys fine bin *spacing*, which is interpolation,
not resolution.

All true, and all nearly irrelevant. Sweeping buffer length:

| buffer / segment | MAE (ungated) |
| :--- | :--- |
| 10 s / 8 s | 12.74 |
| 32 s / 32 s | 11.63 |
| 60 s / 60 s | 11.21 |

MAE stayed 11-13 br/min from 10 s to 60 s. **Resolution cannot help when the
wanted signal is not in the band being analysed.** Lesson repeated from
earlier eras: a correct-sounding mechanism is not the same as the dominant one.

## Finding 3 — the actual root cause: `_safe_butter_sos` silently built a HEART-RATE filter

`_safe_butter_sos` applied pulse-band guards to every caller:

```python
low  = max(_BP_LOW_FLOOR, low)      # _BP_LOW_FLOOR = 0.70 Hz
high = min(_BP_HIGH_CEIL, high)
if high - low < _BP_MIN_WIDTH:
    high = low + _BP_MIN_WIDTH
```

Trace the respiration request `(_RESP_LOW=0.10, _RESP_HIGH=0.50)`:
low clamps UP to 0.70, high clamps DOWN to 0.50, the now-negative width trips
the collapse guard, high becomes 1.00.

Verified directly with `sosfreqz`:

```
requested 0.1-0.5 Hz  ->  actual -3 dB passband 0.701-0.999 Hz  =  42-60 breaths/min
```

**The "respiration filter" was a 42-60 br/min bandpass centred on the heart
rate fundamental.** It was *passing the pulse, not rejecting it*. That is why
the argmax pinned to the top of the search band, and it is the true origin of
that absent correlation. A generic-looking helper with one domain's constants baked in,
used by a second domain, silently producing a physiologically absurd filter
that nothing caught for the project's entire history.

Fixed with an explicit `clamp_to_pulse_band` parameter, defaulting to True so
every heart-rate call site is bit-identical, and passed False for respiration.

## The replacement estimator: RIV fusion

Rather than tune the old approach, RR is now derived from **beat-level
features** — the method validated on this exact dataset by Pimentel et al.
(the BIDMC paper) and Karlen et al. (SmartFusion):

* **RIAV** — respiratory-induced *amplitude* variation (peak-to-trough per beat)
* **RIIV** — respiratory-induced *intensity* variation (baseline at each beat)
* **RIFV** — respiratory-induced *frequency* variation (beat interval = actual RSA)

Each series is resampled onto a uniform 4 Hz grid, linearly detrended,
bandpass-filtered, and transformed; the three candidates are fused by median.
Two edge-lock mechanisms die *structurally* rather than by tuning:

* resampling to 4 Hz puts Nyquist at 2 Hz, so **the pulse is decimated away by
  construction** — no pulse energy remains to leak into the resp band;
* detrending plus a correctly-built 0.1-0.5 Hz filter removes the 1/f pedestal.

Measured design choices (BIDMC, 20 recordings):

* **median fusion beats mean** — within-2-br/min rate 52.8% -> 62.6%
* **agreement tolerance 2.0** beat 3/4/6/8 br/min
* **keeping all three channels** beat RIAV+RIIV: slightly worse window MAE
  (2.89 vs 2.60) but far fewer suppressed readings (27.5% vs 40.3%) and better
  session MAE
* **RIFV alone is nearly useless here** (r = +0.068 vs RIIV's +0.357) —
  expected, since BIDMC patients are critically ill and many are ventilated,
  which blunts respiratory sinus arrhythmia. Notably, RSA is exactly what the
  old docstring *claimed* to be measuring.

## Finding 4 — subharmonic lock-on, and a fix that first destroyed everything

After the filter fix, 11 recordings still failed badly. Diagnosis was
unambiguous: **9 of 11 were 0.5x subharmonic locks** (ratio 0.38-0.62), one
was 2x, e.g. `bidmc_22` gold 19 -> est 7.6, `bidmc_30` gold 19 -> est 7.3.

Two changes went in together — an order-4 RIV bandpass and an asymmetric
subharmonic-promotion check mirroring `_select_dominant_freq_hps`. The result
was catastrophic: **every record, good and bad, pinned to exactly 30.00
br/min.** Cause: the new RIV filter was built through the still-clamped
helper, so it too became a 0.70-1.00 Hz filter.

Because two changes shipped at once, neither was trustworthy. Ablating them
independently (after fixing the clamp, and after fixing a bug in the ablation
harness itself — restoring a `staticmethod` without re-wrapping it made `self`
arrive as `f`, and the broad `except` in `_estimate_rr` swallowed the
TypeError, silently reporting "no readings" for four of six configs):

| config | mean abs error, 8-record probe |
| :--- | :--- |
| A: baseline (validated state) | 5.25 |
| **B: fixed filter only** | **2.32** |
| C: promotion only | 4.41 |
| **D: filter + promotion 0.55 (kept)** | **2.12** |
| F: no filter + promotion 0.90 | 5.13 |

The filter does the heavy lifting; promotion adds a smaller real gain. The
four already-correct records were **identical across every config** — no
regression. `bidmc_22` 7.6 -> 18.2 and `bidmc_30` 7.3 -> 18.1 were fixed at
this point; `bidmc_43` and `bidmc_45` were **not** and remained stuck near 8
against gold 17/16. (`bidmc_45` was later fixed as a side effect of Finding
6's two bugfixes below — final session error 0.27. `bidmc_43` was not:
final session error 8.09, still unresolved — see Open Items.)

Era 4's lesson, re-earned: a fix that looks right and improves its target case
is not verified until the other known-good cases are re-run too.

## Finding 5 — session-level aggregation was missing entirely

HR gets an IQR-filtered median over its whole `bpm_history` in
`get_final_summary()`. RR had **no session history at all** — only a
two-sample EMA — so the session RR was effectively whatever the last frame
said. Three separate call sites had this asymmetry:

* `core/rppg.py` — added `rr_history` + `_summarise_rr()`, returning
  `final_rr=None` (never a guessed 0) when fewer than 3 confident readings exist
* `scripts/analyze_video.py` — `final_rr` was a **mean of the last 10
  readings**, the exact bug already fixed for `final_bpm` in an earlier era
  and left in place for RR
* `agents/tools.py` — the live path fed the **instantaneous** `metrics['rr']`
  to the clinical engine while BPM correctly used a session median, meaning
  the tachypnea threshold was decided by one instant of the noisiest channel

Aggregating is worth more than the estimator rewrite's final increment:
per-window MAE 3.09 -> session MAE 2.14.

## Results — intermediate (3-channel fusion, linear interpolation)

Full 53 recordings, deployment-faithful 30 fps, human-annotated breath gold.
This was the FIRST working state, before Finding 6's two additional bugs were
caught and fixed — kept here as a checkpoint showing the RIV-fusion rewrite's
standalone contribution, separate from the two later bugfixes.

| | legacy | RIV fusion (3-channel, linear) |
| :--- | :--- | :--- |
| per-window MAE (ungated) | 11.88 | 3.27 |
| per-window MAE (gated) | 8.64 | 3.09 |
| correlation vs gold | +0.048 | +0.216 |
| suppressed | 99.7% | 20.2% |
| session MAE | 4.56 (from 1 measurable recording) | 2.14 |
| session median error | — | 0.34 |
| sessions within 2 br/min | 0% | 71.7% |
| worst session | — | 16.62 |

A synthetic RR self-test was added to `scripts/validate_dataset.py` at this
point — RR had no automated regression test at all, which is how a filter
passing 42-60 br/min survived indefinitely. It imposes all three RIVs
(amplitude, baseline, and true phase-integrated RSA) so a broken channel
cannot pass unnoticed. **It caught real regressions almost immediately** —
see Finding 6.

A synthetic RR self-test was added to `scripts/validate_dataset.py` — RR had
no automated regression test at all, which is how a filter passing 42-60
br/min survived indefinitely. It imposes all three RIVs (amplitude, baseline,
and true phase-integrated RSA) so a broken channel cannot pass unnoticed.

## Finding 6 — two more real bugs, found by the synthetic self-test catching a regression

After the 2-channel/cubic-interpolation fix above shipped, a routine re-run of
`scripts/validate_dataset.py` (the "re-run the harness after any core/rppg.py
change" rule this file already insists on) caught a real regression: the
synthetic RR=12 br/min case had jumped from 12.4 (already-decent) to a
confidently-wrong **27.1** — almost exactly 2x — and the RR=18 case had gone
from a bad-but-present 7.8 to **completely unmeasurable**.

**Bug A — `rr_history` was silently duplicating every reading ~15x.**
`_estimate_rr()` is only *recomputed* once every `_RR_UPDATE_HZ` (2 Hz,
i.e. every ~15 frames at 30 fps) — but the append to `rr_history` ran on
every single `process_ppg_signal()` call regardless of whether a fresh
estimate had actually been computed. So one true estimate got appended
once per frame until the next recompute: **not N independent measurements,
but N/15 distinct values each repeated ~15x**, with the repeat count set by
frame timing, not by measurement quality. A single wrong-but-confident
cold-start estimate that happened to persist for one throttle interval
could dominate the IQR-filtered session median purely by having appeared
for longer.

Traced directly: a synthetic RR=12 run's *very first* RR estimate
(28.3 br/min, confidence 83.1% — a confidently-wrong cold-start read) got
appended 15 times in a row before the next recompute, then several later
correct readings (~11.0) appeared, then the tracker drifted into a sustained
harmonic lock (see Bug B) contributing dozens more near-27 entries. Fixed by
gating the append on the same `is_fresh_rr` flag that gates the recompute.
Deduplication alone reduced `rr_history` from ~975 entries to 65 for that run
— confirming the bug was real and large — but did NOT fix the RR=12 case by
itself, because (per the trace) 89% of the now-deduplicated *distinct*
readings were themselves still near 26-28, not just one duplicated outlier.
That pointed to Bug B.

Measured effect on real BIDMC data: session MAE 2.14 -> 1.60 br/min. This
was not a synthetic-only fix — it materially improved the validated
real-ground-truth number.

**Bug B — RIV beat detection over-counted beats by ~55%, independent of
respiration entirely.** Traced by checking `beat_hz` (detected beats per
second) against the known synthetic heart rate: for HR=60 bpm (1.00 Hz),
detected beat rate was **1.55 Hz** — 55% too many "beats". Testing the
*plain* HR-only synthetic generator (no respiration modulation injected at
all) reproduced the identical 1.55 Hz over-count, proving this was never a
respiration-specific bug: `_extract_rivs`'s pulse filter (`_sos_pulse`) was
a **static wide band** (`_BP_LOW_FLOOR`..`_BP_HIGH_CEIL` = 42-180 bpm) that
passes a heart rate's 2nd harmonic completely unattenuated whenever the
harmonic itself also falls in-range — which it almost always does (2x 60 bpm
= 120 bpm, itself a perfectly plausible heart rate to a 42-180 bpm filter).
The synthetic generator's pulse model includes an explicit
`0.25*sin(4*pi*hr_hz*t)` second-harmonic term (added for dicrotic-notch
realism), which created a second local maximum per cardiac cycle that the
generic `prominence=0.20*std` threshold could not reliably reject.

Fixed with `_beat_detection_filter()`: narrows the pulse-band filter to the
CONFIRMED heart rate (`self.prev_bpm`) +-25% once one exists, mirroring how
the main HR pipeline's adaptive filter already solves the identical
harmonic-leakage problem for its own peak-picking (`_update_adaptive_filter`)
— this is the same "kill it structurally, not by tuning" pattern used
throughout this project rather than a special-cased threshold. Falls back to
the old static wide band during cold start, before any HR is confirmed yet.

Verified: the same HR=60, no-respiration synthetic case now detects 58 beats
at 1.02 Hz (was 91 beats / 1.55 Hz). The RR=12 case's confidently-wrong
27.1 br/min resolved to 18.1 (still off, but no longer a ~2x harmonic lock),
and the RR=18 case stopped being completely unmeasurable.

**Honest caveat**: this fix did NOT move the aggregate 53-record BIDMC
number at all (session MAE stayed exactly 1.60 before and after) — real PPG
morphology apparently doesn't trigger the harmonic-leakage bug as severely
as the synthetic generator's explicit 2nd-harmonic term does. It is still a
real, verified robustness fix (confirmed via the beat-rate measurement
above, independent of any RR accuracy number) that closes a documented
failure mode and should not regress in the future — just not one that
happened to move BIDMC's specific number.

## Final Results

Full 53 recordings, deployment-faithful 30 fps, human-annotated breath gold.
This is what ships as of this writing — 3-channel fusion + linear
interpolation, then RIFV dropped + cubic interpolation (earlier in this
document), then the two Finding-6 bugfixes (rr_history dedup + narrowed
beat-detection filter) on top:

| | legacy | RIV fusion (final) |
| :--- | :--- | :--- |
| per-window MAE (gated) | 8.64 | **2.21** |
| correlation vs gold | +0.048 | **+0.389** |
| suppressed | 99.7% | 35.8% |
| **session MAE** | 4.56 (from 1 measurable recording) | **1.60** |
| **session median error** | — | **0.37** |
| sessions within 2 br/min | 0% | **81.1%** |
| worst session | — | 15.88 |
| estimate-vs-truth regression slope (1.0 = unbiased) | n/a | 0.339 -> **0.541** (cubic interp) |

Progression across this era's fixes, session MAE: **4.56 (legacy, barely
functional) -> 2.14 (RIV fusion, 3-channel) -> 1.71 (RIFV dropped + cubic
interpolation) -> 1.60 (rr_history dedup)**. Roughly a 3x improvement over
the original implementation, and the estimator now produces a usable reading
on the clear majority (81.1%) of sessions to within 2 br/min, versus 0% for
the original.

**Heart rate is unchanged and bit-identical throughout every one of these
changes: 0.29 BPM MAE / 0.53 RMSE**, re-verified after every single step in
this era (the long buffer only extends history; `_run_pipeline` still slices
its own 10 s window for HR; the beat-detection filter change only touches
`_extract_rivs`, never the main HR peak-picking path).

## Finding 7 — second independent dataset (CapnoBase), and a literature comparison

BIDMC alone can't distinguish "this method generalizes" from "this method
happens to work on BIDMC's specific population/equipment." CapnoBase (IEEE
TBME RR Benchmark, Karlen 2021) closes that gap: different institution
(Vancouver pediatric anaesthesia research program vs. BIDMC's Boston ICU),
different population (anaesthetised surgical patients, mostly pediatric, vs.
critically-ill ICU adults), different ground-truth principle (direct CO2
capnography measurement vs. two humans annotating breaths by eye/ear), and —
uniquely — the **original paper's own published SmartFusion algorithm
output** for every case, since this project's RIV-fusion method is modeled
directly on SmartFusion. Harness: `scripts/validate_rr_capnobase.py`.

**Format gotcha**: CapnoBase ships as MATLAB v7.3, which is HDF5 under the
hood — `scipy.io.loadmat` raises `NotImplementedError` on it. Needed `h5py`
instead (`pip install h5py`), confirmed not already a project dependency.

**Data-quality gotcha, caught by the harness rather than silently
corrupting results**: record `0133`'s own shipped `reference.rr.co2.y`
ground-truth trend contains 5 literal `inf` values — verified directly with
`np.isinf`, not an artifact of this project's parsing. The first full run
silently produced `RMSE = inf` for the SmartFusion comparison because one
window's mean absorbed an infinite ground-truth sample. Fixed by filtering
non-finite values out of both the ground-truth and SmartFusion aggregation
functions before they can poison a mean.

**Results, all 42 recordings, same production estimator, same
deployment-faithful 30 fps resampling, same IQR-median session aggregation
used throughout this document:**

| | This project | Published SmartFusion (same windows) |
| :--- | :--- | :--- |
| Per-window MAE | 3.03 | 1.69 |
| **Session MAE** | 3.37 | 1.59 |
| **Session median error** | **0.20** | 0.35 |
| Sessions within 2 br/min | 73.2% | 78.6% |

The SmartFusion comparison was deliberately computed with the *same*
IQR-filtered-median session aggregation as our own number, not just its
raw per-window MAE, specifically so this is an apples-to-apples comparison
rather than window-level-us vs. session-level-them.

**Honest reading — this is a more precise finding than "we're better" or
"we're worse":** on the *typical* recording, this project's method is
competitive with, and by median slightly better than, a mature published
algorithm it never had access to during development. But the *mean* is
worse (3.37 vs 1.59) purely because of outlier severity: `0115` errs by
27.48 br/min, `0018` by 21.84, `0370` by 15.18, `0035` by 14.89 — failures
far more severe than SmartFusion produces on the same recordings. This is
the same bimodal shape already documented on BIDMC (`bidmc_32`/`33`/`36`/`43`
as outliers, most other recordings excellent) — CapnoBase confirms this
bimodality is a real property of the method, not a BIDMC-specific fluke.
**The gap to the published state of the art is specifically in outlier
robustness, not in typical-case accuracy** — a much more actionable finding
than a vague "needs to be more accurate."

## Finding 8 — 2x harmonic lock, found by tracing CapnoBase's worst outliers

The user's explicit goal after Finding 7 was concrete: beat published
SmartFusion's session MAE (1.59) on CapnoBase, not just get closer to it.
Session MAE was 3.37 with a median of only 0.20 (already better than
SmartFusion's 0.35) — meaning the whole gap was concentrated in a handful
of severe outliers dragging the mean up, not a systematic problem. That is
a fundamentally different, more tractable target than "improve accuracy
everywhere."

**Traced two distinct outlier mechanisms, not one:**

1. **Physiological range mismatch (not a bug).** `0115` (gold RR ~40.6
   br/min, 99% of its true rate above 30) and `0018` (gold ~34.8, 95% above
   30) are simply outside `_RESP_HIGH`'s 30 br/min ceiling. CapnoBase
   includes pediatric/neonatal surgical patients, whose normal respiratory
   rate genuinely runs 30-60+ br/min — the estimator cannot report the
   correct answer for these patients by construction, independent of any
   algorithm quality. Left unfixed — widening the search band was already
   tested and refuted for a different reason in Finding 5 (bias didn't
   move, synthetic MAE got worse) — but now the reason THIS specific
   failure mode exists is fully understood rather than mysterious.

2. **2x harmonic lock, a genuine bug** — found by tracing `0105` (gold RR a
   rock-steady ~10 br/min for the full 8 minutes). The estimator alternated
   between correct (~10) and exactly double (~20, ratio 1.96-2.02) for
   sustained multi-minute stretches. The damning detail: **confidence was
   HIGH (82-100%) precisely on the wrong windows and LOW (1-3%) on the rare
   correct ones** — RIAV and RIIV agreeing with each other on the same
   wrong harmonic looks exactly like strong agreement to the fusion
   confidence logic. Verified directly by inspecting the raw spectrum: RIIV
   really did have more power at 2x than at the fundamental in the affected
   windows (ratio 2.62), confirmed physiological — breathing waveforms are
   asymmetric (inhale/exhale are not mirror images), so real 2nd-harmonic
   energy exceeding the fundamental is expected, not a leakage artefact.

   `_promote_subharmonic`'s own docstring had already predicted this needed
   a mirror check ("must be one-directional or it will happily walk a
   correct answer up to 2x instead") — this is that mirror, added as
   `_demote_harmonic`: same structure, opposite direction (checks the
   peak's HALF instead of its double), applied strictly after promotion so
   the two cannot fight over the same bin in one call.

**First full-dataset test surfaced a real trade-off, not a free win**: with
a shared 0.55 threshold for both directions, CapnoBase improved sharply
(session MAE 3.37 -> 2.39) but BIDMC regressed (1.60 -> 1.87). Rather than
accept the trade-off or revert, swept a separate `_RR_DEMOTE_RATIO` on
20+20-record subsets of both datasets:

| demote ratio | BIDMC session MAE (subset) | CapnoBase session MAE (subset) |
| :--- | :--- | :--- |
| 0.55 | 0.97 | 3.61 |
| 0.65 | 0.94 | 3.59 |
| **0.75-0.85** | **0.60** | 3.61-3.66 |
| 0.95 | 0.61 | 4.19 |
| 1.10 | 0.62 | 4.21 |
| 1000 (demotion disabled) | 0.61 | 5.65 |

0.75-0.85 is a genuine sweet spot, not a compromise between two competing
optima — BIDMC's subset MAE at 0.80 (0.60) is actually *better* than with
demotion fully disabled (0.61), while CapnoBase keeps nearly all of its
gain relative to disabled (3.61-3.66 vs 5.65). Set `_RR_DEMOTE_RATIO = 0.80`
and confirmed on the FULL datasets (subset numbers are directional, not
final — this project's own rule):

| | Before Finding 8 | After (both datasets, final) |
| :--- | :--- | :--- |
| **BIDMC session MAE** | 1.60 | **1.59** |
| BIDMC within 2 br/min | 81.1% | **83.0%** |
| **CapnoBase session MAE** | 3.37 | **2.51** |
| CapnoBase session median | 0.20 | **0.07** |
| CapnoBase within 2 br/min | 73.2% | **83.3%** (now beats SmartFusion's 78.6%) |

**Both datasets improved simultaneously** — not a trade-off resolved by
splitting the difference, a genuine win-win from picking the threshold on
real measured data rather than reusing the promotion threshold by
assumption. CapnoBase's within-2-br/min rate now exceeds the published
SmartFusion algorithm's own (83.3% vs 78.6%), though session MAE (2.51)
still trails SmartFusion's (1.59) — the remaining gap is smaller but not
closed. HR reconfirmed bit-identical (0.29/0.53) after this change, as
after every change in this document.

## Finding 9 — `bidmc_32` traced: a genuine signal-absence case, not a harmonic bug (negative result, kept for the record)

Following the same win from Finding 8, traced BIDMC's own worst outlier
(`bidmc_32`, session error 15.88, a 90+ year-old MICU patient with a steady
true RR of ~24-25 br/min). The failure pattern looked superficially similar
to Finding 8's: confidently wrong (91-98%) on a stretch of windows,
correctly-but-unconfidently right (2-3%) on others. But the ratio this time
was consistently ~0.31-0.38 — not the 2x/0.5x relationship `_demote_harmonic`
already covers.

**Inspected the raw spectrum directly rather than assuming a 3rd-harmonic
bug and building a fix around that assumption.** Result: the RIAV channel's
power at the true rate (~25 br/min) was only 4-8% of its power at the wrong
peak (~8 br/min) — not "a harmonic slightly stronger than the fundamental"
(Finding 8's pattern) but "the fundamental is barely present at all." This
is consistent with genuinely weak respiratory modulation in a very elderly,
critically-ill patient (the same population this project's own RIFV/RSA
analysis already flagged as blunted) rather than a spectral mislabeling a
smarter peak-picker could fix.

**Tested this conclusion rather than trusting it.** Implemented
`_promote_third_harmonic`, an exact structural mirror of the already-proven
`_promote_subharmonic`/`_demote_harmonic` pattern (same "only fire if the
alternate bin's power is genuinely comparable" gate), which by construction
can only ever be a no-op when the hypothesis is correct — if the true
rate's power isn't there, the threshold can't fire. Verified on the
*specific* case first (bit-for-bit identical trace, confirming the
hypothesis directly), then on **both full datasets** (BIDMC session MAE
1.59 -> 1.59 exactly, CapnoBase 2.51 -> 2.51 exactly, `bidmc_32` unchanged
at 15.88) — a clean, unambiguous null result on real data, not just the one
case it was built for.

**Removed the code rather than ship it.** Zero measured benefit on two real
datasets is a legitimate reason not to carry the added surface area,
independent of how principled the reasoning behind it was — this project's
own standard is verified value, not "seems like it should help." The
*finding* is kept here because it is real, useful information: it rules out
a third harmonic-order class of bug across both real datasets, and narrows
what's left. The remaining CapnoBase/BIDMC outlier tail is now understood
to be exactly two mechanisms — the range-limit cases (Finding 8) and
signal-absence cases like `bidmc_32` — and signal absence has no algorithmic
fix; it needs either a different derived feature or an honest low-confidence
report, not a better peak-picker.

## Finding 10 — live webcam RR returns nothing all session; root-caused to RIIV, not a confidence-gate bug

First real live-webcam test (not BIDMC/CapnoBase, not a recorded UBFC file —
an actual laptop webcam, live, after the 90s acquisition window fix) came
back with `final_rr = None` — "Respiratory rate could not be reliably
measured this session (insufficient RSA-modulation signal)." Worth noting
that message text is itself stale: it names RSA, but RSA/RIFV was dropped
from the fusion back in the original rewrite (see the constant-block note
above) — the real fusion is RIAV+RIIV only. Not fixed here (wording only,
no behavior implication), flagged for whoever next touches `clinical.py`.

**Investigated by reading the fusion branch logic in `_estimate_rr`
(`core/rppg.py` ~1507-1571) before assuming "webcam signal is just noisier"
was the whole story.** With RIFV dropped, exactly two candidates (RIAV,
RIIV) feed the agreement check. The branch structure is a hard binary, not
a gradient:
- `spread <= _RR_AGREE_TOL` (2.0 br/min) → confidence up to 100, enters
  `rr_history` if it clears `_RR_HISTORY_MIN_CONF` (30).
- `spread > 2.0` → falls to the "channels disagree outright" branch, which
  caps confidence at `concs[k] * 0.3`, itself clipped to a max of 25 — below
  30 by construction, regardless of how good the underlying spectral
  concentration is.
- The middle "tightest pair, reduced confidence" branch that could rescue a
  near-miss is **mathematically unreachable with exactly 2 candidates** — for
  two numbers `a,b`, `gap = 2 * spread`, so `spread > 2.0` implies
  `gap > 4.0 > _RR_AGREE_TOL`, meaning that branch's own condition
  (`gaps[j] <= _RR_AGREE_TOL`) can never be true whenever the first branch's
  condition was false. The code's own comment already says as much ("this
  branch only matters if a third channel is reinstated"). So today, with two
  channels, it's agree-within-2-br/min-or-nothing — no partial credit path
  actually exists.

**Confirmed with a temporary diagnostic** (`log.info` added at the point
`cands`/`concs` are built in `_estimate_rr`, removed after this session —
see git history around this commit if it needs re-adding) rather than
trusting the branch-logic reading alone. One live 90s session, real webcam,
real face:

```
RIAV: 12.94, 12.54, 12.07, 11.95, 11.95, 11.95, 12.01, 11.95   br/min
RIIV: 10.37, 10.31, 10.37, 10.61,  6.80,  6.33,  6.39,  6.45   br/min
```

RIAV (amplitude modulation) stayed rock-stable at ~12 br/min the entire
session — plausibly the subject's real breathing rate. RIIV (baseline
intensity modulation) started close to RIAV, then drifted down to ~6.4 and
kept sliding, opening a ~5.5 br/min gap — comfortably past the 2.0 threshold
on every single recompute. Every reading landed in the "disagree outright"
branch and was discarded, which is exactly the observed `final_rr = None` —
not a confidence-gate miscalibration, a genuine one-channel failure.

**Working hypothesis for why RIIV specifically fails on live camera but not
on BIDMC/CapnoBase**: RIIV is derived from mean pixel brightness of the face
ROI, which any consumer webcam's continuous auto-exposure/auto-gain/white-
balance loop will slowly drift independent of actual blood-volume changes —
a real confound this signal has no way to distinguish from true respiratory
baseline modulation. BIDMC and CapnoBase's RIV channels are both derived
from a contact PPG waveform, not a camera frame — there is no auto-exposure
loop anywhere in that signal path, so this specific failure mode had no way
to appear in either validated dataset. This is consistent with the
project's own standing caveat (repeated throughout this doc and in
`CLAUDE.md`) that the 1.59/2.51 br/min MAE figures describe the DSP on a
clean contact-sensor signal, never the product measuring a live face — this
is the first concrete, mechanistic instance of that gap actually manifesting,
rather than just a acknowledged-but-abstract caveat.

**Not yet fixed.** This is a single live session's evidence, not a
validated conclusion — the hypothesis (webcam AGC/auto-exposure corrupting
RIIV specifically) has not been tested against a controlled camera with
exposure/white-balance locked, nor against enough live sessions to rule out
a one-off lighting event. Per this project's standing rule, no threshold or
fusion-logic change should ship without re-running `validate_rr_bidmc.py`
and `validate_rr_capnobase.py` afterward to confirm no regression — that has
deliberately not been done yet. Candidate directions, none implemented:
locking webcam exposure/white-balance via camera API before RIIV extraction;
weighting RIAV more heavily than RIIV when they disagree instead of an
all-or-nothing gate; or deriving RIIV from a channel-ratio (e.g. CHROM's own
projected signal) that's less exposure-dependent than raw ROI brightness.

## Finding 11 — HR-adaptive search band + low-zone peak rescue (and what did NOT work)

Finding 10's RIIV dedrift made live RR possible; this pass attacked the error
that remained on the validated datasets.

**Method change that made this fast:** cache every window's RIV series once
(`scripts/rr_lab/`), then evaluate peak-selection rules on the cache in
seconds. The lab reproduced the shipped harnesses to two decimals before any
experiment was trusted (BIDMC 1.61, CapnoBase 2.45).

**Error budget (before):** a handful of recordings dominated. BIDMC: six
recordings = 63% of total session error (without them MAE ~0.67). CapnoBase:
five = ~87% (without them ~0.36). Oracle check: on BIDMC the true rate was
among the top-4 spectral peaks of at least one channel in **93%** of wrong
windows, so the information was present and the *selection* step was failing.
Wrong-window classes: BIDMC 43% "~0.5x" + 43% other; CapnoBase ~2x 25%,
~0.33x 20%, ~0.5x 12%, other 26%.

**Two fixes shipped (`core/rppg.py`):**
1. *HR-adaptive band.* True RR/HR ratio is tight across populations (5th-95th
   pct 0.08-0.27 CapnoBase, 0.13-0.26 BIDMC), so the search band is 0.06x-0.30x
   the window's own beat-derived HR (clipped 0.10-0.25 / 0.50-0.80 Hz) instead
   of a fixed 6-30 br/min. Rescues children breathing above the old 30 br/min
   ceiling (CapnoBase 0018: 21.9 -> ~0).
2. *Low-zone rescue.* The old `_promote_subharmonic` only inspects the bin at
   EXACTLY 2x (+-0.06 br/min). Slow junk peaks (vasomotor/drift, 6-10 br/min)
   have their true partner at 2.2-2.8x, never rescued. New rule: if the
   winning peak is < 12 br/min, take the strongest peak in [1.6x, 2.8x] when it
   holds >= 30% of the winner's power (bidmc_32: 15.9 -> ~0, bidmc_21, 0035).

**Result (production code through the original harnesses):** BIDMC session MAE
1.61 -> **0.96**, CapnoBase 2.45 -> **1.84**, synthetic RR 7.16 -> 4.44, synthetic
HR unchanged. **Held-out check:** 40 VitalDB surgeries (never used for tuning,
parameters frozen first): 1.21 -> **1.03**, median 0.42 -> 0.24, within-2
77.5% -> 87.5%. Gain is smaller out-of-sample (15% vs 25-40% in-sample), as
expected; results are flat across zone 11-13 / ratio 0.25-0.35.

**Tried and rejected (each traded error between populations, none was a free
win):** lowering the band ceiling floor (helps CapnoBase/VitalDB, BIDMC 0.96 ->
1.26); soft RR/HR log-normal prior; RIFV as a tie-breaker when RIAV/RIIV
disagree (CapnoBase 1.84 -> ~1.45 but BIDMC -> ~1.15-1.3 — RSA is strong in
children, near-noise in ventilated ICU adults); cross-channel verification of the
rescue; session-level spectrum averaging; temporal tracker; top-K peak
matching; fused-spectrum picking. Known regressions of the shipped fix:
CapnoBase 0016/0029 (true ~10 br/min with a genuine 2x harmonic).

**Literature (search snippets only, not verified against the papers; protocols
differ):** deep-learning PPG models report ~1.26-1.33 MAE on BIDMC and ~2.27 on
CapnoBase; best classical pipeline ~1.9 on BIDMC; Charlton 2016 reports limits
of agreement, not MAE. The only like-for-like number is SmartFusion's own
shipped output on the same CapnoBase recordings with the same aggregation:
session 1.59 / per-window 1.69 vs ours 1.84 / 1.95. We beat it on median error
(0.06 vs 0.35) and within-2 rate (85.7% vs 78.6%) but not on mean. Per-window
BIDMC is on par with classical methods (1.76 gated / ~2.2 ungated), behind
deep learning (~1.3). Do not claim to beat the literature across the board.

**Learned peak re-ranker — built, evaluated, NOT shipped.** Candidate peaks
from RIAV/RIIV/RIFV + 34 physiological features (per-channel support, harmonic
structure, RR/HR ratio, spectral flatness, stacking on the shipped rule's pick),
HistGradientBoosting, probability = learned confidence. In-population (grouped
5-fold) it wins everywhere: BIDMC 0.88, CapnoBase 1.50, VitalDB 0.68.
Leave-one-population-out over 3 datasets it LOSES (BIDMC 1.4-2.2, CapnoBase
3.3-3.9), also with population-identifying features removed and as a pure
accept/reject gate. Re-run with 120 more age-stratified VitalDB cases (30 each
children / young / middle / elderly; only 30 children exist with both signals),
leave-one-group-out over 7 groups: mean 1.34 (rule) -> 1.11 (learned), wins 5/7
(VitalDB age groups, -20% to -50%) but is worse on the two independent sources
(BIDMC 0.96 -> 1.62, CapnoBase 1.84 -> 1.92). Conclusion: it generalizes across
age within one data source, not across sources; a webcam is another source.

**Still unvalidated: live webcam RR.** No dataset here is camera-derived.
Recommended next test: paced breathing to a metronome at 6/10/15/20 br/min,
90 s each — cheap real ground truth, and the data a learned model would need.

## Open items — genuinely unresolved, do not overclaim

* **Systematic shrinkage toward the population mode, unresolved (improved,
  not solved).** Reframed partway through this era from "high-RR
  underestimate" to what a direct regression check showed it actually is:
  fitting `estimate = slope * gold + intercept` gives **slope 0.339**
  (1.0 = unbiased) on the original 3-channel/linear config — the estimator
  moves only about a third as much as reality, i.e. it compresses toward
  ~17-18 br/min from both directions (signed error **+4.93 below 14 br/min**,
  negative in every band above 14: -1.21 at 14-18, -1.76 at 18-20, -1.79 at
  20-24, -16.12 at 24+). Switching interpolation from linear to cubic spline
  moved the slope to **0.541** — a real, verified improvement (linear
  interpolation between beat-spaced ~0.7s samples acts as a mild low-pass,
  attenuating higher frequencies more) — but the DOMINANT remaining cause is
  still unknown. Tested and refuted: gate/selection bias (ungated slope 0.339
  vs gated 0.307 essentially unchanged), widening `_RESP_HIGH` (swept
  0.50/0.60/0.70/0.85/0.95 Hz — miss rate stayed at exactly 40%, synthetic MAE
  got *worse*), multi-segment Welch sub-windowing as a variance-reduction
  attempt (measured WORSE than the current full-span periodogram at every
  window length tried — more variance from shorter windows, not less), and
  inadequate beats-per-breath sampling (the `_RR_MIN_BEATS_PER_BREATH` derate
  guard added for it is **inert on BIDMC** and cannot detect the condition
  anyway, since an underestimate makes beats-per-breath look artificially
  healthy).
* **This still biases toward MISSING tachypnea**, the clinically dangerous
  direction, since `clinical.RR_TACHY` is 20 and suppressed tachypnea also
  suppresses the Compensated Shock interaction term. The original 5-recording
  "40% miss rate" figure predates the shrinkage-reduction fixes above and has
  not been re-measured on the final configuration; treat any such figure as a
  flagged risk to re-check, not a stable measured rate — the underlying pool
  of true-tachypnea BIDMC recordings is small (5 out of 53).
* `_RR_MIN_BEATS_PER_BREATH` derating is **unvalidated** — it never fires on
  the available data. Kept as a physically-motivated guard against reporting
  an over-high RR from too few beats, not because it was shown to help.
* **Per-record outliers remain even after all fixes**: `bidmc_32` (session
  error 15.88), `bidmc_36` (11.21), `bidmc_33` (6.60), `bidmc_43` (8.09),
  `bidmc_21` (7.79), `bidmc_40` (5.81). Not individually root-caused the way
  the original `subject41`-style HR harmonic-lock cases were — a real, open
  item, and the natural next place to look if further improvement is wanted.
* The disagreement-fallback ("highest spectral concentration wins") is a weak
  tie-breaker — on synthetic it sometimes picks the worse of the two
  candidates. It survives only because its confidence is always below
  `_RR_HISTORY_MIN_CONF`, so such readings never reach the clinical engine.
* **Still no webcam-video RR validation.** Both BIDMC and CapnoBase are
  contact pulse oximetry. The 1.60/3.37 br/min session MAE figures describe
  the DSP on a clean signal, NOT the product measuring a face. COHFACE
  (permission pending) or a self-collected respiration-belt pilot is
  required before any end-to-end RR accuracy claim.
* **Outlier tail — partially closed (Finding 8), not eliminated.** Two of
  CapnoBase's worst cases are now understood: `0115`/`0018` are a
  physiological range mismatch (true RR ~35-40 br/min, outside the 30 br/min
  ceiling — not fixable without redesigning the search band, and Finding 5
  already showed widening it doesn't help for a different reason), and the
  2x-harmonic-lock class (like `0105`) is now actively corrected by
  `_demote_harmonic`. `0115` itself remains a large outlier post-fix (its
  failure mode is the range mismatch, which this fix does not address) —
  worth tracing directly to confirm, not assumed. CapnoBase session MAE
  improved 3.37 -> 2.51 and now beats SmartFusion on within-2-br/min rate
  (83.3% vs 78.6%), but still trails on raw session MAE (2.51 vs 1.59) — the
  gap is smaller, not closed. `bidmc_32` (BIDMC's own worst outlier, 15.88)
  IS now traced (Finding 9): a third mechanism, genuine signal absence in a
  very elderly critically-ill patient, not a harmonic-order bug — confirmed
  by testing a structurally-analogous fix and measuring a clean null result
  on both full datasets, not by inspection alone. **All three known outlier
  mechanisms across both datasets are now identified**: range-limit,
  harmonic-lock (fixed), and signal-absence (no algorithmic fix exists for
  this one — it would need a different derived feature entirely, or an
  honest low-confidence report instead of a peak-picking trick).
* Synthetic RR MAE (~6.7-6.9) is much worse than real-data MAE (1.60), and the
  gap did not close even after fixing two real bugs the synthetic self-test
  itself caught (rr_history dedup, beat-detection narrowing — see Finding 6).
  Either the generator is unrealistically hard, or it is exposing a weakness
  BIDMC's mostly-15-24 br/min distribution does not probe (BIDMC has almost no
  recordings below 14 or above 24 br/min — see the shrinkage bullet above,
  which by definition cannot be well-measured on a narrow-range dataset). Not
  resolved.
