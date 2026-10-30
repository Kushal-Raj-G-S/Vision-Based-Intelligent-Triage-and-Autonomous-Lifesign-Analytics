# Era 21 — HR: data audit after the re-download, why the number stalled, and the plan

Written 2026-10-30 before any new HR run. **Nothing in this file is a new result**: the numbers are the
existing 71-session run (`test/validation_run11_huber_full71.log`) re-analysed, plus a file audit of what is
on disk now. Anything marked *exploratory* was tuned on the same 71 sessions and must not be quoted.

## 1. Coverage audit (what is on disk vs what the old numbers used)

| | Old 71-session set | On disk now | New |
| :--- | :--- | :--- | :--- |
| UBFC-rPPG dataset_2 | 40 subjects | 42 subjects, all complete | 2 (subject18, subject20) |
| UBFC-rPPG dataset_1 | 1 clip (`after-exercise`) | 8 clips | 7 clips (5,6,7,8,10,11,12) |
| UBFC-Phys | 10 subjects x 3 = 30 sessions | 25 subjects x 3 = 75 sessions, all complete | 15 subjects (45 sessions) |
| **Sessions** | **71** | **125** | **54** |
| **People** | **51** | **67 confirmed (+ up to 8 from dataset_1, not yet confirmed distinct)** | **17 (+ up to 7)** |

- **All 71 old sessions are present and complete** (checked by file existence and size, session by session,
  including the 9 UBFC-Phys sessions of s11, s16, s33). Every old subject is covered.
- Layout problems to fix before any script reads these: folders have spaces (`ubfc rppg`, `ubfc phys`);
  the newer Phys subjects are extracted one level too deep (`s11/s11/vid_s11_T1.avi`); dataset_1 folders carry
  Google-Drive timestamps (`5-gt-20261030T054959Z-1-002`) and mixed file names (`vid.avi`, `vid-001.avi`,
  `vid-001 (1).avi`) with `gtdump.xmp` ground truth, unlike dataset_2's `ground_truth.txt`.
- Still not downloaded: UBFC-Phys s23, s26-s32, s34-s56 (31 more people would complete all 56).
- UBFC-rPPG readme: faces of subjects 21 and 27 must not appear in any publication. We have subject27.

## 2. The numbers (old 71 sessions, hybrid evaluated out-of-fold)

| | MAE | median error | within 3 BPM | within 5 BPM | errors > 20 BPM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Classical DSP (CHROM) | 9.43 | **3.70** | 32/71 | 42/71 | 14 |
| + ML correction (hybrid) | **6.48** | 4.30 | 27/71 | 40/71 | 4 |

By condition (classical -> hybrid MAE): UBFC-rPPG dataset_2 (n=40) 6.8 -> 5.5; Phys rest (10) 5.1 -> **6.1 (worse)**;
Phys speech (10) 21.0 -> 7.8; Phys arithmetic (10) 10.4 -> 7.0; after-exercise (1) 32.9 -> 32.3 (not fixed).

## 3. Why the number would not come down (evidence, not opinion)

1. **The error is a tail, not a typical-case problem.** Classical median error is only 3.7 BPM. The worst 10
   sessions are 57% of all classical error; without them the MAE is 4.76. Fourteen sessions are off by more than
   20 BPM. Lowering MAE means fixing those sessions, not polishing the other 57.
2. **The ML correction repairs the tail but damages the typical case.** It cut >20 BPM failures from 14 to 4 but
   its median error is *worse* (3.70 -> 4.30), sessions within 3 BPM fell 32 -> 27, and Phys rest got worse.
   Cause: it **shrinks toward the population mean**. Slope of estimate vs truth is 0.76 classical but **0.57
   hybrid** (1.0 = unbiased); mean signed error is +7.9 BPM when true HR < 79 and **-7.4 BPM when true HR > 95**.
   This is the same shrinkage seen in RR (Era 19). It is a property of regressing a value from 51 people.
3. **Only 51 independent people.** 242,000 training rows are windows from 51 faces. Bigger models were tested and
   did not help (Era 10). The improvement itself is **not statistically significant**: session-level Wilcoxon
   p = 0.27, subject-level p = 0.24 with a 95% CI that includes zero, and ~5 subjects carry the whole net gain
   (Era 16). Do not claim "the hybrid reliably helps" from this data.
4. **The correction cannot recover what the DSP never measured.** Sessions where the spectrum locked onto a
   harmonic (0.5x / 2x) for the whole clip, or lost the face / were corrupted by speech motion, give the ML model
   a wrong input. Open regressions: after-exercise, subject22, subject24, subject32, s4-speech (CLAUDE.md).
5. **High heart rates are the hard half.** 22 of the 71 sessions have truth > 100 BPM: MAE 12.2 classical /
   9.7 hybrid, versus far lower below 100. Harmonic ambiguity is worst there, and the hybrid under-reads them.
6. **Ground truth and metric definition have never been audited end to end.** The session "truth" is derived
   from the oximeter signal; how (window, peak method) must be shown to match how the estimate is aggregated
   (IQR median of the history). A 1-3 BPM disagreement here is a floor on every number we report.

*Exploratory only (tuned in-sample, do not quote):* using the hybrid only when it disagrees with classical by more
than 8 BPM gives MAE 6.14 and median 3.40 on the same 71 sessions. Choosing the better of the two per session
(an oracle, impossible in practice) gives 4.65, so a gate alone cannot go far: 4.65 is the floor for "pick one
of these two estimators".

## 4. Plan

**Phase 0 — make the data usable and the comparison fair (no model work yet)**
- Rename to `ubfc_rppg` / `ubfc_phys`; flatten the nested Phys folders; keep an untouched copy of the old names
  only if something still references them.
- Extend `scripts/build_manifest.py` for the new layout and dataset_1's `gtdump.xmp`. Emit `manifest_v2.csv` with
  subject_id, dataset, condition, source, split. Keep `manifest.csv` as `manifest_old71`.
- **Freeze the test set now:** the 17 new people (+ their dataset_1 clips once confirmed distinct) were never used
  for any tuning, threshold, fix or model training. They are the held-out test. The 51 old people are
  development data (GroupKFold by subject, as before). Write this down before running anything.
- **Reproduce the old numbers first.** Run the old 71 sessions with the current code and the saved fold models.
  Expect 9.43 / 6.48. Code has changed since (RR work, `ml_compat.py`); if the numbers move, find out why before
  doing anything else.
- **Ground-truth audit:** one function to derive session truth from `ground_truth.txt` / `bvp_*.csv` /
  `gtdump.xmp`; cross-check the three sources against each other on subjects that have more than one.

**Phase 1 — the honest baseline on unseen people**
- Run classical and the *existing* hybrid (trained on the 51 old people) on the 54 new sessions. This is the first
  true out-of-sample number the project has had. Sequential, one process at a time. Measure per-session time first
  (old rPPG sessions took about 45 s each; Phys clips are larger).
- Report MAE, median, within-5-BPM, RMSE, Bland-Altman limits, per dataset, per condition, per HR band, and at
  the subject level with a paired bootstrap CI and Wilcoxon test.

**Phase 2 — diagnose the tail before choosing a fix**
- For every session with error > 10 BPM, label the failure mode automatically: harmonic lock (estimate is about
  0.5x / 2x / 1.5x of truth), whole-clip stable wrong lock, motion/speech (low SQI, high motion), face or ROI
  loss, high-HR band edge, ground-truth mismatch. Count sessions and the share of total error per mode.
- Pick targets by expected MAE reduction. This is what decides the next phase; do not skip it.

**Phase 3 — candidate fixes, in order of expected value per effort** (each: development set with GroupKFold,
then the frozen test set once)
1. **Candidate-selection instead of regression.** Generate the top spectral peaks from CHROM (and POS), and
   train a ranker to *choose* a peak (with an abstain option that keeps classical). The output is always a real
   spectral peak, so it cannot shrink toward the mean, which targets cause 2 directly. Lesson from RR: evaluate
   leave-one-dataset-out (rPPG vs Phys), not only across people from one source.
2. **Cross-algorithm agreement.** CHROM undershoots and POS overshoots to harmonics (Eras 4, 14), so their
   disagreement is a signal. POS is not wired into the app; it only goes in after full re-validation.
3. **Viterbi/Kalman tracking over the spectrogram** instead of per-window argmax plus hysteresis. Fixes brief wrong
   locks; will not fix whole-clip wrong locks (subject25 type).
4. **Speech/motion handling:** exclude mouth-adjacent regions when landmarks show the mouth moving; weight ROIs by
   their own SNR.
5. **Pretrained deep model as an extra candidate source** (not as the whole system). Only with weights trained on
   data that excludes UBFC, otherwise it leaks. Feasibility (licence, weights, GPU time) is unchecked; treat as an
   option to investigate, not a commitment.
- Do **not** repeat: bigger models (Era 10), majority-vote gate (Era 13), RIFV-style extra knobs without a held-out
  check. Every fix is re-run against the known-good cases (subject41, subject25, subject37) — the Era 4 lesson.

**Phase 4 — what the papers need from this**
- HR paper: the contribution is the failure analysis and the validation discipline (three root-caused bugs,
  tail diagnosis, shrinkage finding, held-out new subjects), plus whatever Phase 3 achieves. State plainly that
  published deep-learning results on UBFC-rPPG are far lower than ours.
- System paper: a small live study with a reference oximeter, Fitzpatrick type recorded per person (no skin-tone
  result exists yet; UBFC has no labels).

## 5. Decisions needed from the user
1. Confirm the frozen test set = the 17 new people (+ dataset_1 clips). Everything old stays development data.
2. OK to rename/flatten the dataset folders and extend the manifest builder.
3. Download the remaining 31 UBFC-Phys subjects? (More people is the main lever; they would extend the test set.)
4. Is a pretrained deep model acceptable as a component, or must the HR system stay classical + light ML?
5. What counts as success: lower MAE, or fewer catastrophic (>20 BPM) sessions? They are different targets and
   Phase 3 options trade them off.

---

## Update (2026-10-30, later) — tooling built, and the reference labels turned out to be partly wrong

### New HR lab (everything reproducible from cached traces, no video re-tracking)
| Step | Script | Notes |
| :--- | :--- | :--- |
| Layout | `dataset/hr/ubfc_rppg/{dataset_1,dataset_2}`, `dataset/hr/ubfc_phys/sN` | folders renamed/flattened by moves only; every size checked unchanged |
| Session discovery + truth | `scripts/hr_data.py` | 125 sessions: dataset_1 x8, dataset_2 x42, UBFC-Phys x75 (25 subjects) |
| Face tracking, once | `scripts/build_roi_cache.py [--shard i/n]` | records per-frame pooled RGB exactly as production, plus per-ROI means (with and without the debug outline), mouth opening, face geometry. Resumable; run shards of up to 4 (each uses ~2.5 cores) |
| Replay | `scripts/hr_replay.py --check-old` | **reproduced the old classical result exactly on 9/9 sessions** (including the failures), so the cache is faithful and the engine changes since did not touch HR |
| Features | `scripts/hr_lab_build.py` | CHROM + POS per cycle, lock-independent wide-band spectra (`AdvancedRPPG.capture_spectra`, off by default; synthetic check unchanged at HR 0.29/0.53, RR 4.44) |
| Evaluation | `scripts/hr_lab.py baselines|peaks` | out-of-fold, grouped by subject; three label views (below) |
| Peak selector | `scripts/hr_peaks.py` | chooses among real spectral peaks of CHROM and POS (plus the production estimate) with a classifier, then a Viterbi track |

### Ground-truth audit (reference = pulse oximeter) — an evaluation problem, not a model result
- UBFC-rPPG dataset_2: the oximeter HR column contains dropouts (readings of 0 or 7 BPM). The original session truth was the **mean** of that column, which a dropout corrupts. 11 of 42 sessions look unreliable by some measure; 4 have >10% impossible readings (subject11 20.6%, subject18 12.3%, subject24 16.5%, subject20 51.1%).
- subject11: label 96.2, valid-median 121.0, oximeter's own waveform 127; the system said 119. **The system was right; the label was wrong.** Same pattern for subject24.
- Fixing the label on the 40 old dataset_2 sessions alone (no system change): classical MAE 6.78 -> 5.83. The hybrid goes 5.47 -> 5.67 — it had partly learned the corrupted labels.
- Rule (fixed in advance, from the reference device only, never from our estimates): robust truth = median of readings within 40-180 BPM; session flagged unreliable if >10% of readings are impossible. UBFC-Phys needs no change (its median-beat-interval truth agrees with the median of 10 s windows to ~1.1 BPM).
- Every table now shows three views: original labels (comparable with the old 9.43/6.48), cleaned labels, cleaned labels with unreliable-reference sessions removed. **Report all three; never only the flattering one.**
- Consequence for training: the old correction model was trained on the uncleaned per-window labels. The new model uses cleaned labels.
- Gotcha for anyone editing these scripts through this tooling: `\n` inside an inline Python patch gets collapsed to a real line break and breaks the target file; use `chr(10)` / `chr(92)` or avoid escapes.

---

## Update — the peak selector works on UBFC-rPPG (preview), full run queued

### Preview, 50 UBFC-rPPG sessions only (42 dataset_2 + 8 dataset_1), out-of-fold, grouped by subject
| Estimator | MAE, original labels | MAE, cleaned labels | median (cleaned) | >20 BPM failures (cleaned) |
| :--- | :--- | :--- | :--- | :--- |
| Classical (CHROM) | 11.02 | 8.98 | 2.00 | 8 |
| Old hybrid (regression, 51 people) | 7.58 | 7.83 | 4.29 | 6 |
| Regression retrained on cleaned labels | 6.62 | 6.23 | 2.35 | 4 |
| **Peak selector (new)** | **4.87** | **2.19** | **1.40** | **0** |

Subject-level paired tests (cleaned labels, reliable references, 46 subjects): selector beats classical by 7.40 BPM
(95% CI 3.40-12.21, Wilcoxon p = 0.0017), the old hybrid by 5.24 (CI 2.90-8.07, p = 8e-7) and the retrained
regression by 3.34 (CI 1.25-6.11, p = 0.0001). Examples: subject41 classical 54 -> selector 97 (truth 100);
after-exercise 81 -> 103 (truth 108). Worst remaining selector errors are 8-11 BPM, two of them on sessions whose
reference is flagged unreliable.

**Why it works (hypothesis consistent with the data):** it never regresses a value, so it cannot shrink toward the
mean; it looks at lock-independent spectra of two algorithms (CHROM, POS) that fail in different places; per cycle a
true-rate peak exists ~76-78% of the time while the production estimate is right only ~58-59%.

**NOT yet shown — do not claim before the full run:** UBFC-Phys results (speech/arithmetic), and the cross-dataset
test (train on one source, test on the other). The RR peak re-ranker passed within-source CV and FAILED that test.

### Where the full results land (automatic, no action needed)
- `dataset/cache/run_all.log` -> "ALL DONE" when the main chain finishes
- `dataset/cache/results_baselines.txt` — classical / POS / old hybrid on all 125 sessions, three label views
- `dataset/cache/results_peaks.txt` — candidate recall + peak selector (argmax and Viterbi) on all 125
- `dataset/cache/run_after.log` -> "MORE DONE"; `dataset/cache/results_more.txt` — paired tests, retrained
  regression, per-dataset/condition breakdown, CROSS-DATASET test both directions, worst sessions
- Rerun anything: `python scripts/hr_lab.py baselines|peaks`, `python scripts/hr_more.py` (minutes, no video)

### How to read them
1. Reproduction gate: old 71, original labels, classical should be ~9.4 and old hybrid ~6.5 (UBFC-Phys sessions
   included). If not, investigate before trusting anything else.
2. Headline = cleaned labels, all 125. Always also quote original labels (comparable to old work / literature).
3. The claim stands only if the cross-dataset rows also beat classical on the test source.
4. The live app is NOT changed: the selector exists only in scripts/. Deploying it needs a numpy-only export and
   a live-safety review (project was frozen for the presentation).

### Added to the automatic follow-up (`scripts/hr_more.py`, writes `results_more.txt`)
- Three selector aggregations fixed in advance (argmax / score-weighted / abstain below 0.5) — designed on
  UBFC-rPPG only, so UBFC-Phys is a genuine held-out judge. Report all three; don't pick the best after the fact.
- Agreement metrics papers report: Pearson r and Bland-Altman bias with 95% limits of agreement, for every method.
- Every session's estimate from every method: `dataset/cache/results_sessions.csv` (for tables/plots/slides).
- Preview on 51 sessions (50 UBFC-rPPG + 1 Phys), cleaned labels: selector r = 0.975, bias -0.3, 95% LoA
  [-7.8, +7.3] BPM; classical r = 0.572, LoA [-37.2, +29.9]; old hybrid r = 0.712, LoA [-27.9, +18.0].
