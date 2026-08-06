# data/ — the evidence archive

This folder exists so the full history of this project's validation and
debugging work survives independent of any one conversation, laptop, or
session. Start here:

- **[EVOLUTION.md](EVOLUTION.md)** — index into the full chronological
  story, now split into one file per era under
  [`evolution/`](evolution/): every bug found, every fix attempted
  (including the ones that failed and why), every real measured number, in
  the order it actually happened. Written so a person or an AI with zero
  prior context can read it and fully reconstruct what was done and why.

The raw evidence EVOLUTION.md references lives in [`test/`](../test/), not
duplicated here, to avoid two copies of the same file drifting apart:

| What | Where |
| :--- | :--- |
| Classical-DSP-only 56-session validation history | `test/validation_run5_final.log`, `test/validation_run6_fpsfix.log`, `test/validation_run8_historyfix_v2.log` |
| The naive (broken) history-unstick fix's real-data regression | `test/validation_run7_historyfix.log` |
| Harmonic lock-on root-cause traces (3 fix attempts) | `test/trace_subject41*.log` |
| Hardcoded-FPS bug root-cause trace | `test/trace_subject25.log` |
| History-freeze bug traces (subject37 fixed, subject41 regression + restore) | `test/trace_subject37_regression.log`, `test/trace_subject41_v4_regression.log` |
| Agent narration faithfulness evaluation (raw + aggregated) | `test/agent_reliability_run2.log`, `test/agent_reliability_results.jsonl` |
| Windowed ML-correction dataset build (242K cycles, 51 subjects) | `test/ml_dataset_build.log` |
| Correction-model training (small vs. bigger comparison) | `test/ml_train_run1.log`, `test/ml_train_bigger.log`, `test/ml_train_final.log`, `test/ml_train_foldmodels.log` |
| Classical-vs-hybrid session-level validation (current, 71 sessions, out-of-fold) | `test/validation_run9_hybrid_oof.log` (original 56) + `test/validation_run10_new_subjects.log` + `test/validation_run10b_remaining.log` (the 15 added sessions) |
| Manifest of every real video used, with ground truth (now 71 sessions) | `test/manifest.csv` |
| Deployed correction model + the 5 fold-specific models used for honest out-of-fold evaluation | `models/bpm_correction_gbr.joblib`, `models/cv_folds/` |

For the current, final numbers (not the history of how they were reached),
see [README.md](../README.md)'s Validation section. For guidance on working
in this codebase, see [CLAUDE.md](../CLAUDE.md).
