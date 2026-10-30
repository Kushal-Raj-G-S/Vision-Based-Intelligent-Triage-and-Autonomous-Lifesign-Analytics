# Era 22 — simulation pretraining, a spectro-temporal tracker, and the HR ensemble in the app

2026-10-30. The "frozen app" rule was lifted by the user for HR ("integrate it into the main backend").

## What was tried (all out-of-fold, grouped by subject, 125 sessions / 75 people, cleaned labels)

| Model | MAE | Notes |
| :--- | ---: | :--- |
| Classical CHROM | 11.35 | Haar ROIs, production colour |
| Old hybrid correction | 4.55 | regresses toward the mean |
| Peak selector (HistGB over spectral peaks) | 4.24 | 4.16 on outline-free landmark colour |
| Deep v2 (MLP on spectra) | 4.45 | |
| Deep v3 (MLP pretrained on GPU simulator, then fine-tuned) | 4.20 | zero-shot (simulation only) 7.45 |
| Deep v4 (conv tracker over 11 spectra, simulator-pretrained) | 4.19 | fewer >20 BPM misses (5 vs 7) |
| **Mean of selector + tracker (shipped)** | **3.75** | see caveat |
| Mean of all four (pre-specified) | 3.94 | the unbiased number |

* The simulator (`scripts/hr_deep3.py::simulate`, GPU, ~2000 traces/step): HR 48-180 BPM with drift, pulse harmonics,
  respiration, lighting drift, motion bursts, speech-like artefacts, sensor noise, through the SAME CHROM/POS spectra.
  Regime picked on the original 71 sessions only ("motion"). It buys zero-shot 7.45 vs classical 11.35 — real, but
  fine-tuned it only TIES the selector (4.20 vs 4.24) and loses cross-dataset. Do not claim simulation "beat" anything.
* Both architectures plateau at ~4.2 → the limit is the signal/data (7 speech/arithmetic sessions carry 35% of the
  error), not the model. Their errors are on different sessions, hence the ensemble gain; oracle-per-session is 2.90.
* **Caveat that must travel with the 3.75:** the selector+tracker pair was one of ~22 combinations scored on the same
  data. Paired bootstrap vs the selector alone: -0.49 BPM, 95% CI [+0.03, +1.09], pair better on 51 sessions and worse
  on 35. Quote "~3.8-3.9, video datasets, not live-webcam validated", never 3.75 as a clean result.
* Strict split (fine-tune on the original 71, test on the new 54): tracker 5.74 vs selector 5.16 (out-of-fold over all
  125 — not the same protocol). Cross-dataset: tracker Phys->rPPG 2.94 (selector 3.29), rPPG->Phys 5.80 (selector 4.88).

## What shipped

* `core/hr_ensemble.py` — numpy/scipy only (no torch/sklearn/joblib). `TreeEnsemble` (exported HistGB, parity with
  sklearn `predict_proba`: max diff 0 on 402k rows), `TrackerNet` (numpy forward of `SpecTracker`, 3 seeds averaged,
  parity with torch 3e-4 on softmax), `HREnsembleSession` (second POS engine + feature building + both models on a worker
  thread; `feed()` from the frame loop, `estimate()`, `flush()`, `finalize()`).
* Models: `models/hr_selector.npz` (72 kB), `models/hr_tracker.npz` (2.7 MB), trained on ALL 125 sessions with
  `scripts/export_hr_selector.py` (main venv) and `scripts/export_hr_tracker.py` (GPU venv, needs
  `dataset/models/hr_tracker_pretrained.pt` from `scripts/hr_deep4.py`).
* Wiring: `web/__init__.py` (loads models, `SessionState.hr_ensemble`), `web/routes.py` (`_start_hr_ensemble` beside every
  new `AdvancedRPPG`, `feed()` after `process_ppg_signal()`, gauge prefers the ensemble once >=10 selector decisions,
  video-end summary), `agents/tools.py::get_vitals_dict` (live: ensemble > hybrid > classical; file: `final_bpm_ensemble`),
  `scripts/analyze_video.py` (`use_ensemble=False` default so every validation script is unchanged).
* The ensemble never raises into the app: models missing / `VITAL_HR_ENSEMBLE=0` / worker error -> the old
  hybrid/classical path runs exactly as before.

## Verification done

* `scripts/test_hr_ensemble.py`: tracker numpy==torch; streaming picks == batch research pipeline (295/295 and
  345/346; the one flip is a 0.43-vs-0.42 near-tie from float32 feature storage).
* Product path on `subject5` (truth 98.3): classical 100, hybrid 97.1, **ensemble 99.0** (selector 99.6, tracker 98.4),
  identical to the offline replay. Synthetic HR self-test unchanged: 0.29 / 0.53.
* Live MJPEG path exercised through the real server with a played video: worker thread, gauge, no queue overflow.

## NOT verified — do not overclaim

* No live-webcam ground truth. The models saw UBFC video via the Haar tracker; a webcam with auto-exposure is a different source.
* Live fps != 30: the models count in cycles (training clips were ~30 fps), same limitation as the BPM lock logic.
* `mouth_open` is 0.0 in the app (Haar has no landmarks; training had NaN->0).
* The final models are trained on all 125 sessions, so their in-sample answers are optimistic; only the out-of-fold
  numbers above are evidence. Tracker decisions run every 15 cycles live (evaluated at 5) — not separately re-evaluated.
* Reproducing the research runs: `dataset/cache/results_deep{2,3,4}.txt`. The exploratory landmark-colour tracker run
  (`deep4mp`) and the cheeks/Haar-clean pipelines were stopped to free the machine and are resumable, not deleted.
