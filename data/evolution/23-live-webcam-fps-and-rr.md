# Era 23 — live webcam: the frame-rate bug, and live respiration that is still WRONG

2026-10-30 (night). Found while testing the HR ensemble (Era 22) on the real webcam.

## 1. Frame-rate bug (found, fixed, verified offline)

* Symptom: live sessions showed no RR at all ("Needs ~30 s of signal" to the end), report RR 0.
* Cause: `AdvancedRPPG` converts every frequency with ONE fixed `fps` and ignores timestamps. A webcam driver reports a
  nominal 30 fps (here `CAP_PROP_FPS` = -1 -> default 30) but the live loop delivered **9.6-11.5 fps** (later 4.5 fps with
  several dashboard tabs pulling the same camera). RR's 900-sample buffer (30 s at the assumed rate) took ~90 s to fill, and
  every frequency was scaled by 30/real_fps.
* It also breaks HR, not only RR: recorded sessions thinned to ~10 fps and fed as "30 fps" gave 65 / 138 / 137 / 130 / 55 BPM
  against truths of 98-111. The live HR of 82-85 seen that night may therefore have been wrong too.
* Fix: `core/live_resampler.UniformResampler` — live frames are resampled by timestamp onto a uniform 30 Hz grid
  (`web/routes.py::_run_dsp`, `_start_live_resampler`); the engine and the HR ensemble then run on exactly the timing they were
  validated at. File sources are untouched. Also: camera driver queue = 1 frame, and the 20 ms sleep removed for live.
* Evidence (`scripts/test_live_resampler.py`, cached ROI traces rebuilt as camera streams at 6/8/10/15/20/24/30/45/60/120 fps
  with jitter): HR stays within ~1-2 BPM of the native-30-fps answer at EVERY rate on 3 sessions; the old behaviour is wrong at
  every rate except 30 (51-147 BPM for a true 98-111). RR matches native where the session has one.
* In-process loop speed with a single consumer on the real camera: ~19-21 fps (raw camera 23 fps at 640x480); the HR ensemble
  costs only 1-2 fps. Extra dashboard tabs sharing the one `Camera` object are what pushed the server loop to 4.5 fps.
* Limits: below ~6 fps a pulse cannot be recovered at all. Not yet confirmed with a long live run on the real camera.
* Invalid checks (do not reuse): forcing a recorded file through the live path with wall-clock timestamps while the file plays
  slower than real time, and a paced fake-webcam wrapper that could not keep real-time pace.

## 2. Live respiration is still WRONG (not fixed)

* Live session "one last time for rr": the app showed **RR 7 br/min** (gauge confidence 1%), report RR 7; the subject counted
  **~22** breaths/min. HR 88 was plausible.
* What the channel log showed (`RR candidates (TEMP DIAG)`, order = RIAV, RIIV):
  * RIIV (baseline brightness) sat at **7.2-7.3 br/min for the first ~80 s** — slow drift near the band floor (0.10-0.12 Hz),
    almost certainly the webcam's auto-exposure, not breathing; it read 22-24 only in the last ~25 s.
  * RIAV (pulse amplitude) wandered 6.7 -> 16 -> 13-14 and never read 22.
  * Early on the two coincidentally agreed near 7, so those readings passed the 30% gate into `rr_history`. The session median
    never forgets them, so the gauge and the triage RR kept 7 although the instantaneous confidence was 1%.
* This is the Finding 10 hypothesis (RIIV contaminated by exposure drift, band-edge lock-on) seen again with a counted ground truth.
* Why no quick fix was shipped: the same channel read 7.2 for 80 s and 22 for 25 s in one session, so tuning to this session
  would just fit the one answer (n = 1). Candidate directions, none validated:
  1. suppress band-edge / drift-locked RR and let the live gauge use only recent readings that pass the confidence gate
     (shows "--" instead of 7; does not produce 22; needs BIDMC + CapnoBase re-validation first);
  2. a proper live validation: 10-15 live sessions with counted RR, tune on some, test on the rest;
  3. lock webcam exposure, or derive RIIV from a less exposure-dependent channel.
* Consequence for claims: respiration is validated on RECORDED data only (BIDMC, CapnoBase, VitalDB). Do NOT claim live-webcam RR.
  HR live is plausible and the frame-rate fix is verified offline, but live HR accuracy against a reference is also not measured.

## 3. Final headline numbers (what to say if asked)

* HR: selector+tracker ensemble ~3.8-3.9 BPM MAE, 125 sessions / 75 people, out-of-fold (pre-specified mean of four models: 3.94;
  classical 11.35, old hybrid 6.69 on the same sessions). Video datasets, not live.
* RR: ~1.0 breaths/min session MAE — held-out VitalDB 1.03 (never tuned on), BIDMC 0.96, CapnoBase 1.84. Recorded data, not live.
