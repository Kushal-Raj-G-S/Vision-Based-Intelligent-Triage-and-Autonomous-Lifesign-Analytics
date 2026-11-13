# Proof of concept: how the heart-rate number is produced

One folder per stage of the pipeline, all generated from the real code (`core/camera.py`, `core/rppg.py`, `core/hr_ensemble.py`)
by a single script. Nothing here is re-implemented for the pictures.

```
./venv/Scripts/python.exe poc/run_poc.py "C:/Users/M.Manoj/Downloads/test.avi" --truth 98.3
```

`--truth` is the reference heart rate from the finger oximeter (UBFC-rPPG subject5 = 98.3 BPM); leave it out for a video without one.

| Folder | What it shows |
| :--- | :--- |
| `01_frames_and_rgb/` | video -> frames (`frames/`, one per second, boxes = the 3 skin patches), `contact_sheet.png`, `rgb_per_frame.csv` (average R,G,B of every frame, also per patch), `rgb_trace.png` (level, then the tiny pulsating part), `channel_contribution.txt` (which colour channel and which skin patch carries the pulse, with numbers) |
| `02_pulse_signals_chrom_pos/` | CHROM and POS pulse waveforms: raw projection (grey) vs after the band-pass (green), heartbeats marked; `chrom_vs_pos.png`; `pulse_last10s.csv` |
| `03_welch_spectrum/` | Welch frequency spectrum of the band-passed pulse for CHROM and POS at three moments; `peaks.txt` lists the top peaks (the tallest is not always the heart rate - harmonics) |
| `04_ensemble/` | `selector_candidates.png` (candidate peaks and the probability the selector gives each), `tracker_posterior.png` (what the conv-net believes over time), `decisions_timeline.png` (every decision of both models), `ensemble_result.txt` (classical vs selector vs tracker vs ensemble vs reference) |

## How to read it (the story in one paragraph)

The webcam video is split into frames. For every frame the average R, G, B of the forehead and both cheeks is one number each
(stage 1). Each heartbeat changes skin colour by about half a percent, so the pulse is a tiny wobble on top of a large steady
level. CHROM and POS combine the three colours so lighting and motion cancel and the pulse remains (stage 2). A Welch transform turns
the last 10 seconds of that pulse into a frequency spectrum, whose tallest peak is the heart rate (stage 3) - but harmonics and noise
can create competing peaks, so the ensemble (stage 4) scores several candidate peaks with a tree model and checks which peak
persists over time with a small conv-net, then averages the two answers.

## Honest notes

* The ensemble models were trained on all 125 recorded sessions, which includes this clip's session, so the single-video result
  (99.6 BPM vs 98.3) is optimistic. The unbiased figure is the out-of-fold error, about 3.9 BPM on 125 sessions / 75 people.
* Stage 1's text states what THIS clip shows: the three colour channels pulse almost equally here, so no channel is singled out.
* Recorded video only. Live-webcam accuracy is not validated (see `data/evolution/23-live-webcam-fps-and-rr.md`).

## Automatic runs: `poc/patients/<patient name>/<date_time>_<upload|live>/`

Every finished upload and every live session now builds its own folder like the one above, by itself, in the background
(`core/poc_jobs.py` starts `core/poc_report.py` as a separate low-priority process, at most 2 at a time; the app never waits for it and
any failure is contained). `poc/patients/index.html` lists all runs with their state and numbers; each run also has `summary.txt`,
`result.json`, `status.json` and `build.log`. Stage 1 additionally holds the red / green / blue channel frames and
`channel_contribution.png` + `channel_contribution.txt` (which colour contributes more, the values, and why).

* The patient name comes from the dashboard's name box (sent when you press Start / Load video); it falls back to the file name, then "Unnamed".
* A live session is recorded as numbers every frame plus one clean frame per second, so the report can be built after Stop. For live runs
  the report uses the same uniform 30 Hz resample as the app.
* Needs ~10 s of face video; shorter recordings get a note instead. Heart rate only. A 1-minute video takes a few minutes to build.
* `VITAL_POC=0` switches it off. `poc/patients/` is git-ignored because it holds face images.

## Viewing the runs (for a reviewer or judge)

Double-click **`poc/patients/index.html`**: patient cards with a thumbnail, status badge and headline numbers; **Open report** shows one run as a page
(`report.html` inside the run folder): the result cards, then the four stages with every picture, a plain-language caption and the numbers. Click any
picture to enlarge. Plain HTML with relative links - no server needed, and it prints cleanly. Rebuild every page with `python -m core.poc_site`.
The index refreshes itself every 10 s while a run is still building. To hand it to someone, zip the `poc/patients` folder (it contains face images).

## Conclusions (what was done, what it shows, what is checked, what comes next)

Every report has an **In short** card at the top and a **Conclusion of this stage** box at the end of each stage; the same text is saved as
`conclusions.txt` in the run folder. They are generated by `core/poc_conclusions.py` **from the measured numbers only - no language model**, so a value
cannot be misquoted. Each check is marked pass / note / fail / info and carries its evidence, e.g. "CHROM and POS agree within 10 BPM in 1 of 3 windows",
"the two models agree within 5 BPM", "error vs the reference 1.3 BPM (target <= 5)". Only the signal-quality checks (stage 1) and the final-answer checks
(stage 4) can make the verdict "treat with caution"; disagreement between single methods in the middle stages is reported as a note, because resolving it is
exactly what stage 4 is for. Older runs built before this feature show fewer checks (their per-window numbers were not saved).
