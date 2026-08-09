# CLAUDE.md — VITAL project guide for coding agents

This file orients a coding agent (or a future you) working in this repo. It
is deliberately blunt about what's solid, what's fragile, and what's
genuinely unresolved — this project was put through an honesty audit and
several rounds of real-data validation; don't undo that by re-introducing
claims the code doesn't back up.

**For the full chronological story** (every bug found, every fix tried
including the failed ones, every real number, in the order it happened),
see [`data/EVOLUTION.md`](data/EVOLUTION.md). This file is the short/brief
version — that one is the detailed one.

## What this project is

VITAL (aka "Opti-Screen") is a college project: contactless vitals
extraction via webcam rPPG, feeding a CrewAI 3-agent "triage" pipeline that
classifies ESI acuity and Compensated Shock, displayed on a Next.js
dashboard with a Flask API backend and Neon serverless Postgres. See
[README.md](README.md) for the full architecture, feature list, and
Validation section — read that before this file for context; this file is
about how to work in the code, not what it does for a user.

## The one rule that matters most here

**The deterministic clinical engine (`core/clinical.py`) is the sole source
of truth for ESI level, Compensated Shock flag, and priority score. The LLM
agents (`agents/`) narrate and explain that output — they never compute or
override it.** This was NOT always true; an earlier version let the
Coordinator Agent freely generate these numbers as JSON, parsed with regex,
falling back to a hardcoded `esi_level=3` on any parse failure. If you're
touching `web/routes.py::triage_run_api`, `agents/crew.py`, or
`agents/config/tasks.yaml`, preserve this separation. Don't let an LLM
compute a number that drives real triage priority.

## Where things actually live

| Concern | File |
| :--- | :--- |
| rPPG DSP engine (CHROM/POS, adaptive filter, SQI, HRV) | `core/rppg.py` |
| Camera/MediaPipe ROI extraction | `core/camera.py` |
| Deterministic ESI/shock scoring | `core/clinical.py` |
| Lightweight guideline/literature retrieval | `core/guidelines.py` |
| CrewAI agents/tasks | `agents/crew.py`, `agents/config/*.yaml`, `agents/tools.py` |
| Flask app, shared session state | `web/__init__.py` |
| API routes | `web/routes.py` |
| Neon Postgres access | `web/neon_db.py` |
| Validation harness (synthetic + real-dataset) | `scripts/validate_dataset.py` |
| Multi-dataset manifest builder (UBFC-rPPG + UBFC-Phys) | `scripts/build_manifest.py` |
| DSP-core-only validation for face-incompatible datasets (rPPG-10) | `scripts/validate_cropped_roi.py` |
| Next.js dashboard (single file, ~1700 lines) | `frontend/src/app/page.tsx` |

## Known-good vs. known-fragile (read before touching)

**Solid, tested, trust these:**
- `core/clinical.py` — unit-tested against 8 hand-built clinical scenarios (normal, isolated tachycardia/tachypnea, borderline/severe shock, bradycardia, low-SQI, no-signal). Any Compensated Shock flag forces `needs_manual_review=True` unconditionally — don't remove that, it's a deliberate safety policy, not a confidence heuristic.
- CHROM on synthetic signal: 0.29 BPM MAE / 0.53 RMSE across 50-140 bpm under 2% sensor noise (`python scripts/validate_dataset.py`).
- The concurrency fix in `agents/crew.py` (`build_triage_crew()` factory, not a module-level singleton) — CrewAI `Task` objects store output on `self` and are not safe to re-enter from concurrent Flask threads.
- `web/neon_db.py::get_all_patients()` now raises on a real DB failure instead of returning `[]` — do not revert this. It used to make a transient Neon cold-start connection hiccup indistinguishable from "the queue is genuinely empty," which looked exactly like data loss to an end user. Every caller must catch and surface the error, not swallow it.

**Fragile / partially fixed, do not overclaim:**
- **CHROM on real video (classical DSP only, no ML correction): 9.4 BPM MAE / 16.5 RMSE across 71 real sessions (41 UBFC-rPPG + 30 UBFC-Phys)**. This is *after* three independent DSP-layer fixes below (was 17.4/24.1 MAE/RMSE before any).
- **Hybrid classical-DSP + ML correction layer (`scripts/build_ml_dataset.py`, `scripts/train_correction_model.py`, `scripts/validate_with_correction.py`, `models/`): 6.48 BPM MAE / 9.66 RMSE across the same 71 sessions**, every one evaluated out-of-fold (never on a subject the correction model trained on). Uses Huber loss (`gbr_huber` in `_build_models()`), not squared-error — measured small-but-real win, 8.80→8.67 MAE per-cycle, 6.65→6.48 MAE at the session level (`data/evolution/12-huber-loss.md`). Biggest single win: the speech-condition motion-artifact gap, untouched by every DSP fix below, dropped from ~21-23 to ~8-9 MAE. Bigger models (deeper GBR, bigger regularized MLP, an ensemble) were tested and did NOT beat the small deployed model — real constraint is 56 unique people, not 242K training rows; see `data/evolution/10-bigger-models-dont-help.md` and `data/evolution/11-more-subjects.md`.
- **Bug #1 — harmonic lock-on: two fix attempts failed, a third worked (partially).** Root cause: the peak-picker was locking onto ~0.5×/2× the true rate for sustained stretches, reproduced across all three datasets (first found on `test/dataset_2/subject41`).
  1. Debounced BPM lock/candidate hysteresis (`AdvancedRPPG._resolve_bpm_lock`) — verified via trace to work exactly as designed, **zero aggregate improvement**: the false lock wasn't noise-driven flip-flopping, so debouncing didn't help.
  2. Naive upward-summing Harmonic Product Spectrum (first version of `_select_dominant_freq_hps`) — **zero change**, and provably wrong-direction: for a signal stuck at half-rate, the true signal's own energy sits exactly at that candidate's "2nd harmonic" position, so naive HPS scores it as support *for* the wrong answer.
  3. **What worked**: asymmetric subharmonic-promotion (current `_select_dominant_freq_hps` — misleading name, kept for history; see its docstring) — explicitly checks whether the top candidate's *double* has comparable-or-greater power, and promotes to it if so. Real, measured wins: `subject41` 52.6→8.2, `subject46` 41.2→1.3, `subject49` 36.5→0.4, `subject9` 60.9→2.8, `subject34` 63.0→0.5 BPM error.
  - **Honest downside, do not hide this**: introduced new regressions — `dataset_1/after-exercise` (11.0→32.9 final), `subject22` (41.0→46.8), `subject24` (5.9→20.4), `subject32` (6.0→19.3), `subject37` (4.8→21.0, **since fixed, see below**), UBFC-Phys `s4` speech (9.6→22.5). `after-exercise`/`subject22`/`subject24`/`subject32`/`s4`-speech are still unresolved, not yet root-caused the way subject41/subject25 were — a real, open item.
  - **A third, unrelated bug — found by tracing `subject37`'s regression**: `get_final_summary()`'s history-append gate (in `process_ppg_signal()`) rejects any reading more than 18bpm from the median of the last 5 *accepted* readings — meant to filter single-frame noise, but if the first few accepted readings land during a brief wrong cold-start burst, every later correct reading gets rejected forever (the anchor never updates). `subject37`'s whole-session history ended up with 7 entries, all from an 82-89bpm burst in the first couple of seconds, while the correct ~58bpm signal that dominated the rest of a 2000-frame clip was discarded throughout. **The first fix attempt (unconditionally allow re-anchoring after a long enough sustained-disagreement streak) fixed subject37 but broke subject41** — dropping it from 8.2 back to 41.8 BPM error. Tracing why showed subject41's per-cycle readings are *still* genuinely bimodal across the whole clip even after subharmonic-promotion (~983 vs ~933 reading split in one trace) — meaning the original "subject41 fixed, 8.2 error" result was never a real resolution; it was this exact freeze bug accidentally anchoring on the correct cluster early and never letting go. **Fixed correctly** by only allowing the escape hatch while the frozen anchor is still small (`_HISTORY_UNSTICK_MAX_HISTORY = 10` in `core/rppg.py`) — genuinely-untested cold-start-fluke territory — never once it has accumulated substantial support. Verified: subject37 stays fixed (21.0→**6.0**), subject41 fully restored (**8.2**, unchanged), other regressions untouched either way (confirms they have a different cause). Full story: `data/evolution/09-history-freeze-bug.md`. **Lesson worth repeating from Era 4: a fix that "looks right" and even improves the one case it targets is not verified until you re-run the OTHER known-good cases too — subject41 was the project's flagship success and the naive version silently destroyed it.**
  - **Validation gotcha that cost real debugging time**: `scripts/validate_dataset.py`'s manifest path used to pass `process_stride=2` to `analyze_video_complete()`. The BPM lock/subharmonic logic counts in *cycles* (calls to `process_ppg_signal()`), not real time — striding silently doubled its effective time window and changed tracking *behavior*, not just speed. A direct stride=1 trace showed a real fix; the same video through the strided batch script showed no change. Fixed by keeping `process_stride=1` (default) on this path — UBFC-rPPG/Phys clips are short enough not to need it; stride stays reserved for rPPG-10's genuinely long (10 min) clips in `validate_cropped_roi.py`, which has no cycle-counted hysteresis in play.
  - **Second gotcha, same debugging session**: `scripts/analyze_video.py::analyze_video_complete()` used to compute its final BPM as a plain mean of the *last 10 readings* — a completely different algorithm than `AdvancedRPPG.get_final_summary()`'s IQR-filtered full-history median, which the live webcam path already used. Fixed: `analyze_video_complete()` now sources `final_bpm` from `get_final_summary()` too.
- **Bug #2 — hardcoded fps=30, found by tracing `subject25` (a completely different failure signature from subject41: rock-stable and confident, wrong the whole clip, not oscillating).** Its video is actually 23.2fps; every `AdvancedRPPG` construction site hardcoded `fps=30` regardless of the source's real frame rate, which deterministically scales every frequency by (assumed/actual) — 91.5 true bpm × (30/23.2) ≈ 118, matching the observed error almost exactly. Fixed via `Camera.get_source_fps()` (`core/camera.py`) used at every construction site (`scripts/analyze_video.py`, `web/routes.py::_reset`). Verified: `subject25` 23.5→3.5, `subject26` 25.5→2.5 BPM error. **If you add a new `AdvancedRPPG(...)` construction site anywhere, use `camera.get_source_fps()`, do not hardcode 30 again.**
- **Stratified-by-condition accuracy** (UBFC-Phys labels rest/speech/arithmetic) — after both fixes: arithmetic 3.5 MAE, rest 6.5 MAE, **speech 23.4 MAE, essentially untouched by either fix** (was 26.1/26.6 before). This is real signal, not noise: speech/talking is a *different* problem (motion artifact from mouth movement) than either bug just fixed (frequency-domain ambiguity, fps mismatch) — don't expect future DSP fixes in this family to touch the speech number.
- POS algorithm (`algorithm="POS"` in `AdvancedRPPG`) — present in `core/rppg.py`, **not wired into any API route or UI control** — every instantiation site hardcodes CHROM. Don't "fix" this by wiring POS into the app without full re-validation first. Its harmonic lock-on was root-caused (Era 14): it overshoots to ~2x the true rate from the very first cold-start reading — the mirror image of CHROM's undershoot bug — and a symmetric "demote if the half has comparable power" check in `_select_dominant_freq_hps` fixed that specific pattern (synthetic MAE 26.71→16.00, zero effect on CHROM's numbers). Two of seven synthetic cases and real-video testing (first ever run against real subjects) still show unresolved harmonic errors in both directions — this is a genuine partial fix, not a resolution. See `data/evolution/14-pos-subharmonic-demotion.md`.
- Respiratory rate (`_estimate_rr`/`_smooth_rr` in `core/rppg.py`) — fixed a real bug (fake single-segment "Welch" averaging + zero temporal smoothing, causing RR to bounce across its entire 6-30 br/min search band), verified via synthetic signal to converge instead of edge-locking. **Never validated against real ground truth** — none of the three real datasets in `test/` provide a respiration-rate ground truth (UBFC-rPPG/UBFC-Phys/rPPG-10 all give HR-only or ECG/BVP), so there is still no real-data RR accuracy number. **The display/aggregation suppression gate (`rr_conf < 30` in `web/routes.py` and `scripts/analyze_video.py`) must stay matched to `core/clinical.py`'s `RR_CONFIDENCE_LOW`** — it used to be a looser `< 10`, which let band-edge lock-on readings (the exact "bounces to 6 or 30" failure `_estimate_rr`'s docstring describes) display as solid-looking numbers the clinical engine had already privately flagged as unreliable. If you touch either threshold, touch both, or the display and the clinical reasoning will disagree about the same reading again. See `data/evolution/17-live-session-hardening.md`.
- **`core/clinical.py::assess()`'s entry gate is `if hr <= 0:` only — do not recombine it with `rr <= 0`.** It used to be `hr <= 0 or rr <= 0`, which discarded a perfectly good, confident HR reading entirely any time RR was zeroed (which happens routinely now that the RR gate above is stricter) — never reaching the isolated-tachycardia/bradycardia branches further down that already handle `rr == 0` correctly (shock is auto-disabled since it requires both vitals). Found live: a 107 BPM / 84% stability session triaged to "Insufficient signal for clinical assessment." Verified against all 8 scenarios in `scripts/evaluate_agent_reliability.py::SCENARIOS` (unchanged — none of them exercise "good HR, no RR") plus the regression case itself. Full story: `data/evolution/17-live-session-hardening.md`.
- Skin-tone stratified accuracy — still not computed. None of the three datasets currently in `test/` ship per-subject skin-tone metadata (rPPG-10 does have Fitzpatrick labels in its `Subject Data.xlsx`, but rPPG-10's videos can't run through the face-detection pipeline at all — see `scripts/validate_cropped_roi.py`'s docstring). The README's "Fitzpatrick Inclusivity" language is a design goal, not a validated result — keep it phrased that way.
- **Agent narration faithfulness (`scripts/evaluate_agent_reliability.py`) — the LLM layer restates `priority_score` correctly only 6.2% of the time** (16 real runs, `meta/llama-3.1-8b-instruct`), even when explicitly told to copy the given number verbatim. ESI level (81.2%) and shock flag (93.8%) fare much better. This is the empirical justification for the "one rule that matters most" at the top of this file — full numbers in README's Agent Narration Faithfulness section, full story in `data/EVOLUTION.md` Era 8.
- **`Camera.get_frame()` in dummy_mode never returns `frame_bytes=None`** (by design, for the live MJPEG stream — it should keep showing a placeholder rather than end). Any NEW code that loops on `frame_bytes is None` as its stop condition (like `analyze_video_complete()` used to) will hang forever if the source fails to open. This caused a real 2h20m infinite loop when a test script passed a bad path and the LLM called a tool with it despite being told not to. Fixed in `analyze_video_complete()` with an immediate `camera.dummy_mode` check plus a hard 100,000-frame cap — if you write another function that loops on `camera.get_frame()`, check `dummy_mode` first, don't rely on `None` alone.
- **The hybrid ML correction model is now wired into the live webcam path, not just offline batch validation/uploads.** It used to be true that "the live product app has no ML correction model loaded at runtime" — that changed: `web/__init__.py` loads `models/bpm_correction_gbr.joblib` once at startup, `web/routes.py::generate_frames()` runs every high-SQI live cycle through it (accumulating `session.corrected_bpm_history`), and `agents/tools.py::get_vitals_dict()`'s live branch uses the IQR-filtered median of that history as `bpm` once ≥3 corrected readings exist, falling back to the classical average otherwise. This reuses the exact model/feature schema validated offline at 6.48 BPM MAE (Era 12), but there is no separate live-webcam ground-truth dataset confirming that number holds identically for a live camera stream vs. the UBFC-rPPG/UBFC-Phys source video it was trained/validated on — treat it as the same proven model applied to a structurally similar but not identical input distribution, not a re-validated result. See `data/evolution/17-live-session-hardening.md`.
- Live camera is single-session by design (`web/__init__.py` holds one global `camera`/`rppg_engine`/`session`) — this matches "one workstation, one webcam," but means the "dynamic triage queue" is a sequential log of completed runs, not concurrent multi-patient live monitoring. Don't describe it as the latter.
- Batch validation over very long videos needs `process_stride` (see rPPG-10's `validate_cropped_roi.py`, stride=3) — running the full DSP pipeline (Welch FFT + cubic-spline HRV) every single frame turned a 10-minute video into 13+ minutes with zero results. But `process_stride` is NOT free to use anywhere — see the harmonic-lock-on writeup above for why `validate_dataset.py`'s manifest path deliberately stays at stride=1 (default): the BPM lock/subharmonic logic counts in cycles, not time, so striding changes tracking *behavior*. The live product path (`agents/tools.py` → `analyze_video_complete(video_path)`, no stride arg) is unaffected either way.

## A note on trusting changes in this repo

Mid-session, the BPM lock/candidate hysteresis fix above was implemented, and its *first* version silently regressed two previously-correct synthetic cases (introduced a same-value streak-counter design that was too fragile against ordinary FFT-bin jitter). It was caught immediately by re-running `scripts/validate_dataset.py` after the change, not by inspection. **The lesson: re-run the validation harness after any change to `core/rppg.py` or `core/clinical.py`, every time, before considering a change safe.** A change being reviewed and appearing reasonable is not the same as it being verified.

```bash
python scripts/validate_dataset.py                                          # synthetic self-test — run this after ANY core/rppg.py change
python scripts/validate_dataset.py --manifest test/manifest.csv --skip-synthetic  # real 56-session set, if test/ dataset is present locally
python scripts/validate_cropped_roi.py test/rPPG-10                        # DSP-only, separate — see README for why it's not comparable to the above
```

## Running the app

See README.md's "Local Setup" section. Quick reference:

```bash
python app.py                    # Flask API, port 5002
cd frontend && npm run dev       # Next.js dashboard, port 3000
```

`.env` needs `NVIDIA_API_KEY` and `DATABASE_URL` at minimum. Optional:
`API_KEY` (gates `DELETE /api/triage_queue`, off by default) and
`CORS_ORIGINS` (defaults to the local Next.js dev server only).

## Style/process notes specific to this repo

- The user has been building this iteratively with AI assistance and values direct, unhedged honesty over reassurance — when something doesn't work, say so plainly with the evidence, don't soften it.
- Prefer running the real datasets in `test/` (56 real sessions across UBFC-rPPG + UBFC-Phys, plus 26 more via the separate rPPG-10 DSP-only path, all built via `scripts/build_manifest.py`) over synthetic-only claims whenever making an accuracy statement — synthetic validation only proves the DSP math, not real-world camera+face-detection accuracy.
- When running multiple batch validation jobs, run them **sequentially, not concurrently** — two MediaPipe/OpenCV processes fighting for CPU at once was directly observed to make both crawl to a near-halt (a video that normally processes in ~1-2 minutes took 45+ minutes with zero results when two jobs ran in parallel).
- The project's own `venv/` already has the heavy deps (`opencv-python`, `mediapipe`, etc.) installed and working — use `./venv/Scripts/python.exe` (Windows) rather than trying to reinstall a fresh environment.
- Windows console encoding: avoid non-cp1252 characters (emoji, `✓`, box-drawing beyond em-dash) in anything passed to `print()` in scripts meant to run in a plain Windows terminal — this caused a real crash in `scripts/analyze_video.py` mid-project. Em-dashes (`—`) are fine; checkmarks/emoji are not.
