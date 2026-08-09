# Era 17 — Live-Testing Round 2: Duplicate Records, a Misaligned RR Gate, and a Self-Inflicted Regression

Era 02 was the first live-testing pass, months earlier, before the honesty
audit and the real-dataset validation work. This era is the second one —
after the frontend redesign, with the hybrid ML correction layer already
proven offline, the project got its first real end-to-end live-webcam and
upload retest since then. It surfaced four real problems, one of which was
a bug this same era's own earlier fix introduced — caught by re-testing,
not assumed away.

## Bug 1 — Duplicate triage queue entries, two separate root causes

The symptom: the same session produced two Triage Dispatch entries a few
seconds apart, one under the typed patient name, one under an
auto-generated `Patient <id>`.

**First cause, fixed but insufficient:** the 30-second auto-triage
countdown's session-report dismiss button ("✕") reset the same timer/ref
state (`sessionTimer`, `reportFetchedRef`) that gated the auto-trigger,
silently restarting the whole cycle. Fixed by making that button only
clear the report, not the timer state. Duplicates continued.

**Second cause, the real one:** `handleRunTriage`'s manual "Kickoff
Triage" button was disabled by `triageAutoRan`, but that flag only flipped
to `true` *after* `/api/generate_report` resolved — a fetch that takes a
few real seconds. During that window the button stayed clickable. A
manual click in that window fired a second `/api/triage_run` for the same
session under a different, auto-generated patient ID, moments before the
auto-trigger's own call landed under the typed name. Fixed by flipping
`triageAutoRan` synchronously the instant the countdown hits zero, not
after the report fetch resolves — closing the race window entirely rather
than narrowing it.

## Bug 2 — RR display gate looser than the clinical engine's own floor

Respiration kept displaying as a suspiciously solid-looking 6.9 or 29.9
br/min — the exact band-edge lock-on failure signature `core/rppg.py`'s
own `_estimate_rr` docstring already describes (RSA modulation is weak;
a noisy estimate tends to land at the search-band edge, not mid-band).

The estimator already penalizes an edge-locked peak's confidence
(`rr_conf *= 0.4`), but the display-suppression gate in `web/routes.py`
(and the equivalent one in `scripts/analyze_video.py`) only zeroed the
reading below **10%** confidence — while `core/clinical.py`'s own
`RR_CONFIDENCE_LOW` reliability floor is **30%**. A borderline edge-lock
(confidence 10–29%) sailed past the display gate and showed a plausible
number the clinical engine had already privately flagged as unreliable in
its own reasoning text. Fixed by raising both gates to match the 30%
floor the clinical engine already uses, so `--` displays instead of a
wrong-looking value whenever confidence can't clear the same bar the
downstream engine requires. This is a display/gating change only — no DSP
math in `core/rppg.py` was touched, so no re-validation against the
71-session real-video set was needed for this specific fix.

## Feature — Hybrid ML correction wired into the live path

Documented since Era 10 as **offline-only**: the deployed correction
model (`models/bpm_correction_gbr.joblib`, 6.48 BPM session MAE
out-of-fold) was applied by `scripts/analyze_video.py` for batch
validation and pre-recorded uploads, but the live webcam path in
`web/routes.py::generate_frames` never touched it — the live product
shipped classical-CHROM-only numbers while the offline path was already
proven more accurate.

Fixed: the model now loads once at Flask startup (`web/__init__.py`), and
`generate_frames()` runs every high-SQI live cycle through it too,
accumulating a session-long `corrected_bpm_history`. `agents/tools.py`'s
`get_vitals_dict()` now takes the IQR-filtered median of that history
(same aggregation Era 12's validation used) as the live session's `bpm`
once at least 3 corrected readings exist, falling back to the classical
average otherwise. Uploaded-video triage was also switched on — it had
`analyze_video_complete`'s `correction_model_path` parameter available
the whole time, it just was never being passed from the triage-run code
path.

**Honest caveat:** this reuses the exact model and feature schema already
validated at 6.48 BPM MAE offline, but there is no separate live-webcam
ground-truth dataset to re-confirm that number holds identically for a
live camera stream (different lighting/compression characteristics than
UBFC-rPPG/UBFC-Phys source video) — it's the same trained model applied to
a structurally similar but not identical input distribution.

## Bug 3 — A dead code path that always failed once a video finished

Tracing why "upload a video, let it finish, then click Kickoff" 400'd
found `handleRunTriage`'s fallback branch: when `metrics.status` reached
`VIDEO_ENDED`, the button sent a `FormData` POST containing only
`patient_name` — never a `video` field — to `/api/triage_run`, which
requires one. This branch was reachable from the UI but had **no code
path that ever attached a real file to it**, so it always failed
end-to-end, not intermittently. There is also no UI flow where this
button holds an actual `File` object — the video is uploaded separately,
earlier, via `/upload` — so the fix was to drop the dead branch entirely
and always send the JSON `{source: 'live'}` body, which reads from the
current backend session state regardless of whether that state came from
a webcam or an already-uploaded file.

## Bug 4 (self-inflicted) — Bug 2's fix broke `core/clinical.py` for good HR readings

Live-retesting Bug 3's fix surfaced a new failure: a video with a
confident, stable 107 BPM reading (84% stability, from the terminal's own
`get_final_summary()` log) triage'd to "Insufficient signal for clinical
assessment" — discarding a perfectly good heart-rate reading entirely.

Root cause: `core/clinical.py::assess()`'s entry gate was
`if hr <= 0 or rr <= 0:` — a combined, all-or-nothing check. Before Bug
2's fix, `rr` reached 0 only when confidence was below 10%, a rare event.
After tightening that gate to 30% (the *correct* fix for Bug 2), sessions
with an unreliable-but-real RR reading now hit `rr <= 0` routinely — and
the combined gate discarded the entire assessment, including HR, every
time, never reaching the isolated-tachycardia/bradycardia/normal branches
further down that already handle `rr == 0` correctly (`tachy_rr` becomes
0, `is_shock` is automatically forced false since shock requires both
vitals elevated together).

Fixed by decoupling the two checks: HR alone is the hard requirement (it's
the one vital this system's own validation actually backs), and `rr <= 0`
now adds an honest reasoning note ("assessment based on heart rate alone
... Compensated Shock cannot be evaluated without a reliable RR reading")
instead of disqualifying everything. Verified against all 8 scenarios in
`scripts/evaluate_agent_reliability.py::SCENARIOS` (identical output —
none of them exercise "good HR, no RR", so none relied on the old
combined gate) plus the exact regression case (`hr=107, rr=0`, which now
correctly reports "Isolated Tachycardia" instead of the generic ESI-3
fallback).

## The lesson, again

This is the same lesson Era 09 already learned once, in a different part
of the codebase: a fix that is correct on its own terms (Bug 2's RR gate
tightening was — it does exactly what it was meant to) can still break a
downstream consumer that made an assumption about the old, looser
behavior. It was caught here the same way Era 09 caught its regression:
by actually re-running the affected path live after the fix, not by
inspection or by trusting that the change "looked safe."

## Other, non-DSP changes this era

- `/upload`'s 100MB Flask cap (`MAX_UPLOAD_BYTES`) removed — this app runs
  single-user, localhost-only (see the live-camera single-session note in
  [CLAUDE.md](../../CLAUDE.md)), and GB-sized real test videos were
  hitting a 413 before the upload handler ever ran. If this app is ever
  exposed beyond localhost, a real cap needs to come back first.
- Two dashboard screenshots recaptured directly in-browser (PNG, native
  1905×914 / 1920×914 resolution) to replace low-quality placeholders in
  the IEEE paper's Dashboard Interface figure — verified sharp at 1:1
  pixel crop, not just visually.

**Next: [Era 18 — A Real Uploaded-Video Workflow](18-uploaded-video-workflow.md)**
— decoupling upload from playback, a video picker, playback-progress
display, and auto-triage for uploaded videos too.
