# Era 5 — The Mechanical Bug (hardcoded FPS)

Rather than guess at a second fix for the remaining bad subjects
(subject25/26, at a *different* ratio — 1.26-1.35×, not a clean 2×), the
same discipline from [Era 4](04-harmonic-lock-on-fixes.md) was applied:
trace first, fix second.

## Trace of `subject25`

`test/trace_subject25.log` showed a completely different signature from
subject41 — **rock-stable, high-confidence, wrong the entire clip**, not
oscillating between two states at all (88% stability score). That ruled
out harmonic ambiguity outright.

## Root cause, found by checking the actual video file properties

`subject25`'s video is recorded at **23.2 fps**, not 30 — but every
`AdvancedRPPG` construction site in the codebase hardcoded `fps=30`
regardless of the source's real frame rate. Analyzing a 23.2fps video as
if it were 30fps deterministically scales every frequency by
(30/23.2 ≈ 1.29×): 91.5 true bpm × 1.29 ≈ 118, matching the observed
error almost exactly.

## Fixed

`Camera.get_source_fps()` (new method in `core/camera.py`) reads the actual
frame rate from the opened source; every `AdvancedRPPG` construction site
(`scripts/analyze_video.py`, `web/routes.py::_reset`) now uses it instead of
a hardcoded constant.

**Verified immediately**: `subject25` 23.5 → **3.5** BPM error, `subject26`
25.5 → **2.5** BPM error.

## Full 56-session result with both fixes combined

See `test/validation_run6_fpsfix.log`: MAE 11.36 → **8.69 BPM**, RMSE
18.08 → **15.84 BPM**. UBFC-rPPG subset alone: **7.79 BPM**, crossing into
"research grade" (<8.0) for the first time. Stratified by condition:
arithmetic 3.5, rest 6.5, speech 23.4 (essentially untouched by either fix
— confirming the speech-condition gap is a genuinely separate problem,
motion from talking, not frequency-domain error).

**Combined effect across the entire [Era 4](04-harmonic-lock-on-fixes.md) +
this era: MAE 17.4 → 8.7 BPM, a 50% total reduction, from two independently
root-caused, independently verified bugs.**

**Next: [06 — Grounding Against the Literature](06-literature-comparison.md)**
