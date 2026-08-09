# Era 18 — A Real Uploaded-Video Workflow, and Auto-Triage for Files Too

Era 17 hardened the live-session path. This era did the same for the
uploaded-video path, which had never had a real workflow at all — a video
dropped onto the upload input used to start playing immediately, with no
way to stage several files first or choose which one to run, and once it
finished there was no feedback that it *had* finished or any way to know
triage needed a manual click.

## What was wrong, concretely

- `/upload` saved the file **and immediately called `_reset()`**, starting
  playback the instant the request landed. Uploading a second file
  pre-empted whatever had just started — there was no way to stage videos
  ahead of time and choose one later.
- Once a video reached `VIDEO_ENDED`, nothing on screen said so in an
  actionable way, and nothing triggered triage. The 30-second
  auto-trigger that already existed only fires for `is_live` sessions
  (confirmed by tracing the countdown effect's own guard condition) —
  uploaded videos were correctly excluded from that timer (they don't
  have a fixed 30s window; the whole file should run), but that also
  meant they had **no** auto-trigger at all, live or otherwise.
- There was genuine confusion about whether an uploaded video only
  processed a 30-second window like the live countdown implies. It
  doesn't — `get_final_summary()` already aggregates the entire session's
  `bpm_history`, and the video plays every frame through to
  `VIDEO_ENDED` regardless of length. The 30-second number was never
  connected to file uploads; it just wasn't obvious from the UI.

## What was built

**Decoupled upload from playback.** `/upload` now only saves the file and
returns its filename — `web/routes.py::upload_video()`. Two new
endpoints: `GET /api/uploaded_videos` lists every video sitting in
`uploads/` (filename, size, modified time), and
`POST /api/start_uploaded_video` starts a *specific* one by filename,
validated against path traversal by resolving the path and checking it
stays inside the upload directory (`os.path.commonpath`), not just
trusting `secure_filename()` alone.

**A real device-selection workflow.** The frontend's Optical Device
dropdown gained an "Uploaded Video" entry. Selecting it reveals a second
dropdown listing every file from `/api/uploaded_videos`; picking one and
pressing Play calls the new start endpoint. A fresh upload auto-selects
itself in that list, so the common single-file case is still one click
end to end.

**Playback progress, so "is it still running" has an answer.** Added
`Camera.get_total_frames()` (`core/camera.py`) — total frame count for a
video file, `None` for a live camera (an int source has no fixed
duration). `generate_frames()` now computes `video_current_sec` /
`video_total_sec` every frame and exposes them via `/status`. The
frontend shows them as a `0:32 / 0:52`-style overlay on hover over the
Intake Acquisition Feed, hidden entirely for live sessions since there's
nothing fixed to divide by. Display-only — never touches the DSP or
clinical pipeline, so an occasionally-approximate container frame count
(a known quirk of some codecs) was an acceptable risk here in a way it
would not be anywhere else in this codebase.

**Auto-triage for uploaded videos, mirroring the live countdown's
already-established pattern.** A new effect watches for
`metrics.status === 'VIDEO_ENDED' && !metrics.is_live`, fires
`/api/generate_report` then `handleRunTriage` exactly once per video
(guarded by its own ref, reset when a new video starts), and flips the
same `triageAutoRan` flag the live path uses to disable the manual
Kickoff button *synchronously* — reusing Era 17's own fix for the
duplicate-entry race rather than reintroducing that exact bug in a new
code path. An on-screen banner appears the moment a video ends, showing a
spinner while the auto-run is in flight and a confirmation once it's
done, with a shortcut into the Clinical Crew tab.

## Verified

Confirmed end-to-end by the project owner on a real uploaded video:
upload → select from the Uploaded Video list → Play → full video plays
through → automatic triage fires on `VIDEO_ENDED` with no manual click →
result appears in Clinical Crew. Typechecked clean (`tsc --noEmit`) and
the two new Flask endpoints were exercised directly (including a
path-traversal attempt, correctly rejected) before the live confirmation.

**Next: see the top-level [`data/README.md`](../README.md) for where the
project stands now.**
