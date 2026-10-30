# Era 20 — Voice-guided autonomous kiosk: requirements, voice bake-off, hardware reality check

Status: **ideation + research only. Nothing in the product has been built yet** —
the user wants to approve the design first. This file records what was decided,
measured and found, so it is not lost.

## Requirements stated by the user
- Autonomous live session: the system greets, asks the person's name (speech-to-text),
  guides them to a good position (look at the camera / move closer / hold still),
  runs the 90 s scan, releases the camera, triages, and speaks.
- Voice must be **soothing, natural, not robotic** — taken seriously, downloading
  models is explicitly allowed.
- Everything built is expected to be **state of the world**, including RR and HR,
  and to use the available GPU/CPU fully.
- Modernise the frontend UI (currently judged "boring").
- Keep a record of work + gains in `data/evolution/` (this file is that record).

## Design leaning (not yet approved)
1. **Two-layer voice**: fixed prompts (greeting, "move closer", "hold still",
   countdown, sign-off) are rendered ONCE offline with the best voice and shipped
   as static audio -> instant, studio quality, no internet at demo time. Only
   dynamic lines (the person's name, results) are synthesised on-device.
2. **Guidance from signals the backend already has**: face_detected, estimated_lux
   (existing warnings <50 / >210), motion_delta; add face bounding box
   (size/position) to `/status` so it can say "closer / left / right".
3. **Name capture**: STT + spoken "is that right?" confirmation + typed fallback;
   mute the mic while the system is speaking; local Whisper preferred over browser
   speech (privacy, accuracy on Indian names). Store the name only, not audio, unless
   decided otherwise; show a consent line.
4. Guidance log visible on screen so a review panel can see the reasoning.

## Voice bake-off (setup)
Isolated environment `tools/tts_lab/` (own venv — does not touch the project's
dependencies): `kokoro-onnx` 0.6.1 + `onnxruntime`, `edge-tts` 7.3.1.
Downloaded: `tools/tts_lab/models/kokoro-v1.0.onnx` (325 MB), `voices-v1.0.bin` (28 MB).
`tools/tts_lab/bakeoff.py` renders the same ~30 s calm kiosk script with each voice
into `tools/tts_lab/samples/`:
- Kokoro (local, Apache-2.0, speed 0.92 for a calmer pace): `af_heart`, `af_nicole`,
  `af_sarah`, `af_sky`, `bf_emma`, `bf_isabella`, `am_michael`, `hf_alpha`
  (hf_alpha is a HINDI voice reading English — only there to confirm Hindi exists;
  Kokoro also ships hf_beta/hm_omega/hm_psi). 54 voices available in total.
- Microsoft neural voices via edge-tts (online at render time only): `en-IN-NeerjaNeural`,
  `en-IN-NeerjaExpressiveNeural`, `en-IN-PrabhatNeural`, `en-US-AvaNeural`, `-JennyNeural`,
  `-EmmaNeural`, `en-GB-SoniaNeural`, `-LibbyNeural` (rate -8%, pitch -2 Hz).
Measured: Kokoro renders at roughly 3-4x real time on this CPU (e.g. 33.8 s of audio in
8.6 s) — fine for pre-rendering, borderline-OK for short dynamic lines.
Observation: `af_nicole` produced 53.6 s of audio for the same text (~30 s for the
others) -> very slow/long pauses; may be too sleepy for a kiosk.
**The assistant cannot hear audio — the voice choice must be made by the user
listening to the samples.** No selection has been made yet.

Licensing note: edge-tts uses Microsoft Edge's online read-aloud service through an
unofficial route — fine for a college demo, NOT a safe basis for a commercial
product. Kokoro (Apache-2.0) is safe to ship. Pre-rendering fixed prompts means the
demo needs no internet either way.

## Hardware reality check (important)
- CPU: Intel i5-13450HX (10 cores / 16 threads), RAM 15.6 GB, ASUS TUF Gaming F16.
- A discrete **NVIDIA GPU is present** (PCI `VEN_10DE&DEV_2DD8`) but Windows lists it as
  **"Microsoft Basic Display Adapter", status Error** -> **no NVIDIA driver installed /
  loaded**, so there is no CUDA, `nvidia-smi` does not exist, and VRAM/model name cannot be
  read. Until the driver is installed, everything runs on CPU. (Note: this machine already
  blocked one scikit-learn DLL via an Application Control policy — a driver install may be
  affected by the same policy.)
- `torch` is not installed in the project venv; a CUDA build would be ~2.5 GB and should
  go in a separate environment.

## What the GPU would unlock once the driver works (candidate list, not started)
- Local Whisper large-v3 speech-to-text for names (faster-whisper on CUDA).
- Deep rPPG models for HR (e.g. PhysFormer / EfficientPhys / RhythmFormer via rPPG-Toolbox)
  — HR's 6.48 MAE is far from state of the art for UBFC-rPPG (published deep models report
  roughly 1-3 bpm intra-dataset); this is the realistic route to "state of the world" HR,
  but it needs video data (the 71 real videos are not on this machine).
- A deep RR model only makes sense with webcam paced-breathing data as training/validation.

## Update — voice DECIDED (2026-10-29, by the user listening to a 13-voice test)
- **Primary: `en-GB-SoniaNeural` (Microsoft neural, British). Backup: `en-IN-NeerjaNeural`
  (Indian English).** Both at **rate +10%, pitch -2 Hz** — the pace was tuned by ear in steps
  (-8% too slow -> +5% "good" -> +10% "perfect"). Stored in `tools/tts_lab/voice_config.json`.
- All other bake-off samples were deleted; Kokoro model kept (354 MB) only as a possible
  offline fallback for dynamic lines. Kokoro `bf_emma` would be the closest-sounding local
  voice but is NOT the same voice as Sonia — a name read in a different voice mid-sentence
  will be audible. Cleanest design: keep the person's name out of spoken sentences, or only
  speak it when online.
- **Pre-rendered prompt library:** 36 fixed lines x 2 voices in `tools/tts_lab/prompts/{sonia_primary,
  neerja_backup}/*.mp3` (1.7 MB total) with `tools/tts_lab/prompts/manifest.json` mapping each id
  to its text and the situation that triggers it (welcome, name capture/retry/typed fallback,
  positioning: closer/farther/left/right/keep-in-circle/perfect/more-light/too-bright/hold-still/
  stay-quiet/remove-cover, 3-2-1 countdown, mid-scan encouragement at ~25/50/70 s, "few more
  seconds", done, routine vs urgent hand-off, weak-signal retry, camera/mic errors).
  Re-render with `tools/tts_lab/venv/Scripts/python.exe tools/tts_lab/render_prompts.py`
  (needs internet; edit the P dict to change wording).
- **"stay_quiet" exists on purpose:** speech/mouth motion is this project's worst HR condition
  (UBFC-Phys speech MAE ~23 vs ~3.5-6.5 for the other conditions, see CLAUDE.md), so telling
  the person not to talk during the scan is accuracy work, not politeness.
- **To check at integration:** left/right wording vs the mirrored camera view; whether the
  mic should be muted while any prompt plays (it should, or STT will transcribe the system).
- Still NOT built: kiosk state machine, STT/name capture, guidance from face box, UI. Waiting
  for the user's go-ahead and answers (who stands at the kiosk, languages, dark/light UI).
- GPU: driver still missing as of this update (see above); user is installing it.


## Update — the kiosk was BUILT (2026-10-29/30)

Requirement from the client (relayed by the user): fully AI-automated, patient-facing — the system must
**guide the patient, record their details, assign a token number, and make them wait.**

### What exists now
| Piece | Where |
| :--- | :--- |
| Patient kiosk (voice-guided flow) | `frontend/src/app/kiosk/page.tsx` (UI only) + `frontend/src/lib/kiosk/flow.ts` (state machine) |
| Position-guidance decisions (tested) | `frontend/src/lib/kiosk/guidance.ts` + `guidance.test.ts` (10 tests, `node --test`) |
| Voice engine (clips, mic capture, VAD, WAV) | `frontend/src/lib/kiosk/voice.ts`; 242 Sonia clips + 242 Neerja backup in `frontend/public/voice/` |
| Waiting-room board (+ `?staff=1` controls) | `frontend/src/app/queue/page.tsx` |
| Local GPU speech-to-text (Whisper turbo, CUDA) | `tools/gpu_lab/stt_server.py` (+ `stt_parse.py`, 26 parser tests) |
| Tokens, intake fields, queue, fast triage | `web/routes.py`, `web/neon_db.py` |
| Self-healing LLM model pool | `agents/llm_pool.py` (used by `agents/crew.py` and the ARIA chat) |
| Shared light/dark design system | `frontend/src/app/globals.css`, `src/lib/theme.ts`, `src/components/ThemeToggle.tsx` |

Run: `python app.py` (5002) · `cd frontend && npm run dev` (3000) · `tools/gpu_lab/venv/Scripts/python.exe tools/gpu_lab/stt_server.py` (5003, optional — without it the kiosk falls back to typing).
Open `/kiosk` (patient), `/queue` (waiting-room screen), `/queue?staff=1` (front desk), `/` (staff dashboard). `/kiosk?demo=1&skip=1` rehearses the whole flow with a scripted camera and saves nothing.

### Design decisions worth remembering
- **Voice and touch always race**; typing is the floor. Mic denied / speech server down / noisy room never traps a patient.
- **All speech is pre-rendered** (36 fixed lines + 150 token announcements + 24 wait times + 16 "people ahead" lines, x2 voices). The patient's NAME is never spoken (shown on screen only) — no run-time synthesis, so the voice never changes mid-sentence and nothing needs internet at the demo.
- **Left/right guidance is judged on the MIRRORED preview** (face on the right of the screen => "move to your left"); unit-tested.
- **`/api/scan/begin`** resets only the rPPG engine (not the camera) after positioning, so frames captured while the patient was still moving never reach the session medians.
- **`fast:true` triage**: token + queue position in ~3 s; the 3-agent AI narrative is written into the record afterwards (`update_narrative` touches only 2 columns, so it cannot overwrite a staff status change). The deterministic engine still decides priority.
- **Waiting order**: called first, then ESI asc, priority score desc, then first-come-first-served (the staff dashboard's order puts the newest first inside a tier, which would be unfair in a waiting room). Wait = position x `QUEUE_AVG_SERVICE_MIN` (default 12, a placeholder — every screen labels it an estimate).
- **Public board shows tokens only** — no names, vitals or diagnoses; staff actions work by token (`/api/queue/mark`).

### Measured problems found and fixed while building (each cost real time — keep)
1. **Neon round trips dominate.** From this machine a NEW connection costs ~1.6-2 s and every query ~0.9 s, so the fast path took ~8-9 s. Fix: `ThreadedConnectionPool`, autocommit single statements, token allocation + insert fused into ONE statement (unique index `uq_token_per_day` makes a lost race retry), a light `get_waiting_patients()` instead of `SELECT *`. Result: 3.1 s end to end.
2. **NVIDIA model access is volatile per API key.** In one evening: models went 410 (retired), `nemotron-3-super` flipped 200 -> 404, `gpt-oss-120b`/`mistral-nemotron` -> 410, `llama-3.2-11b` went from ~1 s to timing out. Hardcoding a list fails within hours. `agents/llm_pool.py` probes candidates in the background, ranks live ones (quality tier, then latency), marks failures dead for 10 min, and both LLM call sites walk that ranking with `timeout=30, max_retries=0`.
3. **Stray backend processes**: restarting without killing the old `app.py` left up to three servers fighting for port 5002 and made behaviour inconsistent. Kill by command line (`Get-CimInstance Win32_Process`) before restarting.
4. **psycopg2 `BEGIN`/`ROLLBACK` are extra round trips** — a read that ends with `conn.rollback()` costs double on a slow link.
5. **Windows CRLF**: `frontend/src/app/page.tsx` has CRLF line endings; multi-line scripted edits must normalise `\r\n` first or they silently fail to match.
6. **Two bugs of mine caught by reading the browser console, not the tests**: a React hydration mismatch (server "PM" vs browser "pm" in the board clock) and a raw `<script>` in the layout (React 19 warns it never re-executes) — fixed with a client-only clock and `next/script beforeInteractive`.
7. **Speech chain test (synthesized Indian-English/US voices, GPU Whisper turbo)**: 11/13 correct, ~0.2 s per clip warm (first clip 18 s CUDA start-up — now warmed at server boot). Misses were transcription variants ("Guru Preet" for Gurpreet; "I am me" for "I am male") — exactly what the on-screen confirmation and buttons absorb.

### Theme system
One token set (`--bg --panel --ink --brand --esi-1..5 ...`) in `globals.css`, `[data-theme=dark|light]` on `<html>`, applied before first paint, remembered in localStorage, `?theme=light|dark` overrides. The staff dashboard (2,189 lines of hard-coded Tailwind colours) is re-skinned centrally by a `.vital-app` layer that maps its hex/zinc classes to the tokens in BOTH themes, replacing ~183 lines of ad-hoc light-mode overrides; tiny 9-10 px labels were raised to readable sizes; the idle camera placeholder (bright yellow-on-black) is covered by a proper empty state (`has_source` in `/status`). The font was forced to Arial by the old `globals.css`; it is now Geist.

### NOT verified — do not claim otherwise
- A real person in front of the kiosk: the face-box thresholds (face 17-52% of frame, motion > 8, etc.) are reasoned and unit-tested but have not been tuned on a live face.
- The browser microphone path (MediaRecorder + silence detection + WAV encode) was not exercised by a human voice; the STT server was tested with synthesized speech through the same backend proxy.
- The "please don't talk" detector (`startTalkMonitor`) is untested.
- The ML heart-rate correction model still cannot load (Windows Application Control blocks a scikit-learn file), so live HR is classical-only.
- Pediatric patients: the clinical engine still uses adult HR/RR thresholds (`core/clinical.py`); the kiosk now records age, but nothing adjusts the thresholds yet — a real patient-safety gap for a public-facing kiosk.
