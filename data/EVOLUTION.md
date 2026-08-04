# VITAL — Project Evolution: The Complete Story

This document exists so that **anyone or anything** — a person, a different AI
session, a future collaborator — can read this folder and reconstruct the
entire debugging/validation journey of this project without ever having seen
the original conversation that produced it. Where possible, every claim below
points at the actual evidence file that backs it (in `test/` or `data/`),
not just a description of what happened.

Read this top to bottom as a timeline. Nothing here is invented after the
fact — this is what actually happened, including the mistakes.

---

## Era 0 — Pre-History: what existed before any of this

VITAL ("Opti-Screen") started as a college assignment: extract heart rate
from a webcam via remote photoplethysmography (rPPG), feed it through a
3-agent CrewAI pipeline (Perception → Diagnostic → Coordinator), and display
a triage dashboard. The original assignment brief (`project details.pdf`)
suggested a much simpler scaffold — Streamlit, a basic rule (`HR>100 AND
RR>20 = Compensated Shock`), UBFC-rPPG as the reference dataset.

By the time this evolution begins, the student had already built something
considerably more ambitious than the assignment: a Next.js SaaS dashboard,
Flask API, Neon serverless Postgres, multi-provider LLM fallback chains, a
research corpus of 30 IEEE papers (`research/*.csv`) used to justify design
choices. But none of it had been rigorously validated, and several of the
"novelty" claims in the docs (`implementation_plan.md`) described features
that were never actually built.

---

## Era 1 — The Honesty Audit

**Trigger:** the student asked for an "overall honest truth" on the project's
approach, technique, and architecture — not encouragement, an audit.

**Findings, in priority order:**
1. **The ESI level / Compensated Shock flag / priority score were computed by
   an LLM**, parsed out of free text with regex, with a silent fallback to
   `esi_level=3` on any parse failure. The single most safety-critical number
   in the system was non-deterministic.
2. The "polynomial cross-correlation" and "RAG for missing narrative data"
   claims in `implementation_plan.md` were never implemented — the actual
   logic was a plain `if HR>100 and RR>20` rule.
3. `agents/crew.py` built ONE module-level `Crew` singleton, reused across
   requests — CrewAI `Task` objects store output on `self` and are not safe
   to re-enter from concurrent Flask threads (Flask runs `threaded=True`).
4. `web/neon_db.py::get_all_patients()` returned `[]` on ANY database error,
   indistinguishable from "the queue is genuinely empty."
5. CORS was `origins: "*"`; no auth on any endpoint, including one that wipes
   the entire patient database.
6. The architecture is inherently single-session (one global camera/session)
   — fine for "one workstation, one webcam," but the "dynamic triage queue"
   is a sequential log of completed runs, not concurrent multi-patient
   monitoring. This was being described as the latter.
7. No real accuracy validation existed. `scripts/validate_dataset.py`
   defaulted every unlabeled video's ground truth to `0 bpm` and printed the
   resulting distance as if it were a real error metric.
8. README claimed "Fitzpatrick Inclusivity" (skin-tone robustness) as a
   proven result — no such validation had ever been run.

**Fixes applied this era:**
- `core/clinical.py` (new) — deterministic ESI/shock/priority engine. Any
  Compensated Shock flag unconditionally forces `needs_manual_review=True`.
  Unit-tested against 8 hand-built scenarios (normal, isolated
  tachycardia/tachypnea, mild/severe shock, bradycardia, low-SQI, no-signal).
- `core/guidelines.py` (new) — local term-frequency-cosine retrieval over
  ESI/WHO guideline text + the 30-paper research corpus. Explicitly
  documented as **not** vector-embedding RAG (the claim in
  `implementation_plan.md` was never built) — an honestly-scoped alternative.
- `agents/crew.py` — `build_triage_crew()` factory replacing the singleton.
- `web/neon_db.py` — `get_all_patients()` now raises on failure instead of
  swallowing it into `[]`.
- CORS restricted to an explicit allowlist; optional `API_KEY` gate added on
  the destructive `DELETE /api/triage_queue` endpoint (off by default).
- `scripts/validate_dataset.py` — the `gt_bpm=0` placeholder deleted
  outright, not patched (a wrong number is worse than a missing one).
- README.md and `CLAUDE.md` (new — a guide for future coding-agent sessions)
  rewritten to match what the code actually did, with explicit "known-fragile,
  do not overclaim" sections.

---

## Era 2 — First Contact With Reality (live testing surfaces real bugs)

Once the fixes above were live, the student actually ran the dashboard and
immediately hit real, previously-invisible bugs:

- **A false "Compensated Shock" alert** on a live webcam session that felt
  completely healthy. Root cause: HR=123bpm/RR=30bpm both genuinely crossed
  threshold, but the RR reading was low-confidence and the system presented
  it with full confidence anyway. Fixed: any Compensated Shock flag now
  forces manual review regardless of signal-quality tier, and low-confidence
  RR readings driving a shock flag get an explicit caution in the reasoning
  trace.
- **Respiration rate bouncing between ~6 and ~30 br/min** — turned out the
  RR search band is literally 6-30 br/min, and the estimator was locking
  onto the edges of its own search range. Root cause: `signal.welch(...,
  nperseg=len(resp), noverlap=0)` used the entire buffer as one segment —
  not actually Welch's method (no averaging benefit), just a noisy
  periodogram. Fixed: real overlapping-segment Welch averaging, plus
  confidence-weighted temporal smoothing (RR had none at all before; HR
  already did).
- **The dashboard displaying raw `` ```json {...}` `` text** as the AI
  summary — a bug introduced while removing an old JSON-parsing helper and
  never replacing its actual purpose. Fixed with a proper
  `_extract_triage_summary()`.
- **Patient records appearing to "vanish"** — root cause: Neon's serverless
  Postgres auto-suspends its compute after inactivity; a cold-start
  connection hiccup was being silently swallowed into an empty list,
  indistinguishable from "no patients." Fixed with a retry-once wrapper on
  `get_connection()` and by making `get_all_patients()` raise properly so
  the frontend's existing "only update display on HTTP 200" logic could do
  its job.
- Model swap: primary LLM changed from `meta/llama-3.3-70b-instruct` to
  `meta/llama-3.1-8b-instruct` (cheaper/faster), with 70B demoted to first
  fallback rather than dropped.
- `core/camera.py` — added full 468-point face-mesh dot visualization
  (cosmetic only) and trimmed-mean spatial pooling for the forehead/cheek
  ROIs (excludes the darkest 10%/brightest 5% of pixels per box, verified via
  synthetic blemish-injection test to recover the true clean-skin color
  within 0.1 units where a plain average was off by ~5).

---

## Era 3 — The Data Age

The student obtained the full UBFC-rPPG dataset from its public Google Drive
link (no gated request needed — confirmed by checking directly). Initial
download was partial: 18 of 58 possible sessions had both a video and a
matching ground-truth file; the rest had one or the other (interrupted
download).

**Tooling built:** `scripts/build_manifest.py` — walks a downloaded dataset
folder, auto-detects which of several ground-truth formats is present
(`gtdump.xmp` comma-separated HR/SpO2/PPG rows for "Dataset 1", or the
3-line `ground_truth.txt` BVP/HR/timestamp format for "Dataset 2"), and
emits a `manifest.csv` consumable by the validation harness.

**First real validation ever run** (see `test/validation_run.log`,
`validation_run2.log`): 18 real subjects. **MAE 19.1 BPM, RMSE 26.0 BPM.**

The key discovery from this first real run: **accuracy was bimodal, not
uniformly mediocre.** 9/18 subjects were excellent (<9 BPM error); 7/18
(~39%) showed errors of 20-60+ BPM, in two distinct patterns:
- ~4 subjects landing at almost exactly **half** the true rate (ratio
  0.44-0.59×)
- ~3 subjects landing at roughly **1.26-1.34×** the true rate

**Root-cause trace** (`test/trace_subject41.log`) on the clearest half-rate
case: the pipeline genuinely found the correct ~100bpm rate for a ~7-second
stretch mid-recording, then slid back into a false ~45-50bpm lock for the
rest of the clip. Because `get_final_summary()` takes a history median,
whichever state occupied more total frames won — actively discarding the
correct stretch as a statistical minority.

Later in this era: the student cleaned up the 27 incomplete UBFC-rPPG
sessions (freed ~3.6GB), then downloaded the missing videos for the
remaining subjects, bringing `dataset_2` to a clean 40/40 complete subjects
(41 total with `dataset_1/after-exercise`). Two more datasets were added:
**UBFC-Phys** (5 subjects × 3 tasks — rest/speech/arithmetic — via a free
IEEE DataPort account, no gated request) and **rPPG-10** (Mendeley, direct
download, 26 subjects with real ECG ground truth, but its videos are
pre-cropped 64×64 skin patches with no visible face — structurally
incompatible with MediaPipe face detection, handled via a separate DSP-only
validation path, `scripts/validate_cropped_roi.py`, that bypasses face
detection entirely).

---

## Era 4 — The Age of Experiments (fixing the harmonic lock-on bug)

Three attempts were made, in order, on the exact same evidence
(`test/dataset_2/subject41`, the case root-caused above). **Two failed
completely, verified by testing, not assumed.** This is the most important
methodological lesson of the whole project: *a change that looks reasonable
under code review is not the same as a change that has been verified.*

### Attempt 1 — BPM lock/candidate hysteresis (FAILED, 0% improvement)
`AdvancedRPPG._resolve_bpm_lock` — a debounced state machine requiring
sustained multi-frame agreement (much more for harmonic-ratio-looking
jumps) before the tracker or its adaptive filter would recenter on a new
frequency. Its first version regressed two previously-perfect synthetic
cases (a same-value streak-counter that was too fragile against ordinary
FFT-bin jitter) — caught immediately by re-running the synthetic self-test,
fixed with a rolling-window spread check instead.

Verified via trace (`test/trace_subject41_v2.log`) to work **exactly as
designed**: it correctly resolved transient noise and correctly held
through the genuine mid-recording correct stretch. Re-running the full
18-subject validation afterward showed **zero aggregate improvement**
(still ~19.1 MAE) — because the false lock wasn't noise-driven
flip-flopping in the first place. It was sustained, stable, and spectrally
confident for hundreds of frames at a stretch. Hysteresis debounces
flip-flopping; it cannot tell two stable-but-different answers apart.

### Attempt 2 — Naive upward-summing Harmonic Product Spectrum (FAILED, 0% improvement)
Classic pitch-detection technique: score each candidate frequency by summing
PSD power at its own value plus its integer harmonics (2×, 3×) — a real
periodic signal should show coherent energy at its own harmonics.

Traced (`test/trace_subject41_hps.log`) and found to produce **an
identical result to before the fix** — and reasoned through *why*, not just
observed that it failed: for a signal locked at half the true rate, the
*true* signal's own energy sits exactly where that low candidate's "2nd
harmonic" would be. Naive HPS summation therefore reads the true signal's
energy as supporting evidence *for* the wrong (low) candidate, not against
it. The technique was directionally wrong for this specific failure mode.

### Attempt 3 — Asymmetric subharmonic-promotion (WORKED, partially)
Corrected approach: for the raw top candidate frequency, explicitly check
whether its **double** has comparable-or-greater spectral power; if so,
promote to the double instead of trusting the raw peak. This targets the
*specific, evidenced* failure direction (undershooting to ~half-rate)
instead of a generic harmonic-support heuristic.

Traced (`test/trace_subject41_v3.log`): frames 180-840 now correctly read
~98-127 instead of ~45-70; final result for subject41 went from 52.6 BPM
error to 8.2 BPM error.

**Two additional bugs found and fixed while verifying this, both of which
would have made the measurement itself wrong:**
- `scripts/validate_dataset.py`'s manifest path was passing
  `process_stride=2` to `analyze_video_complete()` (a speed optimization
  added earlier for rPPG-10's 10-minute clips). The BPM lock/subharmonic
  logic counts in *cycles* (calls to `process_ppg_signal()`), not real
  time — striding silently doubled its effective time window, changing
  tracking *behavior*, not just speed. A direct stride=1 trace showed a
  real fix; the identical video through the strided batch script showed no
  change at all. Fixed by keeping stride=1 on this path (short UBFC
  clips don't need the optimization; it stays reserved for rPPG-10's
  genuinely long clips, which have no cycle-counted hysteresis in play).
- `scripts/analyze_video.py::analyze_video_complete()` computed its final
  BPM as a plain mean of the **last 10 readings** — a completely different
  algorithm than `AdvancedRPPG.get_final_summary()`'s IQR-filtered
  full-history median, which the live webcam path already used. Two
  different "final answer" methods for the same concept, silently
  disagreeing. This specifically mattered for subject41: a video that
  settles into a wrong lock right at the very end previously reported that
  wrong tail value no matter how correct the rest of the clip was. Fixed by
  sourcing `final_bpm` from `get_final_summary()` in both places.

**Full 56-session result after fixing both** (`test/validation_run5_final.log`):
MAE 17.4 → **11.36 BPM**, RMSE 24.1 → **18.08 BPM** (35% reduction).

**Honest downside, documented not hidden:** the subharmonic-promotion fix
introduced new regressions on several previously-good sessions —
`dataset_1/after-exercise`, `subject22/24/32/37`, and two UBFC-Phys
sessions. Net effect was clearly positive, but not universal.

---

## Era 5 — The Mechanical Bug (hardcoded FPS)

Rather than guess at a second fix for the remaining bad subjects
(subject25/26, at a *different* ratio — 1.26-1.35×, not a clean 2×), the
same discipline was applied: trace first, fix second.

**Trace of `subject25`** (`test/trace_subject25.log`) showed a completely
different signature from subject41 — **rock-stable, high-confidence,
wrong the entire clip**, not oscillating between two states at all
(88% stability score). That ruled out harmonic ambiguity outright.

**Root cause, found by checking the actual video file properties**:
`subject25`'s video is recorded at **23.2 fps**, not 30 — but every
`AdvancedRPPG` construction site in the codebase hardcoded `fps=30`
regardless of the source's real frame rate. Analyzing a 23.2fps video as
if it were 30fps deterministically scales every frequency by
(30/23.2 ≈ 1.29×): 91.5 true bpm × 1.29 ≈ 118, matching the observed
error almost exactly.

**Fixed**: `Camera.get_source_fps()` (new method in `core/camera.py`)
reads the actual frame rate from the opened source; every `AdvancedRPPG`
construction site (`scripts/analyze_video.py`, `web/routes.py::_reset`) now
uses it instead of a hardcoded constant.

**Verified immediately**: `subject25` 23.5 → **3.5** BPM error, `subject26`
25.5 → **2.5** BPM error.

**Full 56-session result with both fixes combined**
(`test/validation_run6_fpsfix.log`): MAE 11.36 → **8.69 BPM**, RMSE
18.08 → **15.84 BPM**. UBFC-rPPG subset alone: **7.79 BPM**, crossing into
"research grade" (<8.0) for the first time. Stratified by condition:
arithmetic 3.5, rest 6.5, speech 23.4 (essentially untouched by either fix
— confirming the speech-condition gap is a genuinely separate problem,
motion from talking, not frequency-domain error).

**Combined effect across this whole debugging thread: MAE 17.4 → 8.7 BPM,
a 50% total reduction, from two independently root-caused, independently
verified bugs.**

---

## Era 6 — Grounding Against the Literature

Before concluding whether 8.7 BPM MAE was "good" or "bad," the actual
published literature was checked rather than assumed. Found: ["A Unified
Benchmarking Framework for Classical Machine Learning Based Heart Rate
Estimation from RGB and NIR rPPG"](https://doi.org/10.3390/electronics15010218)
(MDPI *Electronics*, 2026) reports, on the same UBFC-rPPG dataset:
**CHROM MAE = 8.20, POS MAE = 8.35**, with a newer hybrid ("ChPOS") at 5.71.

This project's UBFC-rPPG-only result (**7.79 BPM**) is *slightly better*
than the published academic CHROM baseline on the same dataset. Not
best-in-class (deep learning and specialized hybrids do better), but
credible, not embarrassing, and — critically — a legitimate data point a
reviewer checking the literature would accept rather than flag as
suspicious.

---

## Era 7 — Going Public

`git init` was run for the first time this session (the project had never
been version-controlled). A `.gitignore` gap was found and fixed
(`**/venv/` was missing — only `.venv/`/`env/` were excluded, meaning the
entire virtual environment would have been committed). A second, much
larger gap was closed deliberately: dataset video files (100GB+ across
three datasets) were excluded from git entirely, while the small
evidence files (`manifest.csv`, every validation log, trace files) were
kept — verified via `git add -n .` dry run before ever actually staging
anything. First commit: 74 files, ~3MB, zero raw dataset content. Pushed
to [github.com/Kushal-Raj-G-S/VITAL](https://github.com/Kushal-Raj-G-S/VITAL).

This mattered beyond just "backing up code": the project's dataset lives
on a borrowed laptop that has to be returned to its owner. The repository
(README.md's Validation section + `CLAUDE.md` + every log in `test/`) is
what survives that laptop being wiped — not the 100GB+ of video files,
which were never the actual asset.

---

## Era 8 — The Agent Trust Question (and a real mistake)

Every validation up to this point measured the **rPPG signal pipeline**.
None of it measured whether the **multi-agent LLM layer** actually behaves
as instructed — specifically, whether the Diagnostic/Coordinator agents
faithfully restate the deterministic clinical assessment rather than drift
or hallucinate a different number, as `agents/config/tasks.yaml` explicitly
instructs them to.

**A real mistake happened building this test**, and it's recorded here
rather than smoothed over: the first version of
`scripts/evaluate_agent_reliability.py` passed a fake placeholder video
path (`"synthetic:<scenario_name>"`) to the crew, trusting the task
instructions to prevent the agent from re-calling the vitals-extraction
tool. **The LLM did not reliably follow that instruction** — it called the
tool anyway with the fake path. `Camera` in "dummy mode" (source failed to
open) is designed to return an endless placeholder frame forever, correct
behavior for the live MJPEG stream (which should keep displaying
*something*), catastrophic for batch analysis, whose only "stop" signal
(`frame_bytes is None`) dummy mode never sends. The process ran for
**2 hours 20 minutes**, producing a 10.7MB log of nothing, before being
noticed and killed.

**Root-cause fix, not a patch**: `analyze_video_complete()` now checks
`camera.dummy_mode` immediately after construction and fails fast with a
clear error, before ever entering the frame loop. A second, independent
safety net was added regardless: a hard 100,000-frame cap, so any future
bug of this shape cannot silently hang again. The evaluation script itself
was also changed to route through a real, small, already-validated video
file instead of a fake path — three independent layers of protection
against the same class of mistake recurring.

**Verified with a hard OS-level timeout this time** (`timeout 180 ...`,
`timeout 2400 ...` wrapping the real run) — not trusted blind.

**Real results, 16 CrewAI executions across the full clinical severity
range** (`test/agent_reliability_run2.log`,
`test/agent_reliability_results.jsonl`):

| Field | Faithfulness rate |
| :--- | :--- |
| JSON parses at all | 15/16 (93.8%) |
| ESI level restated correctly | 13/16 (81.2%) |
| **Priority score restated correctly** | **1/16 (6.2%)** |
| Shock flag restated correctly | 15/16 (93.8%) |
| Diagnosis — exact wording | 6/16 (37.5%) |
| Diagnosis — correct meaning, different words | 12/16 (75.0%) |

**Why this is the project's strongest single finding, not just a QA
metric**: the 6.2% number means that even when explicitly instructed to
copy a given number verbatim, the LLM almost never does — it repeatedly
stuffed descriptive text into the numeric field instead (e.g.
`"esi_level": "2 (Emergent)"`, `"priority_score": "ESI Level 2 — Emergent /
Compensated Shock (score=0.311)"`). This is **empirical, measured proof**
that the Era-1 decision — never let an LLM compute or hold the
authoritative clinical number, only narrate a deterministically-computed
one — was not a defensive-programming nicety. Had this system trusted the
LLM's own JSON for `priority_score` (as many "agentic AI in healthcare"
demos do), it would have assigned the wrong queue priority to a patient
**over 90% of the time.** The architecture doesn't just avoid this
failure mode by design — it can now be shown, with real data, exactly how
badly the alternative would have failed.

A related, separate finding from the same runs: the LLM occasionally tried
to invoke tools that don't exist at all (`structured_summary_tool`,
`clinical_summary_tool` — "Tool not found") — a second, distinct
reliability gap in the agent layer, worth citing alongside the above.

---

## Where this leaves the project

**Solid and measured:**
- Deterministic clinical engine, unit-tested, provably authoritative over the LLM
- 56-session real-video MAE of 8.7 BPM (research-grade on the UBFC-rPPG subset, competitive with published CHROM baselines)
- Two independently root-caused, independently verified DSP bugs fixed (harmonic lock-on — partial fix; hardcoded FPS — clean fix)
- A measured, quantified justification for the core safety architecture (agent reliability evaluation)

**Open, honestly documented, not hidden:**
- A handful of real-video sessions still show large, unexplained errors (subject22/24/32/37, after-exercise, one UBFC-Phys speech session)
- Speech-condition accuracy (23.4 MAE) is untouched by either DSP fix — a separate, unaddressed motion-artifact problem
- No skin-tone-stratified validation exists (no dataset used ships that metadata compatible with this pipeline)
- Live camera is inherently single-session (one workstation, one webcam) — the "queue" is a sequential log, not concurrent multi-patient monitoring
- The LLM's exact-field restatement reliability (especially for numeric fields) is measurably poor — mitigated by architecture, not solved

Every specific claim above has a corresponding raw log file in `test/` that
can be independently re-checked. Nothing in this document should be taken
on faith over the underlying evidence.
