# Era 8 — The Agent Trust Question (and a real mistake)

Every validation up to this point ([Era 3](03-dataset-acquisition.md)
through [Era 6](06-literature-comparison.md)) measured the **rPPG signal
pipeline**. None of it measured whether the **multi-agent LLM layer**
actually behaves as instructed — specifically, whether the
Diagnostic/Coordinator agents faithfully restate the deterministic clinical
assessment rather than drift or hallucinate a different number, as
`agents/config/tasks.yaml` explicitly instructs them to.

## A real mistake happened building this test

The first version of `scripts/evaluate_agent_reliability.py` passed a fake
placeholder video path (`"synthetic:<scenario_name>"`) to the crew,
trusting the task instructions to prevent the agent from re-calling the
vitals-extraction tool. **The LLM did not reliably follow that
instruction** — it called the tool anyway with the fake path. `Camera` in
"dummy mode" (source failed to open) is designed to return an endless
placeholder frame forever, correct behavior for the live MJPEG stream
(which should keep displaying *something*), catastrophic for batch
analysis, whose only "stop" signal (`frame_bytes is None`) dummy mode never
sends. The process ran for **2 hours 20 minutes**, producing an 11.4MB log
of nothing, before being noticed and killed.

This is recorded here rather than smoothed over on purpose: it wasted real
time, on a borrowed laptop, and the person who caught it (not the AI
running it) was right to be frustrated and right to ask for more visibility
into long-running work going forward.

### Root-cause fix, not a patch

`analyze_video_complete()` now checks `camera.dummy_mode` immediately after
construction and fails fast with a clear error, before ever entering the
frame loop. A second, independent safety net was added regardless: a hard
100,000-frame cap, so any future bug of this shape cannot silently hang
again. The evaluation script itself was also changed to route through a
real, small, already-validated video file instead of a fake path — three
independent layers of protection against the same class of mistake
recurring. Verified with a hard OS-level timeout this time
(`timeout 180 ...`, `timeout 2400 ...` wrapping the real run) — not trusted
blind a second time.

## Real results, 16 CrewAI executions across the full clinical severity range

See `test/agent_reliability_run2.log`, `test/agent_reliability_results.jsonl`:

| Field | Faithfulness rate |
| :--- | :--- |
| JSON parses at all | 15/16 (93.8%) |
| ESI level restated correctly | 13/16 (81.2%) |
| **Priority score restated correctly** | **1/16 (6.2%)** |
| Shock flag restated correctly | 15/16 (93.8%) |
| Diagnosis — exact wording | 6/16 (37.5%) |
| Diagnosis — correct meaning, different words | 12/16 (75.0%) |

## Why this is the project's strongest single finding, not just a QA metric

The 6.2% number means that even when explicitly instructed to copy a given
number verbatim, the LLM almost never does — it repeatedly stuffed
descriptive text into the numeric field instead (e.g.
`"esi_level": "2 (Emergent)"`, `"priority_score": "ESI Level 2 — Emergent /
Compensated Shock (score=0.311)"`). This is **empirical, measured proof**
that the Era-1 decision — never let an LLM compute or hold the
authoritative clinical number, only narrate a deterministically-computed
one — was not a defensive-programming nicety. Had this system trusted the
LLM's own JSON for `priority_score` (as many "agentic AI in healthcare"
demos do), it would have assigned the wrong queue priority to a patient
**over 90% of the time.** The architecture doesn't just avoid this failure
mode by design — it can now be shown, with real data, exactly how badly the
alternative would have failed.

A related, separate finding from the same runs: the LLM occasionally tried
to call tools that don't exist at all (`structured_summary_tool`,
`clinical_summary_tool` — "Tool not found") — a second, distinct
reliability gap in the agent layer.

## Model size is a real, honest factor here, not an excuse

The model under test is `meta/llama-3.1-8b-instruct` — chosen in
[Era 2](02-live-testing-bugs.md) for speed/cost over the larger
`meta/llama-3.3-70b-instruct` (which is still configured as the first
fallback in `agents/crew.py`, not removed). Smaller instruction-tuned models
are well-documented in the field to be measurably worse at strict
structured-output fidelity than larger ones — this isn't a defect unique to
this project's prompting, it's a known, general property of model scale.
That context matters for how this finding should be read: it is not "LLMs
cannot be trusted for this," it is "*this specific 8B model*, chosen for
cost/speed, has a measured 6.2% numeric-fidelity rate — and the system's
architecture is built so that number doesn't matter for safety even though
it's low."

**A natural, not-yet-run follow-up experiment**: re-run the exact same 8
scenarios against the 70B fallback model directly and compare fidelity
rates. If 70B scores meaningfully higher, that becomes a second concrete,
citable data point — a measured cost/reliability tradeoff table
(model size vs. speed vs. faithfulness), not just a single model's score in
isolation. This has not been done yet as of this writing.

**Next: see the top-level [`data/README.md`](../README.md) for where the
project stands now and what's still open.**
