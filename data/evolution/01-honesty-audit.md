# Era 1 — The Honesty Audit

**Trigger:** the student asked for an "overall honest truth" on the project's
approach, technique, and architecture — not encouragement, an audit.

## Findings, in priority order

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

## Fixes applied this era

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

**Next: [02 — First Contact With Reality](02-live-testing-bugs.md)**
