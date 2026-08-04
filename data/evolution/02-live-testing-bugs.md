# Era 2 — First Contact With Reality (live testing surfaces real bugs)

Once the Era 1 fixes were live, the student actually ran the dashboard and
immediately hit real, previously-invisible bugs — the kind that only show up
when a real person uses the real product, not when reading the code.

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
  fallback rather than dropped. **This choice matters a lot later — see
  [Era 8](08-agent-reliability.md).**
- `core/camera.py` — added full 468-point face-mesh dot visualization
  (cosmetic only) and trimmed-mean spatial pooling for the forehead/cheek
  ROIs (excludes the darkest 10%/brightest 5% of pixels per box, verified via
  synthetic blemish-injection test to recover the true clean-skin color
  within 0.1 units where a plain average was off by ~5).

**Next: [03 — The Data Age](03-dataset-acquisition.md)**
