# Era 0 — Pre-History: what existed before any of this

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

**Next: [01 — The Honesty Audit](01-honesty-audit.md)**
