# VITAL — Project Evolution: Index

This document used to be one long file. It's now split into one file per
era under [`evolution/`](evolution/), so each chapter is short enough to
read on its own and easy to link to individually. Read them in order —
each ends with a link to the next:

1. [00 — Pre-History](evolution/00-pre-history.md) — what existed before
   this audit began, vs. what the original assignment asked for.
2. [01 — The Honesty Audit](evolution/01-honesty-audit.md) — the
   architecture had an LLM computing the clinical safety numbers; how that
   got found and fixed.
3. [02 — Live Testing Bugs](evolution/02-live-testing-bugs.md) — a false
   shock alert, an RR-bouncing bug, a vanishing-database-records bug, the
   LLM model swap.
4. [03 — Dataset Acquisition](evolution/03-dataset-acquisition.md) —
   UBFC-rPPG/UBFC-Phys/rPPG-10 integration, the first real-data validation,
   and the discovery of a bimodal harmonic-lock-on failure.
5. [04 — The Age of Experiments](evolution/04-harmonic-lock-on-fixes.md) —
   three attempts at fixing the harmonic lock-on bug; two failed, verified
   by testing, not assumed.
6. [05 — The Mechanical Bug](evolution/05-fps-bug.md) — a hardcoded FPS
   assumption, root-caused and fixed.
7. [06 — Grounding Against the Literature](evolution/06-literature-comparison.md)
   — comparing this project's real numbers against a published benchmark
   paper instead of assuming they're good.
8. [07 — Going Public](evolution/07-going-public.md) — git setup, dataset
   exclusion, first push to GitHub, and why that mattered given the
   borrowed-laptop constraint.
9. [08 — The Agent Trust Question](evolution/08-agent-reliability.md) —
   testing whether the LLM agents faithfully restate the deterministic
   clinical assessment; a real 2h20m mistake made and fixed along the way;
   the strongest single finding of the project.
10. [09 — A Third Bug, and a Fix That Broke the Flagship Case](evolution/09-history-freeze-bug.md)
    — a third, independent bug found by tracing a regression; a first fix
    that worked on its target but silently broke the project's most-cited
    result; what that revealed; the corrected, verified fix. MAE 8.7 → 8.4
    BPM, 52% total reduction across the whole debugging thread.

Nothing here is invented after the fact — this is what actually happened,
including the mistakes. Every claim in each chapter points at the actual
evidence file that backs it (in [`test/`](../test/)), not just a
description of what happened.

For the current, final numbers (not the history of how they were reached),
see [README.md](../README.md)'s Validation section. For guidance on
working in this codebase, see [CLAUDE.md](../CLAUDE.md).
