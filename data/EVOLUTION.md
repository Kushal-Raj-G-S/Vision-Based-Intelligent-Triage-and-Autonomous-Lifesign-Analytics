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
11. [10 — Testing Whether Bigger Helps](evolution/10-bigger-models-dont-help.md)
    — a hybrid classical-DSP + ML correction layer (8.4 → 6.5 BPM session
    MAE, and the speech-condition gap that DSP fixes never touched: 23.4 →
    9.4), then an honest test of whether a bigger model does even better.
    It doesn't — verified, not assumed, and explained by the real
    constraint (51 subjects, not 242K rows).
12. [11 — Finding 5 Subjects That Were Already There](evolution/11-more-subjects.md)
    — "56" was sessions, not people; 5 more UBFC-Phys subjects were
    already on disk and unused. Expanding to 71 real sessions widened the
    hybrid model's advantage (23% → 29% relative MAE reduction) rather
    than narrowing it.
13. [12 — A Small, Honest Win: Huber Loss](evolution/12-huber-loss.md) —
    testing whether the correction model's loss function (not size) had
    room to improve; a real but modest win, tested cheaply first before
    spending the ~4.5 hours to confirm it at the session level. Final
    result: classical 9.43 → hybrid 6.48 BPM, 31% relative reduction.
14. [13 — Two Real Fixes, a Net Wash, and a Reverted Change](evolution/13-majority-vote-gate-reverted.md)
    — tracing the last 4 unresolved classical regressions found at least 3
    genuinely different bug classes. A majority-vote gate fix cleanly
    resolved 2 of them (subject22, subject41) and was carefully verified
    against every known protected case — but the full 71-session
    validation showed a slightly *worse* aggregate classical MAE (9.43 →
    9.54), so it was reverted rather than kept for a net-negative result.
    Current numbers unchanged from Era 12.
15. [14 — POS's Own Bug Is a Mirror Image, Partially Fixed](evolution/14-pos-subharmonic-demotion.md)
    — POS overshoots to ~2x the true rate from cold start, the mirror of
    CHROM's undershoot bug; a symmetric demotion check fixed one pattern
    (synthetic MAE 26.71→16.00) without touching CHROM at all, but real
    video (tested for the first time ever) still shows unresolved harmonic
    errors. A real, partial, honestly-scoped fix — POS stays experimental.
16. [15 — POS's Real-Video Failures Aren't Threshold Problems](evolution/15-pos-real-video-ceiling.md)
    — measured (not guessed) why the remaining real-video failures
    resist threshold tuning: one has no correct candidate present at all,
    the other's correct candidate is real but too weak (median 11% power)
    to ever cross a safe threshold. Traces back to POS's weaker per-window
    normalization (no variance equalization, unlike CHROM) — a projection-
    level limitation, not a peak-selection one.

Nothing here is invented after the fact — this is what actually happened,
including the mistakes. Every claim in each chapter points at the actual
evidence file that backs it (in [`test/`](../test/)), not just a
description of what happened.

For the current, final numbers (not the history of how they were reached),
see [README.md](../README.md)'s Validation section. For guidance on
working in this codebase, see [CLAUDE.md](../CLAUDE.md).
