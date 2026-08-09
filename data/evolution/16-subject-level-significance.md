# Era 16 — The 31% Reduction Is Real, But It Isn't Typical

This one didn't start from a bug. It started from an external review of the
draft IEEE paper (a simulated peer-review pass, not a real submission) that
asked a pointed question the project had never actually checked: **is the
9.43 → 6.48 BPM MAE improvement broad across subjects, or is it a mean
being dragged down by a handful of dramatically-corrected outliers?**

MAE is an average. Averages can be misleading when a few large corrections
dominate a mostly-flat distribution. The honest way to answer this is to
look at the *paired, per-subject* distribution of the improvement, not just
its mean — and to check whether that distribution would survive a
significance test, not just eyeball the aggregate number.

## No rerun needed — the answer was already sitting on disk

The exact run that produced the paper's headline numbers is
`test/validation_run11_huber_full71.log` (Era 12's Huber-loss confirmation
run). It logs classical error and hybrid error for all 71 sessions,
line by line. Parsing that log directly (no reprocessing, no retraining)
gives real per-session and per-subject paired data for free.

## What the numbers actually show

**Session level (n=71, paired classical-error minus hybrid-error):**

| | Value |
| :--- | :--- |
| Mean improvement | 2.95 BPM (matches 9.43−6.48, confirms the parse) |
| Median improvement | **0.50 BPM** |
| IQR | [-2.35, 3.00] |
| Sessions where hybrid is better | 40/71 |
| Sessions where hybrid is worse | 31/71 |
| Bootstrap 95% CI on mean | [0.56, 5.64] |
| Wilcoxon signed-rank test | statistic=1084.5, **p=0.268** |

**Subject level (n=51, averaging each subject's sessions first, so
UBFC-Phys's 3-sessions-per-person structure doesn't triple-count anyone):**

| | Value |
| :--- | :--- |
| Mean improvement | 2.07 BPM |
| Median improvement | **0.60 BPM** |
| IQR | [-1.65, 2.75] |
| Subjects where hybrid is better | 30/51 |
| Subjects where hybrid is worse | 21/51 |
| Bootstrap 95% CI on mean | **[-0.19, 4.54]** — includes zero |
| Wilcoxon signed-rank test | statistic=538.5, **p=0.243** |

**Top-5-subject share of the total subject-level improvement: 101.7%.**
(The top 5 subjects' gains of 15.5–36.2 BPM each sum to slightly *more*
than the entire net improvement, because a large minority of subjects —
21 of 51 — got worse and roughly cancel out the rest of the gainers.)

## The honest interpretation

The 31% relative MAE reduction is **not fabricated or miscalculated** — it
is the correct arithmetic result of averaging 71 real session errors, and
it matches the headline number in every table already in the paper and
README. But "31% relative MAE reduction" and "the hybrid model reliably
helps a session" are two different claims, and only the first one is
actually supported here:

- At the session level, fewer than 4 in 7 sessions improve at all, and a
  paired Wilcoxon test does **not** reach significance (p=0.27).
- At the subject level, barely more than half of subjects (30/51) see any
  net improvement, the 95% CI on the mean gain crosses zero, and the test
  again does not reach significance (p=0.24).
- The entire net gain is carried by roughly 5 subjects with large
  corrections (15–36 BPM), consistent with the project's own repeated
  finding that the hybrid layer's real value is rescuing specific hard
  failure cases (arithmetic/speech motion artifact, the kind of session
  that was 40+ BPM off classically) rather than incrementally improving
  every session a little.

This is consistent with, not contradictory to, everything already
documented: [Era 11](11-more-subjects.md) already showed the classical
baseline is "shakier than the smaller sample suggested" and that the
correction model "held up on exactly the cases that broke it" — this era
just puts a number and a significance test on that same pattern instead of
describing it qualitatively.

## What this means going forward

If this project is ever extended into a fuller (e.g. journal-length)
write-up, the headline claim should be stated more precisely: the hybrid
correction layer produces a real, non-trivial reduction in aggregate MAE,
concentrated in a specific subset of hard failure cases, not a uniform
per-session or per-subject improvement — and that distinction should be
stated in the same breath as the 31% number, not left for a reader to
discover by re-deriving it themselves. This was deliberately **not** folded
into the current draft IEEE paper (already at a tight 6-page limit); it is
recorded here so the finding isn't lost, and so a future revision — on
this machine or any other — knows exactly what data backs it and how to
reproduce it (`test/validation_run11_huber_full71.log`, parsed for
per-session classical/hybrid error pairs, then paired bootstrap + Wilcoxon
at both the session and subject level).

**Next: [Era 17 — Live-Testing Round 2](17-live-session-hardening.md)** —
duplicate triage records, a misaligned RR reliability gate, and a
regression that fix caused, all found and fixed by re-testing the live
product end to end.
