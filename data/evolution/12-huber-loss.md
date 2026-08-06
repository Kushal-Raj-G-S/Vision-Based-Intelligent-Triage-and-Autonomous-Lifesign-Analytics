# Era 12 — A Small, Honest Win: Huber Loss

After Era 10 established that bigger models don't help and Era 11 showed
more real subjects widens the hybrid model's advantage, the next question
was narrower: is the small gradient-boosted correction model's *loss
function* — not its size — leaving anything on the table?

## The reasoning, checked before spending hours on it

The default (squared-error) loss trains a model to minimize the *square*
of its errors. That means a handful of extreme-outlier sessions — the
kind seen throughout this project's real data (a 44 BPM classical error on
one UBFC-Phys arithmetic session, a 41 BPM error on a speech session) —
get disproportionate weight during training, since squaring a big error
makes it much bigger. Huber loss behaves like squared-error for small
errors but like absolute-error for large ones, specifically to stop a few
extreme cases from dominating.

## Cheap test first, expensive test only if it looked worth it

Rather than immediately retrain the deployed fold models and re-run the
full multi-hour session-level validation, Huber vs. squared-error was
compared at the **per-cycle level** first, reusing the already-extracted
242,026-row dataset — no video reprocessing, ~25 minutes total:

| | Per-cycle MAE | RMSE |
| :--- | :--- | :--- |
| Squared-error (original) | 8.80 | 12.03 |
| **Huber** | **8.67** | 12.01 |

A real improvement (~1.5%), but a small one — set the honest expectation
upfront that any session-level effect would likely be smaller still, since
median-aggregating ~150-300 cycles per session already smooths out a lot
of what a per-cycle robustness fix targets.

## The expensive confirmation, run anyway

The fold models and deployed model were retrained with Huber loss, then
the full 71-session out-of-fold validation (~4.5 hours) was re-run to get
the real session-level number rather than assume the per-cycle gain would
translate:

| | MAE | RMSE |
| :--- | :--- | :--- |
| Classical DSP | 9.43 | 16.51 |
| Hybrid, squared-error | 6.65 | 9.66 |
| **Hybrid, Huber** | **6.48** | 9.66 |

Exactly as predicted: real, but modest — a further ~2.6% relative MAE
reduction, RMSE unchanged. **Combined effect of every fix in this
project's hybrid-layer work: classical 9.43 → hybrid 6.48 BPM, a 31%
relative reduction.**

## A process note worth keeping

This session also involved two background jobs getting silently killed
mid-run by unrelated tool/session disconnects (not by any bug in the
scripts themselves) — once 9 of 15 videos in, once 5 of 6. Both times,
checking whether the underlying OS process was still actually alive
(`Get-Process`/`Get-CimInstance`, not just trusting a "stopped" status
label from the tooling layer) revealed the work was still running and
finished on its own — avoiding redoing multiple hours of already-completed
processing. Worth remembering for any future long-running job in this
environment: a "stopped" notification from the harness is not the same
as the process actually being dead.

**Next: see the top-level [`data/README.md`](../README.md) for the
current state of the project.**
