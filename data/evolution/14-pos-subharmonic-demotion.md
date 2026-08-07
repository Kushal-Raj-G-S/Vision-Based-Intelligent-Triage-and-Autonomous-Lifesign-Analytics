# Era 14 — POS's Own Bug Is a Mirror Image, Partially Fixed

CHROM's documented harmonic-ambiguity bug (Era 4) undershoots to roughly
half the true rate. POS, this project's other implemented but never-live
algorithm, was documented as having "its own separate harmonic lock-on
under sensor noise (26.71 BPM MAE at HR≲90bpm)" — but nobody had actually
traced *why*.

## The trace

Running POS against a synthetic 50 BPM signal and watching per-cycle debug
output showed the very first reading right after cold start was already
**99.0 BPM — almost exactly double the true rate.** The adaptive filter
then locked onto that and narrowed its search band around it, cutting the
true 50 BPM component out of consideration entirely rather than
correcting itself over time.

This is the mirror image of CHROM's bug. CHROM undershoots; POS
overshoots. The existing fix for CHROM (`_select_dominant_freq_hps`'s
promotion logic) only ever checks *upward* — "does the top candidate's
double have comparable power, if so promote to it" — which by
construction can never catch an *overshoot*, since the top candidate is
already too high.

## The fix

A mirrored, symmetric check was added: does the top candidate's **half**
have comparable power? If so, demote to it — same tolerance (0.08 Hz) and
threshold (55%) as the existing promotion check, kept symmetric since
there was no prior evidence either direction needed different
sensitivity.

## Verified, honestly, in both directions

- **CHROM (the default, live algorithm): completely unaffected.**
  Synthetic self-test stayed at exactly 0.29 MAE — the shared function is
  used by both algorithms, so this was the one non-negotiable check.
- **POS synthetic: real, partial improvement.** MAE dropped from 26.43 to
  **16.00** (39% reduction). GT=50 (the traced case) went from 74.0 to
  **1.0** BPM error. But GT=75 and GT=90 still fail badly (53.0 and 55.0
  BPM error) — and their wrong values aren't even clean 2x/half ratios of
  the truth (128 vs 75 is a 1.71× relationship, not 2×), meaning this is a
  **third, different bug**, not more of the same ambiguity.
- **POS on real video — tested for the first time ever in this project's
  history.** Mixed: `subject1` was accurate (1.7 BPM error), but
  `subject41` undershot badly (48.8 error) and `subject37` overshot to
  **exactly double** its true rate (126 vs 63.0) — a textbook case the
  promotion check should catch but didn't, likely compounded by the same
  `bpm_history` gate dynamics CHROM has (POS shares that same code path,
  and was never checked against it before).

## Honest bottom line

This is a real, verified, non-regressing fix — worth keeping. It is **not**
"POS is fixed." POS still has unresolved harmonic ambiguity in both
directions on real video, and at least one more synthetic failure mode
(GT=75/90) that isn't a 2x/0.5x relationship at all. POS remains
experimental and stays out of the live product (`CLAUDE.md`'s existing
warning against wiring it in without full re-validation still applies,
now with more evidence backing exactly why).

**Next: see the top-level [`data/README.md`](../README.md) for the
current state of the project.**
