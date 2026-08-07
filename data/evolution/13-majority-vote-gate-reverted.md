# Era 13 — Two Real Fixes, a Net Wash, and a Reverted Change

After the hybrid correction layer and Huber loss (Eras 10-12), the
remaining open items were the still-undiagnosed classical-DSP regressions
from Era 4: `subject22`, `subject24`, `subject32`, and `after-exercise`.
Tracing all four (same discipline as every previous era) revealed they are
**at least three genuinely different bugs**, not one:

| Subject | Signature | Category |
| :--- | :--- | :--- |
| `subject22`, `subject32` | A genuine majority of raw per-cycle readings sits near the correct rate, but a minority got frozen into `bpm_history` by the same gate family fixed (narrowly) in Era 9 | Aggregation-gate bug |
| `subject24` | No clear majority at all — 98% of readings get accepted into history, spread across a wide 80-130 BPM range with ground truth at the low edge | Estimation noise/drift, not a gate bug |
| `after-exercise` | 72.5% of readings lock onto ~70 BPM with a real, strong companion peak — but that companion is at ~140 BPM, not the true rate (113.9). 113.9/70 ≈ 1.63, not a clean 2× ratio | **Not** a subharmonic-lock case at all — a competing non-cardiac periodic component (plausibly motion/breathing artifact, consistent with post-exercise recording conditions) simply overpowers the true pulse signal. Confirmed via instrumenting `_select_dominant_freq_hps` directly: the promotion check's ratio (14-18%) is nowhere near its 55% threshold, and tightening that threshold would not help, because there's no real 2:1 relationship to detect here in the first place. |

## The fix attempted, and why it was hard to get right

A majority-vote re-anchoring mechanism was added to `core/rppg.py`'s
history-append gate: track every raw smoothed BPM reading (not just
gate-accepted ones) across the whole clip, and let a challenger reading
override a stale anchor if it belongs to a clearly dominant (>55%)
cluster, rather than requiring the narrow small-history-only condition
Era 9 was limited to.

**First version regressed `subject41` hard** (8.2 → 41.8 BPM error) using
a fixed 300-cycle rolling window: a genuinely ~50/50 bimodal signal
(subject41's known, still-unresolved ambiguity from Era 4) can drift well
past 55% in a short local window purely by chance, since real clusters
persist in multi-second blocks rather than alternating every cycle.
Switched to an **unbounded, whole-clip-so-far** cumulative window instead
— a true ~50/50 split stays near 50/50 as more data accumulates; a real
majority stays a stable majority. This fixed `subject41` (restored to 8.2,
then actually *improved* to 0.2-0.8 across several re-tests) and fixed
`subject22` cleanly (46.8 → 0.8).

**A second bug in the check itself** (bin-based clustering with too narrow
a tolerance) initially failed to resolve `subject32` even though its
majority cluster was real (56.3% of readings) — a single candidate near
the edge of its own spread-out cluster wasn't being counted as agreeing
with most of that same cluster. Switched to a distance-based tolerance
matching the main gate's own 18 BPM tolerance, for consistency.

**Then, checking the previously-assumed-unaffected cases, `after-exercise`
regressed badly** (32.9 → 54.9 BPM error) — because its "majority" is
exactly backwards: 72.5% of its raw readings are the *wrong* answer (the
competing artifact described above), and trusting a majority blindly
reinforces that instead of recovering a correct signal like it did for
`subject22`/`subject41`. There is no way to distinguish "majority = the
correct signal, minority = a gating artifact" from "majority = a strong
systematic error, minority = the truth" at runtime without already
knowing the ground truth — the two known-good target cases and the one
newly-broken case are contradictory training signals for the same
mechanism.

## The full, honest verification — and why the fix was reverted

Individually-verified deltas looked promising (subject22 −46.0, subject41
−8.0, subject32 ≈0, after-exercise +22.0 — net −32.0 across just the four
traced cases). But the full 71-session out-of-fold validation — the same
discipline that caught Era 9's subject41 regression — told a different
story:

| | MAE | RMSE |
| :--- | :--- | :--- |
| Classical, without the fix (Era 12 baseline) | 9.43 | 16.51 |
| Classical, with the majority-vote fix | **9.54** (worse) | **16.89** (worse) |
| Hybrid, either way | ~6.44-6.48 (unaffected — the correction layer doesn't depend on this gate at all) | ~9.60-9.66 |

Despite two clear, individually-verified wins, the aggregate classical
number got *slightly worse*, not better — meaning sessions beyond the
four traced here (not individually identified) regressed enough to more
than offset the fixes. **The change was reverted** (`core/rppg.py` restored
to its Era 12 state via `git checkout`) rather than kept for a net-negative
result, especially since the number that actually matters for this
project — the hybrid MAE — was unaffected by it either way, so there was
nothing to gain by keeping the added complexity and risk.

## Why this is documented rather than deleted

This is the same lesson Era 4 and Era 9 already taught, in a new form: a
fix that demonstrably helps its target cases is not the same as a fix
that helps in aggregate, and the only way to know the difference is to
check — not assume, not extrapolate from a handful of spot checks. Three
separate times now, across three different fixes, this exact discipline
caught something a smaller check would have missed. The `subject22`/
`subject41`/`subject32` traces and the `after-exercise` frequency-selection
diagnostic remain genuinely useful evidence of what's actually wrong with
this pipeline, even though the fix built from them didn't survive full
validation.

**Current numbers, unchanged by this era**: classical DSP 9.43 BPM MAE,
hybrid (DSP + Huber-loss ML correction) 6.48 BPM MAE, across 71 real
sessions — the same as reported at the end of Era 12.

**Next: see the top-level [`data/README.md`](../README.md) for where the
project stands now.**
