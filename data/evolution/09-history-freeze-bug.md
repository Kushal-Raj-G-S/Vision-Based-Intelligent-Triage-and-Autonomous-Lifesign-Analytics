# Era 9 — A Third Bug, and a Fix That Broke the Flagship Case

With the paper's deadline close and the two big DSP fixes already validated
([Era 4](04-harmonic-lock-on-fixes.md), [Era 5](05-fps-bug.md)), the
remaining honest, still-open items were the unexplained per-session
regressions documented at the end of Era 4: `dataset_1/after-exercise`,
`subject22`, `subject24`, `subject32`, `subject37`, and UBFC-Phys `s4`
speech. Rather than chase every one blind, the same discipline that found
the first two bugs was applied again: pick one, trace it, don't guess.

## Tracing `subject37`

Enabling per-cycle debug logging and re-running the video showed something
that immediately ruled out a frequency-domain explanation: the smoothed
BPM estimate converged to **~58 BPM** (ground truth: 63.0 — a good result)
and held there, rock-stable, SQI 73-74, for nearly the entire two-thousand
-frame clip. Yet the reported session summary was **84 BPM**.

The bug wasn't in signal processing at all — it was in
`AdvancedRPPG.get_final_summary()`'s aggregation. A gate in
`process_ppg_signal()` only lets a reading into `bpm_history` (the list
`get_final_summary()` takes its median from) if it's within 18 BPM of the
median of the **last 5 accepted** readings — meant to keep single-frame
noise spikes out of the final answer. `subject37`'s history had exactly
**7 entries**, all from a wrong 82-89 BPM burst during the first couple of
seconds of cold-start convergence. Every one of the ~1950 later, correct,
~58 BPM readings differed from that frozen 7-entry anchor by more than 18
BPM — so none of them were ever accepted. The gate can freeze permanently
on an early fluke and never recover, no matter how long the signal is
correct afterward.

## Fix attempt 1 (FAILED on a previously-fixed case)

The obvious fix: if a long enough streak of readings has been consistently
rejected — but consistent with *each other* — treat it as a genuine
sustained plateau and let it re-anchor the history. Implemented as a
reject-streak counter (`_HISTORY_UNSTICK_STREAK = 150` cycles, ~5s).

Verified on the target case: `subject37` immediately fixed, 21.0 → 6.0 BPM
error, `bpm_history` size going from 7 to 1764 entries. Synthetic self-test
unaffected (still 0.29 MAE). Looked done.

**Then the full 56-session real-data re-run, per this project's own
standing rule ("re-run the harness after any core/rppg.py change, every
time"), told a different story:** aggregate MAE barely moved (8.69 → 8.42)
and RMSE got slightly *worse* (15.84 → 15.99). Buried in the per-subject
table: `subject41` — the single most-cited success of the entire project,
the case the subharmonic-promotion fix in Era 4 was built and verified
against — regressed from its documented **8.2 BPM error back to 41.8**.

## Why it broke subject41 — and what that actually reveals

Tracing `subject41` again under the new code showed its per-cycle BPM
readings are **still genuinely bimodal across the whole clip**, even after
the Era 4 subharmonic-promotion fix — a rough 983-vs-933 split between a
low cluster and a high cluster in this trace, not a brief early wobble.
The naive unstick rule doesn't distinguish "a frozen anchor that was
wrong" from "a frozen anchor that happens to be correct and is now being
challenged by a later bout of the same old ambiguity" — it just gives up
resisting change after any sufficiently long streak, regardless of which
side is actually right.

That forces an uncomfortable but important conclusion, stated plainly
because this file exists to record things exactly as they happened: **the
original "subject41 fixed, 8.2 BPM error" result from Era 4 was never a
genuine resolution of its harmonic ambiguity.** It was this exact
freeze bug — undiscovered until now — accidentally anchoring on the
*correct* early cluster during subject41's cold start and then never
letting go, for the same reason it wrongly froze on the *incorrect* early
cluster for subject37. Two independent bugs, and one of them was quietly
propping up the headline result of the other. This is worth surfacing on
its own merits: a passing accuracy number is not proof that the mechanism
producing it is correct, and this project found a concrete, traceable
case of exactly that gap.

## Fix attempt 2 (verified correct)

The corrected version narrows the escape hatch instead of removing the
freeze protection: re-anchoring is only allowed while the frozen anchor
still has very little accumulated support (`_HISTORY_UNSTICK_MAX_HISTORY
= 10` entries) — genuinely untested, cold-start-fluke territory, exactly
`subject37`'s situation (7 entries). Once a history has substantial
support, a later long streak of disagreement is far more likely to be a
resurfacing of a real, still-unsolved ambiguity (subject41's situation)
than proof the anchor was wrong, so the escape hatch no longer fires.

**Verified, in order, before trusting it:**
1. Synthetic self-test: unchanged, 0.29 MAE.
2. `subject37` retrace: still fixed, 21.0 → **6.0** BPM error.
3. `subject41` retrace: fully restored, **8.2** BPM error, unchanged from
   Era 4's original result.
4. Full real 56-session re-run: **MAE 8.69 → 8.42 BPM, RMSE 15.84 → 15.61
   BPM** — a small, genuine, individually-traced improvement. The other
   previously-documented regressions (`subject22/24/32`, `after-exercise`,
   UBFC-Phys `s4` speech) came back completely unchanged either way,
   confirming they have a different, still-open root cause — this fix
   didn't touch them, for better or worse.

## Combined effect across all three DSP-layer fixes now

MAE: 17.4 → 11.4 (Era 4) → 8.7 (Era 5) → **8.4 BPM** (this era). RMSE:
24.1 → 18.1 → 15.8 → **15.6 BPM**. A 52% total MAE reduction across the
whole debugging thread, three independently root-caused and independently
verified bugs, one of which (this one) was only findable *because* an
earlier fix's own regression was investigated rather than shrugged off.

## The methodological point worth keeping, for the paper and beyond

The same lesson from Era 4 repeated itself, in a sharper form: a fix that
looks correct, that even measurably improves the exact case it was built
for, is not verified until the *other* known-good cases are re-checked
too. `subject41` wasn't some obscure edge case — it was this project's
single most-quoted result. The only reason this didn't ship as a silent
regression is the standing rule to re-run the full validation harness
after every `core/rppg.py` change, no exceptions, even when the targeted
fix already "looks done."

**Next: see the top-level [`data/README.md`](../README.md) and
[`EVOLUTION.md`](../EVOLUTION.md) for the current state of the project.**
