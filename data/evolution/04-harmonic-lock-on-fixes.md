# Era 4 — The Age of Experiments (fixing the harmonic lock-on bug)

Three attempts were made, in order, on the exact same evidence
(`test/dataset_2/subject41`, the case root-caused in
[Era 3](03-dataset-acquisition.md)). **Two failed completely, verified by
testing, not assumed.** This is the most important methodological lesson of
the whole project: *a change that looks reasonable under code review is not
the same as a change that has been verified.*

## Attempt 1 — BPM lock/candidate hysteresis (FAILED, 0% improvement)

`AdvancedRPPG._resolve_bpm_lock` — a debounced state machine requiring
sustained multi-frame agreement (much more for harmonic-ratio-looking
jumps) before the tracker or its adaptive filter would recenter on a new
frequency. Its first version regressed two previously-perfect synthetic
cases (a same-value streak-counter that was too fragile against ordinary
FFT-bin jitter) — caught immediately by re-running the synthetic self-test,
fixed with a rolling-window spread check instead.

Verified via trace (`test/trace_subject41_v2.log`) to work **exactly as
designed**: it correctly resolved transient noise and correctly held
through the genuine mid-recording correct stretch. Re-running the full
18-subject validation afterward showed **zero aggregate improvement**
(still ~19.1 MAE) — because the false lock wasn't noise-driven
flip-flopping in the first place. It was sustained, stable, and spectrally
confident for hundreds of frames at a stretch. Hysteresis debounces
flip-flopping; it cannot tell two stable-but-different answers apart.

## Attempt 2 — Naive upward-summing Harmonic Product Spectrum (FAILED, 0% improvement)

Classic pitch-detection technique: score each candidate frequency by summing
PSD power at its own value plus its integer harmonics (2×, 3×) — a real
periodic signal should show coherent energy at its own harmonics.

Traced (`test/trace_subject41_hps.log`) and found to produce **an
identical result to before the fix** — and reasoned through *why*, not just
observed that it failed: for a signal locked at half the true rate, the
*true* signal's own energy sits exactly where that low candidate's "2nd
harmonic" would be. Naive HPS summation therefore reads the true signal's
energy as supporting evidence *for* the wrong (low) candidate, not against
it. The technique was directionally wrong for this specific failure mode.

## Attempt 3 — Asymmetric subharmonic-promotion (WORKED, partially)

Corrected approach: for the raw top candidate frequency, explicitly check
whether its **double** has comparable-or-greater spectral power; if so,
promote to the double instead of trusting the raw peak. This targets the
*specific, evidenced* failure direction (undershooting to ~half-rate)
instead of a generic harmonic-support heuristic.

Traced (`test/trace_subject41_v3.log`): frames 180-840 now correctly read
~98-127 instead of ~45-70; final result for subject41 went from 52.6 BPM
error to 8.2 BPM error.

### Two additional bugs found while verifying this, both of which would have made the measurement itself wrong

- `scripts/validate_dataset.py`'s manifest path was passing
  `process_stride=2` to `analyze_video_complete()` (a speed optimization
  added earlier for rPPG-10's 10-minute clips). The BPM lock/subharmonic
  logic counts in *cycles* (calls to `process_ppg_signal()`), not real
  time — striding silently doubled its effective time window, changing
  tracking *behavior*, not just speed. A direct stride=1 trace showed a
  real fix; the identical video through the strided batch script showed no
  change at all. Fixed by keeping stride=1 on this path (short UBFC
  clips don't need the optimization; it stays reserved for rPPG-10's
  genuinely long clips, which have no cycle-counted hysteresis in play).
- `scripts/analyze_video.py::analyze_video_complete()` computed its final
  BPM as a plain mean of the **last 10 readings** — a completely different
  algorithm than `AdvancedRPPG.get_final_summary()`'s IQR-filtered
  full-history median, which the live webcam path already used. Two
  different "final answer" methods for the same concept, silently
  disagreeing. This specifically mattered for subject41: a video that
  settles into a wrong lock right at the very end previously reported that
  wrong tail value no matter how correct the rest of the clip was. Fixed by
  sourcing `final_bpm` from `get_final_summary()` in both places.

## Full 56-session result after fixing both

See `test/validation_run5_final.log`: MAE 17.4 → **11.36 BPM**, RMSE
24.1 → **18.08 BPM** (35% reduction).

## Honest downside, documented not hidden

The subharmonic-promotion fix introduced new regressions on several
previously-good sessions — `dataset_1/after-exercise`, `subject22/24/32/37`,
and two UBFC-Phys sessions. Net effect was clearly positive, but not
universal.

**Next: [05 — The Mechanical Bug (hardcoded FPS)](05-fps-bug.md)**
