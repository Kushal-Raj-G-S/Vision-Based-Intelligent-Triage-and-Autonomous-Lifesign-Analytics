# Era 15 — POS's Real-Video Failures Aren't Threshold Problems

Era 14 fixed one real bug in POS (the overshoot-to-double pattern) with a
symmetric demotion check. Two real-video cases still failed after that
fix — `subject37` (63.0 → 126, an exact 2x error) and `subject41`
(100.8 → 52, roughly half) — and the working theory was that the existing
demotion/promotion threshold (55%) just wasn't tuned right for real,
noisier signals. That theory was tested directly rather than assumed, and
it turned out to be wrong.

## subject37 — no correct candidate exists at all

Tracing the full per-cycle distribution showed POS's raw readings for
this video sit almost entirely in the 80-140 BPM range — essentially none
anywhere near the true 63 BPM rate. This isn't a mislabeling problem (a
correct candidate the algorithm picks the wrong companion for); the true
heart rate essentially never appears as a spectral candidate at all in
POS's output for this video. No peak-selection rule — promotion,
demotion, or anything else — can select a candidate that isn't present.

## subject41 — the correct candidate exists, but it's genuinely weak

This one looked more promising: ~77% of readings cluster near half the
true rate (100.8/2 ≈ 50), and ~18% cluster near the correct rate — the
right shape for the demotion check to catch. Measuring the actual
power ratio between the true-rate candidate and the wrong (dominant) one,
instead of guessing at a better threshold, told the real story: median
ratio **11%**, max ever observed **54%** — never once crossing the
current 55% threshold. Even dropping the threshold aggressively to 30%
(risking new false demotions on already-correct cycles elsewhere) would
only have fixed **11% of the bad cycles**.

The fix's whole premise — "the true signal is present as a comparably
strong second peak, just mislabeled as secondary" — simply doesn't hold
here. The true signal's peak is often just too weak relative to whatever
POS's weaker per-window normalization lets through. This is not a
threshold-tuning problem; forcing one attempts to manufacture a fix where
the actual cause doesn't support one.

## Why this matches the literature, not contradicts it

The benchmarking paper compared against in Era 6
([data/evolution/06-literature-comparison.md](06-literature-comparison.md))
describes CHROM as introducing "stronger illumination compensation" via a
global mean-normalization plus per-window variance equalization across
channels, while POS applies "only per-window mean normalization... without
variance equalization." This project's own implementation matches that
exact distinction: `_project_chrom` detrends and mean-normalizes each
channel over the whole buffer before projecting; `_project_pos` only
divides each windowed channel by its own window mean — no variance
equalization at all. That's a real, structural difference in how much
noise/illumination drift survives into the signal each algorithm hands to
the frequency-selection stage — and it's consistent with POS's peaks
being weaker and less reliable on the exact subjects where lighting/motion
is less clean.

## Where this leaves POS

The Era 14 fix is real and stays — it's a genuine, verified improvement
(synthetic MAE 26.71→16.00) with zero effect on CHROM. But POS's
remaining real-video failures are not fixable by adjusting the
frequency-selection heuristics further; they trace back to the projection
step itself producing a weaker signal on some subjects. A further fix
would need to change `_project_pos`'s normalization to add the same kind
of per-window variance equalization CHROM already does — a bigger,
riskier change to the actual signal-projection math, not the
peak-selection logic, and not yet attempted.

**Next: see the top-level [`data/README.md`](../README.md) for the
current state of the project.**
