# Era 3 — The Data Age

The student obtained the full UBFC-rPPG dataset from its public Google Drive
link (no gated request needed — confirmed by checking directly). Initial
download was partial: 18 of 58 possible sessions had both a video and a
matching ground-truth file; the rest had one or the other (interrupted
download).

## Tooling built

`scripts/build_manifest.py` — walks a downloaded dataset folder, auto-detects
which of several ground-truth formats is present (`gtdump.xmp`
comma-separated HR/SpO2/PPG rows for "Dataset 1", or the 3-line
`ground_truth.txt` BVP/HR/timestamp format for "Dataset 2"), and emits a
`manifest.csv` consumable by the validation harness.

## First real validation ever run

See `test/validation_run.log`, `test/validation_run2.log`: 18 real subjects.
**MAE 19.1 BPM, RMSE 26.0 BPM.**

The key discovery from this first real run: **accuracy was bimodal, not
uniformly mediocre.** 9/18 subjects were excellent (<9 BPM error); 7/18
(~39%) showed errors of 20-60+ BPM, in two distinct patterns:
- ~4 subjects landing at almost exactly **half** the true rate (ratio
  0.44-0.59×)
- ~3 subjects landing at roughly **1.26-1.34×** the true rate

## Root-cause trace

`test/trace_subject41.log`, on the clearest half-rate case: the pipeline
genuinely found the correct ~100bpm rate for a ~7-second stretch
mid-recording, then slid back into a false ~45-50bpm lock for the rest of
the clip. Because `get_final_summary()` takes a history median, whichever
state occupied more total frames won — actively discarding the correct
stretch as a statistical minority. **This became the case study for all of
[Era 4](04-harmonic-lock-on-fixes.md).**

## Dataset cleanup and expansion

Later in this era: the student cleaned up the 27 incomplete UBFC-rPPG
sessions (freed ~3.6GB), then downloaded the missing videos for the
remaining subjects, bringing `dataset_2` to a clean 40/40 complete subjects
(41 total with `dataset_1/after-exercise`). Two more datasets were added:

- **UBFC-Phys** (5 subjects × 3 tasks — rest/speech/arithmetic — via a free
  IEEE DataPort account, no gated request).
- **rPPG-10** (Mendeley, direct download, 26 subjects with real ECG ground
  truth, but its videos are pre-cropped 64×64 skin patches with no visible
  face — structurally incompatible with MediaPipe face detection, handled
  via a separate DSP-only validation path, `scripts/validate_cropped_roi.py`,
  that bypasses face detection entirely).

**Next: [04 — The Age of Experiments (fixing harmonic lock-on)](04-harmonic-lock-on-fixes.md)**
