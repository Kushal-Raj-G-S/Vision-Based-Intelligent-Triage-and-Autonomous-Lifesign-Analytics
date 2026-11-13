# Era 25 — plan: getting camera-based breathing rate right (nothing built yet)

2026-11-04. Planning note only. Nothing was downloaded or changed; the user cannot record new videos.

## The gap
Breathing rate (RR) is hand-built signal processing, validated ONLY on contact pulse recordings (BIDMC 0.96, CapnoBase 1.81,
VitalDB 1.03 br/min session MAE). It has never been scored against a reference on camera video. Live test: app showed 7 br/min at
1% confidence, subject counted ~22 (Era 23). So the camera layer for RR is untested and, in practice, wrong.

## What exists locally (gathered, verified on disk)
* 125 videos with a CONTACT PULSE WAVEFORM as ground truth (HR was the only thing used so far):
  * UBFC-rPPG (50): `ground_truth.txt` row 1 = PPG, one sample per video frame (1550 samples / 1550 frames, ~52 s clips).
  * UBFC-Phys (75 clips, 25 people): `bvp_*.csv`, 64 Hz wrist BVP, 180 s, video 35 fps / 180 s.
* A breathing reference can be DERIVED from that waveform with our validated RIV method (silver standard, not true ground truth).
* Cached per-frame face colour traces for all 125 sessions already exist (`dataset/cache/roi`), so camera-side RR can be replayed in
  minutes without re-tracking faces.

## Public video + breathing datasets (status as found, not downloaded)
| Dataset | What | Access | Verdict |
| :--- | :--- | :--- | :--- |
| COHFACE (Idiap) | 40 people, 160 x 1 min webcam, belt | Zenodo record restricted; Idiap "Get Data"; earlier request never approved | best match, gated |
| Magdeburg (Fiedler et al.) | 12 adults, 4 x 3 min, paced 10/15/20 + spontaneous, belt 512 Hz | no download; contact authors | very good match, needs a request |
| CAST-Phys (IEEE DataPort) | ~60 people, 1,080 face videos, respiration | EULA, academic | likely good, gated; check the respiration sensor |
| OBF (Oulu) | ECG, BVP, respiration 32 Hz + face video | access unclear | ask authors |
| MPSC-RR (Hugging Face) | 31 videos, 24 adults, 2.1 h, CC-BY-4.0 | open | videos OPEN, but the belt signals are NOT in the public repo (only videos + metadata); mostly side/back/lying views, 6 front |
| MCD-rPPG (Hugging Face) | 600 people, 3,600 videos, 135 GB, CC-BY-4.0 | open | its "respiratory" is ONE spot value per person (15-24, std 1.7, method undocumented): unusable as ground truth |
| IEEE DataPort radar+RGB+ECG+belt | 31 people, 451 recordings | account | contains RGB-derived rPPG; raw video not confirmed |
| RR-rPPG | built for this problem | not released | not available |

## Plan (each phase has a gate)
**Phase 0 - safe fixes (hours, no new data).** Reject readings stuck at the bottom band edge; let early readings age out of the session
median; live gauge uses recent confident readings only. Re-run synthetic + BIDMC + CapnoBase before keeping (BIDMC ~1 min, CapnoBase ~2 min).
**Phase 1 - silver-standard benchmark on our own 125 videos (CPU only).** Reference RR from the contact waveform; camera RR replayed
from the cached traces; report MAE, within 2/5 br/min, % of time blank, and which channel (RIAV / RIIV) agrees better. Caveats: UBFC-rPPG
clips (~52 s) are shorter than the 60 s window; wrist BVP is noisier; stress/speech tasks; reference error ~1-2 br/min bounds what can be
measured; use a second, different method on the contact signal as an agreement check. GATE: this is the first real camera-layer number.
**Phase 2 - improve against that benchmark.** Background-reference normalisation (needs one pass over the videos for a wall patch, UBFC-rPPG
first), shoulder/face motion channel (check shoulders are in frame), confidence-weighted vote across three channels, then a small learned
peak selector trained on silver labels and tested leave-subject-out. Keep a change only if it helps held-out people.
**Phase 3 - true-reference external validation (user sends the requests).** COHFACE, Magdeburg, CAST-Phys (EULA), OBF; ask the MPSC-RR
authors for the belt signals. Any one of these becomes the honest held-out test.
**Phase 4 - live check without storing faces.** A paced-breathing session that saves only numbers (colour traces, motion tracks,
timestamps, pacer schedule, taps), no images. Gives labelled webcam data without recording video.

## Success criteria (fixed before looking)
Average error <= 2 br/min and >= 70% of sessions within 2, with the blank rate reported; silver-standard results stay labelled "silver".
Until a true-reference test passes, respiration remains claimed on recorded contact data only.

## Risks
Silver labels carry their own error; the same algorithm family on both sides can flatter agreement; short UBFC-rPPG clips; small
numbers of people; gated datasets may never be granted.
