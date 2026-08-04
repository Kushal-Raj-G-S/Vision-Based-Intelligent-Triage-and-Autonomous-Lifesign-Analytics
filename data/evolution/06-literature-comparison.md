# Era 6 — Grounding Against the Literature

Before concluding whether 8.7 BPM MAE was "good" or "bad," the actual
published literature was checked rather than assumed.

Found: ["A Unified Benchmarking Framework for Classical Machine Learning
Based Heart Rate Estimation from RGB and NIR
rPPG"](https://doi.org/10.3390/electronics15010218) (MDPI *Electronics*,
2026) reports, on the same UBFC-rPPG dataset:

| Method | Published MAE |
| :--- | :--- |
| POS | 8.35 BPM |
| CHROM (what this project uses) | 8.20 BPM |
| ChPOS (a newer hybrid of both) | 5.71 BPM |
| **This project (UBFC-rPPG subset only)** | **7.79 BPM** |

This project's UBFC-rPPG-only result is *slightly better* than the
published academic CHROM baseline on the same dataset. Not best-in-class
(deep learning and specialized hybrids do better), but credible, not
embarrassing, and — critically — a legitimate data point a reviewer
checking the literature would accept rather than flag as suspicious.

**Next: [07 — Going Public](07-going-public.md)**
