# Era 10 — Testing Whether Bigger Helps (it doesn't, and that's real evidence)

After the hybrid correction layer's first real result (56-session classical
8.42 → hybrid 6.46 BPM MAE, [Era 9](09-history-freeze-bug.md) territory —
see the top-level README's Validation section for the full write-up of
that layer), the obvious next question was: would a bigger model do even
better? More parameters, more layers, an ensemble of models — "make it
techier."

## What was actually tested, not just assumed

Four model configurations were trained and compared with the same
GroupKFold-by-subject discipline as every other number in this project —
never evaluating a model on a person it trained on:

| Model | Held-out per-cycle MAE |
| :--- | :--- |
| Small gradient-boosted model (200 trees, depth 3) — the original | **8.80** |
| Bigger gradient-boosted model (600 trees, depth 5, stochastic subsampling) | 8.77 |
| Small neural net (64→32→16) | 9.18 |
| Bigger, more-regularized neural net (128→64→32, 20x the L2 penalty) | 9.38 (**worse**) |
| Ensemble of the two bigger models | 8.87 (worse than the small GBR alone) |

Bigger did not win. In one case (the neural net) bigger was measurably
*worse*, not just flat.

## Why, and how this was reasoned through rather than shrugged off

The dataset has 242,026 rows — but only **51 unique people**. Rows from
the same person are highly correlated (same face, same lighting, same
skin tone), so the real amount of independent information a model can
learn from is much closer to 51 than 242,026. A small model already
extracts most of what 51 people's data can teach; giving it more
parameters mostly gives it more room to fit quirks specific to those 51
faces, which is exactly the shape of the neural net's result — more
capacity, worse held-out accuracy.

This is not a failed experiment to hide — it is itself a legitimate,
citable finding: **model-capacity scaling saturates quickly under a small
real-subject count, and this was verified rather than assumed.** A paper
that only reports "we used a gradient-boosted model" is asserting a
choice; this project can show the comparison that justifies it.

## An honest first attempt to catch a mid-size computational trap

The first version of the bigger neural net (256→128→64→32, max_iter=3000)
was timed on small subsamples before committing to a full run — a
discipline adopted this session after two earlier scares this same day
where a script silently ran for hours with no visible progress. Measured
(not guessed) timing showed that config would scale to roughly 100+
minutes *per fold*, ~8-9 hours total, on this dataset and hardware
(sklearn's single-threaded, no-GPU implementation). It was scaled down to
a still-meaningfully-bigger (128→64→32) config before running the real
comparison, and MLP training rows were capped at 40,000/fold (versus
GBR's full ~194,000/fold, which measured near-linear scaling and needed no
cap) — both decisions made from timed evidence, not intuition.

**Practical result: the original small gradient-boosted model
(`models/bpm_correction_gbr.joblib`, `models/cv_folds/`) remains the
deployed model.** It is simpler, faster, and not measurably beaten by
anything bigger tried here.

**Next: [11 — Finding 5 Subjects That Were Already There](11-more-subjects.md)**
