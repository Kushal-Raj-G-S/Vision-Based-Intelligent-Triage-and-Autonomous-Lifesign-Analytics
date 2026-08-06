# Era 11 — Finding 5 Subjects That Were Already There

Era 10 established the real bottleneck honestly: 51 unique subjects, not
model size. The obvious next lever — get more real people — led to a
useful discovery before any new download was needed.

## "56" was never the number of people

The classical/hybrid session-level comparison had always reported
"n=56" — but that is a count of *sessions* (videos), not people.
UBFC-Phys contributes 3 sessions per person (rest/speech/arithmetic), so
56 sessions was actually only **46 distinct people**: 41 from UBFC-rPPG +
1 + only **5** from UBFC-Phys (s4, s7, s11, s16, s33).

Checking `test/ubfc_phys/` directly turned up **10** UBFC-Phys subject
folders already sitting on disk — s1, s2, s3, s5, s6 had been downloaded
at some point but never made it into `test/manifest.csv`, so they were
never part of the classical-vs-hybrid comparison (though they *were*
already part of the 51-subject ML training set from
[Era 9](09-history-freeze-bug.md), since that dataset was built by walking
the raw folder directly, not from the manifest).

## Regenerating the manifest, testing what was missing

`scripts/build_manifest.py` was re-run against `test/` and correctly
picked up all 10 phys subjects — 71 total sessions instead of 56. Rather
than reprocess the already-known 56 sessions (deterministic, no reason to
distrust them, and expensive to redo), only the 15 new sessions (5
subjects × 3 tasks) were run through
`scripts/validate_with_correction.py`, then pooled arithmetically with the
existing 56-session error list for one honest combined number.

**This run got interrupted mid-way twice by unrelated session
disconnects** — once after 9 of 15 videos, once with 5 of 6 remaining
videos already done. Both times, the actual OS process was still alive
and finished on its own even after the tool session lost track of it;
checking `Get-Process`/`Get-CimInstance` for the real running process,
not just trusting a "stopped" status label, avoided redoing already-
completed work twice.

## The real, combined result across all 71 sessions

| | MAE | RMSE |
| :--- | :--- | :--- |
| Classical DSP | 9.43 BPM | 16.51 |
| Hybrid (out-of-fold ML correction) | **6.65 BPM** | **9.66** |

The classical baseline's error actually **rose** once these 15 new
sessions were included (8.42 → 9.43) — they contained genuinely hard
cases (one subject's arithmetic-task session was off by 44 BPM
classically; another's speech session by 41 BPM). The hybrid model barely
moved (6.46 → 6.65) on those same hard sessions, despite never having
been trained on them in their held-out fold.

**The gap widened, not narrowed: a 23% relative MAE reduction (56
sessions) became a 29% relative reduction (71 sessions).** More real data
didn't just re-confirm the earlier result — it strengthened the case for
the hybrid approach, by revealing the classical baseline was shakier than
the smaller sample suggested, while the correction model held up on
exactly the cases that broke it.

**Still open, stated plainly**: 71 real people remains a modest number
for anything claiming to generalize broadly. UBFC-Phys alone has 56
subjects in total; only 10 have been obtained so far. Getting more
remains the single highest-leverage lever available — bigger models
already failed to substitute for it (Era 10).

**Next: see the top-level [`data/README.md`](../README.md) for where the
project stands now.**
