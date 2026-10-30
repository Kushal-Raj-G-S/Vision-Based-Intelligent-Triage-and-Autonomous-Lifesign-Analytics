# RR lab (Finding 11)

Fast experiment harness used to improve RR peak selection. NOT part of the
product. Scripts have the author's absolute paths and a scratchpad `SP`
argument baked in; adjust before reuse.

Order: `build_cache.py <out.pkl>` (BIDMC+CapnoBase RIV windows) ->
`build_vdb.py` / `build_vdb_age.py <out.pkl>` (VitalDB, needs `dataset/rr/vitaldb/`
downloads + `selected*.json`) -> `lab.py <dir>` (reproduces shipped numbers)
-> `exp*.py` (rule experiments) -> `build_feats.py` / `final_eval.py` (learned
re-ranker, leave-one-group-out). VitalDB PPG gotcha: only the first two rows
carry a timestamp; the rest are uniform 500 Hz — rebuild the time axis from
the row index. See `data/evolution/19-respiratory-rate-ground-truth.md` Finding 11.
