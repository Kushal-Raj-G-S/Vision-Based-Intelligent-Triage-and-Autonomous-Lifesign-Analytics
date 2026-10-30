#!/usr/bin/env python3
"""
Train the final peak-selector on ALL 125 sessions and export it as plain numpy arrays (models/hr_selector.npz).

    ./venv/Scripts/python.exe scripts/export_hr_selector.py

Same recipe the out-of-fold evaluation used (hr_lab.run_peaks): 29 candidate features, HistGradientBoosting(max_depth=4,
lr=0.08, 200 iters, l2=1.0).  The numpy copy (core/hr_ensemble.TreeEnsemble) is checked against sklearn's own
predict_proba before the file is written; a mismatch aborts the export.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hr_lab                      # noqa: E402
import hr_peaks as P               # noqa: E402
from core.hr_ensemble import TreeEnsemble   # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "models" / "hr_selector.npz"


def export_trees(clf):
    feat, thr, left, right, val, leaf, miss, roots, off = [], [], [], [], [], [], [], [], 0
    for per_iter in clf._predictors:
        nodes = per_iter[0].nodes
        roots.append(off)
        feat.append(nodes["feature_idx"].astype(np.int32)); thr.append(nodes["num_threshold"].astype(np.float64))
        left.append(nodes["left"].astype(np.int64) + off); right.append(nodes["right"].astype(np.int64) + off)
        val.append(nodes["value"].astype(np.float64)); leaf.append(nodes["is_leaf"].astype(bool))
        miss.append(nodes["missing_go_to_left"].astype(bool))
        off += len(nodes)
    return dict(feature_idx=np.concatenate(feat), threshold=np.concatenate(thr), left=np.concatenate(left),
                right=np.concatenate(right), value=np.concatenate(val), is_leaf=np.concatenate(leaf),
                missing_left=np.concatenate(miss), roots=np.array(roots, np.int64),
                baseline=np.float64(np.ravel(clf._baseline_prediction)[0]))


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    S = hr_lab.load_all()
    T = hr_lab.build_tables(S)
    X = np.concatenate([T[s["id"]][0] for s in S]); y = np.concatenate([T[s["id"]][1] for s in S])
    keep = np.isfinite(y)
    print(f"{len(S)} sessions, {keep.sum()} labelled candidate rows ({int(y[keep].sum())} positive)", flush=True)
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, l2_regularization=1.0, random_state=0)
    clf.fit(X[keep], y[keep])
    arrays = export_trees(clf)
    te = TreeEnsemble(arrays)
    ref = clf.predict_proba(X)[:, 1]
    got = te.predict_proba(X)
    err = float(np.abs(ref - got).max())
    print(f"numpy trees vs sklearn predict_proba on {len(X)} rows: max |diff| = {err:.2e}", flush=True)
    if err > 1e-6:
        raise SystemExit("MISMATCH — refusing to write the model file")
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print("written", OUT, f"({OUT.stat().st_size / 1e3:.0f} kB, {len(arrays['roots'])} trees, {len(arrays['value'])} nodes)")


if __name__ == "__main__":
    main()
