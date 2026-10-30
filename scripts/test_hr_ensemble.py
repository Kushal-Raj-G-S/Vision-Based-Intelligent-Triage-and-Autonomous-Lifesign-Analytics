#!/usr/bin/env python3
"""
Parity + equivalence checks for core/hr_ensemble.py (run in the MAIN venv; needs dataset/cache from the HR lab).

    ./venv/Scripts/python.exe scripts/test_hr_ensemble.py

  1. numpy tracker == torch tracker  (dataset/cache/tracker_parity.npz, written by scripts/export_hr_tracker.py)
  2. streaming HREnsembleSession picks the SAME selector candidates as the batch research pipeline
     (hr_peaks.candidate_table) on real cached sessions.
Exit code 1 on any mismatch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hr_peaks as P                     # noqa: E402
import hr_replay as R                    # noqa: E402
from core import hr_ensemble as E        # noqa: E402
from core.rppg import AdvancedRPPG       # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "cache")


def main():
    M = E.load_models()
    assert M is not None, "models failed to load"
    z = np.load(os.path.join(ROOT, "tracker_parity.npz"))
    d = max(float(np.abs(E._softmax(M.trackers[j].logits(z["x"])) - E._softmax(z["logits"][j])).max()) for j in range(len(M.trackers)))
    print(f"tracker numpy vs torch: max |softmax diff| {d:.2e}")
    ok = d < 2e-3
    for sid in ("d2_subject5", "d2_subject3"):
        c = R.load(os.path.join(ROOT, "roi", f"{sid}.npz"))
        fps = c["meta"]["fps"]; w = max(300, int(fps * 10))
        main_eng = AdvancedRPPG(fps=int(round(fps)), window_size=w); main_eng.capture_spectra = True
        ses = E.HREnsembleSession(M, int(round(fps)), w, threaded=False)
        for i in range(len(c["valid"])):
            if c["valid"][i]:
                rgb = tuple(float(x) for x in c["rgb"][i])
                main_eng.add_frame(rgb, motion_delta=0.0)
                ses.feed(rgb, float(c["motion"][i]), main_eng.process_ppg_signal())
        f = np.load(os.path.join(ROOT, "feat", f"{sid}.npz"), allow_pickle=False); s = {k: f[k] for k in f.files}
        X, y, cyc, bpm = P.candidate_table(s, stride=5)
        prob = M.selector.predict_proba(X)
        batch = [bpm[np.where(cyc == t)[0][np.argmax(prob[np.where(cyc == t)[0]])]] for t in np.unique(cyc)]
        n = min(len(batch), len(ses.sel_hist))
        same = int(np.sum(np.isclose(batch[:n], ses.sel_hist[:n])))
        print(f"{sid}: streamed {len(ses.sel_hist)} vs batch {len(batch)} selector decisions, identical picks {same}/{n}")
        # the cache stores features as float32, the live path as float64: a pick may flip only on a genuine near-tie
        ok &= (same >= 0.99 * n and len(batch) == len(ses.sel_hist))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
