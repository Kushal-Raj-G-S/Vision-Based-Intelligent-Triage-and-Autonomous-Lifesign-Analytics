#!/usr/bin/env python3
"""
Does the live path give the same heart rate / respiration at ANY camera frame rate?

    ./venv/Scripts/python.exe scripts/test_live_resampler.py [session ids ...]

Rebuilds a recorded session (cached ROI colour trace, dataset/cache/roi) as a camera stream at many real frame rates with
timestamp jitter, pushes it through core/live_resampler.UniformResampler into an AdvancedRPPG(fps=30) exactly as
web/routes.py::_run_dsp does, and compares with (a) the session's own 30 fps result and (b) the OLD behaviour (frames fed
straight in while the engine assumes 30).  Exit code 1 if any resampled rate is off by more than TOL BPM from the native one.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hr_replay as R                                   # noqa: E402
from core.live_resampler import UniformResampler        # noqa: E402
from core.rppg import AdvancedRPPG                      # noqa: E402

RATES = [6, 8, 10, 15, 20, 24, 30, 45, 60, 120]
TOL = 8.0
rng = np.random.default_rng(1)


def camera_stream(c, fps):
    """(timestamps, rgb, motion) a camera at `fps` would deliver, sampling the session's continuous colour trace."""
    idx = np.where(c["valid"])[0]
    t0 = idx / c["meta"]["fps"]
    n = int((t0[-1] - t0[0]) * fps)
    t = t0[0] + (np.arange(n) + rng.normal(0, 0.1, n)) / fps                 # +-10% frame-time jitter
    t = np.clip(np.maximum.accumulate(t), t0[0], t0[-1])
    rgb = np.stack([np.interp(t, t0, c["rgb"][idx, k]) for k in range(3)], 1)
    mot = np.interp(t, t0, c["motion"][idx])
    return t, rgb, mot


def run_live(t, rgb, mot, resample=True):
    eng = AdvancedRPPG(fps=30, window_size=300)
    rs = UniformResampler(30.0) if resample else None
    for ti, ci, mi in zip(t, rgb, mot):
        if rs is None:
            eng.add_frame(tuple(ci), motion_delta=float(mi)); eng.process_ppg_signal()
        else:
            for s, m in rs.push(float(ti), ci, float(mi)):
                eng.add_frame(s, motion_delta=m); eng.process_ppg_signal()
    s = eng.get_final_summary()
    return s.get("final_bpm"), s.get("final_rr")


def main():
    ids = sys.argv[1:] or ["d2_subject5", "d2_subject8", "d2_subject1"]
    bad = 0
    for sid in ids:
        c = R.load(os.path.join("dataset", "cache", "roi", f"{sid}.npz"))
        native = R.replay(c, "CHROM")
        nb, nr = native["final_bpm"], native["summary"].get("final_rr")
        print(f"\n{sid}: truth {c['meta']['gt_session']:.1f} BPM | native 30 fps: HR {nb:.1f}, RR {nr if nr is None else round(nr, 1)}")
        print(f"  {'camera fps':>10s} | {'NEW resample: HR / RR':>22s} | {'OLD (assume 30): HR / RR':>26s}")
        for fps in RATES:
            t, rgb, mot = camera_stream(c, fps)
            hn, rn = run_live(t, rgb, mot, True)
            ho, ro = run_live(t, rgb, mot, False)
            f = lambda v: "None" if v is None else f"{v:.1f}"
            ok = hn is not None and abs(hn - nb) <= TOL
            bad += 0 if ok else 1
            print(f"  {fps:10d} | {f(hn):>8s} / {f(rn):>8s} {'ok ' if ok else 'BAD'} | {f(ho):>11s} / {f(ro):>8s}", flush=True)
    print("\nPASS" if not bad else f"\nFAIL: {bad} rate(s) off by more than {TOL} BPM")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
