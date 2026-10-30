#!/usr/bin/env python3
"""
Replay the heart-rate DSP over cached ROI traces (scripts/build_roi_cache.py) — no video, no face tracking.

`replay()` is the production loop of scripts/analyze_video.py with the camera replaced by the cache: every
frame with a face is fed to AdvancedRPPG, the pipeline runs every frame, the session answer is
`get_final_summary()['final_bpm']`. Fidelity check against the old 71-session log:

    python scripts/hr_replay.py --check-old --cache dataset/cache/roi
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.rppg import AdvancedRPPG        # noqa: E402

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "dataset" / "cache" / "roi"


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def replay(c, algorithm="CHROM", rgb_key="rgb", capture_spectra=False, stride=1, keep_signal=False):
    """Returns dict(final_bpm, readings=[{frame, bpm, sqi, ...}], spectra) for one cached session."""
    fps = c["meta"]["fps"]
    eng = AdvancedRPPG(fps=int(round(fps)), window_size=max(300, int(fps * 10)), algorithm=algorithm)
    eng.capture_spectra = capture_spectra
    rgb, valid, motion = c[rgb_key], c["valid"], c["motion"]
    readings, spectra = [], []
    for i in range(len(valid)):
        if not valid[i]:
            continue
        r, g, b = (float(x) for x in rgb[i])
        eng.add_frame((r, g, b), motion_delta=float(motion[i]))
        if i % stride:
            continue
        res = eng.process_ppg_signal()
        if res["ready"] and res["bpm"] > 0:
            row = {"frame": i, "bpm": res["bpm"], "sqi": res.get("sqi", 0.0), "snr_db": res.get("snr_db", 0.0),
                   "stability": res.get("stability_score", 0.0), "harmonic_ratio": res.get("harmonic_ratio", 0.0),
                   "filt_low": res.get("filter_band_hz", [0, 0])[0], "filt_high": res.get("filter_band_hz", [0, 0])[1],
                   "motion": res.get("motion_delta", 0.0), "rr": res.get("rr", 0.0), "rr_conf": res.get("rr_confidence", 0.0),
                   "hrv": res.get("hrv", 0.0), "stress": res.get("stress_index", 0.0)}
            if keep_signal:
                row["ppg"] = res["ppg_signal"]
            readings.append(row)
            if capture_spectra and "spectrum_wide" in res:
                spectra.append(res["spectrum_wide"])
    summary = eng.get_final_summary()
    final = summary.get("final_bpm")
    if final is None and readings:
        final = float(np.mean([r["bpm"] for r in readings[-10:]]))
    return {"final_bpm": final, "readings": readings, "spectra": np.array(spectra, np.float32) if spectra else None,
            "summary": summary}


def old_log_classical(log_path):
    """{session_id: (gt, classical_estimate)} from the original 71-session validation log."""
    out = {}
    for ln in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^(test[^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", ln)
        if not m:
            continue
        v, gt, est = m.group(1).strip(), float(m.group(2)), float(m.group(3))
        p = re.split(r"[\\/]", v)
        if p[1] == "dataset_2":
            sid = f"d2_{p[2]}"
        elif p[1] == "dataset_1":
            sid = "d1_after-exercise"
        else:
            continue          # Phys rows are cut off in the log's fixed-width column; matched separately
        out[sid] = (gt, est)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--check-old", action="store_true")
    ap.add_argument("--log", default="test/validation_run11_huber_full71.log")
    args = ap.parse_args()
    if args.check_old:
        old = old_log_classical(args.log)
        rows = []
        for f in sorted(Path(args.cache).glob("*.npz")):
            sid = f.stem
            if sid not in old:
                continue
            c = load(f)
            r = replay(c)
            gt_old, est_old = old[sid]
            rows.append((sid, gt_old, c["meta"]["gt_session"], est_old, r["final_bpm"]))
            print(f"{sid:20s} gt old {gt_old:6.1f} / now {c['meta']['gt_session']:6.1f} | classical old {est_old:6.1f}  replay {r['final_bpm']:6.1f}", flush=True)
        if rows:
            d = np.array([abs(a - b) for _, _, _, a, b in rows])
            print(f"\n{len(rows)} sessions: |replay - old| mean {d.mean():.2f} BPM, max {d.max():.2f}, identical to 0.1: {(d < 0.1).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
