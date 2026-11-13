#!/usr/bin/env python3
"""
Feasibility check: can the HR videos (whose ground truth is a CONTACT PULSE WAVEFORM) be used to test camera-based breathing?

    ./venv/Scripts/python.exe scripts/rr_silver_check.py [--phys-people 5] [--out dataset/cache/rr_silver.csv]

For every session:
  reference ("silver") RR : our validated breathing estimator (core/rppg.py, as in validate_rr_bidmc.py) run on the CONTACT pulse waveform
                            in the sliding 60 s windows the product would see; session answer = IQR median of readings with conf >= 30.
  camera RR               : the engine's own session answer after replaying the cached face colour trace (no video, no re-tracking).
Then agreement (MAE, within 2 / 5 br/min, how often each side is blank) and a trivial "always predict the median" baseline.

Silver != true ground truth: the contact estimate has its own error (~1-2 br/min on BIDMC/CapnoBase). Local data only; nothing is downloaded.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hr_data as H                                    # noqa: E402
import hr_replay as R                                  # noqa: E402
from validate_rr_bidmc import _production_estimator    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RR_GATE = 30.0


def iqr_median(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return None
    q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
    keep = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    return float(np.median(keep if len(keep) else x))


def contact_ppg(s):
    """(signal at ~30 Hz, fs) from the session's contact ground-truth file."""
    if s["gt_kind"] == "d2":
        rows = [np.array(l.split(), float) for l in open(s["gt_path"]).read().strip().splitlines()]
        ppg, t = rows[0], rows[2]
        fs = 1.0 / float(np.median(np.diff(t)))
        return ppg, fs
    vals = np.array([float(x) for x in open(s["gt_path"], encoding="utf-8", errors="ignore").read().split()])
    return resample_poly(vals, 15, 32), 30.0          # 64 Hz wrist BVP -> 30 Hz


def reference_rr(ppg, fs, win_s=60.0, step_s=3.0):
    est = _production_estimator(fs)
    L = len(ppg) / fs
    kept, total = [], 0
    e = 30.0
    while e <= L + 1e-6:
        buf = ppg[max(0, int((e - win_s) * fs)):int(e * fs)]
        rr, conf = est(buf)
        total += 1
        if rr > 0 and conf >= RR_GATE:
            kept.append(rr)
        e += step_s
    return iqr_median(kept), len(kept), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phys-people", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "dataset" / "cache" / "rr_silver.csv"))
    a = ap.parse_args()
    sess = [s for s in H.discover() if s["gt_kind"] == "d2"]
    phys = [s for s in H.discover() if s["gt_kind"] == "phys"]
    people = sorted({s["subject"] for s in phys}, key=H._num)[: a.phys_people]
    sess += [s for s in phys if s["subject"] in people]
    rows = []
    t0 = time.time()
    for n, s in enumerate(sess, 1):
        cache = ROOT / "dataset" / "cache" / "roi" / f"{s['id']}.npz"
        if not cache.is_file():
            continue
        ppg, fs = contact_ppg(s)
        ref, nk, nt = reference_rr(ppg, fs)
        r = R.replay(R.load(cache), "CHROM")
        cam = r["summary"].get("final_rr")
        rows.append(dict(id=s["id"], dataset=s["dataset"], condition=s["condition"], ref_rr=ref, ref_windows=f"{nk}/{nt}",
                         cam_rr=cam, cam_readings=r["summary"].get("rr_readings")))
        f = lambda v: "  --" if v is None else f"{v:5.1f}"
        print(f"[{n}/{len(sess)}] {s['id']:16s} contact-ref {f(ref)} ({nk}/{nt} windows)  camera {f(cam)}  ({time.time()-t0:.0f}s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def summarize(name, sel):
        both = [x for x in sel if x["ref_rr"] is not None and x["cam_rr"] is not None]
        print(f"\n{name}: {len(sel)} sessions | reference blank {sum(x['ref_rr'] is None for x in sel)} | camera blank {sum(x['cam_rr'] is None for x in sel)} | both present {len(both)}")
        if len(both) >= 3:
            ref = np.array([x["ref_rr"] for x in both]); cam = np.array([x["cam_rr"] for x in both]); e = np.abs(cam - ref)
            base = np.abs(np.median(ref) - ref)
            print(f"  camera vs contact reference: MAE {e.mean():.2f} | median {np.median(e):.2f} | within 2: {100*np.mean(e<=2):.0f}% | within 5: {100*np.mean(e<=5):.0f}% | corr {np.corrcoef(cam, ref)[0,1]:+.2f}")
            print(f"  trivial baseline (always say {np.median(ref):.1f}): MAE {base.mean():.2f} | reference spread: {ref.min():.1f}-{ref.max():.1f}, std {ref.std():.2f}")
    summarize("UBFC-rPPG (rest, ~52 s clips)", [x for x in rows if x["dataset"] != "ubfc_phys"])
    summarize("UBFC-Phys (3 min clips)", [x for x in rows if x["dataset"] == "ubfc_phys"])
    for cond in ("rest", "speech", "arithmetic"):
        summarize(f"  Phys / {cond}", [x for x in rows if x["dataset"] == "ubfc_phys" and x["condition"] == cond])
    print("\nwritten", a.out)


if __name__ == "__main__":
    main()
