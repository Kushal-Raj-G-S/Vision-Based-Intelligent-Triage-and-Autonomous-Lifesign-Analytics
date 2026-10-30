#!/usr/bin/env python3
"""
Haar (what the app has actually been running) vs MediaPipe landmarks (what it was designed to run), same sessions,
same evaluation. Reads dataset/cache/results_sessions.csv and results_sessions_mp.csv plus both ROI caches;
writes dataset/cache/TRACKER_COMPARISON.md.
"""
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "dataset" / "cache"


def load(name):
    out = {}
    with open(CACHE / name, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["session"]] = r
    return out


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def face_cover(folder):
    cov = {}
    for f in glob.glob(str(CACHE / folder / "*.npz")):
        if ".tmp" in f:
            continue
        z = np.load(f, allow_pickle=False)
        cov[json.loads(str(z["meta"]))["id"]] = float(z["valid"].mean())
    return cov


def main():
    haar, mp = load("results_sessions.csv"), load("results_sessions_mp.csv")
    common = sorted(set(haar) & set(mp))
    ch, cm = face_cover("roi"), face_cover("roi_mp")
    md = [f"# Face tracker comparison — Haar (as deployed) vs MediaPipe landmarks — {len(common)} sessions", "",
          "Cleaned labels, out-of-fold, same sessions and evaluation for both.", ""]
    hc = np.array([ch.get(s, np.nan) for s in common]); mc = np.array([cm.get(s, np.nan) for s in common])
    md += [f"Face found in frames: Haar mean {np.nanmean(hc)*100:.1f}% (worst {np.nanmin(hc)*100:.1f}%), "
           f"MediaPipe mean {np.nanmean(mc)*100:.1f}% (worst {np.nanmin(mc)*100:.1f}%).", ""]
    groups = [("all", lambda r: True), ("UBFC-rPPG", lambda r: r["dataset"] != "ubfc_phys"),
              ("UBFC-Phys", lambda r: r["dataset"] == "ubfc_phys"),
              ("Phys rest", lambda r: r["dataset"] == "ubfc_phys" and r["condition"] == "rest"),
              ("Phys speech", lambda r: r["dataset"] == "ubfc_phys" and r["condition"] == "speech"),
              ("Phys arithmetic", lambda r: r["dataset"] == "ubfc_phys" and r["condition"] == "arithmetic")]
    for col, name in (("classical", "Classical (CHROM)"), ("selector_argmax", "Peak selector")):
        md += [f"## {name}", "", "| Group | n | MAE Haar | MAE MediaPipe | >20 BPM Haar | >20 BPM MediaPipe |", "| :--- | ---: | ---: | ---: | ---: | ---: |"]
        for g, fn in groups:
            eh, em = [], []
            for s in common:
                rh, rm = haar[s], mp[s]
                t = num(rh["truth_cleaned"])
                if not fn(rh) or not np.isfinite(t):
                    continue
                a, b = num(rh[col]), num(rm[col])
                if np.isfinite(a) and np.isfinite(b):
                    eh.append(abs(a - t)); em.append(abs(b - t))
            if eh:
                eh, em = np.array(eh), np.array(em)
                md.append(f"| {g} | {len(eh)} | {eh.mean():.2f} | **{em.mean():.2f}** | {(eh > 20).sum()} | {(em > 20).sum()} |")
        md.append("")
    md += ["## Sessions where Haar lost the face most", "", "| Session | face found Haar | face found MediaPipe | classical err Haar | classical err MediaPipe |", "| :--- | ---: | ---: | ---: | ---: |"]
    for s in sorted(common, key=lambda s: ch.get(s, 1))[:12]:
        t = num(haar[s]["truth_cleaned"])
        md.append(f"| {s} | {ch.get(s, np.nan)*100:.0f}% | {cm.get(s, np.nan)*100:.0f}% | "
                  f"{abs(num(haar[s]['classical']) - t):.1f} | {abs(num(mp[s]['classical']) - t):.1f} |")
    (CACHE / "TRACKER_COMPARISON.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("written", CACHE / "TRACKER_COMPARISON.md")


if __name__ == "__main__":
    main()
