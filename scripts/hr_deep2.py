#!/usr/bin/env python3
"""
Deep model v2: a neural net over lock-independent SPECTRA (not raw colour traces).

    tools/gpu_lab/venv/Scripts/python.exe scripts/hr_deep2.py [--feat dataset/cache/feat] [--roi dataset/cache/roi] [--tag deep2]

Input per cycle: CHROM and POS wide-band spectra (116 bins, 0.70-3.00 Hz) averaged over the last 3 s and the last 10 s
        (so the net sees which peak PERSISTS), plus SQI and motion.  ->  MLP  ->  posterior over the 116 frequency bins.
Training: Gaussian-smoothed target at the CLEANED reference HR (readings outside 40-180 BPM dropped); frequency-axis
        stretch augmentation (spectra resampled by a random factor, label scaled by the same factor).
Protocol: out-of-fold GROUPED BY SUBJECT (8 folds) plus both cross-dataset directions; session answer = IQR median of
        per-cycle argmax, same as production and the peak selector.  Same labels and scoring as hr_deep.py.
"""
import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hr_deep import CACHE, iqr_median, load_truth, show          # noqa: E402

GRID = np.arange(0.70, 3.0001, 0.02)
GBPM = GRID * 60.0
NB = len(GRID)
STEP = 5                                   # one training/eval point every 5 cycles (~0.17 s)


def cum_avg(S, w):
    c = np.cumsum(np.vstack([np.zeros((1, S.shape[1])), S]), axis=0)
    n = len(S)
    idx = np.arange(1, n + 1)
    lo = np.maximum(idx - w, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


def load_session(feat_path, roi_dir):
    z = np.load(feat_path, allow_pickle=False)
    meta = ast.literal_eval(str(z["meta"]))
    sid = Path(feat_path).stem
    r = np.load(Path(roi_dir) / f"{sid}.npz", allow_pickle=False)
    gt_t, gt_hr = r["gt_t"], np.where((r["gt_hr"] >= 40) & (r["gt_hr"] <= 180), r["gt_hr"], np.nan)
    gv = np.isfinite(gt_hr)
    t = z["t"].astype(float)
    gt = np.interp(t, gt_t[gv], gt_hr[gv]) if gv.sum() > 1 else np.full(len(t), np.nan)
    near = np.array([np.min(np.abs(gt_t[gv] - q)) if gv.any() else 99 for q in t])
    gt = np.where(near <= 3.0, gt, np.nan)
    c, p = z["chrom_spec"].astype(np.float32), z["pos_spec"].astype(np.float32)
    # four spectra stacked: (cycles, 4, 116)
    S = np.stack([cum_avg(c, 90), cum_avg(p, 90), cum_avg(c, 300), cum_avg(p, 300)], 1).astype(np.float32)
    extra = np.stack([z["sqi"] / 100.0, np.clip(z["motion"] / 20.0, 0, 3)], 1).astype(np.float32)
    return dict(id=sid, subject=meta["subject"], dataset=meta["dataset"], S=S, extra=extra, gt=gt)


def target_bins(hz):
    d = (GRID[None, :] - hz[:, None]) / 0.02
    p = np.exp(-0.5 * (d / 1.2) ** 2)
    s = p.sum(1, keepdims=True)
    return (p / np.maximum(s, 1e-9)).astype(np.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(4 * NB + 2, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.GELU(),
                               nn.Dropout(0.3), nn.Linear(256, NB))

    def forward(self, x):
        return self.f(x)


def prep(S, extra):
    S = S / (S.max(2, keepdims=True) + 1e-9)                         # per-spectrum max normalisation
    return np.concatenate([S.reshape(len(S), -1), extra], 1).astype(np.float32)


def stretch(S, alpha):
    """resample every spectrum on the frequency axis: s'(f) = s(f / alpha)."""
    out = np.empty_like(S)
    q = GRID / alpha
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            out[i, j] = np.interp(q, GRID, S[i, j], left=0.0, right=0.0)
    return out


def train(sessions, epochs, dev, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pts = [(si, k) for si, s in enumerate(sessions) for k in range(150, len(s["gt"]), STEP) if np.isfinite(s["gt"][k])]
    net = Net().to(dev)
    steps = max(1, len(pts) // 256)
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=epochs * steps)
    net.train()
    for _ in range(epochs):
        for _ in range(steps):
            sel = rng.integers(0, len(pts), 256)
            S = np.stack([sessions[pts[i][0]]["S"][pts[i][1]] for i in sel])
            E = np.stack([sessions[pts[i][0]]["extra"][pts[i][1]] for i in sel])
            hz = np.array([sessions[pts[i][0]]["gt"][pts[i][1]] / 60.0 for i in sel])
            a = np.where(rng.random(256) < 0.8, np.exp(rng.uniform(np.log(0.85), np.log(1.2), 256)), 1.0)
            S2 = np.empty_like(S)
            for i in range(256):
                S2[i] = stretch(S[i:i + 1], a[i])[0] if a[i] != 1.0 else S[i]
            hz2 = hz * a
            ok = (hz2 >= GRID[0]) & (hz2 <= GRID[-1])
            if not ok.any():
                continue
            x = torch.from_numpy(prep(S2[ok], E[ok])).to(dev)
            y = torch.from_numpy(target_bins(hz2[ok])).to(dev)
            loss = -(y * F.log_softmax(net(x), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


@torch.no_grad()
def predict(net, s, dev):
    ks = np.arange(150, len(s["S"]), STEP)
    if len(ks) == 0:
        return np.nan
    x = torch.from_numpy(prep(s["S"][ks], s["extra"][ks])).to(dev)
    p = F.softmax(net(x), 1).cpu().numpy()
    return iqr_median(GBPM[p.argmax(1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(CACHE / "feat"))
    ap.add_argument("--roi", default=str(CACHE / "roi"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--tag", default="deep2")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    truth = load_truth()
    sessions = []
    for p in sorted(Path(a.feat).glob("*.npz")):
        if ".tmp" in p.name or p.stem not in truth:
            continue
        try:
            sessions.append(load_session(p, a.roi))
        except Exception as e:
            print("skip", p.stem, repr(e)[:80])
    print(len(sessions), "sessions,", len({s["subject"] for s in sessions}), "subjects", flush=True)
    ids = [s["id"] for s in sessions]
    subj = sorted({s["subject"] for s in sessions})
    np.random.default_rng(0).shuffle(subj)
    fold_of = {u: i % a.folds for i, u in enumerate(subj)}
    est, t0 = {}, time.time()
    for k in range(a.folds):
        net = train([s for s in sessions if fold_of[s["subject"]] != k], a.epochs, dev, seed=k)
        for s in sessions:
            if fold_of[s["subject"]] == k:
                est[s["id"]] = predict(net, s, dev)
        print(f"fold {k + 1}/{a.folds} ({time.time() - t0:.0f}s)", flush=True)
    sel = {i: truth[i]["sel"] for i in ids}
    cls = {i: truth[i]["cls"] for i in ids}
    show("DEEP v2 (spectra network), out-of-fold grouped by subject", est, truth, ids)
    show("PEAK SELECTOR (reference, same sessions)", sel, truth, ids)
    show("CLASSICAL (reference)", cls, truth, ids)
    isp = lambda s: "UBFC-Phys" if s["dataset"] == "ubfc_phys" else "UBFC-rPPG"
    cross = {}
    for trn, ten in (("UBFC-rPPG", "UBFC-Phys"), ("UBFC-Phys", "UBFC-rPPG")):
        net = train([s for s in sessions if isp(s) == trn], a.epochs, dev, seed=100)
        te = [s for s in sessions if isp(s) == ten]
        e = {s["id"]: predict(net, s, dev) for s in te}
        cross.update(e)
        tid = [s["id"] for s in te]
        show(f"CROSS-DATASET deep v2: trained ONLY on {trn}, tested on {ten}", e, truth, tid)
        show(f"   (peak selector on the same {ten} sessions)", sel, truth, tid)
    out = CACHE / f"results_{a.tag}_sessions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["session", "deep2_oof", "deep2_cross_dataset"])
        for i in ids:
            w.writerow([i, "" if not np.isfinite(est.get(i, np.nan)) else round(est[i], 2),
                        "" if not np.isfinite(cross.get(i, np.nan)) else round(cross[i], 2)])
    print("\nwritten", out)


if __name__ == "__main__":
    main()
