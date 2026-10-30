#!/usr/bin/env python3
"""
Deep heart-rate model on the cached ROI colour traces (numpy + PyTorch only; uses the GPU when one is visible).

    tools/gpu_lab/venv/Scripts/python.exe scripts/hr_deep.py [--epochs 30] [--roi dataset/cache/roi] [--tag deep]

Input : 10 s windows (300 samples at 30 Hz) of the per-frame pooled colour: normalised R, G, B, the two CHROM
        chrominance signals and their combination (6 channels).
Output: a probability over 141 heart-rate bins (40..180 BPM), trained with a Gaussian-smoothed target against the
        CLEANED reference (hr_data.reference_info rule: readings outside 40-180 BPM are dropped).
Key trick: time-stretch augmentation. A window is resampled by a random factor and the label is scaled by the same
        factor, so the net must learn "heart rate = frequency" rather than memorise people. (A standard rPPG trick.)
Session answer = IQR median of per-window argmax, exactly like production and the peak selector.
Protocol: out-of-fold GROUPED BY SUBJECT (8 folds) plus both cross-dataset directions; nothing is ever scored by a model
        that trained on the same person. Labels for scoring come from results_sessions.csv (original / cleaned).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "dataset" / "cache"
FS, WIN, STRIDE_S = 30.0, 300, 1.0
BINS = np.arange(40, 181)                    # 141 classes, 1 BPM apart
SIGMA = 1.5


# ----------------------------------------------------------------------------------------------- data
def load_session(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    fps, valid, rgb = float(meta["fps"]), z["valid"], z["rgb"].astype(np.float64)
    n = len(valid)
    t = np.arange(n) / fps
    ok = valid & np.isfinite(rgb[:, 0])
    if ok.sum() < 300:
        return None
    tg = np.arange(t[ok][0], t[ok][-1], 1.0 / FS)
    grid = np.stack([np.interp(tg, t[ok], rgb[ok, c]) for c in range(3)], 1)          # uniform 30 Hz, gaps interpolated
    cover = np.interp(tg, t, ok.astype(float))                                       # local share of real frames
    gt_t, gt_hr = z["gt_t"], np.where((z["gt_hr"] >= 40) & (z["gt_hr"] <= 180), z["gt_hr"], np.nan)
    gv = np.isfinite(gt_hr)
    gt = np.interp(tg, gt_t[gv], gt_hr[gv]) if gv.sum() > 1 else np.full(len(tg), np.nan)
    near = np.array([np.min(np.abs(gt_t[gv] - q)) if gv.any() else 99 for q in tg[::30]])
    near = np.repeat(near, 30)[: len(tg)]
    if len(near) < len(tg):
        near = np.pad(near, (0, len(tg) - len(near)), mode="edge")
    gt = np.where(near <= 3.0, gt, np.nan)
    return dict(id=meta["id"], subject=meta["subject"], dataset=meta["dataset"], rgb=grid, cover=cover, gt=gt)


def features(seg):
    """seg: (L, 3) raw rgb -> (6, L) channels, each standardised over the window."""
    m = seg.mean(0, keepdims=True) + 1e-6
    n = seg / m - 1.0
    r, g, b = n[:, 0], n[:, 1], n[:, 2]
    xs, ys = 3 * r - 2 * g, 1.5 * r + g - 1.5 * b
    s = xs - (xs.std() / (ys.std() + 1e-9)) * ys
    ch = np.stack([r, g, b, xs, ys, s], 0)
    return (ch - ch.mean(1, keepdims=True)) / (ch.std(1, keepdims=True) + 1e-6)


def window_starts(sess):
    n = len(sess["rgb"])
    return np.arange(0, n - WIN, int(STRIDE_S * FS))


def make_eval(sess):
    st = window_starts(sess)
    keep = [i for i, s in enumerate(st) if sess["cover"][s:s + WIN].mean() >= 0.5]
    X = np.stack([features(sess["rgb"][st[i]:st[i] + WIN]) for i in keep]).astype(np.float32) if keep else np.zeros((0, 6, WIN), np.float32)
    return X


def target(hr):
    d = BINS[None, :] - hr[:, None]
    p = np.exp(-0.5 * (d / SIGMA) ** 2)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


class Sampler:
    """Draws augmented training windows from a list of sessions."""

    def __init__(self, sessions, rng):
        self.rng = rng
        self.items = []
        for si, s in enumerate(sessions):
            for st in window_starts(s):
                c = st + WIN // 2
                if s["cover"][st:st + WIN].mean() >= 0.5 and np.isfinite(s["gt"][c]):
                    self.items.append((si, st))
        self.sessions = sessions

    def batch(self, n, stretch=True):
        X, y = [], []
        for k in self.rng.integers(0, len(self.items), n):
            si, st = self.items[k]
            s = self.sessions[si]
            c = st + WIN // 2
            hr = float(s["gt"][c])
            L = WIN
            if stretch and self.rng.random() < 0.85:
                f = float(np.exp(self.rng.uniform(np.log(0.8), np.log(1.25))))
                L = int(round(WIN * f))
                hr2 = hr * WIN / L
                lo = c - L // 2
                if 40.0 <= hr2 <= 180.0 and lo >= 0 and lo + L <= len(s["rgb"]):
                    seg = s["rgb"][lo:lo + L]
                    xi = np.linspace(0, L - 1, WIN)
                    seg = np.stack([np.interp(xi, np.arange(L), seg[:, j]) for j in range(3)], 1)
                    hr = hr2
                else:
                    seg = s["rgb"][st:st + WIN]
            else:
                seg = s["rgb"][st:st + WIN]
            X.append(features(seg)); y.append(hr)
        return np.stack(X).astype(np.float32), target(np.array(y))


# ----------------------------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, c_in=6, n_out=len(BINS)):
        super().__init__()
        ch = [32, 64, 96, 128, 128]
        layers, c = [], c_in
        for i, o in enumerate(ch):
            layers += [nn.Conv1d(c, o, 7 if i == 0 else 5, stride=1 if i == 0 else 2, padding=3 if i == 0 else 2),
                       nn.BatchNorm1d(o), nn.GELU()]
            c = o
        self.f = nn.Sequential(*layers)
        t = WIN
        for i in range(1, len(ch)):
            t = (t + 1) // 2
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c * t, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n_out))

    def forward(self, x):
        return self.head(self.f(x).flatten(1))


def train(sessions, epochs, dev, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    samp = Sampler(sessions, rng)
    net = Net().to(dev)
    steps = max(1, len(samp.items) // 128)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps)
    net.train()
    for _ in range(epochs):
        for _ in range(steps):
            x, y = samp.batch(128)
            x, y = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
            loss = -(y * F.log_softmax(net(x), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


@torch.no_grad()
def predict_session(net, sess, dev):
    X = make_eval(sess)
    if len(X) == 0:
        return np.nan
    p = F.softmax(net(torch.from_numpy(X).to(dev)), 1).cpu().numpy()
    est = BINS[p.argmax(1)].astype(float)
    return iqr_median(est)


def iqr_median(x):
    x = np.asarray(x, float)
    if len(x) >= 4:
        q1, q3 = np.percentile(x, [25, 75]); i = q3 - q1
        k = x[(x >= q1 - 1.5 * i) & (x <= q3 + 1.5 * i)]
        x = k if len(k) else x
    return float(np.median(x))


# ----------------------------------------------------------------------------------------------- evaluation
def load_truth():
    t = {}
    with open(CACHE / "results_sessions.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t[r["session"]] = dict(orig=float(r["truth_original"]), clean=float(r["truth_cleaned"]),
                                   bad=int(r["reference_unreliable"]), old=int(r["in_old_71"]),
                                   sel=float(r["selector_argmax"]) if r["selector_argmax"] else np.nan,
                                   cls=float(r["classical"]) if r["classical"] else np.nan,
                                   hyb=float(r["old_hybrid"]) if r["old_hybrid"] else np.nan)
    return t


def line(est, truth, key, ids):
    e = np.array([abs(est[i] - truth[i][key]) for i in ids if np.isfinite(est.get(i, np.nan)) and np.isfinite(truth[i][key])])
    if not len(e):
        return "n=0"
    return f"n={len(e):3d}  MAE {e.mean():5.2f}  median {np.median(e):5.2f}  RMSE {np.sqrt((e**2).mean()):5.2f}  <=5: {int((e<=5).sum()):3d}  >20: {int((e>20).sum()):2d}"


def show(name, est, truth, ids):
    print(f"\n{name}")
    for tag, sel in (("all sessions", ids), ("original 71", [i for i in ids if truth[i]["old"]]), ("new sessions", [i for i in ids if not truth[i]["old"]])):
        print(f"  {tag}")
        print(f"     original labels          {line(est, truth, 'orig', sel)}")
        print(f"     cleaned labels           {line(est, truth, 'clean', sel)}")
        print(f"     cleaned, reliable only   {line(est, truth, 'clean', [i for i in sel if not truth[i]['bad']])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--roi", default=str(CACHE / "roi"))
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--tag", default="deep")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, "| torch", torch.__version__, flush=True)
    truth = load_truth()
    sessions = []
    for p in sorted(Path(a.roi).glob("*.npz")):
        if ".tmp" in p.name:
            continue
        s = load_session(p)
        if s is not None and s["id"] in truth:
            sessions.append(s)
    print(len(sessions), "sessions,", len({s["subject"] for s in sessions}), "subjects", flush=True)
    ids = [s["id"] for s in sessions]
    subj = sorted({s["subject"] for s in sessions})
    rng = np.random.default_rng(0)
    rng.shuffle(subj)
    fold_of = {u: i % a.folds for i, u in enumerate(subj)}
    est = {}
    t0 = time.time()
    for k in range(a.folds):
        tr = [s for s in sessions if fold_of[s["subject"]] != k]
        te = [s for s in sessions if fold_of[s["subject"]] == k]
        net = train(tr, a.epochs, dev, seed=k)
        for s in te:
            est[s["id"]] = predict_session(net, s, dev)
        print(f"fold {k + 1}/{a.folds} done ({time.time() - t0:.0f}s)", flush=True)
    show("DEEP MODEL (time-stretch trained), out-of-fold grouped by subject", est, truth, ids)
    sel = {i: truth[i]["sel"] for i in ids}
    show("PEAK SELECTOR (for reference, same sessions)", sel, truth, ids)
    cls = {i: truth[i]["cls"] for i in ids}
    show("CLASSICAL (for reference)", cls, truth, ids)

    cross = {}
    for tr_name, te_name in (("UBFC-rPPG", "UBFC-Phys"), ("UBFC-Phys", "UBFC-rPPG")):
        isp = lambda s: "UBFC-Phys" if s["dataset"] == "ubfc_phys" else "UBFC-rPPG"
        tr = [s for s in sessions if isp(s) == tr_name]
        te = [s for s in sessions if isp(s) == te_name]
        net = train(tr, a.epochs, dev, seed=100)
        e = {s["id"]: predict_session(net, s, dev) for s in te}
        cross.update(e)
        tid = [s["id"] for s in te]
        show(f"CROSS-DATASET deep model: trained ONLY on {tr_name}, tested on {te_name}", e, truth, tid)
        show(f"   (peak selector, in-fold, on the same {te_name} sessions)", sel, truth, tid)
    out = CACHE / f"results_{a.tag}_sessions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["session", "deep_oof", "deep_cross_dataset"])
        for i in ids:
            w.writerow([i, "" if not np.isfinite(est.get(i, np.nan)) else round(est[i], 2), "" if not np.isfinite(cross.get(i, np.nan)) else round(cross[i], 2)])
    print("\nwritten", out)


if __name__ == "__main__":
    main()
