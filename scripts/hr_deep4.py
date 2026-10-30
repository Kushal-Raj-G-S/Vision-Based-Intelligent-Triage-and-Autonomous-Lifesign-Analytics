#!/usr/bin/env python3
"""
Deep model v4: frequency-equivariant spectro-temporal tracker, pretrained on the GPU simulator.

    tools/gpu_lab/venv/Scripts/python.exe scripts/hr_deep4.py [--pre-steps 4000] [--ft-epochs 8] [--seeds 1] [--tag deep4]

Input per decision: the last 11 CHROM+POS spectra (1 s apart, each a 116-bin Welch spectrum of the current buffer),
each normalised by its own maximum -> tensor (11, 2, 116).  No scalar features: the net judges the SHAPE of the evidence.
  1. frequency encoder  : Conv1d across bins (shared weights => a shifted/stretched heart rate looks the same)
  2. temporal encoder   : Conv1d across the 11 steps, independently for every bin (does a peak PERSIST here?)
  3. context head       : dilated Conv1d across bins + global pooling (sees harmonic structure and competing peaks)
  4. learned bin prior  : per-bin bias (the plausible-HR prior)  ->  posterior over the 116 bins.
Pretraining: GPU simulator (hr_deep3.simulate) mixing regimes.  Fine-tuning: real people, frequency-stretch augmentation.

Evaluation protocol is identical to v2/v3: out-of-fold grouped by subject on all 125 sessions, both cross-dataset
directions, plus a STRICT split (fine-tune only on the original 71, test only on the new 54).
"""
import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hr_deep2 as D2                                          # noqa: E402
import hr_deep3 as H3                                          # noqa: E402
from hr_deep import CACHE, iqr_median, load_truth, show        # noqa: E402

NB, L, SPACING = D2.NB, 11, 30
GRID_T = torch.tensor(D2.GRID, dtype=torch.float32)
MIX = [("clean", 0.15), ("motion", 0.30), ("motion+speech", 0.30), ("harsh", 0.25)]


class SpecTracker(nn.Module):
    def __init__(self, ch=48):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(2, ch, 7, padding=3), nn.GELU(), nn.Conv1d(ch, ch, 5, padding=2), nn.GELU())
        self.tmp = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.GELU(), nn.Conv1d(ch, ch, 3, padding=1), nn.GELU())
        c3 = 3 * ch
        self.head = nn.Sequential(
            nn.Conv1d(2 * c3, 96, 5, padding=2), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(96, 96, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(96, 64, 5, padding=8, dilation=4), nn.GELU(),
            nn.Conv1d(64, 1, 1))
        self.bias = nn.Parameter(torch.zeros(NB))

    def forward(self, x):                                       # x (B, L, 2, NB)
        B, T_, _, N = x.shape
        h = self.enc(x.reshape(B * T_, 2, N))                    # (B*L, ch, NB)
        ch = h.shape[1]
        h = h.reshape(B, T_, ch, N).permute(0, 3, 2, 1).reshape(B * N, ch, T_)
        h = self.tmp(h)                                          # temporal conv per bin
        h = torch.cat([h[:, :, -1], h.mean(2), h.amax(2)], 1)    # (B*NB, 3ch)
        h = h.reshape(B, N, -1).permute(0, 2, 1)                 # (B, 3ch, NB)
        g = h.mean(2, keepdim=True).expand(-1, -1, N)
        return self.head(torch.cat([h, g], 1)).squeeze(1) + self.bias


def norm_spec(x):
    return x / (x.amax(-1, keepdim=True) + 1e-9)


# ---------- simulation -------------------------------------------------------------------------------------------
def sim_seq(B, dev, regime):
    rgb, hz = H3.simulate(B, dev, regime)
    win = torch.hann_window(300, device=dev)[None, :]
    idx = (GRID_T.to(dev) * 4096 / H3.FS)
    lo, hi = idx.floor().long(), idx.ceil().long().clamp(max=2048)
    wgt = (idx - idx.floor())[None, :]
    steps = []
    for e in H3.END:
        seg = rgb[:, :, e - 300:e]
        n = seg / (seg.mean(2, keepdim=True) + 1e-9)
        r, g, b = n[:, 0], n[:, 1], n[:, 2]
        xs, ys = 3 * r - 2 * g, 1.5 * r + g - 1.5 * b
        chrom = xs - (xs.std(1, keepdim=True) / (ys.std(1, keepdim=True) + 1e-9)) * ys
        s1, s2 = g - b, -2 * r + g + b
        pos = s1 + (s1.std(1, keepdim=True) / (s2.std(1, keepdim=True) + 1e-9)) * s2
        pair = []
        for sgl in (chrom, pos):
            P = torch.fft.rfft((sgl - sgl.mean(1, keepdim=True)) * win, n=4096, dim=1).abs() ** 2
            pair.append(P[:, lo] * (1 - wgt) + P[:, hi] * wgt)
        steps.append(torch.stack(pair, 1))
    return norm_spec(torch.stack(steps, 1)), hz                  # (B, 11, 2, NB), (B,)


def pretrain(net, steps, dev, B=512, lr=2e-3):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    names, probs = zip(*MIX)
    rng = np.random.default_rng(0)
    net.train()
    for k in range(steps):
        x, hz = sim_seq(B, dev, names[rng.choice(len(names), p=probs)])
        loss = -(H3.target_t(hz, dev) * F.log_softmax(net(x), 1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (k + 1) % 250 == 0:
            print(f"    pretrain step {k + 1}/{steps} loss {loss.item():.3f}", flush=True)
    return net.eval()


# ---------- real data --------------------------------------------------------------------------------------------
def load_real(a, truth):
    out = []
    for p in sorted(Path(a.feat).glob("*.npz")):
        if ".tmp" in p.name or p.stem not in truth:
            continue
        try:
            s = D2.load_session(p, a.roi)
            z = np.load(p, allow_pickle=False)
            s["X"] = np.stack([z["chrom_spec"], z["pos_spec"]], 1).astype(np.float32)      # (n, 2, NB)
            s["X"] = s["X"] / (s["X"].max(2, keepdims=True) + 1e-9)
            del s["S"], s["extra"]
            out.append(s)
        except Exception as e:
            print("skip", p.stem, repr(e)[:70])
    return out


def seq_index(k, n):
    return np.clip(k[:, None] - (L - 1) * SPACING + SPACING * np.arange(L)[None, :], 0, n - 1)


@torch.no_grad()
def posterior(nets, s, dev):
    ks = np.arange(150, len(s["X"]), D2.STEP)
    if len(ks) == 0:
        return None
    X = torch.from_numpy(s["X"][seq_index(ks, len(s["X"]))]).to(dev)       # (K, L, 2, NB)
    return sum(F.softmax(n(X), 1) for n in nets).cpu().numpy() / len(nets)


def answer(p):
    return np.nan if p is None else iqr_median(D2.GBPM[p.argmax(1)])


def finetune(net0, sessions, epochs, dev, seed=0, lr=4e-4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = copy.deepcopy(net0).train()
    Xs, idxs, hzs, off = [], [], [], 0
    for s in sessions:
        ks = np.array([k for k in range(150, len(s["gt"]), D2.STEP) if np.isfinite(s["gt"][k])], int)
        if len(ks):
            Xs.append(s["X"]); idxs.append(seq_index(ks, len(s["X"])) + off); hzs.append(s["gt"][ks] / 60.0)
            off += len(s["X"])
    X = torch.from_numpy(np.concatenate(Xs)).to(dev)
    I = torch.from_numpy(np.concatenate(idxs)).to(dev)
    Hz = torch.from_numpy(np.concatenate(hzs).astype(np.float32)).to(dev)
    n, bs = len(I), 128
    steps = max(1, n // bs)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    for _ in range(epochs):
        for _ in range(steps):
            i = torch.randint(0, n, (bs,), device=dev)
            x = X[I[i]]                                                       # (bs, L, 2, NB)
            a = torch.where(torch.rand(bs, device=dev) < 0.8, torch.exp(torch.empty(bs, device=dev).uniform_(np.log(0.85), np.log(1.2))), torch.ones(bs, device=dev))
            x = H3.stretch_t(x.reshape(bs, L * 2, NB), a).reshape(bs, L, 2, NB)
            hz2 = Hz[i] * a
            ok = (hz2 >= GRID_T[0]) & (hz2 <= GRID_T[-1])
            if not ok.any():
                continue
            loss = -(H3.target_t(hz2, dev)[ok] * F.log_softmax(net(x[ok]), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(CACHE / "feat"))
    ap.add_argument("--roi", default=str(CACHE / "roi"))
    ap.add_argument("--pre-steps", type=int, default=4000)
    ap.add_argument("--ft-epochs", type=int, default=8)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="deep4")
    ap.add_argument("--load", default="", help="reuse saved pretrained weights instead of pretraining")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    truth = load_truth()
    sessions = load_real(a, truth)
    ids = [s["id"] for s in sessions]
    old = [i for i in ids if truth[i]["old"]]
    new = [i for i in ids if not truth[i]["old"]]
    print(len(sessions), "sessions;", len(old), "original,", len(new), "new", flush=True)
    sel = {i: truth[i]["sel"] for i in ids}
    cls = {i: truth[i]["cls"] for i in ids}
    t0 = time.time()

    torch.manual_seed(3)
    if a.load:
        net0 = SpecTracker().to(dev)
        net0.load_state_dict(torch.load(a.load, map_location=dev))
        net0.eval()
    else:
        net0 = pretrain(SpecTracker().to(dev), a.pre_steps, dev)
    print(f"params {sum(p.numel() for p in net0.parameters()):,}   pretraining {time.time() - t0:.0f}s", flush=True)
    zero = {s["id"]: answer(posterior([net0], s, dev)) for s in sessions}
    show("ZERO-SHOT tracker (simulation only)", zero, truth, ids)
    show("PEAK SELECTOR (reference)", sel, truth, ids)

    subj = sorted({s["subject"] for s in sessions})
    np.random.default_rng(0).shuffle(subj)
    fold_of = {u: i % a.folds for i, u in enumerate(subj)}
    ft = {}
    for k in range(a.folds):
        nets = [finetune(net0, [s for s in sessions if fold_of[s["subject"]] != k], a.ft_epochs, dev, seed=10 * k + r) for r in range(a.seeds)]
        for s in sessions:
            if fold_of[s["subject"]] == k:
                ft[s["id"]] = answer(posterior(nets, s, dev))
        print(f"fine-tune fold {k + 1}/{a.folds} ({time.time() - t0:.0f}s)", flush=True)
    show("TRACKER (sim-pretrained + fine-tuned), out-of-fold grouped by subject", ft, truth, ids)
    show("CLASSICAL (reference)", cls, truth, ids)

    cross = {}
    isp = lambda s: "UBFC-Phys" if s["dataset"] == "ubfc_phys" else "UBFC-rPPG"
    for trn, ten in (("UBFC-rPPG", "UBFC-Phys"), ("UBFC-Phys", "UBFC-rPPG")):
        nets = [finetune(net0, [s for s in sessions if isp(s) == trn], a.ft_epochs, dev, seed=100 + r) for r in range(a.seeds)]
        te = [s for s in sessions if isp(s) == ten]
        e = {s["id"]: answer(posterior(nets, s, dev)) for s in te}
        cross.update(e)
        tid = [s["id"] for s in te]
        show(f"CROSS-DATASET tracker: fine-tuned ONLY on {trn}, tested on {ten}", e, truth, tid)
        show(f"   (peak selector on the same {ten} sessions)", sel, truth, tid)

    nets = [finetune(net0, [s for s in sessions if truth[s["id"]]["old"]], a.ft_epochs, dev, seed=200 + r) for r in range(a.seeds)]
    strict = {s["id"]: answer(posterior(nets, s, dev)) for s in sessions if not truth[s["id"]]["old"]}
    show("STRICT: fine-tuned ONLY on the original 71, tested ONLY on the new 54", strict, truth, new)
    show("   (peak selector on the same 54, out-of-fold)", sel, truth, new)

    out = CACHE / f"results_{a.tag}_sessions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["session", "zero_shot", "oof", "cross_dataset", "strict_71_to_54"])
        for i in ids:
            cell = lambda d: "" if not np.isfinite(d.get(i, np.nan)) else round(d[i], 2)
            w.writerow([i, cell(zero), cell(ft), cell(cross), cell(strict)])
    if not a.load:
        torch.save(net0.state_dict(), CACHE.parent / "models" / "hr_tracker_pretrained.pt")
    print("\nwritten", out, flush=True)


if __name__ == "__main__":
    main()
