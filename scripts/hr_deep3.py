#!/usr/bin/env python3
"""
Deep model v3: pretrain the spectra network on a GPU physiological SIMULATOR, then fine-tune on real people.

    tools/gpu_lab/venv/Scripts/python.exe scripts/hr_deep3.py [--search-steps 800] [--full-steps 2500] [--tag deep3]

Simulator (all on the GPU, batches of thousands): heart rate 48-180 BPM with slow drift; pulse waveform = fundamental + 2nd
+ 3rd harmonic with random phases (dicrotic shapes); skin/lighting RGB pulse signature with jitter; respiration modulation;
multiplicative lighting drift; motion bursts (band-limited, correlated across channels); speech-like periodic artifact;
sensor noise.  Each simulated 20 s trace is pushed through the SAME representation the real system uses: CHROM and POS
projections -> power spectra on the 116-bin 0.70-3.00 Hz grid at 11 time points -> averages over the last 3 and 10 s.

Honest protocol:
  * simulator REGIME is chosen by zero-shot MAE on the ORIGINAL 71 sessions only; the 54 new sessions are only reported.
  * zero-shot = trained on simulation only, never saw a real person (extras zeroed).
  * fine-tuned = pretrained net, then trained on real people OUT-OF-FOLD grouped by subject (8 folds), plus both
    cross-dataset directions.  Session answer = IQR median of per-cycle argmax, as everywhere else.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hr_deep2 as D2                                          # noqa: E402
from hr_deep import CACHE, iqr_median, load_truth, show        # noqa: E402

FS, T = 30.0, 600                                             # 20 s at 30 Hz
NB, GRID = D2.NB, torch.tensor(D2.GRID, dtype=torch.float32)
END = [300 + 30 * k for k in range(11)]                       # window ends: 10 s ... 20 s (10 s windows)

REGIMES = {                                                    # (motion_p, speech_p, motion_amp_max, noise_max)
    "clean": (0.10, 0.00, 0.02, 0.002),
    "motion": (0.60, 0.00, 0.05, 0.004),
    "motion+speech": (0.60, 0.40, 0.05, 0.004),
    "harsh": (0.90, 0.60, 0.10, 0.008),
}


def logu(lo, hi, shape, dev):
    return torch.exp(torch.rand(shape, device=dev) * (np.log(hi) - np.log(lo)) + np.log(lo))


def simulate(B, dev, regime, g=None):
    mp, sp, mamax, nmax = REGIMES[regime]
    t = (torch.arange(T, device=dev) / FS)[None, :]
    U = lambda lo, hi, shape=(B, 1): torch.rand(shape, device=dev) * (hi - lo) + lo
    f0 = U(0.8, 3.0)
    drift = U(0.0, 0.06) * torch.sin(2 * np.pi * U(0.02, 0.15) * t + U(0, 6.28)) + 0.01 * torch.cumsum(torch.randn(B, T, device=dev), 1) / np.sqrt(T) * 3
    inst = f0 * (1 + drift)
    ph = 2 * np.pi * torch.cumsum(inst, 1) / FS
    w = torch.cos(ph) + U(0.1, 0.7) * torch.cos(2 * ph + U(0, 6.28)) + U(0, 0.3) * torch.cos(3 * ph + U(0, 6.28))
    w = w / w.std(1, keepdim=True)
    sig = torch.tensor([0.33, 0.77, 0.53], device=dev)[None, :, None] * (1 + 0.25 * torch.randn(B, 3, 1, device=dev))
    pa = logu(1e-3, 1.2e-2, (B, 1, 1), dev)
    pulse = pa * sig * w[:, None, :]
    resp = logu(1e-3, 2e-2, (B, 1, 1), dev) * (1 + 0.2 * torch.randn(B, 3, 1, device=dev)) * torch.cos(2 * np.pi * U(0.1, 0.5)[:, :, None] * t[:, None, :] + U(0, 6.28)[:, :, None])
    il = sum(logu(1e-3, 3e-2, (B, 1, 1), dev) * torch.sin(2 * np.pi * U(0.02, 0.4)[:, :, None] * t[:, None, :] + U(0, 6.28)[:, :, None]) for _ in range(3))
    # motion bursts: band-limited noise, common mixing vector, gaussian time envelope
    noise = torch.randn(B, 3, T, device=dev)
    Fz = torch.fft.rfft(noise, dim=2)
    fr = torch.fft.rfftfreq(T, 1 / FS).to(dev)[None, None, :]
    lo = U(0.3, 2.0)[:, :, None]
    mask = ((fr >= lo) & (fr <= lo + U(0.3, 1.5)[:, :, None])).float()
    mot = torch.fft.irfft(Fz * mask, n=T, dim=2)
    mot = mot / (mot.std(2, keepdim=True) + 1e-9) * (1 + 0.5 * torch.randn(B, 3, 1, device=dev))
    env = torch.exp(-0.5 * ((t - U(2, 18)) / U(1.5, 8)) ** 2)[:, None, :]
    mot = mot * env * logu(1e-3, mamax, (B, 1, 1), dev) * (torch.rand(B, 1, 1, device=dev) < mp).float()
    # speech-like periodic artifact (2-5 Hz, amplitude modulated)
    sp_f = U(2.0, 5.0)[:, :, None]
    spe = torch.sin(2 * np.pi * sp_f * t[:, None, :] + U(0, 6.28)[:, :, None]) * (0.6 + 0.4 * torch.sin(2 * np.pi * U(0.3, 1.0)[:, :, None] * t[:, None, :]))
    spe = spe * (1 + 0.5 * torch.randn(B, 3, 1, device=dev)) * logu(1e-3, mamax, (B, 1, 1), dev) * (torch.rand(B, 1, 1, device=dev) < sp).float()
    sens = torch.randn(B, 3, T, device=dev) * logu(5e-4, nmax, (B, 1, 1), dev)
    rgb = (1 + pulse + resp) * (1 + il) + mot + spe + sens
    return rgb, inst[:, -180:].mean(1)                                         # (B,3,T), mean true heart rate (Hz) over the last 6 s


def spectra(rgb, dev):
    """(B,3,T) -> (B, 4, NB): CHROM/POS spectra averaged over the last 3 s and 10 s of 10 s windows."""
    win = torch.hann_window(300, device=dev)[None, :]
    idx = (GRID * 4096 / FS).to(dev)
    lo, hi = idx.floor().long(), idx.ceil().long().clamp(max=2048)
    wgt = (idx - idx.floor())[None, :]
    out_c, out_p = [], []
    for e in END:
        seg = rgb[:, :, e - 300:e]
        n = seg / (seg.mean(2, keepdim=True) + 1e-9)
        r, g, b = n[:, 0], n[:, 1], n[:, 2]
        xs, ys = 3 * r - 2 * g, 1.5 * r + g - 1.5 * b
        chrom = xs - (xs.std(1, keepdim=True) / (ys.std(1, keepdim=True) + 1e-9)) * ys
        s1, s2 = g - b, -2 * r + g + b
        pos = s1 + (s1.std(1, keepdim=True) / (s2.std(1, keepdim=True) + 1e-9)) * s2
        for sgl, store in ((chrom, out_c), (pos, out_p)):
            x = (sgl - sgl.mean(1, keepdim=True)) * win
            P = torch.fft.rfft(x, n=4096, dim=1).abs() ** 2
            Pg = P[:, lo] * (1 - wgt) + P[:, hi] * wgt
            store.append(Pg / (Pg.sum(1, keepdim=True) + 1e-12))
    C, P = torch.stack(out_c, 1), torch.stack(out_p, 1)            # (B, 11, NB)
    return torch.stack([C[:, -3:].mean(1), P[:, -3:].mean(1), C[:, 1:].mean(1), P[:, 1:].mean(1)], 1)


def prep_t(S, extra):
    S = S / (S.amax(2, keepdim=True) + 1e-9)
    return torch.cat([S.reshape(len(S), -1), extra], 1)


def target_t(hz, dev):
    d = (GRID.to(dev)[None, :] - hz[:, None]) / 0.02
    p = torch.exp(-0.5 * (d / 1.2) ** 2)
    return p / (p.sum(1, keepdim=True) + 1e-9)


def pretrain(net, steps, regime, dev, B=2048, lr=1.5e-3, log=None):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    net.train()
    for k in range(steps):
        rgb, hz = simulate(B, dev, regime)
        with torch.no_grad():
            S = spectra(rgb, dev)
            x = prep_t(S, torch.zeros(B, 2, device=dev))
            y = target_t(hz, dev)
        loss = -(y * F.log_softmax(net(x), 1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and (k + 1) % 200 == 0:
            print(f"    [{regime}] step {k + 1}/{steps} loss {loss.item():.3f}", flush=True)
    return net.eval()


@torch.no_grad()
def predict_zero(net, s, dev):
    ks = np.arange(150, len(s["S"]), D2.STEP)
    if len(ks) == 0:
        return np.nan
    x = torch.from_numpy(D2.prep(s["S"][ks], np.zeros_like(s["extra"][ks]))).to(dev)
    p = F.softmax(net(x), 1).cpu().numpy()
    return iqr_median(D2.GBPM[p.argmax(1)])


def stretch_t(S, alpha):
    """GPU frequency-axis stretch: s'(f) = s(f / alpha), zero outside the grid.  S (N,4,NB), alpha (N,)."""
    g = GRID.to(S.device)
    pos = (g[None, :] / alpha[:, None] - g[0]) / 0.02            # fractional source bin
    lo = pos.floor().clamp(0, NB - 1).long()
    hi = (lo + 1).clamp(max=NB - 1)
    w = (pos - lo.float()).clamp(0, 1)[:, None, :]
    inside = ((pos >= 0) & (pos <= NB - 1)).float()[:, None, :]
    return (S.gather(2, lo[:, None, :].expand(-1, S.shape[1], -1)) * (1 - w)
            + S.gather(2, hi[:, None, :].expand(-1, S.shape[1], -1)) * w) * inside


def finetune(net0, sessions, epochs, dev, seed=0, lr=5e-4):
    import copy
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = copy.deepcopy(net0).train()
    Ss, Es, Hs = [], [], []
    for s in sessions:
        ks = np.array([k for k in range(150, len(s["gt"]), D2.STEP) if np.isfinite(s["gt"][k])], int)
        if len(ks):
            Ss.append(s["S"][ks]); Es.append(s["extra"][ks]); Hs.append(s["gt"][ks] / 60.0)
    S = torch.from_numpy(np.concatenate(Ss)).to(dev)
    E = torch.from_numpy(np.concatenate(Es)).to(dev)
    Hz = torch.from_numpy(np.concatenate(Hs).astype(np.float32)).to(dev)
    n = len(S)
    steps = max(1, n // 256)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    for _ in range(epochs):
        for _ in range(steps):
            i = torch.randint(0, n, (256,), device=dev)
            a = torch.where(torch.rand(256, device=dev) < 0.8, torch.exp(torch.empty(256, device=dev).uniform_(np.log(0.85), np.log(1.2))), torch.ones(256, device=dev))
            hz2 = Hz[i] * a
            ok = (hz2 >= GRID[0]) & (hz2 <= GRID[-1])
            if not ok.any():
                continue
            x = prep_t(stretch_t(S[i], a), E[i])[ok]
            loss = -(target_t(hz2, dev)[ok] * F.log_softmax(net(x), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


def mae(est, truth, ids, key="clean"):
    e = [abs(est[i] - truth[i][key]) for i in ids if np.isfinite(est.get(i, np.nan)) and np.isfinite(truth[i][key])]
    return (float(np.mean(e)), len(e)) if e else (np.nan, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(CACHE / "feat"))
    ap.add_argument("--roi", default=str(CACHE / "roi"))
    ap.add_argument("--search-steps", type=int, default=800)
    ap.add_argument("--full-steps", type=int, default=2500)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--tag", default="deep3")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, "|", torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    truth = load_truth()
    sessions = []
    for p in sorted(Path(a.feat).glob("*.npz")):
        if ".tmp" in p.name or p.stem not in truth:
            continue
        try:
            sessions.append(D2.load_session(p, a.roi))
        except Exception as e:
            print("skip", p.stem, repr(e)[:70])
    ids = [s["id"] for s in sessions]
    old = [i for i in ids if truth[i]["old"]]
    new = [i for i in ids if not truth[i]["old"]]
    print(len(sessions), "sessions;", len(old), "original,", len(new), "new", flush=True)

    # 1) choose the simulator regime using ONLY the original 71 sessions (zero-shot)
    t0 = time.time()
    results = {}
    for reg in REGIMES:
        torch.manual_seed(1)
        net = pretrain(D2.Net().to(dev), a.search_steps, reg, dev)
        est = {s["id"]: predict_zero(net, s, dev) for s in sessions}
        results[reg] = (mae(est, truth, old), mae(est, truth, new), est, net)
        print(f"regime {reg:14s} zero-shot MAE  original-71: {results[reg][0][0]:5.2f}   new-54: {results[reg][1][0]:5.2f}   ({time.time() - t0:.0f}s)", flush=True)
    best = min(results, key=lambda r: results[r][0][0])
    print(f"\nchosen regime (by original-71 zero-shot only): {best}", flush=True)

    # 2) full pretraining in the chosen regime
    torch.manual_seed(2)
    net0 = pretrain(D2.Net().to(dev), a.full_steps, best, dev, log=True)
    zero = {s["id"]: predict_zero(net0, s, dev) for s in sessions}
    show(f"ZERO-SHOT: trained on simulation only ({best}), never saw a real person", zero, truth, ids)
    sel = {i: truth[i]["sel"] for i in ids}
    cls = {i: truth[i]["cls"] for i in ids}
    show("PEAK SELECTOR (reference)", sel, truth, ids)
    show("CLASSICAL (reference)", cls, truth, ids)

    # 3) fine-tune on real people, out-of-fold by subject
    subj = sorted({s["subject"] for s in sessions})
    np.random.default_rng(0).shuffle(subj)
    fold_of = {u: i % a.folds for i, u in enumerate(subj)}
    ft = {}
    for k in range(a.folds):
        net = finetune(net0, [s for s in sessions if fold_of[s["subject"]] != k], a.ft_epochs, dev, seed=k)
        for s in sessions:
            if fold_of[s["subject"]] == k:
                ft[s["id"]] = D2.predict(net, s, dev)
        print(f"fine-tune fold {k + 1}/{a.folds} ({time.time() - t0:.0f}s)", flush=True)
    show("SIMULATION-PRETRAINED + FINE-TUNED, out-of-fold grouped by subject", ft, truth, ids)

    isp = lambda s: "UBFC-Phys" if s["dataset"] == "ubfc_phys" else "UBFC-rPPG"
    cross = {}
    for trn, ten in (("UBFC-rPPG", "UBFC-Phys"), ("UBFC-Phys", "UBFC-rPPG")):
        net = finetune(net0, [s for s in sessions if isp(s) == trn], a.ft_epochs, dev, seed=100)
        te = [s for s in sessions if isp(s) == ten]
        e = {s["id"]: D2.predict(net, s, dev) for s in te}
        cross.update(e)
        tid = [s["id"] for s in te]
        show(f"CROSS-DATASET (sim-pretrained): fine-tuned ONLY on {trn}, tested on {ten}", e, truth, tid)
        show(f"   (peak selector on the same {ten} sessions)", sel, truth, tid)
    out = CACHE / f"results_{a.tag}_sessions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["session", "sim_zero_shot", "sim_finetuned_oof", "sim_cross_dataset"])
        for i in ids:
            cell = lambda d: "" if not np.isfinite(d.get(i, np.nan)) else round(d[i], 2)
            w.writerow([i, cell(zero), cell(ft), cell(cross)])
    torch.save(net0.state_dict(), CACHE.parent / "models" / "hr_sim_pretrained.pt")
    print("\nwritten", out, flush=True)


if __name__ == "__main__":
    main()
