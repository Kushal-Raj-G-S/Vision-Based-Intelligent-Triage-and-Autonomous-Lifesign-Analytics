#!/usr/bin/env python3
"""
Peak-selection HR estimator.

Instead of regressing a corrected value (which shrinks toward the population mean), CHOOSE among real spectral
peaks. Candidates per cycle come from the lock-independent wide-band spectra of CHROM and POS plus the production
estimate (so the pool always contains what the classical system would have said). A classifier scores each
candidate; a Viterbi track over the 0.02 Hz frequency grid links the choices through time.

Everything here is evaluated by scripts/hr_lab.py out-of-fold, grouped by subject.
"""
import numpy as np
from scipy.signal import find_peaks

GRID_HZ = np.arange(0.70, 3.0001, 0.02)          # 116 bins, matches rppg._wide_spectrum
GRID_BPM = GRID_HZ * 60.0
TOL_BPM = 4.0                                     # a candidate "matches" the truth within this
K_PEAKS = 3


def _peaks(spec, k=K_PEAKS):
    sm = np.convolve(spec, np.array([0.25, 0.5, 0.25]), mode="same")
    idx, props = find_peaks(sm, prominence=sm.max() * 0.05)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(sm))])
        props = {"prominences": np.array([sm[idx[0]]])}
    order = np.argsort(-sm[idx])[:k]
    return idx[order], sm


def _bin(bpm):
    return int(np.clip(round((bpm / 60.0 - GRID_HZ[0]) / 0.02), 0, len(GRID_HZ) - 1))


def candidate_table(s, stride=5, roll=30):
    """Per-cycle candidate rows for one session. Returns (X, y, cycle_index, bpm) with y = matches the truth."""
    n = len(s["bpm"])
    rows, ys, cyc, bpms, meta = [], [], [], [], []
    prod = s["bpm"]
    for t in range(0, n, stride):
        cs, cp = s["chrom_spec"][t], s["pos_spec"][t]
        ic, smc = _peaks(cs)
        ip, smp = _peaks(cp)
        cands = {}
        for rank, i in enumerate(ic):
            cands.setdefault(int(i), {})["rc"] = rank
        for rank, i in enumerate(ip):
            cands.setdefault(int(i), {})["rp"] = rank
        pb = _bin(prod[t])
        cands.setdefault(pb, {})["prod"] = True
        pb2 = _bin(float(s["pos_bpm"][t]))
        cands.setdefault(pb2, {})["posprod"] = True
        # merge candidates closer than 2 bins
        merged = []
        for k in sorted(cands):
            if merged and k - merged[-1][0] <= 1:
                old_k, flags = merged[-1]
                flags.update(cands[k])                              # union of the flags of the two neighbouring bins
                if smc[k] + smp[k] > smc[old_k] + smp[old_k]:
                    merged[-1][0] = k                               # keep the stronger bin as the candidate frequency
            else:
                merged.append([k, dict(cands[k])])
        lo = max(0, t - roll)
        rmean, rstd = float(np.mean(prod[lo:t + 1])), float(np.std(prod[lo:t + 1]))
        for b, flag in merged:
            f = GRID_BPM[b]
            b2 = _bin(f * 2.0) if f * 2.0 <= 180 else None
            bh = _bin(f / 2.0) if f / 2.0 >= 42 else None
            pc, pp = float(smc[b]), float(smp[b])
            row = [
                f, pc, pp, pc + pp,
                flag.get("rc", 9), flag.get("rp", 9), float("prod" in flag), float("posprod" in flag),
                float(smc[b2]) if b2 is not None else 0.0, float(smc[bh]) if bh is not None else 0.0,
                float(smp[b2]) if b2 is not None else 0.0, float(smp[bh]) if bh is not None else 0.0,
                abs(f - prod[t]), abs(f - float(s["pos_bpm"][t])), abs(prod[t] - float(s["pos_bpm"][t])),
                f - rmean, rstd, rmean,
                float(s["sqi"][t]), float(s["snr_db"][t]), float(s["stability"][t]), float(s["harmonic_ratio"][t]),
                float(s["motion"][t]), float(s["pos_sqi"][t]),
                float(np.nan_to_num(s["mouth_open"][t], nan=0.0)),
                float(smc.max()), float(smp.max()),
                pc / (smc.max() + 1e-9), pp / (smp.max() + 1e-9),
            ]
            rows.append(row)
            ys.append(float(abs(f - s["gt"][t]) <= TOL_BPM) if np.isfinite(s["gt"][t]) else np.nan)
            cyc.append(t)
            bpms.append(f)
    return np.array(rows, np.float32), np.array(ys, np.float32), np.array(cyc), np.array(bpms, np.float32)


FEATURE_NAMES = ["bpm", "pow_chrom", "pow_pos", "pow_sum", "rank_chrom", "rank_pos", "is_prod", "is_posprod",
                 "chrom_pow_2x", "chrom_pow_half", "pos_pow_2x", "pos_pow_half", "d_prod", "d_pos", "prod_vs_pos",
                 "vs_roll_mean", "roll_std", "roll_mean", "sqi", "snr", "stability", "harm_ratio", "motion", "pos_sqi",
                 "mouth", "chrom_max", "pos_max", "chrom_rel", "pos_rel"]


def viterbi_track(cyc, bpm, prob, n_cycles, stride, max_jump_bins=2, jump_pen=1.6, eps=1e-3):
    """Link the per-cycle candidate scores into one frequency track over the 116-bin grid."""
    T = int(np.ceil(n_cycles / stride))
    nb = len(GRID_HZ)
    emit = np.full((T, nb), eps, np.float64)
    for c, b, p in zip(cyc, bpm, prob):
        t = c // stride
        k = _bin(b)
        for d in (-1, 0, 1):
            if 0 <= k + d < nb:
                emit[t, k + d] = max(emit[t, k + d], float(p) * (0.6 if d else 1.0) + eps)
    logE = np.log(emit)
    delta = logE[0].copy()
    back = np.zeros((T, nb), np.int16)
    jump = np.arange(-max_jump_bins * 6, max_jump_bins * 6 + 1)
    pen = -jump_pen * (np.abs(jump) / float(max_jump_bins)) ** 1.0
    for t in range(1, T):
        best = np.full(nb, -1e18)
        arg = np.zeros(nb, np.int16)
        for d, pn in zip(jump, pen):
            lo, hi = max(0, -d), min(nb, nb - d)
            cand = delta[lo:hi] + pn
            tgt = np.arange(lo, hi) + d
            upd = cand > best[tgt]
            best[tgt[upd]] = cand[upd]
            arg[tgt[upd]] = (np.arange(lo, hi))[upd]
        delta = best + logE[t]
        back[t] = arg
    path = np.zeros(T, int)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return GRID_BPM[path]
