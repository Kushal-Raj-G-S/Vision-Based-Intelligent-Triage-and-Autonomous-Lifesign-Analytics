"""Candidate-peak feature table for the learned RR peak re-ranker.

One row per (window, candidate peak). Stateless: uses only that window's own
RIV spectra + its beat-derived heart rate, so it is deployable live."""
import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

SPd = Path(sys.argv[1])
sys.argv = [sys.argv[0], sys.argv[1]]
import lab  # noqa: E402
import core.rppg as R  # noqa: E402

A = R.AdvancedRPPG
CH = ("riav", "riiv", "rifv")
FEATS = None


def q(x):
    return round(x * 100) / 100


def rule_pick(f, p, lo, hi, zone=12.0, ratio=0.30):
    kk = int(np.argmax(p))
    if f[kk] * 60 < zone:
        m = (f >= 1.6 * f[kk]) & (f <= 2.8 * f[kk])
        if m.any():
            idx = np.where(m)[0]; k2 = int(idx[np.argmax(p[idx])])
            if p[k2] >= ratio * p[kk]:
                kk = k2
    kk = A._promote_subharmonic(f, p, kk, hi); kk = A._demote_harmonic(f, p, kk, lo)
    return float(f[kk] * 60), float(np.sum(p[max(0, kk - 2):kk + 3]) / (p.sum() + 1e-12) * 100)


def near(f, p, bpm, tol):
    m = np.abs(f * 60 - bpm) <= tol
    return float(p[m].max()) if m.any() else 0.0


def peaks_of(f, p, K=3):
    idx, _ = find_peaks(p)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(p))])
    idx = idx[np.argsort(p[idx])[::-1]][:K]
    return [float(f[i] * 60) for i in idx], idx


def build_window(w):
    if w["y"] is None:
        return None
    bh = w.get("beat_hz", 0.0); hr = bh * 60
    if hr <= 0:
        return None
    lo = q(min(max(0.06 * hr / 60, 0.10), 0.25)); hi = q(min(max(0.30 * hr / 60, 0.50), 0.80))
    S = {}
    for ch in CH:
        r = lab.psd(w["y"][ch], lo=lo, hi=hi)
        if r is None:
            return None
        S[ch] = r
    f = S["riav"][0]
    pn = {c: S[c][1] / S[c][1].sum() for c in CH}
    pr = {c: S[c][1] / S[c][1].max() for c in CH}
    F2 = pn["riav"] + pn["riiv"]; F3 = F2 + pn["rifv"]
    # current shipped rule's pick (as a stacking feature)
    ra, ca = rule_pick(f, S["riav"][1], lo, hi); ri, ci = rule_pick(f, S["riiv"][1], lo, hi)
    rule_rr, rule_conf = lab.fuse_prod([ra, ri], [ca, ci], bh)
    # candidate set
    cand = []
    for c in CH:
        cand += peaks_of(f, S[c][1])[0]
    cand += peaks_of(f, F2)[0] + peaks_of(f, F3)[0] + [ra, ri]
    cand = sorted(cand)
    uniq = []
    for c in cand:
        if not uniq or c - uniq[-1] > 0.75:
            uniq.append(c)
    uniq = uniq[:14]
    am = {c: float(f[int(np.argmax(S[c][1]))] * 60) for c in CH}
    fa = float(f[int(np.argmax(F2))] * 60)
    flat = {c: float(np.exp(np.mean(np.log(S[c][1] + 1e-20))) / (np.mean(S[c][1]) + 1e-20)) for c in CH}
    rank_cache = {c: np.argsort(-S[c][1]) for c in CH}
    rows = []
    for fc in uniq:
        k = int(np.argmin(np.abs(f * 60 - fc)))
        x = [fc, fc / hr, hr, hr / max(fc, 1e-6), lo * 60, hi * 60, 1.0 if fc < 12 else 0.0]
        for c in CH:
            rel = near(f, pr[c], fc, 1.5)
            conc = float(pn[c][np.abs(f * 60 - fc) <= 1.5].sum())
            x += [rel, conc, abs(am[c] - fc), flat[c], float(np.where(rank_cache[c] == k)[0][0]) / len(f)]
        r2 = near(f, F2 / F2.max(), fc, 1.5); r3 = near(f, F3 / F3.max(), fc, 1.5)
        x += [r2, r3, fc / max(fa, 1e-6),
              near(f, F2 / F2.max(), 2 * fc, 1.0), near(f, F2 / F2.max(), fc / 2, 1.0),
              near(f, F2 / F2.max(), 3 * fc, 1.0),
              abs(am["riav"] - am["riiv"]), abs(ra - ri), rule_conf, abs(fc - rule_rr), 1.0 if abs(fc - rule_rr) <= 1.0 else 0.0,
              bh]
        rows.append(x)
    return np.asarray(rows, dtype=np.float32), np.asarray(uniq, dtype=np.float32), (rule_rr, rule_conf)


def main(rows_all, out):
    X, meta = [], []
    for wi, w in enumerate(rows_all):
        b = build_window(w)
        if b is None:
            continue
        Xw, cands, (rr_rule, conf_rule) = b
        y = (np.abs(cands - w["gold"]) <= 2.0).astype(np.int8)
        for j in range(len(cands)):
            X.append(Xw[j]); meta.append((wi, w["ds"], w["rec"], w["t0"], float(cands[j]), int(y[j]), w["gold"], rr_rule, conf_rule))
    pickle.dump(dict(X=np.vstack(X), meta=meta), open(out, "wb"))
    print("candidate rows", len(meta), "windows", len({m[0] for m in meta}), "features", len(X[0]))


if __name__ == "__main__":
    allrows = lab.load() + pickle.load(open(SPd / "cache_vdb.pkl", "rb"))
    main(allrows, SPd / "feats.pkl")
