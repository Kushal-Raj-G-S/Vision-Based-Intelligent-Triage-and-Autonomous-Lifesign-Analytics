"""RR peak-selection lab: evaluate rules on cached RIV series."""
import pickle
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import signal

ROOT = Path(r"C:\Users\M.Manoj\OneDrive\Desktop\vision 1")
sys.path.insert(0, str(ROOT))
from core.rppg import (AdvancedRPPG, _safe_butter_sos, _RIV_FS, _RESP_LOW,  # noqa: E402
                       _RESP_HIGH, _RR_AGREE_TOL)

SP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
GATE = 30.0

_sos_cache = {}


def sos_for(lo=_RESP_LOW, hi=_RESP_HIGH, order=4):
    key = (lo, hi, order)
    if key not in _sos_cache:
        _sos_cache[key] = _safe_butter_sos(lo, hi, _RIV_FS, order=order, clamp_to_pulse_band=False)
    return _sos_cache[key]


def psd(y, lo=_RESP_LOW, hi=_RESP_HIGH, sos=None):
    y = y.astype(float)
    sos = sos if sos is not None else sos_for(lo, hi)
    if len(y) > 24:
        try:
            y = signal.sosfiltfilt(sos, y)
        except Exception:
            pass
    n = len(y)
    if n < 8:
        return None
    f, p = signal.welch(y, fs=_RIV_FS, window="hann", nperseg=n, noverlap=n // 2, nfft=4096)
    m = (f >= lo) & (f <= hi)
    f, p = f[m], p[m]
    if len(p) == 0 or not np.any(p > 0):
        return None
    return f, p


def pick(f, p):
    """production per-channel peak: argmax -> promote -> demote"""
    k = int(np.argmax(p))
    k = AdvancedRPPG._promote_subharmonic(f, p, k)
    k = AdvancedRPPG._demote_harmonic(f, p, k)
    conc = float(np.sum(p[max(0, k - 2):k + 3]) / (float(np.sum(p)) + 1e-12) * 100.0)
    return float(f[k] * 60.0), conc


derate = AdvancedRPPG._sampling_derate


def fuse_prod(cands, concs, beat_hz):
    cands = np.asarray(cands); concs = np.asarray(concs)
    if len(cands) == 0:
        return 0.0, 0.0
    if len(cands) == 1:
        return derate(float(cands[0]), float(np.clip(concs[0] * 0.5, 0, 40)), beat_hz)
    spread = float(np.std(cands))
    if spread <= _RR_AGREE_TOL:
        return derate(float(np.median(cands)), float(np.clip(100 - spread * 12, 20, 100)), beat_hz)
    cs = np.sort(cands); gaps = np.diff(cs); j = int(np.argmin(gaps))
    if gaps[j] <= _RR_AGREE_TOL:
        return derate(float(np.mean(cs[j:j + 2])), float(np.clip(60 - gaps[j] * 8, 15, 60)), beat_hz)
    k = int(np.argmax(concs))
    return derate(float(cands[k]), float(np.clip(concs[k] * 0.3, 0, 25)), beat_hz)


def rule_prod(w, ctx=None):
    if w["y"] is None:
        return 0.0, 0.0
    cands, concs = [], []
    for ch in ("riav", "riiv"):
        r = psd(w["y"][ch])
        if r is None:
            continue
        rr, c = pick(*r)
        cands.append(rr); concs.append(c)
    return fuse_prod(cands, concs, w.get("beat_hz", 0.0))


def session_err(est, gold):
    if len(est) < 3:
        return float("nan")
    a = np.asarray(est)
    q1, q3 = np.percentile(a, [25, 75]); iqr = q3 - q1
    keep = a[(a >= q1 - 1.5 * iqr) & (a <= q3 + 1.5 * iqr)]
    if len(keep) == 0:
        keep = a
    return abs(float(np.median(keep)) - float(np.median(gold)))


def evaluate(rows, rule, stateful=False, verbose=False, label=""):
    """rule(window, ctx)->(rr,conf). ctx is a per-record dict if stateful."""
    byrec = defaultdict(list)
    for w in rows:
        byrec[(w["ds"], w["rec"])].append(w)
    out = defaultdict(list)
    perrec = {}
    werr = defaultdict(list)
    nwin = defaultdict(int); nsup = defaultdict(int)
    for (ds, rec), ws in byrec.items():
        ctx = {} if stateful else None
        acc, gl = [], []
        for w in ws:
            nwin[ds] += 1
            rr, conf = rule(w, ctx)
            if rr <= 0 or conf < GATE:
                nsup[ds] += 1
            else:
                acc.append(rr); gl.append(w["gold"])
                werr[ds].append(abs(rr - w["gold"]))
        e = session_err(acc, gl)
        perrec[(ds, rec)] = e
        out[ds].append(e)
    res = {}
    for ds in out:
        a = np.asarray([e for e in out[ds] if not np.isnan(e)])
        res[ds] = dict(
            mae=float(a.mean()) if len(a) else float("nan"),
            med=float(np.median(a)) if len(a) else float("nan"),
            w2=float((a <= 2).mean() * 100) if len(a) else float("nan"),
            n=len(a), tot=len(out[ds]),
            wmae=float(np.mean(werr[ds])) if werr[ds] else float("nan"),
            sup=100.0 * nsup[ds] / max(1, nwin[ds]))
    if label:
        parts = []
        for ds in sorted(res):
            r = res[ds]
            parts.append(f"{ds.upper()} mae {r['mae']:5.2f} med {r['med']:4.2f} w2 {r['w2']:4.1f}% n {r['n']}/{r['tot']} "
                         f"wmae {r['wmae']:4.2f} sup {r['sup']:4.1f}%")
        allv = [e for ds in out for e in out[ds] if not np.isnan(e)]
        print(f"{label:34s} " + " | ".join(parts) + f" | ALL {np.mean(allv):5.2f}")
    if verbose:
        return res, perrec
    return res


def load():
    with open(SP / "cache.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    rows = load()
    print(len(rows), "windows")
    # sanity: cached production output vs lab reproduction
    cache_rule = lambda w, ctx=None: (w["est"], w["conf"])
    evaluate(rows, cache_rule, label="cached production output")
    evaluate(rows, rule_prod, label="lab reproduction (rule_prod)")
