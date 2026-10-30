"""
Respiratory-rate validation against the CapnoBase IEEE TBME RR Benchmark
=========================================================================

Second independent real-data RR validation, run for the same reason
scripts/validate_rr_bidmc.py exists: one dataset's numbers could be a
fluke of that dataset's specific population/equipment/quirks. CapnoBase
differs from BIDMC on every axis that matters for that check:

  * Different population: anaesthesia/surgical patients (mostly pediatric,
    from a Vancouver hospital's pediatric anaesthesia research program),
    vs. BIDMC's critically-ill ICU adults.
  * Different ground-truth principle: capnography (CO2 waveform), a direct
    physical measurement of each exhaled breath, vs. BIDMC's two human
    annotators marking breaths by eye/ear on an impedance-pneumography trace.
  * Different equipment/institution entirely.

It also carries something BIDMC does not: the ORIGINAL PAPER'S OWN
SmartFusion algorithm output for every case (Karlen et al. 2013), which
this project's RIV-fusion method is directly modelled on. That gives a
real literature comparison point, not just a before/after self-comparison.

Format note: unlike BIDMC (WFDB/CSV), CapnoBase ships as MATLAB v7.3
(HDF5) files, which need `h5py` (not scipy.io.loadmat) to read.

Usage
─────
    python scripts/validate_rr_capnobase.py --stride 5.0
    python scripts/validate_rr_capnobase.py --stride 5.0 --per-record
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rppg import AdvancedRPPG  # noqa: E402

_DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "dataset" / "doi-10.5683-sp2-nlb8it" / "doi-10.5683-sp2-nlb8it" / "data" / "mat"
)

_TARGET_FPS = 30           # deployment-faithful, same as validate_rr_bidmc.py
_EVAL_STRIDE_SEC = 5.0
_RR_CONF_GATE = 30.0       # must match core.clinical.RR_CONFIDENCE_LOW
_WINDOW_S = 60.0           # matches core.rppg._RR_WINDOW_SEC


@dataclass
class Record:
    name: str
    ppg: np.ndarray
    fps: float
    rr_t: np.ndarray     # reference.rr.co2.x, seconds
    rr_v: np.ndarray     # reference.rr.co2.y, br/min
    sf_t: np.ndarray     # SFresults.x, seconds
    sf_v: np.ndarray     # SFresults.SmartFusion.y, br/min (may contain NaN)


def _get(h5, *path):
    node = h5
    for p in path:
        node = node[p]
    return np.array(node).ravel().astype(float)


def load_records(root: Path, limit: int | None = None) -> list:
    files = sorted(root.glob("*_8min.mat"))
    if limit:
        files = files[:limit]
    up, down = _TARGET_FPS, 300
    g = np.gcd(up, down)
    up, down = up // g, down // g

    out = []
    for fp in files:
        try:
            with h5py.File(fp, "r") as h5:
                pleth = _get(h5, "signal", "pleth", "y")
                rr_t = _get(h5, "reference", "rr", "co2", "x")
                rr_v = _get(h5, "reference", "rr", "co2", "y")
                try:
                    sf_t = _get(h5, "SFresults", "x")
                    sf_v = _get(h5, "SFresults", "SmartFusion", "y")
                except Exception:
                    sf_t, sf_v = np.array([]), np.array([])
            ppg = sps.resample_poly(pleth, up, down)
            out.append(Record(fp.stem, ppg, float(_TARGET_FPS), rr_t, rr_v, sf_t, sf_v))
        except Exception as exc:
            print(f"  skip {fp.name}: {exc}")
    return out


def gold_rr(record: Record, t0: float, t1: float) -> float | None:
    """Mean of the reference instantaneous RR trend within [t0, t1).

    Defensive: record 0133's reference.rr.co2.y contains 5 literal
    infinite values (a real artifact in the shipped ground truth, verified
    directly — not a parsing bug here). Any window touching one of those
    samples must not be allowed to poison the aggregate with inf/nan.
    """
    m = (record.rr_t >= t0) & (record.rr_t < t1)
    vals = record.rr_v[m]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return None
    return float(np.mean(vals))


def smartfusion_rr(record: Record, t0: float, t1: float) -> float | None:
    if len(record.sf_t) == 0:
        return None
    m = (record.sf_t >= t0) & (record.sf_t < t1)
    if m.sum() == 0:
        return None
    vals = record.sf_v[m]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def evaluate(record: Record, window_s: float, stride_s: float):
    fps = record.fps
    win_n = int(window_s * fps)
    stride_n = max(1, int(stride_s * fps))
    eng = AdvancedRPPG(fps=int(round(fps)), window_size=int(fps * 10))

    win_errs, sf_errs = [], []
    acc, golds = [], []
    sf_acc, sf_golds = [], []
    n_win = n_sup = 0

    for end in range(win_n, len(record.ppg) + 1, stride_n):
        s = end - win_n
        t0, t1 = s / fps, end / fps
        g = gold_rr(record, t0, t1)
        if g is None:
            continue
        n_win += 1
        rr, conf = eng._estimate_rr(record.ppg[s:end])
        if rr <= 0 or conf < _RR_CONF_GATE:
            n_sup += 1
        else:
            win_errs.append(abs(rr - g))
            acc.append(rr); golds.append(g)

        sf = smartfusion_rr(record, t0, t1)
        if sf is not None:
            sf_errs.append(abs(sf - g))
            sf_acc.append(sf); sf_golds.append(g)

    def _session_err(vals, gold_vals):
        if len(vals) < 3:
            return float("nan")
        a = np.asarray(vals)
        q1, q3 = np.percentile(a, [25, 75]); iqr = q3 - q1
        keep = a[(a >= q1 - 1.5 * iqr) & (a <= q3 + 1.5 * iqr)]
        if len(keep) == 0:
            keep = a
        return abs(float(np.median(keep)) - float(np.median(gold_vals)))

    return {
        "win_errs": win_errs, "sf_errs": sf_errs, "n_win": n_win, "n_sup": n_sup,
        "sess_err": _session_err(acc, golds), "n_accepted": len(acc),
        "sf_sess_err": _session_err(sf_acc, sf_golds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=float, default=_EVAL_STRIDE_SEC)
    ap.add_argument("--window", type=float, default=_WINDOW_S)
    ap.add_argument("--per-record", action="store_true")
    args = ap.parse_args()

    print(f"Loading CapnoBase from {args.root}")
    records = load_records(args.root, args.limit)
    print(f"Loaded {len(records)} records\n")

    all_win, all_sf, all_sess, all_sf_sess = [], [], [], []
    total_win = total_sup = 0
    for r in records:
        res = evaluate(r, args.window, args.stride)
        all_win.extend(res["win_errs"])
        all_sf.extend(res["sf_errs"])
        total_win += res["n_win"]
        total_sup += res["n_sup"]
        if not np.isnan(res["sess_err"]):
            all_sess.append(res["sess_err"])
        if not np.isnan(res["sf_sess_err"]):
            all_sf_sess.append(res["sf_sess_err"])
        if args.per_record:
            print(f"  {r.name:<14} win-MAE {np.mean(res['win_errs']) if res['win_errs'] else float('nan'):6.2f}  "
                  f"session-err {res['sess_err']:6.2f}  n_acc={res['n_accepted']:4d}  "
                  f"supp {100*res['n_sup']/max(1,res['n_win']):5.1f}%")

    win_errs = np.asarray(all_win)
    sf_errs = np.asarray(all_sf)
    sess = np.asarray(all_sess)

    print("\n" + "=" * 70)
    print(f"PRODUCTION estimator (core/rppg.py as it ships) — {len(records)} CapnoBase recordings")
    print(f"  per-window MAE : {np.mean(win_errs):.2f}  RMSE {np.sqrt(np.mean(win_errs**2)):.2f}  "
          f"n={len(win_errs)}  suppressed {100*total_sup/max(1,total_win):.1f}%")
    print(f"  SESSION MAE    : {np.mean(sess):.2f}  median {np.median(sess):.2f}  "
          f"<=2 {100*np.mean(sess<=2):.1f}%  worst {np.max(sess):.2f}  (n={len(sess)})")
    if len(sf_errs):
        sf_sess = np.asarray(all_sf_sess)
        print(f"\nPublished SmartFusion (Karlen et al. 2013), same windows:")
        print(f"  per-window MAE : {np.mean(sf_errs):.2f}  RMSE {np.sqrt(np.mean(sf_errs**2)):.2f}  n={len(sf_errs)}")
        if len(sf_sess):
            print(f"  SESSION MAE    : {np.mean(sf_sess):.2f}  median {np.median(sf_sess):.2f}  "
                  f"<=2 {100*np.mean(sf_sess<=2):.1f}%  (n={len(sf_sess)})  "
                  f"[same IQR-median aggregation applied to SmartFusion's own output, for a fair comparison]")
    print("=" * 70)


if __name__ == "__main__":
    main()
