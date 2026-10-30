"""Build the held-out VitalDB window cache (same row format as cache.pkl)."""
import gzip
import io
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter

ROOT = Path(r"C:\Users\M.Manoj\OneDrive\Desktop\vision 1")
sys.path.insert(0, str(ROOT))
from core.rppg import AdvancedRPPG, _RIV_FS, _RIIV_DEDRIFT_WINDOW_SEC  # noqa: E402

OUT = Path(sys.argv[1])
TR = ROOT / "dataset/rr/vitaldb/tracks"
FPS = 30
SEG_S = 600.0


def pre(grid, t, series, dedrift):
    y = CubicSpline(t, series)(grid) if len(t) >= 4 else np.interp(grid, t, series)
    if dedrift:
        win = int(_RIIV_DEDRIFT_WINDOW_SEC * _RIV_FS); win += 1 - (win % 2)
        if win >= 3 and len(y) > win:
            y = y - median_filter(y, size=win, mode="nearest")
    return signal.detrend(y, type="linear").astype(np.float32)


def read(path):
    return pd.read_csv(io.BytesIO(gzip.decompress(open(path, "rb").read())))


rows = []
SEL = json.load(open(ROOT / "dataset/rr/vitaldb/selected_age.json"))
for case in sorted(SEL, key=int):
    try:
        rr = read(TR / f"{case}_rr.csv.gz"); rr.columns = ["t", "v"]
        rr = rr[np.isfinite(rr.v) & np.isfinite(rr.t)]
        rr = rr[(rr.v >= 5) & (rr.v <= 45)]
        if len(rr) < 120:
            print(case, "skip: too little valid RR"); continue
        pl = read(TR / f"{case}_pleth.csv.gz"); pl.columns = ["t", "v"]
        # VitalDB writes the timestamp only on the first two rows (t0, t0+dt);
        # every later row has a blank Time and a uniform sample spacing.
        t0_, t1_ = float(pl.t.iloc[0]), float(pl.t.iloc[1])
        dt_ = t1_ - t0_
        tt_all = t0_ + dt_ * np.arange(len(pl))
        okv = np.isfinite(pl.v.values)
        t = tt_all[okv]; v = pl.v.values[okv]
        # best 600 s block: high valid-PPG fraction, RR present throughout
        best = None
        for s in np.arange(rr.t.min(), rr.t.max() - SEG_S, 120.0):
            nrr = ((rr.t >= s) & (rr.t < s + SEG_S)).sum()
            if nrr < 0.8 * SEG_S / 2.0:
                continue
            m = (t >= s) & (t < s + SEG_S)
            frac = m.sum() / (SEG_S * 500)
            if best is None or frac > best[0]:
                best = (frac, s)
        if best is None or best[0] < 0.9:
            print(case, "skip: no clean block", None if best is None else round(best[0], 2)); continue
        s0 = best[1]
        m = (t >= s0) & (t < s0 + SEG_S)
        grid = np.arange(s0, s0 + SEG_S, 1 / 500.0)
        x = np.interp(grid, t[m], v[m])
        ppg = signal.resample_poly(x, 3, 50)             # 500 Hz -> 30 Hz
        eng = AdvancedRPPG(fps=FPS, window_size=FPS * 10)
        win_n, stride_n = int(60 * FPS), int(5 * FPS)
        nrows = 0
        for end in range(win_n, len(ppg) + 1, stride_n):
            a = end - win_n
            w0, w1 = s0 + a / FPS, s0 + end / FPS
            mm = (rr.t >= w0) & (rr.t < w1)
            vals = rr.v[mm].values
            if len(vals) < 15 or np.std(vals) > 3.0:
                continue
            gold = float(np.mean(vals))
            seg = ppg[a:end]
            est, conf = eng._estimate_rr(seg)
            row = dict(ds="vdb_" + SEL[case]["band"], rec=f"vdb_{case}", t0=a / FPS, gold=gold, est=est, conf=conf, y=None)
            got = eng._extract_rivs(seg)
            if got is not None:
                tt, riav, riiv, rifv = got
                g = np.arange(tt[0], tt[-1], 1.0 / _RIV_FS)
                if len(g) >= 16:
                    span = float(tt[-1] - tt[0])
                    row["y"] = dict(riav=pre(g, tt, riav, False), riiv=pre(g, tt, riiv, True),
                                    rifv=pre(g, tt, rifv, False))
                    row["beat_hz"] = len(tt) / span if span > 0 else 0.0
            rows.append(row); nrows += 1
        print(f"{case}: block@{s0:.0f}s frac={best[0]:.2f} windows={nrows}", flush=True)
    except Exception as exc:
        print(case, "ERROR", exc, flush=True)
pickle.dump(rows, open(OUT, "wb"))
print("DONE", len(rows), "windows ->", OUT)
