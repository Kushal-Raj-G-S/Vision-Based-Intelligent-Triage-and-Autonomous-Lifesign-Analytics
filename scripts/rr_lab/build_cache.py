"""Cache per-window RIV series (pre-bandpass) for BIDMC + CapnoBase so RR
peak-selection rules can be tested in seconds instead of re-running beat
detection for every experiment."""
import pickle
import sys
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter

ROOT = Path(r"C:\Users\M.Manoj\OneDrive\Desktop\vision 1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_rr_bidmc as vb  # noqa: E402
import validate_rr_capnobase as vc  # noqa: E402
from core.rppg import AdvancedRPPG, _RIV_FS, _RIIV_DEDRIFT_WINDOW_SEC  # noqa: E402

OUT = Path(sys.argv[1])
BIDMC = ROOT / "dataset/rr/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv"
CAPNO = ROOT / "dataset/rr/doi-10.5683-sp2-nlb8it/doi-10.5683-sp2-nlb8it/data/mat"


def pre(eng, grid, t, series, dedrift):
    if len(t) >= 4:
        try:
            y = CubicSpline(t, series)(grid)
        except Exception:
            y = np.interp(grid, t, series)
    else:
        y = np.interp(grid, t, series)
    if dedrift:
        win = int(_RIIV_DEDRIFT_WINDOW_SEC * _RIV_FS)
        win += 1 - (win % 2)
        if win >= 3 and len(y) > win:
            y = y - median_filter(y, size=win, mode="nearest")
    try:
        y = signal.detrend(y, type="linear")
    except Exception:
        y = y - float(np.mean(y))
    return y.astype(np.float32)


def run(ds, records, gold_fn, stride_s, window_s=60.0):
    rows = []
    for rec in records:
        fps = rec.fps
        win_n = int(round(window_s * fps))
        stride_n = max(1, int(round(stride_s * fps)))
        eng = AdvancedRPPG(fps=int(round(fps)), window_size=int(fps * 10))
        for end in range(win_n, len(rec.ppg) + 1, stride_n):
            s = end - win_n
            g = gold_fn(rec, s / fps, end / fps)
            if g is None:
                continue
            seg = rec.ppg[s:end]
            est, conf = eng._estimate_rr(seg)
            row = dict(ds=ds, rec=rec.name, t0=s / fps, gold=g, est=est, conf=conf, y=None)
            got = eng._extract_rivs(seg)
            if got is not None:
                t, riav, riiv, rifv = got
                grid = np.arange(t[0], t[-1], 1.0 / _RIV_FS)
                if len(grid) >= 16:
                    span = float(t[-1] - t[0])
                    row["y"] = dict(
                        riav=pre(eng, grid, t, riav, False),
                        riiv=pre(eng, grid, t, riiv, True),
                        rifv=pre(eng, grid, t, rifv, False),
                    )
                    row["beat_hz"] = len(t) / span if span > 0 else 0.0
            rows.append(row)
        print(f"{ds} {rec.name}: {sum(1 for r in rows if r['rec']==rec.name)} windows", flush=True)
    return rows


rows = []
rows += run("capno", vc.load_records(CAPNO), vc.gold_rr, 5.0)
rows += run("bidmc", vb.load_records(BIDMC), vb.gold_rr, 3.0)
with open(OUT, "wb") as f:
    pickle.dump(rows, f)
print("DONE", len(rows), "windows ->", OUT)
