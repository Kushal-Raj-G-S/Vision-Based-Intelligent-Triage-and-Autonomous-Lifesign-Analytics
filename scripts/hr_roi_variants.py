#!/usr/bin/env python3
"""
ROI pooling variants, replayed from a landmark ROI cache (dataset/cache/roi_mp, built with --tracker tasks).

Every variant builds a per-frame (r, g, b) from the recorded per-region means and runs the production CHROM replay.
  prod          : the camera's own pooled value (0.5 forehead + 0.25 + 0.25 cheeks, measured after the debug outline)
  clean         : same weights, means taken BEFORE the outline is drawn on the ROI border
  forehead      : forehead only (clean)
  cheeks        : mean of the two cheeks only (clean)
  talk_forehead : clean pooled, but forehead only on frames where the mouth is open (speech motion lives in the cheeks)

    python scripts/hr_roi_variants.py [session ids ...]          # default: every session in the landmark cache
    python scripts/hr_roi_variants.py --workers 12               # parallel over sessions
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hr_data as H          # noqa: E402
import hr_replay as R        # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "dataset" / "cache"
W = np.array([0.5, 0.25, 0.25], np.float32)
MOUTH_OPEN = 0.05          # inner-lip gap / inter-ocular distance; closed mouth ~0.00-0.02 (fixed in advance)
VARIANTS = ("prod", "clean", "forehead", "cheeks", "talk_forehead")


def variant_rgb(c, name):
    pp, pc = c["parts_prod"], c["parts_clean"]            # (frames, [fh, lc, rc], [b, g, r])
    if name == "prod":
        return c["rgb"]
    if name == "clean":
        bgr = (pc * W[None, :, None]).sum(1)
    elif name == "forehead":
        bgr = pc[:, 0, :]
    elif name == "cheeks":
        bgr = (pc[:, 1, :] + pc[:, 2, :]) / 2.0
    elif name == "talk_forehead":
        pooled = (pc * W[None, :, None]).sum(1)
        talking = np.nan_to_num(c["mouth_open"], nan=0.0) > MOUTH_OPEN
        bgr = np.where(talking[:, None], pc[:, 0, :], pooled)
    else:
        raise ValueError(name)
    return bgr[:, ::-1].astype(np.float32)                 # -> r, g, b


def run_session(path):
    c = R.load(path)
    ok = c["valid"] & np.isfinite(c["parts_clean"][:, 0, 0])
    c["valid"] = ok
    res = {}
    for v in VARIANTS:
        c2 = dict(c)
        c2["rgb_v"] = variant_rgb(c, v)
        res[v] = R.replay(c2, rgb_key="rgb_v")["final_bpm"]
    talk_share = float(np.mean(np.nan_to_num(c["mouth_open"], nan=0.0) > MOUTH_OPEN))
    return Path(path).stem, res, talk_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", default=str(CACHE / "roi_mp"))
    args = ap.parse_args()
    files = sorted(p for p in Path(args.cache).glob("*.npz") if ".tmp" not in p.name)
    if args.ids:
        files = [p for p in files if p.stem in args.ids]
    by = {s["id"]: s for s in H.discover()}
    with Pool(min(args.workers, max(1, len(files)))) as pool:
        out = pool.map(run_session, [str(p) for p in files])
    errs = {v: [] for v in VARIANTS}
    print(f"{'session':22s} {'truth':>6s} " + " ".join(f"{v:>13s}" for v in VARIANTS) + "  talking")
    for sid, res, talk in out:
        truth = H.reference_info(by[sid])["robust"]
        cells = []
        for v in VARIANTS:
            e = abs(res[v] - truth) if res[v] is not None else np.nan
            errs[v].append(e)
            cells.append(f"{res[v]:6.1f} ({e:4.1f})" if res[v] is not None else "      n/a    ")
        print(f"{sid:22s} {truth:6.1f} " + " ".join(cells) + f"  {talk*100:4.0f}%")
    print("\nMAE per variant (classical CHROM replay, landmark ROIs, cleaned truth):")
    for v in VARIANTS:
        e = np.array(errs[v], float)
        print(f"  {v:14s} MAE {np.nanmean(e):6.2f}  median {np.nanmedian(e):5.2f}  >20 BPM: {int(np.nansum(e > 20))}/{np.isfinite(e).sum()}")


if __name__ == "__main__":
    main()
