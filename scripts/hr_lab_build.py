#!/usr/bin/env python3
"""
Turn cached ROI traces into the HR "lab" feature tables (one .npz per session in dataset/cache/feat).

For every DSP cycle (frame) of each session, with CHROM and POS run side by side on the identical frames:
  scalar features (the 16 the old correction model used + extras), both algorithms' lock-independent wide-band
  spectra (116 bins, 0.70-3.00 Hz), the ground-truth HR at that moment, and the session-level classical answers.
Everything downstream (new models, peak selection, tracking, ablations) works from these tables in seconds.

    python scripts/hr_lab_build.py --workers 6        # resumable; re-run after more sessions finish caching
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hr_replay as R                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "dataset" / "cache"
SCALARS = ["bpm", "sqi", "snr_db", "stability", "harmonic_ratio", "filt_low", "filt_high", "motion", "rr", "rr_conf", "hrv", "stress"]


RGB_MODE = "prod"        # which per-frame colour to use; see scripts/hr_roi_variants.py (needs a landmark cache)


def build_one(args):
    src, dst, rgb_mode = args
    t0 = time.time()
    try:
        c = R.load(src)
        key = "rgb"
        if rgb_mode != "prod":
            from hr_roi_variants import variant_rgb
            c["valid"] = c["valid"] & np.isfinite(c["parts_clean"][:, 0, 0])
            c["rgb_v"] = variant_rgb(c, rgb_mode)
            key = "rgb_v"
        ch = R.replay(c, "CHROM", capture_spectra=True, rgb_key=key)
        po = R.replay(c, "POS", capture_spectra=True, rgb_key=key)
        fps = c["meta"]["fps"]
        # align POS to CHROM by frame index (both run every frame once ready, so the frame sets are identical in
        # practice; guard anyway)
        pf = {r["frame"]: i for i, r in enumerate(po["readings"])}
        keep = [i for i, r in enumerate(ch["readings"]) if r["frame"] in pf]
        frames = np.array([ch["readings"][i]["frame"] for i in keep])
        t = frames / fps
        gt = np.interp(t, c["gt_t"], np.nan_to_num(c["gt_hr"], nan=np.nan))
        out = dict(
            meta=np.array(str(c["meta"])), frame=frames, t=t.astype(np.float32), gt=gt.astype(np.float32),
            chrom_spec=ch["spectra"][keep], pos_spec=po["spectra"][[pf[ch["readings"][i]["frame"]] for i in keep]],
            pos_bpm=np.array([po["readings"][pf[ch["readings"][i]["frame"]]]["bpm"] for i in keep], np.float32),
            pos_sqi=np.array([po["readings"][pf[ch["readings"][i]["frame"]]]["sqi"] for i in keep], np.float32),
            final_chrom=np.float32(ch["final_bpm"] if ch["final_bpm"] is not None else np.nan),
            final_pos=np.float32(po["final_bpm"] if po["final_bpm"] is not None else np.nan),
            gt_session=np.float32(c["meta"]["gt_session"]),
            mouth_open=c["mouth_open"][frames], motion_frame=c["motion"][frames],
        )
        for k in SCALARS:
            out[k] = np.array([ch["readings"][i][k] for i in keep], np.float32)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **out)
        os.replace(tmp, dst)
        return src.stem, True, f"{len(keep)} cycles, chrom {ch['final_bpm']:.1f} pos {po['final_bpm']:.1f} gt {c['meta']['gt_session']:.1f}", time.time() - t0
    except Exception as exc:
        return src.stem, False, repr(exc)[:160], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--roi", default=str(ROOT / "roi"))
    ap.add_argument("--out", default=str(ROOT / "feat"))
    ap.add_argument("--rgb", default="prod", choices=["prod", "clean", "forehead", "cheeks", "talk_forehead"],
                    help="per-frame colour source; anything but prod needs per-region values (landmark cache)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, out / p.name, args.rgb) for p in sorted(Path(args.roi).glob("*.npz")) if ".tmp" not in p.name and not (out / p.name).exists()]
    print(f"{len(jobs)} sessions to build with {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        for n, (name, ok, msg, dt) in enumerate(pool.imap_unordered(build_one, jobs), 1):
            print(f"[{n}/{len(jobs)}] {'OK  ' if ok else 'FAIL'} {name:22s} {msg}  ({dt:.0f}s)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
