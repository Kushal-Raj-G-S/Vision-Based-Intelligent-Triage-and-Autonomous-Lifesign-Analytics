"""
HR datasets: session discovery + ground truth for everything under dataset/hr.

    dataset/hr/ubfc_rppg/dataset_1/<clip>/{vid.avi, gtdump.xmp}
    dataset/hr/ubfc_rppg/dataset_2/subjectN/{vid.avi, ground_truth.txt}
    dataset/hr/ubfc_phys/sN/{vid_sN_T{1,2,3}.avi, bvp_sN_T{1,2,3}.csv, ...}

The session-level ground truth deliberately reuses scripts/build_manifest.py's methods, so every number
stays comparable with the original 71-session results (see data/evolution/21-hr-data-audit-and-plan.md):
  dataset_1 / dataset_2: mean of the oximeter's per-sample HR;  UBFC-Phys: median beat interval of the BVP.
`gt_t / gt_hr` is the same ground truth as a 1 Hz series, for per-window labels.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from build_manifest import bpm_from_waveform, _parse_ground_truth_txt, _parse_gtdump_xmp          # noqa: E402
from build_ml_dataset import (_instantaneous_bpm_from_bvp_window, _load_dataset1_gt,              # noqa: E402
                              _load_dataset2_gt)

HR_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "hr"
CONDITION = {"T1": "rest", "T2": "speech", "T3": "arithmetic"}
PHYS_FS = 64.0


def _num(s):
    m = re.search(r"\d+", s)
    return int(m.group()) if m else 0


def discover(root: Path = HR_ROOT):
    """Yield one dict per usable session: id, video, dataset, subject, condition, gt_session, gt_kind."""
    out = []
    d1 = root / "ubfc_rppg" / "dataset_1"
    for clip in sorted(p for p in d1.glob("*") if p.is_dir()) if d1.is_dir() else []:
        v, g = clip / "vid.avi", clip / "gtdump.xmp"
        if v.is_file() and g.is_file():
            out.append(dict(id=f"d1_{clip.name}", video=v, dataset="ubfc_rppg_d1", subject=f"d1_{clip.name}",
                            condition="exercise" if "exercise" in clip.name else "rest", gt_path=g, gt_kind="d1"))
    d2 = root / "ubfc_rppg" / "dataset_2"
    for sub in sorted((p for p in d2.glob("subject*") if p.is_dir()), key=lambda p: _num(p.name)) if d2.is_dir() else []:
        v, g = sub / "vid.avi", sub / "ground_truth.txt"
        if v.is_file() and g.is_file():
            out.append(dict(id=f"d2_{sub.name}", video=v, dataset="ubfc_rppg_d2", subject=sub.name,
                            condition="rest", gt_path=g, gt_kind="d2"))
    ph = root / "ubfc_phys"
    for sub in sorted((p for p in ph.glob("s*") if p.is_dir()), key=lambda p: _num(p.name)) if ph.is_dir() else []:
        for t in ("T1", "T2", "T3"):
            v, b = sub / f"vid_{sub.name}_{t}.avi", sub / f"bvp_{sub.name}_{t}.csv"
            if v.is_file() and b.is_file():
                out.append(dict(id=f"ph_{sub.name}_{t}", video=v, dataset="ubfc_phys", subject=sub.name,
                                condition=CONDITION[t], gt_path=b, gt_kind="phys"))
    return out


def session_gt(s):
    """Whole-session ground-truth BPM, by the same method as the original manifest."""
    if s["gt_kind"] == "d2":
        return _parse_ground_truth_txt(str(s["gt_path"]))
    if s["gt_kind"] == "d1":
        return _parse_gtdump_xmp(str(s["gt_path"]))
    vals = [float(x) for x in open(s["gt_path"], encoding="utf-8", errors="ignore").read().split()]
    return bpm_from_waveform(vals, fs=PHYS_FS)


def gt_series(s, duration_s: float, step_s: float = 1.0):
    """Ground-truth HR on a regular grid (t seconds, bpm); NaN where unavailable."""
    t = np.arange(0.0, duration_s, step_s)
    if s["gt_kind"] == "d2":
        ts, hr = _load_dataset2_gt(str(s["gt_path"]))
        return t, np.interp(t, ts, hr)
    if s["gt_kind"] == "d1":
        ts, hr = _load_dataset1_gt(str(s["gt_path"]))
        return t, np.interp(t, ts, hr)
    vals = [float(x) for x in open(s["gt_path"], encoding="utf-8", errors="ignore").read().split()]
    hr = np.array([(_instantaneous_bpm_from_bvp_window(vals, PHYS_FS, int(x * PHYS_FS)) or np.nan) for x in t])
    return t, hr


UNRELIABLE_BAD_FRAC = 0.10        # fixed in advance, from the reference device only (never from our estimates)


def reference_info(s, duration_s=None):
    """Ground-truth variants for one session.
      old      : the original definition (mean of the oximeter HR column; Phys: median beat interval of the BVP)
      robust   : median of the VALID oximeter readings (40-180 BPM). The oximeter drops out in some UBFC-rPPG sessions
                 (readings of 0 or 7 BPM), which corrupts a mean. Phys is already robust, so robust == old there.
      bad_frac : share of impossible readings; > UNRELIABLE_BAD_FRAC marks the reference itself as unreliable
      t, hr    : cleaned HR series (invalid readings = NaN) for per-window labels
    """
    if s["gt_kind"] in ("d1", "d2"):
        if s["gt_kind"] == "d2":
            ts, hr = _load_dataset2_gt(str(s["gt_path"]))
        else:
            ts, hr = _load_dataset1_gt(str(s["gt_path"]))
        hr = np.asarray(hr, float)
        ok = (hr >= 40) & (hr <= 180)
        robust = float(np.median(hr[ok])) if ok.any() else float("nan")
        return dict(old=float(hr.mean()), robust=robust, bad_frac=float(1 - ok.mean()),
                    unreliable=bool(1 - ok.mean() > UNRELIABLE_BAD_FRAC), t=np.asarray(ts, float), hr=np.where(ok, hr, np.nan))
    g = session_gt(s)
    return dict(old=g, robust=g, bad_frac=0.0, unreliable=False, t=None, hr=None)


def clean_series_at(ref, t_query, max_gap_s=3.0):
    """Cleaned reference HR interpolated at t_query; NaN where no valid reading lies within max_gap_s."""
    ok = np.isfinite(ref["hr"])
    if not ok.any():
        return np.full(len(t_query), np.nan)
    tv, hv = ref["t"][ok], ref["hr"][ok]
    out = np.interp(t_query, tv, hv)
    near = np.array([np.min(np.abs(tv - q)) for q in t_query]) <= max_gap_s
    return np.where(near, out, np.nan)


if __name__ == "__main__":
    ss = discover()
    print(len(ss), "sessions;", len({s["subject"] for s in ss}), "distinct subject ids")
    for ds in sorted({s["dataset"] for s in ss}):
        sub = [s for s in ss if s["dataset"] == ds]
        print(f"  {ds}: {len(sub)} sessions, {len({s['subject'] for s in sub})} subjects")
