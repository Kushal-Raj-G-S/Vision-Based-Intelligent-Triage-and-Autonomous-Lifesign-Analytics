#!/usr/bin/env python3
"""
VITAL — Multi-Dataset Manifest Builder
=========================================
Scans a downloaded dataset folder tree and emits a manifest.csv compatible
with `scripts/validate_dataset.py --manifest`, containing one row per
session that has BOTH a video file and a ground-truth file. Sessions
missing either are skipped and reported — a partial/in-progress dataset
download (some subjects with video but no ground truth, or vice versa) is
expected, not an error.

Ground-truth formats handled
-----------------------------
UBFC-rPPG "Dataset 1" (gtdump.xmp): comma-separated rows of
    [timestamp_ms, HR_bpm, SpO2, ppg_signal]
    gt_bpm = mean of column 2 (HR) across the file.

UBFC-rPPG "Dataset 2" (ground_truth.txt): 3 space-separated numeric rows —
    line 1 = BVP waveform, line 2 = per-sample HR (bpm), line 3 = timestamps.
    gt_bpm = mean of line 2.

UBFC-Phys (bvp_<subject>_<task>.csv alongside vid_<subject>_<task>.avi):
    single-column raw BVP waveform, no header/timestamps. Sample rate is NOT
    given in the file — empirically confirmed at 64 Hz by cross-checking
    sample count against matching video duration (11520 samples / 180.03s
    video = 63.99 Hz). gt_bpm is derived by bandpass-filtering + peak-picking
    the raw waveform (this dataset ships a BVP *signal*, not a pre-computed
    per-sample HR series like UBFC-rPPG's ground_truth.txt does).

Note: rPPG-10 (ECG .npy ground truth) is NOT handled here — its videos are
pre-cropped 64x64 skin patches with no visible face, incompatible with this
project's MediaPipe-based face/ROI detection. See scripts/validate_cropped_roi.py
for that dataset's separate DSP-only validation path.

Usage
-----
    python scripts/build_manifest.py test/ --out test/manifest.csv
    python scripts/validate_dataset.py --manifest test/manifest.csv
"""

import argparse
import csv
import os
import re
import statistics

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".webm")

_UBFC_PHYS_BVP_RE = re.compile(r"^bvp_(?P<subject>.+)_(?P<task>T\d+)\.csv$", re.IGNORECASE)


def bpm_from_waveform(values, fs, low_hz=0.7, high_hz=3.0, min_bpm=40.0, max_bpm=180.0):
    """
    Estimate a representative BPM from a raw physiological waveform (BVP or
    ECG) via bandpass filtering + peak detection. Used for ground-truth
    sources that ship a raw signal rather than a pre-computed per-sample HR
    series (unlike UBFC-rPPG's ground_truth.txt, which already gives HR
    directly). Mirrors the peak-detection approach already used for HRV in
    core/rppg.py — same 400ms refractory-period assumption.
    """
    import numpy as np
    from scipy import signal as sig

    x = np.asarray(values, dtype=np.float64)
    if len(x) < fs * 3:
        return None
    x = sig.detrend(x)
    try:
        sos = sig.butter(3, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
        filtered = sig.sosfiltfilt(sos, x)
    except Exception:
        filtered = x

    min_dist = max(1, int(fs * 0.4))
    std = float(np.std(filtered))
    if std < 1e-9:
        return None
    peaks, _ = sig.find_peaks(filtered, distance=min_dist, prominence=std * 0.3)
    if len(peaks) < 3:
        return None

    ibis = np.diff(peaks) / fs
    ibis = ibis[(ibis > 60.0 / max_bpm) & (ibis < 60.0 / min_bpm)]
    if len(ibis) < 2:
        return None
    return float(60.0 / np.median(ibis))


def _find_video(folder, filenames):
    for fname in sorted(filenames):
        if fname.lower().endswith(VIDEO_EXTS):
            return os.path.join(folder, fname)
    return None


def _parse_gtdump_xmp(path):
    """Dataset 1 format: rows of timestamp_ms,HR,SpO2,ppg_signal."""
    hrs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                hrs.append(float(parts[1]))
            except ValueError:
                continue
    return statistics.mean(hrs) if hrs else None


def _parse_ground_truth_txt(path):
    """Dataset 2 format: line1=BVP, line2=HR per sample, line3=timestamps."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    try:
        hrs = [float(x) for x in lines[1].split()]
    except ValueError:
        return None
    return statistics.mean(hrs) if hrs else None


def _scan_ubfc_phys_tasks(dirpath, filenames):
    """
    UBFC-Phys stores 3 tasks (T1=rest, T2=speech, T3=arithmetic) per subject
    as separate bvp_<subj>_<task>.csv / vid_<subj>_<task>.avi pairs in the
    same folder — handled separately from the single-file-per-folder scan
    below since one folder can yield up to 3 manifest rows.
    """
    found = []
    for fname in filenames:
        m = _UBFC_PHYS_BVP_RE.match(fname)
        if not m:
            continue
        subject, task = m.group("subject"), m.group("task")
        vid_name = f"vid_{subject}_{task}.avi"
        if vid_name not in filenames:
            found.append((None, fname, "video missing for this task"))
            continue
        found.append((os.path.join(dirpath, vid_name), fname, None))
    return found


def build_manifest(root, out_path):
    rows = []
    skipped = []

    for dirpath, _dirnames, filenames in os.walk(root):
        # UBFC-Phys: multiple task sessions can live in one folder.
        phys_tasks = _scan_ubfc_phys_tasks(dirpath, filenames)
        if phys_tasks:
            session_name = os.path.relpath(dirpath, root)
            for video_path, bvp_fname, problem in phys_tasks:
                task_label = _UBFC_PHYS_BVP_RE.match(bvp_fname).group("task")
                if problem:
                    skipped.append((f"{session_name}/{bvp_fname}", problem))
                    continue
                bvp_path = os.path.join(dirpath, bvp_fname)
                try:
                    with open(bvp_path, "r", encoding="utf-8", errors="ignore") as f:
                        values = [float(line.strip()) for line in f if line.strip()]
                except Exception:
                    skipped.append((f"{session_name}/{bvp_fname}", "BVP file unreadable"))
                    continue
                gt_bpm = bpm_from_waveform(values, fs=64.0)
                if gt_bpm is None:
                    skipped.append((f"{session_name}/{bvp_fname}", "BVP peak-detection failed (no stable rate found)"))
                    continue
                rows.append({
                    "video": video_path.replace("\\", "/"),
                    "gt_bpm": round(gt_bpm, 1),
                    "skin_tone": "",
                    "motion": {"T1": "rest", "T2": "speech", "T3": "arithmetic"}.get(task_label, task_label),
                    "notes": f"ubfc_phys:{session_name}:{task_label}",
                })
            continue  # this folder was UBFC-Phys — don't also try the single-file formats below

        has_video = any(f.lower().endswith(VIDEO_EXTS) for f in filenames)
        gt_path = gt_kind = None
        if "ground_truth.txt" in filenames:
            gt_path, gt_kind = os.path.join(dirpath, "ground_truth.txt"), "dataset_2"
        elif "gtdump.xmp" in filenames:
            gt_path, gt_kind = os.path.join(dirpath, "gtdump.xmp"), "dataset_1"

        if gt_path is None and not has_video:
            continue

        session_name = os.path.relpath(dirpath, root)

        if gt_path is None:
            skipped.append((session_name, "video present, ground truth missing"))
            continue

        video_path = _find_video(dirpath, filenames)
        if video_path is None:
            skipped.append((session_name, "ground truth present, video missing"))
            continue

        gt_bpm = (
            _parse_ground_truth_txt(gt_path) if gt_kind == "dataset_2"
            else _parse_gtdump_xmp(gt_path)
        )
        if gt_bpm is None:
            skipped.append((session_name, "ground truth file unreadable/empty"))
            continue

        rows.append({
            "video": video_path.replace("\\", "/"),
            "gt_bpm": round(gt_bpm, 1),
            "skin_tone": "",
            "motion": "",
            "notes": f"{gt_kind}:{session_name}",
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "gt_bpm", "skin_tone", "motion", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    return rows, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a validate_dataset.py manifest from a UBFC-rPPG folder tree."
    )
    parser.add_argument("root", help="Root folder to scan (e.g. test/)")
    parser.add_argument("--out", default="manifest.csv", help="Output manifest CSV path")
    args = parser.parse_args()

    rows, skipped = build_manifest(args.root, args.out)

    print(f"Usable sessions (video + ground truth): {len(rows)}")
    for r in rows:
        print(f"  {r['video']}  gt_bpm={r['gt_bpm']}  ({r['notes']})")

    print(f"\nSkipped sessions (incomplete download): {len(skipped)}")
    for name, reason in skipped:
        print(f"  {name}: {reason}")

    print(f"\nManifest written to {args.out}")
