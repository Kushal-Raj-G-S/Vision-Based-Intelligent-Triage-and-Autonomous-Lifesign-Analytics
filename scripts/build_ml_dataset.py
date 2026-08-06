#!/usr/bin/env python3
"""
VITAL — Windowed ML Correction Dataset Builder
================================================
The DSP pipeline (core/rppg.py) reports one final BPM per SESSION when
validated via scripts/validate_dataset.py — 56 sessions, 56 data points.
Too few to train anything on honestly.

But UBFC-rPPG and UBFC-Phys ship dense, per-frame or near-per-frame
ground-truth HR, not just one number per session:
  - dataset_2 (ground_truth.txt): a per-SAMPLE HR value, one per video
    frame, with explicit timestamps — genuinely instantaneous ground truth.
  - dataset_1 (gtdump.xmp): [timestamp_ms, HR, SpO2, ppg] rows — same idea.
  - UBFC-Phys (bvp_<subj>_<task>.csv): raw 64Hz BVP waveform, no
    precomputed HR — instantaneous HR is derived per-window via the same
    bandpass+peak-detection approach build_manifest.py already uses for
    the session-level number, just applied locally around each timestamp
    instead of over the whole file.

Running the real DSP pipeline (AdvancedRPPG) over each video and pairing
every process_ppg_signal() cycle's output against the ground-truth HR at
that exact moment in time turns "56 sessions" into thousands of labeled
(features -> true instantaneous HR) examples — enough to train a small
correction model on, honestly, IF the eventual train/test split is done
by SUBJECT (not by row), since consecutive rows from the same session are
highly correlated and would leak across a random split otherwise.

Output: a CSV with one row per DSP cycle, feature columns, gt_hr label,
and a subject_id column for grouped cross-validation.

Usage:
    python scripts/build_ml_dataset.py test/ --out test/ml_dataset.csv
"""
import argparse
import csv
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy import signal as sig

from core.camera import Camera
from core.rppg import AdvancedRPPG

_UBFC_PHYS_BVP_RE = re.compile(r"^bvp_(?P<subject>.+)_(?P<task>T\d+)\.csv$", re.IGNORECASE)

FEATURE_COLUMNS = [
    "bpm", "sqi", "snr_db", "stability_score", "rr", "rr_confidence",
    "hrv", "stress_index", "filt_low", "filt_high",
]


def _instantaneous_bpm_from_bvp_window(values, fs, center_idx, half_window_sec=5.0,
                                        low_hz=0.7, high_hz=3.0, min_bpm=40.0, max_bpm=180.0):
    """Same peak-detection approach as build_manifest.py's bpm_from_waveform,
    applied to a local window around `center_idx` instead of the whole file,
    to get an instantaneous rather than whole-session HR estimate."""
    half_n = int(fs * half_window_sec)
    lo = max(0, center_idx - half_n)
    hi = min(len(values), center_idx + half_n)
    window = np.asarray(values[lo:hi], dtype=np.float64)
    if len(window) < fs * 3:
        return None
    window = sig.detrend(window)
    try:
        sos = sig.butter(3, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
        filtered = sig.sosfiltfilt(sos, window)
    except Exception:
        filtered = window
    std = float(np.std(filtered))
    if std < 1e-9:
        return None
    peaks, _ = sig.find_peaks(filtered, distance=max(1, int(fs * 0.4)), prominence=std * 0.3)
    if len(peaks) < 3:
        return None
    ibis = np.diff(peaks) / fs
    ibis = ibis[(ibis > 60.0 / max_bpm) & (ibis < 60.0 / min_bpm)]
    if len(ibis) < 2:
        return None
    return float(60.0 / np.median(ibis))


def _load_dataset2_gt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    hr = np.array([float(x) for x in lines[1].split()])
    ts = np.array([float(x) for x in lines[2].split()])
    return ts, hr


def _load_dataset1_gt(path):
    ts, hr = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                ts.append(float(parts[0]) / 1000.0)  # ms -> s
                hr.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(ts), np.array(hr)


def _process_video_dense_gt(video_path, gt_timestamps_sec, gt_hr, subject_id, group, rows_out):
    camera = Camera(source=video_path)
    if camera.dummy_mode:
        print(f"  [skip] could not open {video_path}")
        return 0
    fps = camera.get_source_fps(default=30.0)
    rppg = AdvancedRPPG(fps=int(round(fps)), window_size=max(300, int(fps * 10)))

    n_added = 0
    frame_idx = 0
    MAX_FRAMES = 100_000
    while frame_idx < MAX_FRAMES:
        frame_bytes, roi_data, _is_moving, _motion = camera.get_frame()
        if frame_bytes is None:
            break
        if roi_data is not None:
            r, g, b = roi_data
            rppg.add_frame((r, g, b))
            result = rppg.process_ppg_signal()
            if result.get("ready") and result.get("bpm", 0) > 0:
                t_sec = frame_idx / fps
                gt_now = float(np.interp(t_sec, gt_timestamps_sec, gt_hr))
                if 40.0 <= gt_now <= 180.0:
                    rows_out.append({
                        "bpm": result["bpm"], "sqi": result.get("sqi", 0.0),
                        "snr_db": result.get("snr_db", 0.0),
                        "stability_score": result.get("stability_score", 0.0),
                        "rr": result.get("rr", 0.0), "rr_confidence": result.get("rr_confidence", 0.0),
                        "hrv": result.get("hrv", 0.0), "stress_index": result.get("stress_index", 0.0),
                        "filt_low": result.get("filter_band_hz", [0, 0])[0],
                        "filt_high": result.get("filter_band_hz", [0, 0])[1],
                        "gt_hr": gt_now, "subject_id": subject_id, "group": group,
                    })
                    n_added += 1
        frame_idx += 1
    return n_added


def _process_video_bvp_gt(video_path, bvp_values, bvp_fs, subject_id, group, rows_out):
    camera = Camera(source=video_path)
    if camera.dummy_mode:
        print(f"  [skip] could not open {video_path}")
        return 0
    fps = camera.get_source_fps(default=30.0)
    rppg = AdvancedRPPG(fps=int(round(fps)), window_size=max(300, int(fps * 10)))

    n_added = 0
    frame_idx = 0
    MAX_FRAMES = 100_000
    while frame_idx < MAX_FRAMES:
        frame_bytes, roi_data, _is_moving, _motion = camera.get_frame()
        if frame_bytes is None:
            break
        if roi_data is not None:
            r, g, b = roi_data
            rppg.add_frame((r, g, b))
            result = rppg.process_ppg_signal()
            if result.get("ready") and result.get("bpm", 0) > 0:
                t_sec = frame_idx / fps
                center_idx = int(t_sec * bvp_fs)
                gt_now = _instantaneous_bpm_from_bvp_window(bvp_values, bvp_fs, center_idx)
                if gt_now is not None and 40.0 <= gt_now <= 180.0:
                    rows_out.append({
                        "bpm": result["bpm"], "sqi": result.get("sqi", 0.0),
                        "snr_db": result.get("snr_db", 0.0),
                        "stability_score": result.get("stability_score", 0.0),
                        "rr": result.get("rr", 0.0), "rr_confidence": result.get("rr_confidence", 0.0),
                        "hrv": result.get("hrv", 0.0), "stress_index": result.get("stress_index", 0.0),
                        "filt_low": result.get("filter_band_hz", [0, 0])[0],
                        "filt_high": result.get("filter_band_hz", [0, 0])[1],
                        "gt_hr": gt_now, "subject_id": subject_id, "group": group,
                    })
                    n_added += 1
        frame_idx += 1
    return n_added


def build_dataset(root, out_path):
    rows = []

    for dirpath, _dirnames, filenames in os.walk(root):
        phys_bvps = [f for f in filenames if _UBFC_PHYS_BVP_RE.match(f)]
        if phys_bvps:
            for bvp_fname in sorted(phys_bvps):
                m = _UBFC_PHYS_BVP_RE.match(bvp_fname)
                subject, task = m.group("subject"), m.group("task")
                vid_name = f"vid_{subject}_{task}.avi"
                video_path = os.path.join(dirpath, vid_name)
                if not os.path.exists(video_path):
                    continue
                bvp_path = os.path.join(dirpath, bvp_fname)
                try:
                    with open(bvp_path, "r", encoding="utf-8", errors="ignore") as f:
                        bvp_values = [float(line.strip()) for line in f if line.strip()]
                except Exception:
                    continue
                print(f"Processing {video_path} (UBFC-Phys, windowed BVP ground truth)...")
                n = _process_video_bvp_gt(video_path, bvp_values, 64.0, subject, "ubfc_phys", rows)
                print(f"  -> {n} labeled cycles")
            continue

        has_video = any(f.lower().endswith((".avi", ".mp4", ".mov")) for f in filenames)
        gt_kind = None
        if "ground_truth.txt" in filenames:
            gt_kind = "dataset_2"
        elif "gtdump.xmp" in filenames:
            gt_kind = "dataset_1"
        if not (has_video and gt_kind):
            continue

        video_path = None
        for f in filenames:
            if f.lower().endswith((".avi", ".mp4", ".mov")):
                video_path = os.path.join(dirpath, f)
                break
        if video_path is None:
            continue

        subject_id = os.path.basename(dirpath.rstrip("/\\")) or dirpath
        gt_path = os.path.join(dirpath, "ground_truth.txt" if gt_kind == "dataset_2" else "gtdump.xmp")
        try:
            ts, hr = (_load_dataset2_gt(gt_path) if gt_kind == "dataset_2" else _load_dataset1_gt(gt_path))
        except Exception as exc:
            print(f"  [skip] {gt_path}: {exc}")
            continue
        if len(ts) < 3:
            continue

        print(f"Processing {video_path} ({gt_kind}, dense ground truth)...")
        n = _process_video_dense_gt(video_path, ts, hr, subject_id, gt_kind, rows)
        print(f"  -> {n} labeled cycles")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = FEATURE_COLUMNS + ["gt_hr", "subject_id", "group"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a windowed ML correction dataset from real rPPG videos.")
    parser.add_argument("root", help="Root folder to scan (e.g. test/)")
    parser.add_argument("--out", default="test/ml_dataset.csv")
    args = parser.parse_args()

    rows = build_dataset(args.root, args.out)
    subjects = {r["subject_id"] for r in rows}
    print(f"\nTotal labeled cycles: {len(rows)} across {len(subjects)} unique subjects")
    print(f"Written to {args.out}")
