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

Feature set v2 — adds signals the model previously never saw, each tied
to a documented, real failure mode rather than added speculatively:
  - harmonic_ratio: the top spectral candidate's 2x-companion power ratio,
    already computed inside _select_dominant_freq_hps and previously
    discarded once the promote/don't-promote decision was made. A high
    ratio means THIS cycle is ambiguous between two candidates ~2x apart,
    independent of which one the hand-written 0.55 threshold picked —
    exactly the signature behind subject41's still-unresolved bimodality
    (see data/evolution/13-majority-vote-gate-reverted.md).
  - motion_delta: Camera's own frame-to-frame face-centroid drift, already
    computed for the live is_moving flag and previously never reaching
    the correction model at all — a direct, real-time motion-artifact
    signal, relevant to the speech/after-exercise failure modes.
  - pos_bpm / chrom_pos_disagreement: POS (Wang et al. 2017) is already
    implemented in this codebase but unwired from the live product
    (core/rppg.py's own docs note it has its own separate failure mode).
    Running it alongside CHROM on the same frames gives the model a
    second, independently-computed estimate — large disagreement between
    the two is itself informative, and the literature comparison in this
    project's README shows a CHROM+POS hybrid (ChPOS) beating either
    alone (5.71 MAE vs 8.20/8.35).
  - rolling_bpm_mean / rolling_bpm_std: temporal context. Every prior
    feature describes ONE cycle in isolation; the model has never seen
    whether the last several cycles have been stable or swinging wildly,
    which is exactly what a human tracing a trace log looks at first.

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
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy import signal as sig

from core.camera import Camera
from core.rppg import AdvancedRPPG

_UBFC_PHYS_BVP_RE = re.compile(r"^bvp_(?P<subject>.+)_(?P<task>T\d+)\.csv$", re.IGNORECASE)

FEATURE_COLUMNS = [
    "bpm", "sqi", "snr_db", "stability_score", "rr", "rr_confidence",
    "hrv", "stress_index", "filt_low", "filt_high",
    "harmonic_ratio", "motion_delta", "pos_bpm", "chrom_pos_disagreement",
    "rolling_bpm_mean", "rolling_bpm_std",
]

_ROLLING_WINDOW = 30  # cycles (~1s @ 30fps) of recent CHROM bpm for temporal-context features


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


def _process_video(video_path, gt_fn, subject_id, group, rows_out):
    """
    Shared extraction loop for both ground-truth sources (dense per-sample
    HR, or windowed BVP peak-detection) — `gt_fn(t_sec)` returns the ground
    truth HR at a given timestamp, or None if unavailable there.

    Runs CHROM and POS side by side on the identical frame stream so the
    model can see both algorithms' independent opinions per cycle.
    """
    camera = Camera(source=video_path)
    if camera.dummy_mode:
        print(f"  [skip] could not open {video_path}")
        return 0
    fps = camera.get_source_fps(default=30.0)
    window_size = max(300, int(fps * 10))
    rppg_chrom = AdvancedRPPG(fps=int(round(fps)), window_size=window_size, algorithm="CHROM")
    rppg_pos = AdvancedRPPG(fps=int(round(fps)), window_size=window_size, algorithm="POS")
    recent_bpm = deque(maxlen=_ROLLING_WINDOW)

    n_added = 0
    frame_idx = 0
    MAX_FRAMES = 100_000
    while frame_idx < MAX_FRAMES:
        frame_bytes, roi_data, _is_moving, motion_delta = camera.get_frame()
        if frame_bytes is None:
            break
        if roi_data is not None:
            r, g, b = roi_data
            rppg_chrom.add_frame((r, g, b), motion_delta=motion_delta)
            rppg_pos.add_frame((r, g, b))
            result = rppg_chrom.process_ppg_signal()
            pos_result = rppg_pos.process_ppg_signal()

            if result.get("ready") and result.get("bpm", 0) > 0:
                recent_bpm.append(result["bpm"])
                t_sec = frame_idx / fps
                gt_now = gt_fn(t_sec)
                if gt_now is not None and 40.0 <= gt_now <= 180.0:
                    pos_bpm = pos_result.get("bpm", 0.0) if pos_result.get("ready") else 0.0
                    rows_out.append({
                        "bpm": result["bpm"], "sqi": result.get("sqi", 0.0),
                        "snr_db": result.get("snr_db", 0.0),
                        "stability_score": result.get("stability_score", 0.0),
                        "rr": result.get("rr", 0.0), "rr_confidence": result.get("rr_confidence", 0.0),
                        "hrv": result.get("hrv", 0.0), "stress_index": result.get("stress_index", 0.0),
                        "filt_low": result.get("filter_band_hz", [0, 0])[0],
                        "filt_high": result.get("filter_band_hz", [0, 0])[1],
                        "harmonic_ratio": result.get("harmonic_ratio", 0.0),
                        "motion_delta": result.get("motion_delta", 0.0),
                        "pos_bpm": pos_bpm,
                        "chrom_pos_disagreement": abs(result["bpm"] - pos_bpm) if pos_bpm > 0 else 0.0,
                        "rolling_bpm_mean": float(np.mean(recent_bpm)),
                        "rolling_bpm_std": float(np.std(recent_bpm)) if len(recent_bpm) > 1 else 0.0,
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
                gt_fn = lambda t_sec, _v=bvp_values: _instantaneous_bpm_from_bvp_window(_v, 64.0, int(t_sec * 64.0))
                n = _process_video(video_path, gt_fn, subject, "ubfc_phys", rows)
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
        gt_fn = lambda t_sec, _ts=ts, _hr=hr: float(np.interp(t_sec, _ts, _hr))
        n = _process_video(video_path, gt_fn, subject_id, gt_kind, rows)
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
