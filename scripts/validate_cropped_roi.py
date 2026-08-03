#!/usr/bin/env python3
"""
VITAL — Cropped-ROI Real-Video DSP Validation (rPPG-10)
==========================================================
rPPG-10 ships pre-cropped 64x64 skin patches (forehead/cheek), not full-face
frames — confirmed by direct inspection (frame shape (64, 64, 3), no visible
face structure). MediaPipe's face mesh needs a whole face to anchor its
landmarks, so this dataset is NOT compatible with the standard
camera+face-detection pipeline (`scripts/analyze_video.py` /
`validate_dataset.py --manifest`) — do not force it through that path, it
will just report 0 valid readings for every subject and look like a total
accuracy failure when it's actually an input-format mismatch.

This script bypasses face detection entirely: each frame IS already the
ROI, so it's spatially averaged directly and fed straight into
AdvancedRPPG. That validates the DSP core (CHROM projection, adaptive
filter, peak detection) against real camera footage and real ECG-derived
ground truth — a genuine middle category between the fully synthetic
self-test (no camera at all) and the full pipeline real-video test (camera
+ face detection together). It does NOT validate face detection or
ROI-selection accuracy, since there is no face to detect here — label any
result from this script accordingly, don't fold it into the main
--manifest real-video numbers.

Ground truth: ECG waveform (.npy, ~1000 Hz per Subject_N_ECG.npy), reduced
to a single gt_bpm via bandpass + peak detection
(scripts/build_manifest.py::bpm_from_waveform) — this dataset ships a raw
ECG signal, not a pre-computed HR series.

Usage
-----
    python scripts/validate_cropped_roi.py path/to/extracted/Dataset_rPPG-10
"""

import argparse
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.build_manifest import bpm_from_waveform  # noqa: E402


def _iter_subjects(root):
    for name in sorted(os.listdir(root)):
        subj_dir = os.path.join(root, name)
        if not os.path.isdir(subj_dir):
            continue
        ecg_path = os.path.join(subj_dir, f"{name}_ECG.npy")
        vid_path = os.path.join(subj_dir, f"{name}_Forehead_.avi")
        if os.path.exists(ecg_path) and os.path.exists(vid_path):
            yield name, ecg_path, vid_path


def _run_cropped_roi(video_path):
    import cv2
    from core.rppg import AdvancedRPPG

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Some backends return 0, -1, or NaN for an unreadable/corrupt file
    # instead of raising — `fps or 30.0` only catches exactly-0, not
    # negative/NaN, which crashed the whole batch on one bad file
    # (deque(maxlen=negative) -> ValueError) instead of just failing that
    # one subject.
    if not fps or fps != fps or fps <= 0:  # `fps != fps` is the NaN check
        fps = 30.0
    rppg = AdvancedRPPG(fps=int(round(fps)), window_size=int(fps * 10), algorithm="CHROM")

    # These clips run ~10 minutes each (18,000 frames at 30fps). No face
    # detection here to offset the cost, so the Welch+cubic-spline HRV pass
    # inside process_ppg_signal() is the entire per-frame cost — running it
    # on every single frame turned one subject into ~13+ minutes with zero
    # results yet. add_frame() still runs every frame so the signal window
    # stays complete; only the expensive recompute is decimated.
    process_stride = 3
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Frame IS the ROI already (pre-cropped skin patch) — direct
        # spatial mean, deliberately no face detection here.
        b = float(frame[:, :, 0].mean())
        g = float(frame[:, :, 1].mean())
        r = float(frame[:, :, 2].mean())
        rppg.add_frame((r, g, b))
        if frame_count % process_stride == 0:
            rppg.process_ppg_signal()
        frame_count += 1

    cap.release()
    return rppg.get_final_summary()


def main():
    parser = argparse.ArgumentParser(
        description="DSP-only validation on rPPG-10's pre-cropped skin-patch videos (bypasses face detection)."
    )
    parser.add_argument("root", help="Path to the extracted Dataset_rPPG-10 folder")
    args = parser.parse_args()

    import numpy as np

    print("=" * 90)
    print("CROPPED-ROI DSP VALIDATION (rPPG-10) — bypasses face detection, DSP core only")
    print("Not directly comparable to the full-pipeline real-video numbers elsewhere —")
    print("there is no face-detection/ROI-selection step being tested here.")
    print("=" * 90)
    print(f"{'Subject':<14} | {'GT BPM':<8} | {'Our BPM':<8} | {'Abs Err':<8}")
    print("-" * 90)

    errors = []
    for name, ecg_path, vid_path in _iter_subjects(args.root):
        ecg = np.load(ecg_path)
        gt_bpm = bpm_from_waveform(ecg, fs=1000.0)
        if gt_bpm is None:
            print(f"{name:<14} | GT PEAK-DETECTION FAILED — skipped")
            continue

        try:
            summary = _run_cropped_roi(vid_path)
        except Exception as exc:
            # One corrupt/unreadable file shouldn't take down a 26-subject
            # batch run — report it and keep going.
            print(f"{name:<14} | {gt_bpm:<8.1f} | ERROR: {exc}")
            continue

        our_bpm = summary.get("final_bpm")
        if our_bpm is None:
            print(f"{name:<14} | {gt_bpm:<8.1f} | {'FAILED':<8}")
            continue

        err = abs(our_bpm - gt_bpm)
        errors.append(err)
        print(f"{name:<14} | {gt_bpm:<8.1f} | {our_bpm:<8.1f} | {err:<8.1f}")

    print("-" * 90)
    if errors:
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
        print(f"MAE:  {mae:.2f} BPM  (n={len(errors)})")
        print(f"RMSE: {rmse:.2f} BPM")
    else:
        print("No valid results.")
    print("=" * 90)


if __name__ == "__main__":
    main()
