#!/usr/bin/env python3
"""
Complete Video Analysis for Final Accurate Results
Processes entire video and returns final vital signs
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from core.rppg import AdvancedRPPG
from core.camera import Camera

def analyze_video_complete(video_path, process_stride: int = 1):
    """
    Analyze complete video and return final results

    Args:
        video_path: path to the video file.
        process_stride: run the full DSP pipeline (Welch PSD + cubic-spline
            HRV — the expensive parts of process_ppg_signal()) only every
            Nth frame instead of every single frame. `add_frame()` still
            runs every frame regardless, so the signal buffer stays
            complete — this only reduces how often we re-run the FFT/HRV
            fit on a window that hasn't meaningfully changed between
            adjacent frames. Default 1 preserves exact prior behavior (used
            by the live product path via agents/tools.py). Batch validation
            over long videos (e.g. 10-minute clips) should pass a higher
            stride — recomputing a full Welch+cubic-spline pass on every
            single one of 18,000 frames is what turned validating a handful
            of long real videos into an hours-long run in practice.

    Returns:
        dict with final BPM, confidence, and quality metrics
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {video_path}")
    print(f"{'='*70}\n")

    # Initialize — camera FIRST, so we can read the video's ACTUAL frame
    # rate rather than assuming 30fps. Videos not encoded at ~30fps (several
    # real UBFC-rPPG subjects are ~23.2fps) were getting a deterministic,
    # confidently-wrong reading scaled by (30 / real_fps) — a completely
    # different bug from the harmonic lock-on issue, found by tracing why
    # subject25's reading was rock-stable and confident, yet still wrong.
    camera = Camera(source=video_path)
    video_fps = camera.get_source_fps(default=30.0)
    rppg = AdvancedRPPG(fps=int(round(video_fps)), window_size=max(300, int(video_fps * 10)))
    print(f"Detected source FPS: {video_fps:.2f}")

    frame_count = 0
    bpm_readings = []

    print("Processing video frames...")

    # Process entire video
    while True:
        frame_bytes, roi_data, is_moving, motion_delta = camera.get_frame()

        if frame_bytes is None:
            break

        if roi_data is not None:
            avg_r, avg_g, avg_b = roi_data

            # Add to rPPG — every frame, regardless of stride, so the
            # buffer/window content is unaffected by the stride.
            rppg.add_frame((avg_r, avg_g, avg_b))

            if frame_count % process_stride != 0:
                frame_count += 1
                continue

            # Process with the configured projection algorithm (CHROM/POS)
            result = rppg.process_ppg_signal()

            if result['ready'] and result['bpm'] > 0:
                # Capture the FULL vitals set, not just bpm/confidence — rr,
                # hrv and stress_index were previously discarded here, which
                # silently meant every file-based triage run reported
                # rr=hrv=stress_index=0 regardless of what the DSP pipeline
                # actually computed (making Compensated Shock detection
                # impossible for any uploaded video).
                bpm_readings.append({
                    'bpm': result['bpm'],
                    'confidence': result['confidence'],
                    'sqi': result.get('sqi', result['confidence']),
                    'rr': result.get('rr', 0.0),
                    'rr_confidence': result.get('rr_confidence', 0.0),
                    'hrv': result.get('hrv', 0.0),
                    'stress_index': result.get('stress_index', 0.0),
                    'frame': frame_count,
                })

        frame_count += 1
        if frame_count % 30 == 0:
            # Plain ASCII — Windows console (cp1252) can't encode U+2713 (checkmark),
            # which crashed batch validation runs before any results printed.
            print(f"  [ok] Processed {frame_count} frames...")

    print(f"\n[ok] Complete: {frame_count} frames analyzed\n")

    # Calculate final results
    if len(bpm_readings) > 0:
        # Use last 10 readings for confidence/SQI/stability display stats
        recent_readings = bpm_readings[-10:] if len(bpm_readings) >= 10 else bpm_readings

        # final_bpm specifically comes from rppg.get_final_summary() — its
        # IQR-filtered median over the engine's OWN admitted bpm_history,
        # not a plain mean of whichever 10 readings happen to fall at the
        # end of the clip. This used to be two different "final answer"
        # algorithms for the same concept: the live webcam path (web/routes.py)
        # already used get_final_summary(); this file-based/batch path used
        # its own "mean of last 10" instead. That mattered in practice — on a
        # video where the BPM tracker settles into a wrong lock right at the
        # very end (a real case found via subject41), "last 10 readings"
        # reports whatever the tail happens to be, while the full-history
        # median correctly reflects that the correct rate dominated most of
        # the clip. Two call sites silently disagreeing about something this
        # safety-relevant is its own bug, independent of any DSP accuracy fix.
        summary = rppg.get_final_summary()
        final_bpm = summary.get('final_bpm')
        if final_bpm is None:
            final_bpm = np.mean([r['bpm'] for r in recent_readings])

        final_confidence = np.mean([r['confidence'] for r in recent_readings])
        stability = np.std([r['bpm'] for r in recent_readings])
        final_sqi = np.mean([r['sqi'] for r in recent_readings])

        # Respiration/HRV/stress readings are noisier and take longer to
        # stabilise than HR — only average over readings where the engine
        # itself reported reasonable confidence, so a few noisy early frames
        # don't null out a perfectly good final reading.
        rr_readings_valid = [r for r in recent_readings if r['rr'] > 0 and r['rr_confidence'] >= 10]
        rr_valid = [r['rr'] for r in rr_readings_valid]
        rr_conf_valid = [r['rr_confidence'] for r in rr_readings_valid]
        hrv_valid = [r['hrv'] for r in recent_readings if r['hrv'] > 0]
        stress_valid = [r['stress_index'] for r in recent_readings if r['stress_index'] > 0]

        final_rr = round(float(np.mean(rr_valid)), 1) if rr_valid else 0.0
        # Aggregate rr_confidence over the SAME readings actually averaged
        # into final_rr, not all recent_readings — so a downstream consumer
        # (core.clinical.assess) can tell how trustworthy final_rr really is,
        # instead of assuming it's fully reliable by default.
        final_rr_confidence = round(float(np.mean(rr_conf_valid)), 1) if rr_conf_valid else 0.0
        final_hrv = round(float(np.mean(hrv_valid)), 1) if hrv_valid else 0.0
        final_stress = round(float(np.mean(stress_valid)), 1) if stress_valid else 0.0

        return {
            'success': True,
            'bpm': round(final_bpm, 1),
            'confidence': round(final_confidence, 1),
            'sqi': round(float(final_sqi), 1),
            'stability': round(stability, 1),
            'rr': final_rr,
            'rr_confidence': final_rr_confidence,
            'hrv': final_hrv,
            'stress_index': final_stress,
            'quality': round(float(final_sqi), 1),
            'frames_analyzed': frame_count,
            'valid_readings': len(bpm_readings),
            'mode': 'FACE'
        }
    else:
        return {
            'success': False,
            'error': 'No valid BPM readings detected',
            'frames_analyzed': frame_count
        }


if __name__ == '__main__':
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'uploads/face.mp4'
    
    results = analyze_video_complete(video_path)
    
    print(f"{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}\n")
    
    if results['success']:
        print(f"HEART RATE: {results['bpm']} BPM")
        print(f"CONFIDENCE: {results['confidence']}%")
        print(f"STABILITY: +/-{results['stability']} BPM")
        print(f"SIGNAL QUALITY: {results['quality']}%")
        print(f"\nFrames Analyzed: {results['frames_analyzed']}")
        print(f"Valid Readings: {results['valid_readings']}")

        # Health assessment
        bpm = results['bpm']
        if 60 <= bpm <= 100:
            print(f"\nSTATUS: NORMAL (Resting heart rate)")
        elif bpm < 60:
            print(f"\nSTATUS: BRADYCARDIA (Low heart rate)")
        else:
            print(f"\nSTATUS: TACHYCARDIA (Elevated heart rate)")
    else:
        print(f"ERROR: {results['error']}")
        print(f"   Frames analyzed: {results['frames_analyzed']}")
    
    print(f"\n{'='*70}\n")
