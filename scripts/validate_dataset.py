#!/usr/bin/env python3
"""
VITAL — rPPG Validation Harness
================================
Two independent validation modes, kept clearly separate so results are
never conflated:

1. SYNTHETIC SIGNAL VALIDATION (always runs, no external data required)
   Feeds a known-BPM, physiologically-modelled RGB trace directly into
   AdvancedRPPG — bypassing camera capture and face detection entirely —
   to sanity-check the DSP pipeline in isolation (CHROM/POS projection,
   adaptive bandpass, Welch PSD peak-finding, temporal smoothing). This
   validates the *signal-processing math*, not end-to-end camera+face
   accuracy. Treat it as a unit test with a real, reportable MAE/RMSE,
   not a substitute for real-video validation.

2. REAL VIDEO VALIDATION (opt-in, requires a labeled manifest)
   Runs the FULL camera+face-detection+DSP pipeline against real video
   files with independently-known ground-truth BPM (e.g. a pulse-oximeter
   or chest-strap reading taken during recording), via:

       python scripts/validate_dataset.py --manifest path/to/manifest.csv

   Manifest CSV columns: video,gt_bpm[,skin_tone,motion,notes]
   (skin_tone/motion are optional — if present, a stratified per-group MAE
   breakdown is printed, since an aggregate MAE can hide large subgroup
   disparities).

   This repository does NOT ship with UBFC-rPPG / PURE / MMSE-HR video
   data — those datasets require a manual request to the dataset owners
   (see README.md's Validation section for pointers). Build a manifest CSV
   from whichever of those you obtain, pointing `video` at the extracted
   clips and `gt_bpm` at their accompanying ground-truth physiological
   file, then re-run this script for a real, citable accuracy number.

Previous behaviour (removed)
─────────────────────────────
This script used to silently default every discovered video's ground truth
to 0 BPM when no labels were supplied, then print the resulting nonsense
"error" as if it were a real accuracy metric. That number was never
measuring anything — it has been deleted rather than patched, because a
wrong metric is worse than a missing one.
"""

import argparse
import csv
import math
import os
import random
import sys


class _HiddenPrints:
    """Suppress analyze_video_complete()'s per-frame progress spam so a
    multi-video manifest run prints one clean table instead of ~2000 lines
    of frame-count chatter per video drowning out the actual results."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.rppg import AdvancedRPPG  # noqa: E402  (needs sys.path fixup above)


# ── Synthetic signal generator ────────────────────────────────────────────────

def _synthetic_rgb_stream(bpm, duration_s, fps=30, noise_level=0.02, seed=0):
    """
    Generate a physiologically-modelled RGB stream for a known BPM.

    Baseline DC levels and per-channel AC modulation depths follow the
    well-established rPPG finding (see this project's own research corpus,
    e.g. Pirzada et al. 2024) that the green channel carries the strongest
    pulsatile signal — hemoglobin absorbs green light most strongly — red
    is weaker, and blue weakest. A slow illumination-drift term rides on
    top of the periodic pulse component, plus i.i.d. sensor noise, so the
    adaptive-bandpass/SQI stages are meaningfully exercised rather than
    handed a pristine sinusoid.
    """
    rng = random.Random(seed)
    n = int(duration_s * fps)
    hr_hz = bpm / 60.0

    base_r, base_g, base_b = 180.0, 130.0, 110.0
    amp_r, amp_g, amp_b = 0.010, 0.018, 0.006  # AC depth as fraction of DC

    drift_freq = rng.uniform(0.02, 0.05)
    drift_phase = rng.uniform(0, 2 * math.pi)

    stream = []
    for i in range(n):
        t = i / fps
        # Weak 2nd harmonic gives a mild systolic/dicrotic-notch asymmetry
        # instead of a pure sine.
        pulse = math.sin(2 * math.pi * hr_hz * t) + 0.25 * math.sin(4 * math.pi * hr_hz * t)
        drift = 0.01 * math.sin(2 * math.pi * drift_freq * t + drift_phase)

        r = base_r * (1 + amp_r * pulse + drift) + rng.gauss(0, noise_level * base_r)
        g = base_g * (1 + amp_g * pulse + drift) + rng.gauss(0, noise_level * base_g)
        b = base_b * (1 + amp_b * pulse + drift) + rng.gauss(0, noise_level * base_b)
        stream.append((r, g, b))
    return stream


def _synthetic_rgb_stream_resp(bpm, rr_brpm, duration_s, fps=30,
                               noise_level=0.02, seed=0):
    """
    Like _synthetic_rgb_stream, but with a KNOWN respiratory rate imposed on
    the pulse via all three respiratory-induced variations the RR estimator
    fuses (see core.rppg's _RR_* block):

      RIAV — pulse amplitude is modulated by the respiratory waveform
      RIIV — the DC baseline is offset by the respiratory waveform
      RIFV — instantaneous pulse RATE is modulated (respiratory sinus
             arrhythmia), implemented by integrating phase rather than
             evaluating sin(2*pi*f*t), which is what actually makes the
             beat-to-beat interval vary

    Exercising all three matters: a generator that only did amplitude
    modulation would let a broken RIIV/RIFV channel pass unnoticed.
    """
    rng = random.Random(seed)
    n = int(duration_s * fps)
    hr_hz = bpm / 60.0
    rr_hz = rr_brpm / 60.0

    base_r, base_g, base_b = 180.0, 130.0, 110.0
    amp_r, amp_g, amp_b = 0.010, 0.018, 0.006

    drift_freq = rng.uniform(0.02, 0.05)
    drift_phase = rng.uniform(0, 2 * math.pi)

    stream = []
    phase = 0.0
    for i in range(n):
        t = i / fps
        resp = math.sin(2 * math.pi * rr_hz * t)

        # RIFV: respiratory sinus arrhythmia — +/-6% instantaneous rate.
        phase += 2.0 * math.pi * hr_hz * (1.0 + 0.06 * resp) / fps
        pulse = math.sin(phase) + 0.25 * math.sin(2.0 * phase)

        # RIAV: +/-30% pulse amplitude with respiration.
        amp_mod = 1.0 + 0.30 * resp
        # RIIV: baseline rides on respiration (comparable in size to the
        # illumination drift, so it cannot simply be filtered away as trend).
        riiv = 0.008 * resp
        drift = 0.01 * math.sin(2 * math.pi * drift_freq * t + drift_phase)

        r = base_r * (1 + amp_r * pulse * amp_mod + drift + riiv) + rng.gauss(0, noise_level * base_r)
        g = base_g * (1 + amp_g * pulse * amp_mod + drift + riiv) + rng.gauss(0, noise_level * base_g)
        b = base_b * (1 + amp_b * pulse * amp_mod + drift + riiv) + rng.gauss(0, noise_level * base_b)
        stream.append((r, g, b))
    return stream


def run_synthetic_rr_validation(algorithm="CHROM"):
    """
    Synthetic self-test for RESPIRATORY rate.

    This exists because RR had no automated regression test at all while HR
    had one, which is how a badly broken RR estimator survived: its
    "respiration filter" was actually passing 42-60 breaths/min (the
    pulse-band clamps in _safe_butter_sos were silently rewriting the
    requested 0.10-0.50 Hz band to 0.70-1.00 Hz), and nothing caught it.

    Synthetic only proves the DSP maths. The real-data RR number comes from
    scripts/validate_rr_bidmc.py (53 recordings, human-annotated breaths).
    """
    print("=" * 80)
    print(f"SYNTHETIC RESPIRATORY-RATE VALIDATION — algorithm={algorithm}")
    print("Known-RR streams with all three RIVs (amplitude/baseline/RSA) imposed.")
    print("DSP-only: see scripts/validate_rr_bidmc.py for real ground-truth RR.")
    print("=" * 80)
    print(f"{'GT RR':<8} | {'HR':<6} | {'Est. RR':<9} | {'Abs Err':<9} | {'Readings':<9} | {'Stab':<5}")
    print("-" * 80)

    # RR needs a long window by physics (see _RR_WINDOW_SEC); 90 s gives the
    # 60 s buffer room to fill and then contribute several readings.
    cases = [(12.0, 60), (15.0, 75), (18.0, 72), (22.0, 90), (27.0, 105)]
    errors = []

    for gt_rr, hr in cases:
        rppg = AdvancedRPPG(fps=30, window_size=300, algorithm=algorithm)
        stream = _synthetic_rgb_stream_resp(hr, gt_rr, duration_s=90.0,
                                            seed=int(gt_rr * 10))
        for rgb in stream:
            rppg.add_frame(rgb)
            rppg.process_ppg_signal()

        summary = rppg.get_final_summary()
        est_rr = summary.get('final_rr')
        if est_rr is None:
            print(f"{gt_rr:<8.1f} | {hr:<6} | {'NOT MEAS':<9} | {'-':<9} | "
                  f"{summary.get('rr_readings', 0):<9} | {'-':<5}")
            continue
        err = abs(est_rr - gt_rr)
        errors.append(err)
        print(f"{gt_rr:<8.1f} | {hr:<6} | {est_rr:<9.1f} | {err:<9.2f} | "
              f"{summary.get('rr_readings', 0):<9} | "
              f"{summary.get('rr_stability_percent', 0):<5}")

    print("-" * 80)
    if errors:
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        print(f"Synthetic RR MAE:  {mae:.2f} br/min across {len(errors)} known-RR trials")
        print(f"Synthetic RR RMSE: {rmse:.2f} br/min")
    else:
        print("Synthetic RR: NO CASE PRODUCED A READING — RR path is broken.")
    print("=" * 80)
    return errors


def run_synthetic_validation(algorithm="CHROM"):
    print("=" * 80)
    print(f"SYNTHETIC SIGNAL VALIDATION — algorithm={algorithm}")
    print("Known-BPM RGB traces fed directly into AdvancedRPPG, bypassing camera/face")
    print("detection. Validates the DSP pipeline in isolation, not end-to-end camera")
    print("accuracy — see README's Validation section for that distinction.")
    print("=" * 80)
    print(f"{'GT BPM':<10} | {'Est. BPM':<10} | {'Abs Error':<12} | {'Final SQI':<10}")
    print("-" * 80)

    test_bpms = [50, 65, 75, 90, 105, 120, 140]
    errors, sq_errors = [], []

    for gt_bpm in test_bpms:
        rppg = AdvancedRPPG(fps=30, window_size=300, algorithm=algorithm)
        stream = _synthetic_rgb_stream(gt_bpm, duration_s=15.0, seed=int(gt_bpm))
        last_result = None
        for rgb in stream:
            rppg.add_frame(rgb)
            last_result = rppg.process_ppg_signal()

        summary = rppg.get_final_summary()
        est_bpm = summary.get('final_bpm')
        if est_bpm is None:
            print(f"{gt_bpm:<10} | {'FAILED':<10} | {'-':<12} | {'-':<10}")
            continue

        err = abs(est_bpm - gt_bpm)
        errors.append(err)
        sq_errors.append(err ** 2)
        sqi = last_result.get('sqi', 0) if last_result else 0
        print(f"{gt_bpm:<10} | {est_bpm:<10.1f} | {err:<12.1f} | {sqi:<10.1f}")

    print("-" * 80)
    if errors:
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
        print(f"Synthetic MAE:  {mae:.2f} BPM across {len(errors)} known-BPM trials")
        print(f"Synthetic RMSE: {rmse:.2f} BPM")
    else:
        print("No successful synthetic trials — the DSP pipeline itself is broken, "
              "not just its accuracy.")
    print("=" * 80)
    return errors


# ── Real video (manifest-driven) validation ───────────────────────────────────

def run_manifest_validation(manifest_path):
    # Imported lazily — cv2/mediapipe are only needed for this path, so the
    # synthetic self-test above still runs in environments without them.
    from scripts.analyze_video import analyze_video_complete

    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return []

    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"Manifest {manifest_path} contained no rows.")
        return []

    print("=" * 90)
    print("REAL VIDEO VALIDATION")
    print("=" * 90)
    print(f"{'Video':<28} | {'GT BPM':<8} | {'Our BPM':<8} | {'Abs Err':<8} | {'Group':<16}")
    print("-" * 90)

    errors, sq_errors = [], []
    group_errors = {}

    for row in rows:
        video_path = (row.get('video') or '').strip()
        gt_raw = (row.get('gt_bpm') or '').strip()
        if not video_path or not gt_raw:
            continue
        try:
            gt_bpm = float(gt_raw)
        except ValueError:
            print(f"{video_path[:26]:<28} | SKIPPED — invalid gt_bpm '{gt_raw}'")
            continue
        if not os.path.exists(video_path):
            print(f"{video_path[:26]:<28} | {'NOT FOUND':<8}")
            continue

        print(f"Processing {video_path}...", end="\r")
        with _HiddenPrints():
            # stride=1 (default) — NOT 2. Found the hard way: the BPM
            # lock/candidate hysteresis and subharmonic-promotion logic in
            # core/rppg.py count in *cycles* (calls to process_ppg_signal()),
            # not real time. Striding silently doubles their effective time
            # window, changing tracking BEHAVIOR, not just speed — a direct
            # trace at stride=1 showed a real fix; a batch run at stride=2
            # showed no change, on the exact same video. UBFC-rPPG/UBFC-Phys
            # clips are short enough (1-3 min) that stride=1 is fine here;
            # the stride optimization stays reserved for rPPG-10's genuinely
            # long (10 min) clips in validate_cropped_roi.py, where no
            # cycle-counted hysteresis is in play.
            result = analyze_video_complete(video_path)
        sys.stdout.write("\033[K")  # clear the "Processing..." line
        group = ",".join(filter(None, [row.get('skin_tone', ''), row.get('motion', '')])) or "unlabeled"

        if result.get('success'):
            our_bpm = result['bpm']
            err = abs(our_bpm - gt_bpm)
            errors.append(err)
            sq_errors.append(err ** 2)
            group_errors.setdefault(group, []).append(err)
            print(f"{video_path[:26]:<28} | {gt_bpm:<8.1f} | {our_bpm:<8.1f} | {err:<8.1f} | {group:<16}")
        else:
            print(f"{video_path[:26]:<28} | {gt_bpm:<8.1f} | {'FAILED':<8} | {'-':<8} | {group:<16}")

    print("-" * 90)
    if errors:
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
        print(f"Overall MAE:  {mae:.2f} BPM  (n={len(errors)})")
        print(f"Overall RMSE: {rmse:.2f} BPM")

        if len(group_errors) > 1:
            print("\nPer-group breakdown — an aggregate MAE can hide large subgroup")
            print("disparities (this is exactly the skin-tone/motion fairness gap this")
            print("project previously claimed was handled without evidence):")
            for group, errs in sorted(group_errors.items()):
                g_mae = sum(errs) / len(errs)
                print(f"  {group:<20} MAE={g_mae:.2f} BPM  (n={len(errs)})")

        if mae < 5.0:
            print("\nStatus: CLINICAL-ADJACENT ACCURACY (MAE < 5 BPM)")
        elif mae < 8.0:
            print("\nStatus: RESEARCH GRADE (MAE < 8 BPM)")
        else:
            print("\nStatus: CONSUMER GRADE (MAE >= 8 BPM)")
    else:
        print("No valid readings — check manifest paths and gt_bpm values.")
    print("=" * 90)
    return errors


def main():
    parser = argparse.ArgumentParser(description="VITAL rPPG validation harness")
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="CSV with columns: video,gt_bpm[,skin_tone,motion,notes]. See README's "
             "Validation section for how to build one from UBFC-rPPG/PURE/MMSE-HR.",
    )
    parser.add_argument("--algorithm", type=str, default="CHROM", choices=["CHROM", "POS"])
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Skip the synthetic self-test (only run --manifest).",
    )
    parser.add_argument(
        "--skip-rr", action="store_true",
        help="Skip the synthetic respiratory-rate self-test. It needs 5 x 90 s "
             "of simulated signal, so it is slower than the HR self-test.",
    )
    args = parser.parse_args()

    if not args.skip_synthetic:
        run_synthetic_validation(algorithm=args.algorithm)
        print()
        if not args.skip_rr:
            run_synthetic_rr_validation(algorithm=args.algorithm)
            print()

    if args.manifest:
        run_manifest_validation(args.manifest)
    elif args.skip_synthetic:
        print("Nothing to do: --skip-synthetic given without --manifest.")
    else:
        print(
            "No --manifest given — skipping real-video validation.\n"
            "The numbers above are DSP-pipeline-only (synthetic signal), NOT\n"
            "end-to-end camera+face-detection accuracy on real subjects. See\n"
            "README's Validation section for how to obtain UBFC-rPPG/PURE and\n"
            "build a manifest CSV."
        )


if __name__ == '__main__':
    main()
