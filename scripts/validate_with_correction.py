#!/usr/bin/env python3
"""
VITAL — Session-Level Classical vs. Hybrid-Corrected Validation (out-of-fold)
================================================================================
Runs the full real-video manifest through analyze_video_complete() twice per
video — once for the classical DSP final_bpm (already validated, e.g. 8.4
BPM MAE), once for the ML-corrected final_bpm — and reports both, honestly,
against the same ground truth.

Critical correctness requirement: every manifest session must be corrected
using a model that NEVER saw that session's subject during training. The
"final" deployed model (train_correction_model.py --save-final) is trained
on ALL subjects with no holdout — using it here would be evaluating the
model on its own training data, which is exactly the kind of leakage this
project has refused to accept anywhere else (see the GroupKFold discipline
in train_correction_model.py's own docstring). Instead, this script uses
the 5 per-fold models saved via --save-fold-models: for each manifest
session, it looks up which fold held that session's subject OUT of
training, and applies only that fold's model — an honest out-of-fold
prediction for every single session, not just an aggregate estimate.

Usage:
    python scripts/train_correction_model.py test/ml_dataset.csv --save-fold-models models/cv_folds
    python scripts/validate_with_correction.py test/manifest.csv models/cv_folds
"""
import argparse
import csv
import math
import os
import re
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_UBFC_PHYS_VID_RE = re.compile(r"^vid_(?P<subject>.+)_(?P<task>T\d+)\.avi$", re.IGNORECASE)


class _HiddenPrints:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout


def _subject_id_for_video(video_path):
    """Must match scripts/build_ml_dataset.py's subject_id assignment exactly,
    or fold lookup below will silently mismatch."""
    fname = os.path.basename(video_path)
    m = _UBFC_PHYS_VID_RE.match(fname)
    if m:
        return m.group("subject")
    # dataset_1 / dataset_2: subject_id was the parent folder's basename
    return os.path.basename(os.path.dirname(video_path).rstrip("/\\"))


def _load_fold_models(fold_dir):
    from core.ml_compat import load_joblib_compat
    fold_models = []
    for fname in sorted(os.listdir(fold_dir)):
        if fname.endswith(".joblib"):
            bundle = load_joblib_compat(os.path.join(fold_dir, fname))
            fold_models.append(bundle)
    return fold_models


def _model_path_for_subject(fold_models, subject_id):
    for bundle in fold_models:
        if subject_id in bundle["held_out_subjects"]:
            return bundle
    return None


def main(manifest_path, fold_models_dir):
    from scripts.analyze_video import analyze_video_complete, _correction_model_cache

    fold_models = _load_fold_models(fold_models_dir)
    print(f"Loaded {len(fold_models)} fold models from {fold_models_dir}")
    all_held_out = set()
    for b in fold_models:
        all_held_out |= set(b["held_out_subjects"])
    print(f"Covering {len(all_held_out)} unique subjects across all folds\n")

    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print("=" * 100)
    print("SESSION-LEVEL VALIDATION — CLASSICAL DSP vs. HYBRID (out-of-fold ML correction)")
    print("=" * 100)
    print(f"{'Video':<26} | {'GT':<7} | {'Classical':<10} | {'C.Err':<7} | {'Corrected':<10} | {'Cr.Err':<7} | {'Group':<12}")
    print("-" * 100)

    classical_errs, corrected_errs = [], []
    group_classical, group_corrected = {}, {}
    skipped_no_fold = []

    for row in rows:
        video_path = (row.get('video') or '').strip()
        gt_raw = (row.get('gt_bpm') or '').strip()
        if not video_path or not gt_raw or not os.path.exists(video_path):
            continue
        try:
            gt_bpm = float(gt_raw)
        except ValueError:
            continue

        subject_id = _subject_id_for_video(video_path)
        bundle = _model_path_for_subject(fold_models, subject_id)
        if bundle is None:
            skipped_no_fold.append((video_path, subject_id))

        # Inject this session's correct out-of-fold model into
        # analyze_video's cache under a fixed key, bypassing its normal
        # load-from-path caching (we're switching models per session here,
        # not loading once for a whole batch).
        _correction_model_cache["__oof__"] = bundle

        # Explicit flush, no \r trick — when stdout is redirected to a file
        # (not a real terminal), Python fully buffers it, and this script's
        # entire output (56 short rows) is small enough to never hit a
        # size-triggered flush on its own. A run redirected to a log file
        # could look like a silent 0-byte hang for its ENTIRE duration even
        # if it's working perfectly the whole time, purely from buffering —
        # found the hard way when a run looked stuck for 2h43m. Flushing
        # after every single video makes real progress actually visible in
        # the log file as it happens, not just at process exit.
        print(f"[{time.strftime('%H:%M:%S')}] Processing {video_path}...", flush=True)
        with _HiddenPrints():
            result = analyze_video_complete(
                video_path,
                correction_model_path=("__oof__" if bundle is not None else None),
            )
        if not result.get('success'):
            print(f"{video_path[:24]:<26} | {gt_bpm:<7.1f} | FAILED", flush=True)
            continue

        group = ",".join(filter(None, [row.get('skin_tone', ''), row.get('motion', '')])) or "unlabeled"
        classical_bpm = result.get('final_bpm_classical', result['bpm'])
        corrected_bpm = result.get('final_bpm_corrected')

        c_err = abs(classical_bpm - gt_bpm)
        classical_errs.append(c_err)
        group_classical.setdefault(group, []).append(c_err)

        if corrected_bpm is not None:
            cr_err = abs(corrected_bpm - gt_bpm)
            corrected_errs.append(cr_err)
            group_corrected.setdefault(group, []).append(cr_err)
            cr_str, cr_err_str = f"{corrected_bpm:<10.1f}", f"{cr_err:<7.1f}"
        else:
            cr_str, cr_err_str = f"{'N/A':<10}", f"{'-':<7}"

        print(f"{video_path[:24]:<26} | {gt_bpm:<7.1f} | {classical_bpm:<10.1f} | {c_err:<7.1f} | {cr_str} | {cr_err_str} | {group:<12}", flush=True)

    print("-" * 100)
    if classical_errs:
        c_mae = sum(classical_errs) / len(classical_errs)
        c_rmse = math.sqrt(sum(e**2 for e in classical_errs) / len(classical_errs))
        print(f"Classical DSP        MAE: {c_mae:.2f} BPM   RMSE: {c_rmse:.2f} BPM   (n={len(classical_errs)})")
    if corrected_errs:
        cr_mae = sum(corrected_errs) / len(corrected_errs)
        cr_rmse = math.sqrt(sum(e**2 for e in corrected_errs) / len(corrected_errs))
        print(f"Hybrid (out-of-fold) MAE: {cr_mae:.2f} BPM   RMSE: {cr_rmse:.2f} BPM   (n={len(corrected_errs)})")

    print("\nPer-group breakdown:")
    for group in sorted(set(group_classical) | set(group_corrected)):
        c = group_classical.get(group, [])
        cr = group_corrected.get(group, [])
        c_str = f"{sum(c)/len(c):.2f}" if c else "-"
        cr_str = f"{sum(cr)/len(cr):.2f}" if cr else "-"
        print(f"  {group:<14} classical={c_str:<8} corrected={cr_str:<8} (n={len(c)})")

    if skipped_no_fold:
        print(f"\n{len(skipped_no_fold)} sessions had no matching fold (subject_id mismatch) — "
              f"classical-only, no correction applied:")
        for vp, sid in skipped_no_fold:
            print(f"  {vp}  (subject_id parsed as: {sid})")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("fold_models_dir")
    args = parser.parse_args()
    main(args.manifest, args.fold_models_dir)
