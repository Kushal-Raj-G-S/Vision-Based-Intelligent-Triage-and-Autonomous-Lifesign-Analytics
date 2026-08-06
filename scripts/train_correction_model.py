#!/usr/bin/env python3
"""
VITAL — Hybrid Classical-DSP + Neural-Network Correction Layer
=================================================================
Trains a small multi-layer neural network (sklearn MLPRegressor) to
correct the classical CHROM/POS pipeline's per-cycle BPM estimate, using
the windowed dataset from scripts/build_ml_dataset.py (thousands of
per-cycle (feature, ground-truth-instantaneous-HR) pairs derived from
real UBFC-rPPG/UBFC-Phys ground truth, not the 56-session-scalar numbers).

The one rule that makes this honest rather than a leaderboard trick:
cross-validation MUST be grouped by subject_id. Rows from the same
session (or the same person's other sessions, for UBFC-Phys's 3 tasks)
are highly correlated — a random row-level split would let the model see
a person's own signal during training and "predict" their held-out rows
almost by memorization, not generalization. GroupKFold prevents that: a
subject's rows are entirely in one fold or the other, never split.

Usage:
    python scripts/train_correction_model.py test/ml_dataset.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

FEATURE_COLUMNS = [
    "bpm", "sqi", "snr_db", "stability_score", "rr", "rr_confidence",
    "hrv", "stress_index", "filt_low", "filt_high",
]


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _build_models():
    """
    Small vs. bigger variants of each model type, all with regularization
    turned UP alongside capacity — the point of this comparison is to test
    whether more parameters actually helps once the real constraint (51
    independent subjects, not 242K independent rows) is respected, not to
    blindly scale up and hope. Reported honestly against each other; there
    is no guarantee bigger wins here, and this script says so either way.
    """
    return {
        "mlp_small": lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32, 16), activation="relu", solver="adam",
            alpha=1e-3, max_iter=2000, early_stopping=True, n_iter_no_change=20, random_state=42,
        ),
        "mlp_big_reg": lambda: MLPRegressor(
            # (256,128,64,32) was tried first and measured (not guessed) to
            # scale worse than linearly with sklearn's single-threaded,
            # no-GPU MLP implementation — extrapolated to ~100+ min/fold,
            # ~8-9 hours total, on this dataset size. Not worth it. This is
            # still meaningfully bigger than mlp_small (128,64,32 vs 64,32,16
            # — roughly 3x the parameters) while staying tractable.
            hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
            alpha=2e-2,  # 20x the L2 penalty of mlp_small — the explicit anti-overfitting lever
            max_iter=1500, early_stopping=True, n_iter_no_change=20,
            validation_fraction=0.15, random_state=42,
        ),
        "gbr_small": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        ),
        "gbr_big_reg": lambda: GradientBoostingRegressor(
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.7,          # stochastic gradient boosting — each tree sees 70% of rows, reduces overfitting
            min_samples_leaf=30,    # forbids trees from carving out leaves small enough to memorize a few rows
            max_features=0.8, random_state=42,
        ),
    }


def main(csv_path, n_folds=5, save_fold_models_dir=None, save_fold_model_name="gbr_small"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURE_COLUMNS + ["gt_hr", "subject_id"])
    print(f"Loaded {len(df)} rows, {df['subject_id'].nunique()} unique subjects, "
          f"{df['group'].nunique()} dataset groups")
    print(df.groupby("group").size())

    X = df[FEATURE_COLUMNS].values
    y = df["gt_hr"].values
    groups = df["subject_id"].values
    baseline_pred = df["bpm"].values  # the classical pipeline's own cycle-level output, unmodified

    n_groups = df["subject_id"].nunique()
    n_folds = min(n_folds, n_groups)
    gkf = GroupKFold(n_splits=n_folds)

    model_builders = _build_models()
    all_err = {name: [] for name in model_builders}
    all_err["baseline"] = []
    all_err["ensemble_big"] = []

    fold_i = 0
    for train_idx, test_idx in gkf.split(X, y, groups):
        fold_i += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        base_test = baseline_pred[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        # MLP training rows are capped, GBR's are not — measured directly
        # (not guessed): sklearn's single-threaded MLP scales worse than
        # linearly with row count on this dataset (74.9s at 20K rows, would
        # extrapolate to ~1.7 HOURS total across 5 folds at the full ~194K
        # rows/fold). This subsample is still ~40K rows drawn from ~41
        # training subjects — a lot more than "51 subjects" worth of
        # independent signal, since rows from the same person are highly
        # redundant to begin with. GBR has no such wall (measured near-linear
        # scaling) so it keeps the full training set.
        MLP_TRAIN_CAP = 40_000
        if len(X_train) > MLP_TRAIN_CAP:
            rng = np.random.RandomState(42)
            mlp_idx = rng.choice(len(X_train), size=MLP_TRAIN_CAP, replace=False)
        else:
            mlp_idx = np.arange(len(X_train))
        X_train_mlp_s, y_train_mlp = X_train_s[mlp_idx], y_train[mlp_idx]

        preds = {}
        for name, builder in model_builders.items():
            model = builder()
            if name.startswith("mlp"):
                model.fit(X_train_mlp_s, y_train_mlp)
                preds[name] = model.predict(X_test_s)
            else:
                model.fit(X_train, y_train)
                preds[name] = model.predict(X_test)

            if save_fold_models_dir and name == save_fold_model_name:
                import joblib
                os.makedirs(save_fold_models_dir, exist_ok=True)
                held_out_subjects_full = sorted(set(groups[test_idx]))
                dump = {"model": model, "feature_columns": FEATURE_COLUMNS, "held_out_subjects": held_out_subjects_full}
                if name.startswith("mlp"):
                    dump["scaler"] = scaler
                joblib.dump(dump, os.path.join(save_fold_models_dir, f"fold_{fold_i}.joblib"))

        # "More networks together" — simple unweighted ensemble of the two
        # biggest/most-regularized models. No meta-learner fit on this fold's
        # own predictions (that would need its own inner CV to stay honest);
        # a plain average is the safe, leakage-free way to combine them.
        preds["ensemble_big"] = 0.5 * preds["mlp_big_reg"] + 0.5 * preds["gbr_big_reg"]

        all_err["baseline"].extend(np.abs(base_test - y_test))
        for name in model_builders:
            all_err[name].extend(np.abs(preds[name] - y_test))
        all_err["ensemble_big"].extend(np.abs(preds["ensemble_big"] - y_test))

        held_out_subjects = sorted(set(groups[test_idx]))
        print(f"\nFold {fold_i}/{n_folds}  (held-out subjects: {held_out_subjects[:5]}"
              f"{'...' if len(held_out_subjects) > 5 else ''}, n={len(test_idx)} rows)")
        print(f"  Classical DSP (no correction) : MAE={mae(base_test, y_test):6.2f}  RMSE={rmse(base_test, y_test):6.2f}")
        for name in model_builders:
            print(f"  + {name:<20}         : MAE={mae(preds[name], y_test):6.2f}  RMSE={rmse(preds[name], y_test):6.2f}")
        print(f"  + ensemble_big (mlp+gbr)      : MAE={mae(preds['ensemble_big'], y_test):6.2f}  RMSE={rmse(preds['ensemble_big'], y_test):6.2f}")

    print("\n" + "=" * 70)
    print(f"OVERALL (pooled across {n_folds} subject-grouped folds, {len(all_err['baseline'])} rows)")
    print("=" * 70)
    print(f"  Classical DSP (no correction) : MAE={np.mean(all_err['baseline']):6.2f}  RMSE={np.sqrt(np.mean(np.square(all_err['baseline']))):6.2f}")
    for name in list(model_builders) + ["ensemble_big"]:
        print(f"  + {name:<20}         : MAE={np.mean(all_err[name]):6.2f}  RMSE={np.sqrt(np.mean(np.square(all_err[name]))):6.2f}")
    print("\nAll numbers are on HELD-OUT subjects never seen during training (GroupKFold by subject_id).")


def train_final_model(csv_path, out_path, model_name="gbr_small"):
    """
    Train the deployed correction model on ALL available data (no holdout).
    The honest generalization estimate for this model is the GroupKFold CV
    number from main() above — NOT re-measured here, since evaluating this
    model on data it was trained on would be meaningless. Standard practice:
    cross-validation estimates how a model of this type/size generalizes;
    the final deployed model then uses all available data to make the best
    use of everything on hand, since none of it needs to be held back once
    the generalization estimate is already in hand.
    """
    import joblib

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURE_COLUMNS + ["gt_hr", "subject_id"])
    X = df[FEATURE_COLUMNS].values
    y = df["gt_hr"].values

    builders = _build_models()
    model = builders[model_name]()
    dump = {"model": model, "feature_columns": FEATURE_COLUMNS}
    if model_name.startswith("mlp"):
        scaler = StandardScaler().fit(X)
        model.fit(scaler.transform(X), y)
        dump["scaler"] = scaler
    else:
        model.fit(X, y)
    joblib.dump(dump, out_path)
    print(f"Final '{model_name}' correction model trained on {len(df)} rows, {df['subject_id'].nunique()} subjects.")
    print(f"Saved to {out_path}")
    print(f"Expected generalization for '{model_name}' — see this run's own GroupKFold results above, "
          f"NOT re-measured on this training data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--save-final", type=str, default=None,
                         help="Also train a final model on ALL data (no holdout) and save it here for deployment.")
    parser.add_argument("--final-model-name", type=str, default="gbr_small",
                         choices=["mlp_small", "mlp_big_reg", "gbr_small", "gbr_big_reg"])
    parser.add_argument("--save-fold-models", type=str, default=None,
                         help="Save each fold's model (+ its held-out subject list) to this directory, "
                              "for honest out-of-fold session-level evaluation (see validate_with_correction.py).")
    parser.add_argument("--fold-model-name", type=str, default="gbr_small",
                         choices=["mlp_small", "mlp_big_reg", "gbr_small", "gbr_big_reg"])
    args = parser.parse_args()
    main(args.csv_path, n_folds=args.folds, save_fold_models_dir=args.save_fold_models,
         save_fold_model_name=args.fold_model_name)
    if args.save_final:
        train_final_model(args.csv_path, args.save_final, model_name=args.final_model_name)
