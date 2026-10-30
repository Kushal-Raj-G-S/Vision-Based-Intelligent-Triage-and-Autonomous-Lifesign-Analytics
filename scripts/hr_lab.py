#!/usr/bin/env python3
"""
HR lab: load the cached feature tables and evaluate estimators at the SESSION level, honestly.

  python scripts/hr_lab.py baselines        # classical, POS, and the OLD hybrid (out-of-fold) on every session

Rules baked in:
  * every learned estimator is evaluated out-of-fold, with folds grouped by SUBJECT (never by row);
  * the session answer is aggregated exactly like production: IQR-filtered median over the session's cycles;
  * results are reported for all sessions, the original 71 ("old") and the newly added ones ("new") separately,
    plus per-condition, per-dataset, and per-HR-band, and at subject level with paired tests.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = Path(__file__).resolve().parent.parent
# HR_TAG selects a parallel experiment (e.g. HR_TAG=mp -> dataset/cache/feat_mp); unset = the main (production) run
TAG = os.environ.get("HR_TAG", "").strip()
FEAT = ROOT / "dataset" / "cache" / ("feat_" + TAG if TAG else "feat")

# --- the original 71-session set (what 9.43 / 6.48 were measured on) -----------------------------------------
OLD_D2 = {f"subject{n}" for n in (1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34,
                                  35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49)}
OLD_PHYS = {"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s11", "s16", "s33"}


def is_old(meta):
    if meta["dataset"] == "ubfc_rppg_d2":
        return meta["subject"] in OLD_D2
    if meta["dataset"] == "ubfc_phys":
        return meta["subject"] in OLD_PHYS
    return meta["subject"] == "d1_after-exercise"


def iqr_median(x):
    """Production aggregation (see agents/tools.py / rppg.get_final_summary): median after an IQR outlier filter."""
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return np.nan
    if len(x) >= 4:
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        keep = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
        if len(keep):
            x = keep
    return float(np.median(x))


def load_all(feat_dir=FEAT):
    S = []
    for f in sorted(Path(feat_dir).glob("*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        import ast
        d["meta"] = ast.literal_eval(str(d["meta"]))
        d["id"] = f.stem
        d["old"] = is_old(d["meta"])
        S.append(d)
    return attach_reference(S)


def attach_reference(S):
    """Add the original and the cleaned ground truth to every session (see hr_data.reference_info)."""
    import hr_data
    by_id = {s["id"]: s for s in hr_data.discover()}
    for d in S:
        ref = hr_data.reference_info(by_id[d["id"]])
        d["gt_session_old"] = float(d["gt_session"])            # as cached (original definition)
        d["gt_session"] = ref["robust"]                          # primary label from here on
        d["unreliable"] = ref["unreliable"]
        d["bad_frac"] = ref["bad_frac"]
        if ref["hr"] is not None:
            d["gt"] = hr_data.clean_series_at(ref, d["t"]).astype(np.float32)
    return S


# ---------------------------------------------------------------------------------------------------------------
def old_hybrid_estimates(S):
    """The ORIGINAL hybrid correction, out-of-fold, reproduced from the cached features.
    Production applies a model per cycle on 10 features only when sqi > 15, then takes the IQR median."""
    from core.ml_compat import load_joblib_compat
    folds = [load_joblib_compat(str(p)) for p in sorted((ROOT / "models" / "cv_folds").glob("*.joblib"))]
    est = {}
    for s in S:
        subj = s["meta"]["subject"].replace("d1_", "")
        own = [b for b in folds if subj in b["held_out_subjects"]]
        use = own if own else folds            # old subject -> its out-of-fold model; unseen subject -> all five
        X = np.stack([s["bpm"], s["sqi"], s["snr_db"], s["stability"], s["rr"], s["rr_conf"], s["hrv"], s["stress"],
                      s["filt_low"], s["filt_high"]], axis=1)
        m = s["sqi"] > 15
        if not m.any():
            est[s["id"]] = np.nan
            continue
        preds = []
        for b in use:
            Xs = b["scaler"].transform(X[m]) if b.get("scaler") is not None else X[m]
            preds.append(b["model"].predict(Xs))
        p = np.mean(preds, axis=0)
        p = p[(p >= 40) & (p <= 180)]
        est[s["id"]] = iqr_median(p)
    return est


def summary3(name, S, est):
    """Compact table: original labels / cleaned labels / cleaned labels without unreliable-reference sessions."""
    def line(label, sel, key):
        rows = [(s, est[s["id"]]) for s in sel if np.isfinite(est[s["id"]]) and np.isfinite(s[key])]
        e = np.array([abs(v - s[key]) for s, v in rows])
        return f"n={len(e):3d}  MAE {e.mean():5.2f}  median {np.median(e):5.2f}  RMSE {np.sqrt((e**2).mean()):5.2f}  <=5: {int((e<=5).sum()):3d}  >20: {int((e>20).sum()):2d}" if len(e) else "n=0"
    print()
    print(name)
    for tag, sel in (("all sessions", S), ("old 71 (original set)", [s for s in S if s["old"]]), ("new sessions", [s for s in S if not s["old"]])):
        print(f"  {tag}")
        print(f"     original labels          {line('o', sel, 'gt_session_old')}")
        print(f"     cleaned labels           {line('c', sel, 'gt_session')}")
        print(f"     cleaned, reliable only   {line('r', [s for s in sel if not s['unreliable']], 'gt_session')}")


def report(name, S, est):
    rows = [(s, est[s["id"]]) for s in S if np.isfinite(est[s["id"]])]
    def stats(sub):
        e = np.array([abs(v - s["gt_session"]) for s, v in sub])
        if len(e) == 0:
            return "n=0"
        return (f"n={len(e):3d}  MAE {e.mean():5.2f}  median {np.median(e):5.2f}  RMSE {np.sqrt((e**2).mean()):5.2f}  "
                f"<=5: {int((e <= 5).sum()):3d}/{len(e)}  >20: {int((e > 20).sum())}")
    print(f"\n{name}")
    print("  all     ", stats(rows))
    print("  old 71  ", stats([r for r in rows if r[0]["old"]]))
    print("  new     ", stats([r for r in rows if not r[0]["old"]]))
    for ds in sorted({r[0]["meta"]["dataset"] for r in rows}):
        print(f"  {ds:14s}", stats([r for r in rows if r[0]["meta"]["dataset"] == ds]))
    for cond in ("rest", "speech", "arithmetic", "exercise"):
        sub = [r for r in rows if r[0]["meta"]["condition"] == cond and r[0]["meta"]["dataset"] == "ubfc_phys"]
        if sub:
            print(f"  phys {cond:10s}", stats(sub))


# ---------------------------------------------------------------------------------------------------------------
def build_tables(S, stride=5):
    import hr_peaks as P
    T = {}
    for s in S:
        T[s["id"]] = P.candidate_table(s, stride=stride)
    return T


def recall_diagnostic(S, T):
    """Ceiling/floor per cycle: does ANY candidate sit within 4 BPM of the truth, and does the production estimate?"""
    print("\nCANDIDATE RECALL (per cycle; ceiling for any selector vs what production already achieves)")
    agg = {}
    for s in S:
        X, y, cyc, bpm = T[s["id"]]
        ok = np.isfinite(s["gt"][cyc])
        if not ok.any():
            continue
        any_hit, prod_hit, tot = 0, 0, 0
        for t in np.unique(cyc):
            m = cyc == t
            g = s["gt"][t]
            if not np.isfinite(g):
                continue
            tot += 1
            any_hit += bool(y[m].max() > 0)
            prod_hit += bool(abs(s["bpm"][t] - g) <= 4.0)
        key = s["meta"]["dataset"] + (":" + s["meta"]["condition"] if s["meta"]["dataset"] == "ubfc_phys" else "")
        a = agg.setdefault(key, [0, 0, 0]); a[0] += any_hit; a[1] += prod_hit; a[2] += tot
    for k, (a, p_, t) in sorted(agg.items()):
        print(f"  {k:24s} cycles {t:6d} | a candidate within 4 BPM: {100*a/t:5.1f}% | production within 4 BPM: {100*p_/t:5.1f}%")


def run_peaks(S, T, n_splits=8, seed=0, tau=None):
    """Grouped-by-subject CV of the peak selector. Returns {id: (argmax_estimate, viterbi_estimate, mean_confidence)}."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    import hr_peaks as P
    ids = [s["id"] for s in S]
    subj = np.array([s["meta"]["subject"] for s in S])
    out = {}
    gkf = GroupKFold(n_splits=min(n_splits, len(set(subj))))
    for k, (tr, te) in enumerate(gkf.split(ids, groups=subj)):
        X = np.concatenate([T[ids[i]][0] for i in tr]); y = np.concatenate([T[ids[i]][1] for i in tr])
        keep = np.isfinite(y)
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, l2_regularization=1.0, random_state=seed)
        clf.fit(X[keep], y[keep])
        for i in te:
            s = S[i]
            Xs, ys, cyc, bpm = T[ids[i]]
            prob = clf.predict_proba(Xs)[:, 1]
            sel_b, sel_p = [], []
            for t in np.unique(cyc):
                m = np.where(cyc == t)[0]
                j = m[np.argmax(prob[m])]
                sel_b.append(bpm[j]); sel_p.append(prob[j])
            est_a = iqr_median(sel_b)
            track = P.viterbi_track(cyc, bpm, prob, len(s["bpm"]), stride=5)
            out[ids[i]] = (est_a, float(np.median(track)), float(np.mean(sel_p)))
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baselines"
    S = load_all()
    print(f"{len(S)} sessions loaded ({sum(s['old'] for s in S)} old, {sum(not s['old'] for s in S)} new)")
    if cmd == "baselines":
        summary3("CLASSICAL (CHROM final_bpm)", S, {s["id"]: float(s["final_chrom"]) for s in S})
        summary3("POS final_bpm", S, {s["id"]: float(s["final_pos"]) for s in S})
        summary3("OLD HYBRID (out-of-fold ML correction, trained on the original 51 people)", S, old_hybrid_estimates(S))
    if cmd == "peaks":
        T = build_tables(S)
        recall_diagnostic(S, T)
        res = run_peaks(S, T)
        summary3("PEAK SELECTOR, per-cycle argmax + IQR median (out-of-fold, grouped by subject)", S, {k: v[0] for k, v in res.items()})
        summary3("PEAK SELECTOR + Viterbi track (out-of-fold, grouped by subject)", S, {k: v[1] for k, v in res.items()})


if __name__ == "__main__":
    main()
