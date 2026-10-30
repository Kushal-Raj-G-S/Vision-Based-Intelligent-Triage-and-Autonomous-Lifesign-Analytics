#!/usr/bin/env python3
"""
Follow-up HR experiments, run automatically after the main chain (dataset/cache/run_after.sh).

Questions it answers about the peak selector (scripts/hr_peaks.py):
  1. Is it better than classical / the old hybrid at the SUBJECT level, with a CI and a paired test?
  2. Does it beat a fair rival: the regression correction retrained on the cleaned labels?
  3. Does it GENERALISE across data sources (train on UBFC-rPPG -> test on UBFC-Phys, and the reverse)?
     A learned RR peak re-ranker passed within-source CV and failed this test (CLAUDE.md), so this is decisive.
  4. Where does it still fail?
Also reports the metrics papers use besides MAE (Pearson r, Bland-Altman bias and 95% limits of agreement) and
writes every session's estimates to dataset/cache/results_sessions.csv for tables and plots.

Selector aggregation variants were fixed in advance (designed on UBFC-rPPG only; UBFC-Phys never seen):
  argmax   : IQR median of each cycle's top-scored candidate (the original)
  weighted : median weighted by the top candidate's score
  abstain  : like argmax but only cycles whose top score >= 0.5 (falls back to all cycles if fewer than 20%)

    python scripts/hr_more.py > dataset/cache/results_more.txt
"""
import csv
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hr_lab as L          # noqa: E402
import hr_peaks as P        # noqa: E402

REG_FEATS = ["bpm", "sqi", "snr_db", "stability", "harmonic_ratio", "filt_low", "filt_high", "motion", "rr", "rr_conf", "hrv", "stress"]
OUT_CSV = L.ROOT / "dataset" / "cache" / ("results_sessions_" + L.TAG + ".csv" if L.TAG else "results_sessions.csv")


def fit_selector(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    keep = np.isfinite(y)
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, l2_regularization=1.0,
                                          random_state=0).fit(X[keep], y[keep])


def weighted_median(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2.0)])


def aggregate(clf, table):
    """Session estimates from one fitted selector: (argmax, weighted, abstain)."""
    X, _y, cyc, bpm = table
    prob = clf.predict_proba(X)[:, 1]
    sb, sp = [], []
    for t in np.unique(cyc):
        m = np.where(cyc == t)[0]
        j = m[np.argmax(prob[m])]
        sb.append(bpm[j]); sp.append(prob[j])
    sb, sp = np.array(sb), np.array(sp)
    a = L.iqr_median(sb)
    w = weighted_median(sb, sp + 1e-6) if len(sb) else np.nan
    keep = sp >= 0.5
    ab = L.iqr_median(sb[keep]) if keep.mean() >= 0.2 else a
    return a, w, ab


def selector_oof(S, T, n_splits=8):
    from sklearn.model_selection import GroupKFold
    ids = [s["id"] for s in S]
    subj = np.array([s["meta"]["subject"] for s in S])
    out = {"argmax": {}, "weighted": {}, "abstain": {}}
    for tr, te in GroupKFold(min(n_splits, len(set(subj)))).split(ids, groups=subj):
        clf = fit_selector(np.concatenate([T[ids[i]][0] for i in tr]), np.concatenate([T[ids[i]][1] for i in tr]))
        for i in te:
            a, w, ab = aggregate(clf, T[ids[i]])
            out["argmax"][ids[i]], out["weighted"][ids[i]], out["abstain"][ids[i]] = a, w, ab
    return out


def cross_dataset(S, T):
    groups = {"UBFC-rPPG": [s for s in S if s["meta"]["dataset"].startswith("ubfc_rppg")],
              "UBFC-Phys": [s for s in S if s["meta"]["dataset"] == "ubfc_phys"]}
    res = {}
    for train_name, test_name in (("UBFC-rPPG", "UBFC-Phys"), ("UBFC-Phys", "UBFC-rPPG")):
        tr, te = groups[train_name], groups[test_name]
        if not tr or not te:
            print(f"\nCROSS-DATASET {train_name} -> {test_name}: skipped (no sessions)")
            continue
        clf = fit_selector(np.concatenate([T[s["id"]][0] for s in tr]), np.concatenate([T[s["id"]][1] for s in tr]))
        est = {s["id"]: aggregate(clf, T[s["id"]])[0] for s in te}
        res.update({k: v for k, v in est.items()})
        L.summary3(f"CROSS-DATASET: selector trained ONLY on {train_name}, tested on {test_name}", te, est)
        L.summary3(f"   (classical on the same {test_name} sessions, for reference)", te, {s["id"]: float(s["final_chrom"]) for s in te})
        for cond in ("rest", "speech", "arithmetic"):
            sub = [s for s in te if s["meta"]["condition"] == cond and s["meta"]["dataset"] == "ubfc_phys"]
            if sub:
                e = np.array([abs(est[s["id"]] - s["gt_session"]) for s in sub if np.isfinite(s["gt_session"])])
                c = np.array([abs(float(s["final_chrom"]) - s["gt_session"]) for s in sub if np.isfinite(s["gt_session"])])
                print(f"     {test_name} {cond:10s} n={len(e):2d}  selector MAE {e.mean():5.2f}  classical MAE {c.mean():5.2f}")
    return res


def regression_oof(S, n_splits=8):
    """The fair rival: a regression correction (like the old hybrid) but retrained on the cleaned labels."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    def feats(s):
        cols = [s[k] for k in REG_FEATS] + [s["pos_bpm"], s["pos_sqi"], np.abs(s["bpm"] - s["pos_bpm"])]
        return np.stack(cols, 1)[::5], s["gt"][::5]

    ids = [s["id"] for s in S]
    subj = np.array([s["meta"]["subject"] for s in S])
    est = {}
    for tr, te in GroupKFold(min(n_splits, len(set(subj)))).split(ids, groups=subj):
        X = np.concatenate([feats(S[i])[0] for i in tr])
        y = np.concatenate([feats(S[i])[1] for i in tr])
        k = np.isfinite(y)
        m = HistGradientBoostingRegressor(loss="absolute_error", max_depth=4, learning_rate=0.08, max_iter=300,
                                          random_state=0).fit(X[k], y[k])
        for i in te:
            est[ids[i]] = L.iqr_median(m.predict(feats(S[i])[0]))
    return est


def agreement(S, est, name):
    """Pearson r and Bland-Altman (bias, 95% limits of agreement) on cleaned labels, all sessions."""
    pairs = np.array([(est[s["id"]], s["gt_session"]) for s in S
                      if np.isfinite(est.get(s["id"], np.nan)) and np.isfinite(s["gt_session"])])
    if len(pairs) < 3:
        return
    e, g = pairs[:, 0], pairs[:, 1]
    d = e - g
    r = np.corrcoef(e, g)[0, 1]
    print(f"  {name:28s} n={len(d):3d}  MAE {np.abs(d).mean():5.2f}  RMSE {np.sqrt((d**2).mean()):5.2f}  Pearson r {r:5.3f}  "
          f"bias {d.mean():+5.2f}  95% LoA [{d.mean() - 1.96 * d.std(ddof=1):+6.2f}, {d.mean() + 1.96 * d.std(ddof=1):+6.2f}]")


def paired(S, a, b, name_a, name_b):
    """Subject-level paired comparison on cleaned labels, reliable references only (positive = b better)."""
    from scipy.stats import wilcoxon
    by = {}
    for s in S:
        if s["unreliable"] or not np.isfinite(s["gt_session"]):
            continue
        ea, eb = abs(a.get(s["id"], np.nan) - s["gt_session"]), abs(b.get(s["id"], np.nan) - s["gt_session"])
        if np.isfinite(ea) and np.isfinite(eb):
            by.setdefault(s["meta"]["subject"], []).append((ea, eb))
    d = np.array([np.mean([x[0] for x in v]) - np.mean([x[1] for x in v]) for v in by.values()])
    if len(d) < 3:
        print(f"\nPAIRED {name_a} vs {name_b}: too few subjects")
        return
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
    print(f"\nPAIRED, subject level ({name_a} error minus {name_b} error; cleaned labels, reliable references only)")
    print(f"  subjects {len(d)} | mean {d.mean():.2f} BPM, 95% CI [{lo:.2f}, {hi:.2f}] | median {np.median(d):.2f} | "
          f"{name_b} better for {int((d > 0).sum())}, worse for {int((d < 0).sum())} | Wilcoxon p = {p:.2g}")


def worst(S, est, name, n=12):
    rows = [(abs(est[s["id"]] - s["gt_session"]), s) for s in S
            if np.isfinite(est.get(s["id"], np.nan)) and np.isfinite(s["gt_session"])]
    rows.sort(key=lambda r: -r[0])
    print(f"\nWORST {n} SESSIONS for {name} (cleaned labels)")
    for e, s in rows[:n]:
        flag = "  (unreliable reference)" if s["unreliable"] else ""
        print(f"  {s['id']:22s} {s['meta']['condition']:10s} truth {s['gt_session']:6.1f}  {name} {est[s['id']]:6.1f}  "
              f"classical {float(s['final_chrom']):6.1f}  error {e:5.1f}{flag}")


def write_csv(S, cols):
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session", "dataset", "subject", "condition", "in_old_71", "reference_unreliable",
                    "truth_original", "truth_cleaned"] + list(cols))
        for s in S:
            w.writerow([s["id"], s["meta"]["dataset"], s["meta"]["subject"], s["meta"]["condition"], int(s["old"]),
                        int(s["unreliable"]), round(float(s["gt_session_old"]), 2), round(float(s["gt_session"]), 2)]
                       + [("" if not np.isfinite(cols[c].get(s["id"], np.nan)) else round(float(cols[c][s["id"]]), 2)) for c in cols])
    print(f"\nper-session estimates written to {OUT_CSV}")


def main():
    S = L.load_all()
    print(f"{len(S)} sessions, {len({s['meta']['subject'] for s in S})} subjects")
    T = L.build_tables(S)
    classical = {s["id"]: float(s["final_chrom"]) for s in S}
    pos = {s["id"]: float(s["final_pos"]) for s in S}
    sel = selector_oof(S, T)
    old_h = L.old_hybrid_estimates(S)
    reg = regression_oof(S)

    for k in ("argmax", "weighted", "abstain"):
        L.summary3(f"PEAK SELECTOR [{k}] (out-of-fold, grouped by subject)", S, sel[k])
    L.summary3("RETRAINED REGRESSION CORRECTION on cleaned labels (out-of-fold) - the fair rival", S, reg)
    L.report("PEAK SELECTOR [argmax] by dataset and condition (cleaned labels)", S, sel["argmax"])

    print("\nAGREEMENT WITH THE REFERENCE (cleaned labels, all sessions)")
    for name, est in (("classical (CHROM)", classical), ("POS", pos), ("old hybrid", old_h),
                      ("retrained regression", reg), ("selector [argmax]", sel["argmax"]),
                      ("selector [weighted]", sel["weighted"]), ("selector [abstain]", sel["abstain"])):
        agreement(S, est, name)

    paired(S, classical, sel["argmax"], "classical", "selector")
    paired(S, old_h, sel["argmax"], "old hybrid", "selector")
    paired(S, reg, sel["argmax"], "retrained regression", "selector")
    xds = cross_dataset(S, T)
    worst(S, sel["argmax"], "selector")
    write_csv(S, {"classical": classical, "pos": pos, "old_hybrid": old_h, "retrained_regression": reg,
                  "selector_argmax": sel["argmax"], "selector_weighted": sel["weighted"],
                  "selector_abstain": sel["abstain"], "selector_cross_dataset": xds})


if __name__ == "__main__":
    main()
