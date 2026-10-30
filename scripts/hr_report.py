#!/usr/bin/env python3
"""
Turn dataset/cache/results_sessions.csv (written by scripts/hr_more.py) into:
  * dataset/cache/HR_RESULTS.md      — slide-ready tables + the verdict, rule by rule (rules fixed in advance)
  * dataset/cache/figures/*.png      — estimate-vs-truth scatter and Bland-Altman plots, MAE by condition

Verdict rules (written down BEFORE the full 125-session results existed):
  R1 reproduction : on the original 71 sessions with the original labels, classical ~= 9.43 and old hybrid ~= 6.48
                    (tolerance 0.75 BPM) — proves the evaluation setup is sound
  R2 it wins      : on all sessions (cleaned labels, reliable references) the selector beats classical AND the old
                    hybrid at the subject level, Wilcoxon p < 0.05 and a positive mean gain
  R3 generalises  : trained on one dataset, tested on the other, it beats classical on the test dataset in BOTH
                    directions (lower MAE, cleaned labels)
"""
import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "dataset" / "cache"
TAG = os.environ.get("HR_TAG", "").strip()          # parallel experiment (e.g. mp = MediaPipe landmarks)
CSV = CACHE / ("results_sessions_" + TAG + ".csv" if TAG else "results_sessions.csv")
OUT = CACHE / ("HR_RESULTS_" + TAG.upper() + ".md" if TAG else "HR_RESULTS.md")
FIG = CACHE / ("figures_" + TAG if TAG else "figures")

METHODS = [("classical", "Classical (CHROM)"), ("old_hybrid", "Old ML correction"),
           ("retrained_regression", "Regression retrained on clean labels"),
           ("selector_argmax", "Peak selector (new)"), ("selector_weighted", "Peak selector, weighted"),
           ("selector_abstain", "Peak selector, abstain")]


def load():
    rows = []
    with open(CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                if k not in ("session", "dataset", "subject", "condition"):
                    r[k] = float(v) if v not in ("", None) else np.nan
            r["source"] = "UBFC-Phys" if r["dataset"] == "ubfc_phys" else "UBFC-rPPG"
            rows.append(r)
    return rows


def metrics(rows, col, truth):
    p = np.array([(r[col], r[truth]) for r in rows if np.isfinite(r.get(col, np.nan)) and np.isfinite(r[truth])])
    if len(p) < 2:
        return None
    d = p[:, 0] - p[:, 1]
    a = np.abs(d)
    r = np.corrcoef(p[:, 0], p[:, 1])[0, 1] if len(p) > 2 else np.nan
    return dict(n=len(d), mae=a.mean(), med=np.median(a), rmse=np.sqrt((d ** 2).mean()), r=r, bias=d.mean(),
                loa=1.96 * d.std(ddof=1), w5=int((a <= 5).sum()), f20=int((a > 20).sum()))


def row_md(name, m):
    if m is None:
        return f"| {name} | - | - | - | - | - | - | - |"
    return (f"| {name} | {m['n']} | **{m['mae']:.2f}** | {m['med']:.2f} | {m['rmse']:.2f} | {m['r']:.3f} | "
            f"{m['bias']:+.2f} ± {m['loa']:.2f} | {m['w5']} / {m['f20']} |")


def table(rows, truth, title):
    out = [f"\n#### {title}", "", "| Method | n | MAE | median | RMSE | Pearson r | bias ± 95% LoA | ≤5 BPM / >20 BPM |",
           "| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |"]
    for col, name in METHODS:
        out.append(row_md(name, metrics(rows, col, truth)))
    return out


def paired(rows, a, b):
    """Subject-level paired gain of b over a (positive = b better). Cleaned labels, reliable references."""
    from scipy.stats import wilcoxon
    by = {}
    for r in rows:
        if r["reference_unreliable"] or not np.isfinite(r["truth_cleaned"]):
            continue
        ea, eb = abs(r[a] - r["truth_cleaned"]), abs(r[b] - r["truth_cleaned"])
        if np.isfinite(ea) and np.isfinite(eb):
            by.setdefault(r["subject"], []).append((ea, eb))
    d = np.array([np.mean([x[0] for x in v]) - np.mean([x[1] for x in v]) for v in by.values()])
    if len(d) < 5:
        return None
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(n=len(d), mean=d.mean(), lo=lo, hi=hi, p=p, better=int((d > 0).sum()), worse=int((d < 0).sum()))


def verdict(rows):
    lines = ["\n## Verdict (rules fixed before the full results)", ""]
    ok = []
    old = [r for r in rows if r["in_old_71"] == 1]
    c = metrics(old, "classical", "truth_original")
    h = metrics(old, "old_hybrid", "truth_original")
    if c and h:
        r1 = abs(c["mae"] - 9.43) <= 0.75 and abs(h["mae"] - 6.48) <= 0.75
        lines.append(f"- **R1 reproduction — {'PASS' if r1 else 'FAIL'}**: old {c['n']} sessions, original labels: classical "
                     f"{c['mae']:.2f} (was 9.43), old hybrid {h['mae']:.2f} (was 6.48).")
        ok.append(r1)
    pc, ph = paired(rows, "classical", "selector_argmax"), paired(rows, "old_hybrid", "selector_argmax")
    if pc and ph:
        r2 = pc["p"] < 0.05 and pc["mean"] > 0 and ph["p"] < 0.05 and ph["mean"] > 0
        lines.append(f"- **R2 it wins — {'PASS' if r2 else 'FAIL'}**: vs classical {pc['mean']:+.2f} BPM per subject "
                     f"(95% CI {pc['lo']:+.2f} to {pc['hi']:+.2f}, p = {pc['p']:.2g}, better/worse {pc['better']}/{pc['worse']} of {pc['n']}); "
                     f"vs old hybrid {ph['mean']:+.2f} (CI {ph['lo']:+.2f} to {ph['hi']:+.2f}, p = {ph['p']:.2g}).")
        ok.append(r2)
    r3_parts, r3 = [], True
    for test_src in ("UBFC-Phys", "UBFC-rPPG"):
        sub = [r for r in rows if r["source"] == test_src and np.isfinite(r.get("selector_cross_dataset", np.nan))]
        mx, mc = metrics(sub, "selector_cross_dataset", "truth_cleaned"), metrics(sub, "classical", "truth_cleaned")
        if mx is None or mc is None:
            r3 = False
            r3_parts.append(f"tested on {test_src}: not available")
            continue
        win = mx["mae"] < mc["mae"]
        r3 = r3 and win
        r3_parts.append(f"tested on {test_src}: selector {mx['mae']:.2f} vs classical {mc['mae']:.2f} ({'better' if win else 'NOT better'})")
    lines.append(f"- **R3 generalises across datasets — {'PASS' if r3 else 'FAIL'}**: " + "; ".join(r3_parts) + ".")
    ok.append(r3)
    overall = all(ok) and len(ok) == 3
    lines += ["", f"**Overall: {'all three rules pass — the improvement can be claimed.' if overall else 'not every rule passes — claim only what passed (see above).'}**"]
    return lines


def figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    show = [("classical", "Classical (CHROM)", "#8a8f98"), ("old_hybrid", "Old ML correction", "#d98b2b"),
            ("selector_argmax", "Peak selector (new)", "#f26a1b")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    for j, (col, name, colr) in enumerate(show):
        p = np.array([(r["truth_cleaned"], r[col], r["source"] == "UBFC-Phys") for r in rows
                      if np.isfinite(r.get(col, np.nan)) and np.isfinite(r["truth_cleaned"])])
        if len(p) < 3:
            continue
        g, e, ph = p[:, 0], p[:, 1], p[:, 2].astype(bool)
        m = metrics(rows, col, "truth_cleaned")
        a = ax[0, j]
        a.scatter(g[~ph], e[~ph], s=22, color=colr, alpha=0.85, label="UBFC-rPPG")
        a.scatter(g[ph], e[ph], s=22, facecolors="none", edgecolors=colr, alpha=0.9, label="UBFC-Phys")
        lim = [40, 150]
        a.plot(lim, lim, color="#333", lw=1)
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_title(f"{name}\nMAE {m['mae']:.2f} BPM, r = {m['r']:.3f}")
        a.set_xlabel("Reference heart rate (BPM)"); a.set_ylabel("Estimated heart rate (BPM)")
        a.legend(loc="upper left", fontsize=8); a.grid(alpha=0.25)
        b = ax[1, j]
        mean, diff = (g + e) / 2, e - g
        b.scatter(mean[~ph], diff[~ph], s=22, color=colr, alpha=0.85)
        b.scatter(mean[ph], diff[ph], s=22, facecolors="none", edgecolors=colr, alpha=0.9)
        for y, ls in ((m["bias"], "-"), (m["bias"] + m["loa"], "--"), (m["bias"] - m["loa"], "--")):
            b.axhline(y, color="#333", ls=ls, lw=1)
        b.set_ylim(-60, 60)
        b.set_title(f"Bland-Altman: bias {m['bias']:+.2f}, 95% LoA ±{m['loa']:.1f} BPM")
        b.set_xlabel("Mean of estimate and reference (BPM)"); b.set_ylabel("Estimate − reference (BPM)")
        b.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "hr_agreement.png", dpi=150)
    plt.close(fig)

    groups = [("UBFC-rPPG", lambda r: r["source"] == "UBFC-rPPG"),
              ("Phys rest", lambda r: r["source"] == "UBFC-Phys" and r["condition"] == "rest"),
              ("Phys speech", lambda r: r["source"] == "UBFC-Phys" and r["condition"] == "speech"),
              ("Phys arithmetic", lambda r: r["source"] == "UBFC-Phys" and r["condition"] == "arithmetic")]
    fig, a = plt.subplots(figsize=(10, 5))
    w = 0.26
    for k, (col, name, colr) in enumerate(show):
        vals = []
        for gname, fn in groups:
            m = metrics([r for r in rows if fn(r)], col, "truth_cleaned")
            vals.append(m["mae"] if m else 0)
        x = np.arange(len(groups)) + (k - 1) * w
        a.bar(x, vals, w, color=colr, label=name)
        for xi, v in zip(x, vals):
            a.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    a.set_xticks(np.arange(len(groups))); a.set_xticklabels([g for g, _ in groups])
    a.set_ylabel("MAE (BPM), cleaned labels"); a.set_title("Heart-rate error by dataset and condition")
    a.legend(); a.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "hr_mae_by_condition.png", dpi=150)
    plt.close(fig)


def main():
    rows = load()
    md = [f"# HR results — {len(rows)} sessions, {len({r['subject'] for r in rows})} subjects",
          "", "Out-of-fold, grouped by subject (no person is ever scored by a model that trained on them).",
          "Labels: *original* = old definition (comparable to earlier numbers); *cleaned* = median of valid oximeter readings;",
          "*reliable* = cleaned, excluding sessions whose reference had >10% impossible readings."]
    md += verdict(rows)
    md += ["\n## Tables"]
    for src in ("all", "UBFC-rPPG", "UBFC-Phys"):
        sub = rows if src == "all" else [r for r in rows if r["source"] == src]
        if not sub:
            continue
        md += table(sub, "truth_cleaned", f"{src} — cleaned labels")
        md += table(sub, "truth_original", f"{src} — original labels")
        md += table([r for r in sub if not r["reference_unreliable"]], "truth_cleaned", f"{src} — cleaned, reliable references only")
    old = [r for r in rows if r["in_old_71"] == 1]
    md += table(old, "truth_original", f"Original 71-session set — original labels (compare with 9.43 / 6.48)")
    try:
        figures(rows)
        md += ["\n## Figures", "", "- `figures/hr_agreement.png` — estimate vs reference, and Bland-Altman, for classical / old ML / peak selector",
               "- `figures/hr_mae_by_condition.png` — MAE by dataset and UBFC-Phys condition"]
    except Exception as exc:          # figures are a convenience; never block the tables/verdict
        md += [f"\n(figures not drawn: {exc!r})"]
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    # the Windows console is cp1252 and cannot print ≤ / ± / −; the .md file keeps them
    import sys
    sys.stdout.buffer.write(("\n".join(md) + "\n").encode("utf-8", "replace"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
