"""
Subject-level significance check for the classical-vs-hybrid MAE claim.

Parses the existing per-session run log (no reprocessing/retraining) and
answers the question raised in data/evolution/16-subject-level-significance.md:
is the aggregate 31% MAE reduction driven broadly, or by a few outlier
subjects? See that file for the full writeup and interpretation.

Usage:
    python scripts/analyze_subject_significance.py
"""
import os
import re
import numpy as np
from scipy import stats

path = os.path.join(os.path.dirname(__file__), "..", "test", "validation_run11_huber_full71.log")

rows = []
with open(path, encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

video_line_re = re.compile(r"^\[\d\d:\d\d:\d\d\] Processing (.+)\.\.\.$")
data_line_re = re.compile(r"^test/\S*\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\S+)")

pending_path = None
for line in lines:
    m = video_line_re.match(line.strip())
    if m:
        pending_path = m.group(1)
        continue
    m = data_line_re.match(line.strip())
    if m and pending_path:
        gt, classical, c_err, hybrid, h_err, group = m.groups()
        rows.append({
            "path": pending_path,
            "gt": float(gt), "classical": float(classical), "c_err": float(c_err),
            "hybrid": float(hybrid), "h_err": float(h_err), "group": group,
        })
        pending_path = None

print(f"Parsed {len(rows)} sessions")

def subject_id(path):
    # test/dataset_1/after-exercise/vid-001.avi -> dataset_1/after-exercise
    # test/dataset_2/subject1/vid.avi -> dataset_2/subject1
    # test/ubfc_phys/s4/s4/vid_s4_T3.avi -> ubfc_phys/s4
    parts = path.split("/")
    if parts[1] == "ubfc_phys":
        return f"ubfc_phys/{parts[2]}"
    else:
        return f"{parts[1]}/{parts[2]}"

for r in rows:
    r["subject"] = subject_id(r["path"])

n_subjects = len(set(r["subject"] for r in rows))
print(f"Unique subjects: {n_subjects}")

# ---- session-level paired delta ----
c_errs = np.array([r["c_err"] for r in rows])
h_errs = np.array([r["h_err"] for r in rows])
delta = c_errs - h_errs  # positive = hybrid better

print("\n=== SESSION-LEVEL (n=71, paired classical-error minus hybrid-error) ===")
print(f"Mean delta: {delta.mean():.2f}")
print(f"Median delta: {np.median(delta):.2f}")
q1, q3 = np.percentile(delta, [25, 75])
print(f"IQR: [{q1:.2f}, {q3:.2f}]")
print(f"Sessions where hybrid is better (delta>0): {(delta>0).sum()}/{len(delta)}")
print(f"Sessions where hybrid is worse (delta<0): {(delta<0).sum()}/{len(delta)}")

# bootstrap CI on mean delta
rng = np.random.default_rng(42)
boot_means = []
for _ in range(10000):
    sample = rng.choice(delta, size=len(delta), replace=True)
    boot_means.append(sample.mean())
boot_means = np.array(boot_means)
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
print(f"Bootstrap 95% CI on mean delta: [{ci_lo:.2f}, {ci_hi:.2f}]")

wilcoxon_stat, wilcoxon_p = stats.wilcoxon(c_errs, h_errs)
print(f"Wilcoxon signed-rank test (classical vs hybrid session errors): statistic={wilcoxon_stat:.1f}, p={wilcoxon_p:.6f}")

# ---- subject-level aggregation (mean error per subject, to avoid within-subject correlation) ----
subjects = {}
for r in rows:
    subjects.setdefault(r["subject"], {"c": [], "h": []})
    subjects[r["subject"]]["c"].append(r["c_err"])
    subjects[r["subject"]]["h"].append(r["h_err"])

subj_c = np.array([np.mean(v["c"]) for v in subjects.values()])
subj_h = np.array([np.mean(v["h"]) for v in subjects.values()])
subj_delta = subj_c - subj_h

print(f"\n=== SUBJECT-LEVEL (n={len(subjects)}, mean error per subject, paired) ===")
print(f"Mean delta: {subj_delta.mean():.2f}")
print(f"Median delta: {np.median(subj_delta):.2f}")
q1, q3 = np.percentile(subj_delta, [25, 75])
print(f"IQR: [{q1:.2f}, {q3:.2f}]")
print(f"Subjects where hybrid is better: {(subj_delta>0).sum()}/{len(subj_delta)}")
print(f"Subjects where hybrid is worse: {(subj_delta<0).sum()}/{len(subj_delta)}")

boot_means_subj = []
for _ in range(10000):
    sample = rng.choice(subj_delta, size=len(subj_delta), replace=True)
    boot_means_subj.append(sample.mean())
boot_means_subj = np.array(boot_means_subj)
ci_lo_s, ci_hi_s = np.percentile(boot_means_subj, [2.5, 97.5])
print(f"Bootstrap 95% CI on mean subject-level delta: [{ci_lo_s:.2f}, {ci_hi_s:.2f}]")

w_stat_s, w_p_s = stats.wilcoxon(subj_c, subj_h)
print(f"Wilcoxon signed-rank test (subject-level): statistic={w_stat_s:.1f}, p={w_p_s:.6f}")

# ---- check if the gain is driven by a handful of catastrophic subjects ----
sorted_delta = np.sort(subj_delta)[::-1]
top5_sum = sorted_delta[:5].sum()
total_sum = subj_delta.sum()
print(f"\nTop-5-subject share of total subject-level delta: {top5_sum/total_sum*100:.1f}%")
print("Top 5 deltas:", [round(x,1) for x in sorted_delta[:5]])
print("Bottom 5 deltas:", [round(x,1) for x in sorted_delta[-5:]])
