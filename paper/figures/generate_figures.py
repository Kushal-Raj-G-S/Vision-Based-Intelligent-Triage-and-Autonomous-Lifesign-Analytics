"""
Generates the data figures for paper/vital_paper.tex from the project's own
validated numbers (README.md / CLAUDE.md / data/evolution/*). No numbers here
are invented -- each one matches a table already in the paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
})

# Vivid palette shared with the paper's TikZ diagram (blue / teal / purple / coral / pink)
SLATE = "#073B4C"
BLUE = "#3A86FF"
TEAL = "#06D6A0"
PURPLE = "#7209B7"
CORAL = "#EF476F"
AMBER = "#FFD166"
PINK = "#F72585"
GRAY = "#8D99AE"


def per_condition_chart():
    conditions = ["Speech", "Arithmetic", "UBFC-rPPG\n(unlabeled)", "Rest"]
    classical = [21.0, 10.4, 7.4, 5.1]
    hybrid = [7.8, 7.0, 6.1, 6.1]
    ns = [10, 10, 41, 10]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    b1 = ax.bar(x - width / 2, classical, width, label="Classical DSP", color=PURPLE, edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + width / 2, hybrid, width, label="Hybrid (DSP + ML)", color=TEAL, edgecolor="white", linewidth=0.6)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(conditions, ns)], fontsize=9)
    ax.set_ylabel("MAE (BPM)")
    ax.set_ylim(0, 24)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("per_condition_mae.png", dpi=400)
    plt.close(fig)


def faithfulness_chart():
    fields = ["JSON\nparses", "ESI level\ncorrect", "Priority score\ncorrect", "Shock flag\ncorrect",
              "Diagnosis\nexact wording", "Diagnosis\ncorrect meaning"]
    rates = [93.8, 81.2, 6.2, 93.8, 37.5, 75.0]
    colors = [BLUE, BLUE, PINK, BLUE, BLUE, BLUE]

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    bars = ax.bar(fields, rates, color=colors, edgecolor="white", linewidth=0.6, width=0.6)
    for bar, r in zip(bars, rates):
        ax.annotate(f"{r:.1f}%", (bar.get_x() + bar.get_width() / 2, r),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")

    ax.set_ylabel("Faithfulness rate (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("agent_faithfulness.png", dpi=400)
    plt.close(fig)


def bugfix_chart():
    subjects = ["subject41", "subject46", "subject49", "subject9", "subject34"]
    before = [52.6, 41.2, 36.5, 60.9, 63.0]
    after = [8.2, 1.3, 0.4, 2.8, 0.5]

    x = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    b1 = ax.bar(x - width / 2, before, width, label="Before fix", color=CORAL, edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + width / 2, after, width, label="After fix", color=TEAL, edgecolor="white", linewidth=0.6)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=9)
    ax.set_ylabel("BPM error")
    ax.set_ylim(0, 78)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("harmonic_fix_subjects.png", dpi=400)
    plt.close(fig)


def _box(ax, cx, cy, w, h, color, text, textcolor="white", fontsize=13.5, zorder=3):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.16",
                          facecolor=color, edgecolor="white", linewidth=1.6,
                          zorder=zorder)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", color=textcolor,
             fontsize=fontsize, fontweight="bold", linespacing=1.7, zorder=zorder + 1)
    return box


def _arrow(ax, p1, p2, color=SLATE, lw=2.6, zorder=2):
    arr = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=20,
                           linewidth=lw, color=color, zorder=zorder,
                           shrinkA=3, shrinkB=3)
    ax.add_patch(arr)


def _cylinder(ax, cx, cy, w, h, color, text, fontsize=12.5):
    body = Rectangle((cx - w / 2, cy - h / 2), w, h * 0.78, facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(body)
    ax.add_patch(Ellipse((cx, cy - h / 2 + h * 0.78), w, h * 0.22, facecolor=color, edgecolor="white", linewidth=1.4, zorder=3))
    ax.add_patch(Ellipse((cx, cy - h / 2), w, h * 0.22, facecolor=color, edgecolor="white", linewidth=1.4, zorder=4))
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h * 0.78, facecolor="none", edgecolor="white", linewidth=1.4, zorder=4))
    ax.text(cx, cy + 0.08, text, ha="center", va="center", color="white", fontsize=fontsize,
            fontweight="bold", linespacing=1.6, zorder=5)


def architecture_diagram():
    # ---- lay out every x-position first, so the canvas width is exact, not guessed ----
    y = 4.4
    in_x, roi_x, chrom_x, ml_x, clin_x, crew_x = 2.3, 7.6, 12.9, 18.2, 23.9, 29.8
    dash_x = 35.3
    in_hw, roi_hw, chrom_hw, ml_hw, clin_hw, crew_hw, dash_hw = 1.7, 1.8, 1.8, 1.8, 2.0, 1.9, 1.6
    TOTAL_W = dash_x + dash_hw + 0.7
    H = 9.6

    fig, ax = plt.subplots(figsize=(TOTAL_W * 0.56, 6.8))
    ax.set_xlim(0, TOTAL_W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- legend strip along the top: fixed-width slots spanning the FULL canvas ----
    ax.add_patch(Rectangle((0, H - 1.15), TOTAL_W, 1.15, facecolor="#F1F3F5", edgecolor="none", zorder=1))
    legend_items = [
        (BLUE, "Perception input"),
        (TEAL, "Classical rPPG signal"),
        (AMBER, "Learned correction"),
        (CORAL, "Deterministic clinical safety"),
        (PINK, "LLM narration (restates only)"),
    ]
    slot = TOTAL_W / len(legend_items)
    for i, (color, label) in enumerate(legend_items):
        lx = i * slot + 0.5
        ax.add_patch(Rectangle((lx, H - 0.78), 0.5, 0.5, facecolor=color, edgecolor="none", zorder=2))
        ax.text(lx + 0.68, H - 0.53, label, ha="left", va="center", fontsize=13, fontweight="bold", color=SLATE, zorder=2)

    # ---- input fork ----
    _box(ax, in_x, 6.75, 3.4, 1.8, BLUE, "Webcam\n(live)")
    _box(ax, in_x, 2.1, 3.4, 1.8, BLUE, "Uploaded\nVideo")
    ax.add_patch(Circle((in_x - 1.35, 7.3), 0.17, facecolor="white", edgecolor=SLATE, linewidth=1.5, zorder=4))
    ax.add_patch(Circle((in_x - 1.35, 7.3), 0.075, facecolor=SLATE, edgecolor="none", zorder=4))
    ax.add_patch(mpatches.RegularPolygon((in_x - 1.35, 2.65), numVertices=3, radius=0.18,
                                          orientation=np.pi, facecolor="white", edgecolor=SLATE,
                                          linewidth=1.5, zorder=4))

    # ---- pipeline row ----
    _box(ax, roi_x, y, 3.6, 2.2, TEAL, "MediaPipe\nROI Extraction")
    _box(ax, chrom_x, y, 3.6, 2.2, TEAL, "CHROM\nrPPG Engine")
    _box(ax, ml_x, y, 3.6, 2.2, AMBER, "ML\nCorrection", textcolor=SLATE)
    _box(ax, clin_x, y, 4.0, 3.0, CORAL, "Deterministic\nClinical Engine", fontsize=15.5)
    _box(ax, crew_x, y, 3.8, 2.6, PINK, "CrewAI\nNarration", fontsize=14)

    # ---- output fork ----
    _box(ax, dash_x, 6.75, 3.2, 1.7, SLATE, "Dashboard\n/ ARIA", fontsize=13.5)
    _cylinder(ax, dash_x, 2.1, 2.9, 2.1, SLATE, "Neon\nPostgres")

    # ---- arrows: each endpoint sits exactly on the neighbor's own edge ----
    _arrow(ax, (in_x + in_hw, 6.35), (roi_x - roi_hw + 0.1, 5.15))
    _arrow(ax, (in_x + in_hw, 2.5), (roi_x - roi_hw + 0.1, 3.65))
    _arrow(ax, (roi_x + roi_hw, y), (chrom_x - chrom_hw, y))
    _arrow(ax, (chrom_x + chrom_hw, y), (ml_x - ml_hw, y))
    _arrow(ax, (ml_x + ml_hw, y), (clin_x - clin_hw, y))
    _arrow(ax, (clin_x + clin_hw, y), (crew_x - crew_hw, y))
    _arrow(ax, (crew_x + crew_hw, 5.0), (dash_x - dash_hw, 6.3))
    _arrow(ax, (crew_x + crew_hw, 3.8), (dash_x - 1.45, 2.65))

    # ---- measured-number badges: narrow, tucked under their own box only, well clear of neighbors ----
    for (bx, by, txt, w) in [(ml_x, y - 1.85, "9.43 -> 6.48 MAE (31%)", 3.6),
                              (crew_x, y - 2.15, "6.2% faithful", 2.6)]:
        ax.add_patch(FancyBboxPatch((bx - w / 2, by - 0.3), w, 0.6,
                                     boxstyle="round,pad=0.02,rounding_size=0.1",
                                     facecolor=SLATE, edgecolor="none", zorder=5))
        ax.text(bx, by, txt, ha="center", va="center", color="white", fontsize=10.5,
                fontweight="bold", zorder=6)

    fig.tight_layout(pad=0.3)
    fig.savefig("architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    per_condition_chart()
    faithfulness_chart()
    bugfix_chart()
    architecture_diagram()
    print("done")
