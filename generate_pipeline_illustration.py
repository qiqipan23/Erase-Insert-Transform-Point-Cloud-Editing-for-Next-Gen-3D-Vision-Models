"""
Generate a two-row diagram illustrating the two evaluation pipelines.

  Row 1 – Hypo3D baseline : original scene image + text change → GPT-4o → ✗ answer
  Row 2 – Our pipeline    : edited labelled scene image        → GPT-4o → ✓ answer

Example used: scene0000_00 / job005_backpack_to_on_top_of_bed
  Change : "The backpack has been moved onto the bed behind the pillow."
  Q      : "How many objects are now situated on the bed following the backpack's movement?"
  Ref    : 2   |   Baseline pred: "Three" (✗)   |   Ours: "Two" (✓)

Output: figs/fig_pipeline_illustration.pdf / .png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import numpy as np

HYPO = Path(__file__).parent
FIGS = HYPO / "figs"; FIGS.mkdir(exist_ok=True)

# ── Image paths ────────────────────────────────────────────────────────────────
BASE_DIR = HYPO / "dataset/2D_VLM_data/move_edits_top_view/scene0000_00"
LAB_DIR  = HYPO / "dataset/2D_VLM_data/move_edits_top_view_labelled/scene0000_00"

img_orig  = np.array(Image.open(BASE_DIR / "original_scene0000_00.png").convert("RGB"))
img_edit  = np.array(Image.open(LAB_DIR  / "job005_backpack_to_on_top_of_bed.png").convert("RGB"))

# ── Layout constants ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig = plt.figure(figsize=(13, 5.8), facecolor="white")

C_BASE = "#2E7D32"   # green  – baseline row
C_OURS = "#1565C0"   # blue   – ours row
C_WRONG = "#C62828"
C_RIGHT = "#1B5E20"
C_GPT  = "#FF8F00"   # amber  – GPT-4o box

CHANGE_TEXT  = "The backpack has been\nmoved onto the bed\nbehind the pillow."
QUESTION     = "How many objects are now\nsituated on the bed following\nthe backpack's movement?"
REF_ANS      = "2"
BASE_PRED    = '"Three"'
OURS_PRED    = '"Two"'

# ── Helper: draw a rounded text box ───────────────────────────────────────────
def textbox(ax, x, y, w, h, text, facecolor, edgecolor, textcolor="white",
            fontsize=8.5, bold=False):
    box = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02", linewidth=1.2,
        facecolor=facecolor, edgecolor=edgecolor,
        transform=ax.transAxes, clip_on=False, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center")

def arrow(ax, x0, x1, y, color="#555555"):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14),
                zorder=5)

# ── Row heights (axes fraction of a single-axes figure) ───────────────────────
# We'll use a single big axes and draw everything with transAxes.
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

# Row centres
Y_TOP = 0.73
Y_BOT = 0.27

# ── Row labels ─────────────────────────────────────────────────────────────────
ax.text(0.01, Y_TOP, "Hypo3D\nbaseline", transform=ax.transAxes,
        ha="left", va="center", fontsize=9, color=C_BASE, fontweight="bold",
        rotation=90)
ax.text(0.01, Y_BOT, "Our\npipeline", transform=ax.transAxes,
        ha="left", va="center", fontsize=9, color=C_OURS, fontweight="bold",
        rotation=90)

# ── Horizontal divider ─────────────────────────────────────────────────────────
ax.plot([0, 1], [0.5, 0.5], color="#dddddd", linewidth=1.0, linestyle="--",
        transform=ax.transAxes, zorder=1)

# ── Column x-positions ─────────────────────────────────────────────────────────
X_IMG   = 0.13   # scene image
X_TEXT  = 0.30   # text change box (baseline only)
X_GPT   = 0.52   # GPT-4o box
X_QA    = 0.73   # question text
X_ANS   = 0.92   # answer box

# ── Image sizes (axes fraction) ────────────────────────────────────────────────
IW = 0.16; IH = 0.36

# ── Inset axes for scene images ───────────────────────────────────────────────
ax_orig = fig.add_axes([X_IMG - IW/2, Y_TOP - IH/2, IW, IH])
ax_orig.imshow(img_orig); ax_orig.axis("off")
for spine in ax_orig.spines.values():
    spine.set_edgecolor(C_BASE); spine.set_linewidth(2)
ax_orig.set_title("Original scene", fontsize=8, color=C_BASE, pad=3)

ax_edit = fig.add_axes([X_IMG - IW/2, Y_BOT - IH/2, IW, IH])
ax_edit.imshow(img_edit); ax_edit.axis("off")
for spine in ax_edit.spines.values():
    spine.set_edgecolor(C_OURS); spine.set_linewidth(2)
ax_edit.set_title("Edited scene\n(with labels)", fontsize=8, color=C_OURS, pad=3)

# ── Baseline row: change-text box + arrow ─────────────────────────────────────
textbox(ax, X_TEXT, Y_TOP, 0.15, 0.30, CHANGE_TEXT,
        facecolor="#E8F5E9", edgecolor=C_BASE, textcolor="#1B5E20", fontsize=8)
ax.text(X_TEXT, Y_TOP + 0.20, "Context change\n(text only)",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.5, color=C_BASE, fontstyle="italic")

arrow(ax, X_IMG + IW/2 + 0.005, X_TEXT - 0.075, Y_TOP, C_BASE)
arrow(ax, X_TEXT + 0.075, X_GPT - 0.065, Y_TOP, C_BASE)

# ── Our-pipeline row: arrow from image straight to GPT-4o ─────────────────────
arrow(ax, X_IMG + IW/2 + 0.005, X_GPT - 0.065, Y_BOT, C_OURS)

# ── GPT-4o boxes ──────────────────────────────────────────────────────────────
for y in (Y_TOP, Y_BOT):
    textbox(ax, X_GPT, y, 0.11, 0.14, "GPT-4o",
            facecolor=C_GPT, edgecolor="#E65100", fontsize=10, bold=True)

# ── Question text (shared, centred between rows) ──────────────────────────────
for y, col in ((Y_TOP, C_BASE), (Y_BOT, C_OURS)):
    arrow(ax, X_GPT + 0.055, X_QA - 0.09, y, col)

ax.text(X_QA, 0.5, "Q: " + QUESTION,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=8.5, color="#333333", multialignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                  edgecolor="#BBBBBB", linewidth=1))

# ── Answer boxes ──────────────────────────────────────────────────────────────
for y, pred, color, tick in (
        (Y_TOP, BASE_PRED, C_WRONG, "✗"),
        (Y_BOT, OURS_PRED, C_RIGHT, "✓")):
    arrow(ax, X_QA + 0.09, X_ANS - 0.06, y, color)
    textbox(ax, X_ANS, y, 0.11, 0.16,
            f"[{tick}]  {pred}\n(ref: {REF_ANS})",
            facecolor=color, edgecolor=color, fontsize=9, bold=True)

# ── Ref label ─────────────────────────────────────────────────────────────────
ax.text(0.5, 0.97,
        "Evaluation pipeline comparison — same question, different visual input",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10.5, fontweight="bold", color="#222222")

ax.text(0.5, 0.02,
        'Example: scene0000_00  |  change: "The backpack has been moved onto the bed behind the pillow."',
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, color="#666666", fontstyle="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out = FIGS / f"fig_pipeline_illustration.{ext}"
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
plt.close()
