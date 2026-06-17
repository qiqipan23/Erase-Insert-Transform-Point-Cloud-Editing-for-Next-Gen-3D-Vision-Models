"""
Pipeline diagram — properly spaced so all arrows are clearly visible.

Fixes:
  - Boxes narrower + wider spacing → arrows have real width
  - Left box no longer clipped
  - All arrows use high-contrast colours and lw=2.2
  - MOVE branch horizontal arrow clearly visible
  - Compass rose repositioned to not overlap
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT_DIR = Path("/rds/general/user/qp23/home/Hypo3D/figs")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 10})

C = {
    "input":    "#2C3E50",
    "parse":    "#7D3C98",
    "lookup":   "#1A5276",
    "remove":   "#922B21",
    "bkgd":     "#1E8449",
    "spatial":  "#D35400",
    "complete": "#117A65",
    "output":   "#1C2833",
    "arr_main": "#333333",
    "arr_move": "#888888",
}

fig, ax = plt.subplots(figsize=(16, 6.8), facecolor="white")
ax.set_xlim(-0.3, 16.3)
ax.set_ylim(0, 6.8)
ax.axis("off")

# ── Layout ────────────────────────────────────────────────────────────────────
BW   = 1.85    # box width  (narrower → more room for arrows)
BH   = 0.85    # box height
STEP = 2.65    # centre-to-centre distance  (gap = 2.65-1.85 = 0.80)
g    = BW / 2 + 0.12   # arrow start/end offset from box edge

Y1   = 4.60   # top row
Y2   = 2.10   # MOVE branch row
X0   = 1.20   # first box centre

Xs   = [X0 + i * STEP for i in range(6)]
# Xs ≈ 1.20, 3.85, 6.50, 9.15, 11.80, 14.45

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(cx, cy, color, line1, line2=None, fs=9.2, w=BW, h=BH):
    rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                          boxstyle="round,pad=0.10",
                          facecolor=color, edgecolor="white",
                          linewidth=2.0, zorder=3, alpha=0.95)
    ax.add_patch(rect)
    if line2:
        ax.text(cx, cy+0.16, line1, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", zorder=4,
                multialignment="center")
        ax.text(cx, cy-0.20, line2, ha="center", va="center",
                fontsize=7.0, color="white", alpha=0.88, zorder=4,
                style="italic")
    else:
        ax.text(cx, cy, line1, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", zorder=4,
                multialignment="center")

def arr(x0, y0, x1, y1, color="#333", lw=2.2, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color, lw=lw,
                    mutation_scale=16,
                    connectionstyle=f"arc3,rad={rad}"
                ), zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# TOP ROW  (REMOVE path)
# ══════════════════════════════════════════════════════════════════════════════
box(Xs[0], Y1, C["input"],   "Natural-language\nInstruction")
box(Xs[1], Y1, C["parse"],   "Language\nParser",            "GPT-4o (few-shot)")
box(Xs[2], Y1, C["lookup"],  "Instance\nLookup",            "Segmentation match")
box(Xs[3], Y1, C["remove"],  "Object\nRemoval",             "AABB / OBB excision")
box(Xs[4], Y1, C["bkgd"],    "Background\nReconstruction",  "Structural + ConvONet")
box(Xs[5], Y1, C["output"],  "Edited\nScene\n(Output)", fs=9.5)

# Top-row arrows  ── clearly visible with 0.80 gap each ──
for i in range(5):
    arr(Xs[i]+g, Y1, Xs[i+1]-g, Y1, color=C["arr_main"], lw=2.2)

# ══════════════════════════════════════════════════════════════════════════════
# MOVE BRANCH  (bottom row, cols 3→4→5)
# ══════════════════════════════════════════════════════════════════════════════
box(Xs[4], Y2, C["spatial"],  "Spatial Relation\nResolver",       "8-direction vocab")
box(Xs[5], Y2, C["complete"], "Instance Completion\n& Placement",  "PBIC / PointR")

# 1. Object Removal → down ↓
arr(Xs[3], Y1-BH/2, Xs[3], Y2+BH/2,
    color=C["remove"], lw=2.0)

# 2. Horizontal at Y2: col-3 → Spatial (col-4)
arr(Xs[3]+g, Y2, Xs[4]-g, Y2,
    color=C["arr_move"], lw=2.0)

# 3. Spatial → Completion
arr(Xs[4]+g, Y2, Xs[5]-g, Y2,
    color=C["arr_move"], lw=2.0)

# 4. Completion → up ↑ to Output (col-5)
arr(Xs[5], Y2+BH/2, Xs[5], Y1-BH/2,
    color=C["complete"], lw=2.0)

# ── Annotations ───────────────────────────────────────────────────────────────
# "MOVE branch" beside the vertical drop
ax.text(Xs[3]-0.18, (Y1-BH/2 + Y2+BH/2)/2,
        "MOVE\nbranch", ha="right", va="center",
        fontsize=8, color=C["remove"], style="italic", fontweight="bold")

# "MOVE merges back" beside vertical up arrow
ax.text(Xs[5]+0.18, (Y1-BH/2 + Y2+BH/2)/2,
        "MOVE\nmerges\nback", ha="left", va="center",
        fontsize=7.5, color=C["complete"], style="italic")

# Parser output JSON
record = ('{ "op": "MOVE",\n'
          '  "target": "desk",\n'
          '  "relation": "RIGHT_OF",\n'
          '  "anchor": "fridge" }')
ax.text(Xs[1], Y1-BH/2-0.14, record,
        ha="center", va="top", fontsize=6.8, color="#333",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.28", fc="#F7F7F7",
                  ec="#BBBBBB", lw=0.9),
        zorder=5)

# ── Compass rose (below Spatial Resolver) ────────────────────────────────────
cx, cy = Xs[4], Y2 - BH/2 - 0.58
r = 0.30
for lbl, (dx, dy) in [("F",(0,1)),("B",(0,-1)),
                       ("L",(-1,0)),("R",(1,0)),
                       ("FL",(-0.71,0.71)),("FR",(0.71,0.71)),
                       ("BL",(-0.71,-0.71)),("BR",(0.71,-0.71))]:
    ax.annotate("", xy=(cx+dx*r, cy+dy*r), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=C["spatial"],
                                lw=0.9, mutation_scale=9), zorder=4)
    ax.text(cx+dx*(r+0.11), cy+dy*(r+0.11), lbl,
            ha="center", va="center", fontsize=6.2,
            color=C["spatial"], fontweight="bold", zorder=5)
ax.plot(cx, cy, "o", color=C["spatial"], ms=5, zorder=5)

# ── Row separator ─────────────────────────────────────────────────────────────
sep_y = (Y1 + Y2) / 2
ax.plot([0.10, 16.10], [sep_y, sep_y],
        color="#DDDDDD", lw=1.0, linestyle="--", zorder=1)
ax.text(0.08, Y1, "REMOVE\npath", ha="right", va="center",
        fontsize=7, color="#BBBBBB", style="italic")
ax.text(0.08, Y2, "MOVE\nbranch", ha="right", va="center",
        fontsize=7, color="#BBBBBB", style="italic")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(8.0, 6.48,
        "Language-Driven 3D Scene Editing — System Pipeline",
        ha="center", va="center", fontsize=13.5,
        fontweight="bold", color="#1C2833")

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(fc=C["parse"],    ec="white", label="Language parsing  (GPT-4o)"),
    mpatches.Patch(fc=C["lookup"],   ec="white", label="Instance lookup"),
    mpatches.Patch(fc=C["remove"],   ec="white", label="Object removal  (AABB / OBB)"),
    mpatches.Patch(fc=C["bkgd"],     ec="white", label="Background reconstruction  (ConvONet + structural fill)"),
    mpatches.Patch(fc=C["spatial"],  ec="white", label="Spatial relation resolver"),
    mpatches.Patch(fc=C["complete"], ec="white", label="Instance completion & placement  (PBIC / PointR)"),
]
ax.legend(handles=patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.01), ncol=3,
          fontsize=8.5, framealpha=0.96,
          edgecolor="#cccccc", facecolor="white",
          columnspacing=1.0, handlelength=1.2)

plt.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_pipeline.{ext}"
    plt.savefig(out, facecolor="white", dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close()
