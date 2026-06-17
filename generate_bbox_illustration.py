"""
Generate a figure illustrating AABB vs OBB bounding boxes.
Output: figs/fig_bbox_illustration.pdf / .png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("/rds/general/user/qp23/home/Hypo3D/figs")
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── Object: desk-like rectangle rotated 35° ────────────────────────────────────
ANGLE_DEG = 35.0
OBJ_W, OBJ_H = 2.2, 1.0
cx, cy = 0.0, 0.0

theta = np.radians(ANGLE_DEG)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

local_corners = np.array([[-OBJ_W,-OBJ_H],[ OBJ_W,-OBJ_H],
                           [ OBJ_W, OBJ_H],[-OBJ_W, OBJ_H]])
world_corners = (R @ local_corners.T).T + [cx, cy]

OBB_PAD = 0.12
obb_local = np.array([[-OBJ_W-OBB_PAD, -OBJ_H-OBB_PAD],
                       [ OBJ_W+OBB_PAD, -OBJ_H-OBB_PAD],
                       [ OBJ_W+OBB_PAD,  OBJ_H+OBB_PAD],
                       [-OBJ_W-OBB_PAD,  OBJ_H+OBB_PAD]])
obb_corners = (R @ obb_local.T).T + [cx, cy]

aabb_min = world_corners.min(axis=0)
aabb_max = world_corners.max(axis=0)
aabb_w   = aabb_max[0] - aabb_min[0]
aabb_h   = aabb_max[1] - aabb_min[1]

C_OBJ_F = "#CFD8DC"
C_OBJ_E = "#455A64"
C_AABB  = "#C62828"
C_OBB   = "#1565C0"

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), facecolor="white")
LIM = 3.3

def draw_panel(ax, title, box_patch, box_color):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM + 0.4, LIM - 0.1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    ax.add_patch(box_patch)

    obj = plt.Polygon(world_corners, closed=True,
                      facecolor=C_OBJ_F, edgecolor=C_OBJ_E,
                      linewidth=1.8, zorder=3)
    ax.add_patch(obj)
    ax.text(cx, cy, "object", ha="center", va="center",
            fontsize=9, color=C_OBJ_E, fontstyle="italic", zorder=5)

import matplotlib.patches as mpatches

# Panel (a) — AABB
aabb_rect = mpatches.Rectangle(
    aabb_min, aabb_w, aabb_h,
    linewidth=2.5, linestyle="--", edgecolor=C_AABB,
    facecolor="#FFEBEE", alpha=0.45, zorder=2)
draw_panel(axes[0], "(a)  Axis-Aligned Bounding Box (AABB)", aabb_rect, C_AABB)

# Panel (b) — OBB
obb_patch = plt.Polygon(obb_corners, closed=True,
                        linewidth=2.5, linestyle="--",
                        edgecolor=C_OBB, facecolor="#E3F2FD",
                        alpha=0.45, zorder=2)
draw_panel(axes[1], "(b)  Oriented Bounding Box (OBB)", obb_patch, C_OBB)

plt.tight_layout()

for ext in ("pdf", "png"):
    out = FIGS / f"fig_bbox_illustration.{ext}"
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
plt.close()
