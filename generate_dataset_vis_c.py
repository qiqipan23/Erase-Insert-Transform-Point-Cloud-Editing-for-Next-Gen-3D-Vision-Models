"""
Option C (v3) — ConvONet dataset examples only.

4 rows × 3 cols:
  Col 0  Before removal  — original RGB crop
  Col 1  Network input   — void crop + fill-region box
  Col 2  Training target — faint input + occupied GT query pts

Output: figs/fig_dataset_vis_c.pdf / .png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

HYPO    = Path("/rds/general/user/qp23/home/Hypo3D")
PROC    = HYPO / "repo/scannet/processed"
DS_DIR  = HYPO / "repo/scannet/completion/convonet_scannet_remove_train/scannet_remove"
JOBS_F  = HYPO / "repo/scannet/derived/generated_jobs_clean.jsonl"
OUT_DIR = HYPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

RNG = np.random.default_rng(42)

# ── 4 representative examples ─────────────────────────────────────────────────
EXAMPLES = [
    ("Bed",     "scene0233_00", 941,  38),
    ("Desk",    "scene0131_00", 609,  15),
    ("Cabinet", "scene0172_00", 768,   5),
    ("Couch",   "scene0151_00", 714,   6),
]

C_INPUT = "#78909C"
C_OBJ   = "#EF5350"
C_FILL  = "#1E88E5"
MAX_PTS = 12_000

# ── Helpers ───────────────────────────────────────────────────────────────────
def crop_mask(xyz, center, half):
    return np.all((xyz >= center - half) & (xyz <= center + half), axis=1)

def sub(arr, n):
    if arr.shape[0] > n:
        return arr[RNG.choice(arr.shape[0], n, replace=False)]
    return arr

def sc(ax, xyz, color, alpha=0.6, ms=0.9, proj=(0, 2)):
    if xyz.shape[0] == 0: return
    ax.scatter(xyz[:, proj[0]], xyz[:, proj[1]], s=ms, c=color,
               alpha=alpha, linewidths=0, rasterized=True)

def sc_rgb(ax, xyz, rgb, alpha=0.70, ms=1.0, proj=(0, 2)):
    if xyz.shape[0] == 0: return
    ax.scatter(xyz[:, proj[0]], xyz[:, proj[1]],
               s=ms, c=rgb.astype(np.float32)/255,
               alpha=alpha, linewidths=0, rasterized=True)

def draw_box(ax, ctr, sz, proj, color, lw=1.4, ls="--"):
    i, j = proj
    cx, cy = ctr[i], ctr[j]
    hx, hy = sz[i]/2, sz[j]/2
    ax.add_patch(plt.Rectangle((cx-hx, cy-hy), 2*hx, 2*hy,
                                fill=False, edgecolor=color,
                                linewidth=lw, linestyle=ls, zorder=5))

def style_cell(ax):
    ax.set_facecolor("#F8F9FA")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#DDDDDD"); sp.set_linewidth(0.6)

def set_lim(ax, center, half, proj, pad=0.12):
    ax.set_xlim(center[proj[0]]-half[proj[0]]-pad,
                center[proj[0]]+half[proj[0]]+pad)
    ax.set_ylim(center[proj[1]]-half[proj[1]]-pad,
                center[proj[1]]+half[proj[1]]+pad)
    ax.set_aspect("equal")

# ── Figure layout ─────────────────────────────────────────────────────────────
N     = len(EXAMPLES)
FIG_W = 11.0
FIG_H = N * 2.1 + 0.6

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

gs_grid = GridSpec(
    N, 3, figure=fig,
    hspace=0.10, wspace=0.05,
    top=0.88, bottom=0.06, left=0.09, right=0.99,
)

COL_HEADERS = [
    "(a) Before removal\n(original scan, RGB)",
    "(b) Network input\n(void crop  ·  fill region  ▭ )",
    "(c) Training target\n(occupied query pts = surface to predict)",
]
COL_COLORS = ["#1565C0", "#C62828", "#2E7D32"]
PROJ = (0, 2)

for r, (cat, scene_id, job_idx, inst_id) in enumerate(EXAMPLES):
    scene_dir = PROC / scene_id
    xyz_all  = np.load(scene_dir / "xyz.npy").astype(np.float32)
    rgb_all  = np.load(scene_dir / "rgb.npy").astype(np.uint8)

    model_id = f"{scene_id}_job{job_idx:03d}"
    meta     = json.loads((DS_DIR / model_id / "metadata.json").read_text())
    pc       = np.load(DS_DIR / model_id / "pointcloud.npz")
    qi       = np.load(DS_DIR / model_id / "points_iou.npz")

    center = np.array(meta["remove_center"],   dtype=np.float32)
    half   = np.array(meta["crop_half_extent"], dtype=np.float32)
    fill_c = np.array(meta.get("fill_region_center", center.tolist()), dtype=np.float32)
    fill_s = np.array(meta.get("fill_region_size",   (half*2).tolist()), dtype=np.float32)

    m_crop     = crop_mask(xyz_all, center, half)
    idx_crop   = np.where(m_crop)[0]
    if len(idx_crop) > MAX_PTS:
        idx_crop = RNG.choice(idx_crop, MAX_PTS, replace=False)
    before_xyz = xyz_all[idx_crop]
    before_rgb = rgb_all[idx_crop]

    input_pts = sub(pc["points"].astype(np.float32), MAX_PTS)
    occ_all   = np.unpackbits(qi["occupancies"])[:qi["points"].shape[0]].astype(bool)
    occ_pts   = sub(qi["points"][occ_all].astype(np.float32),  MAX_PTS)
    occ_pct   = 100 * meta["occupancy_positive"] / meta["points_iou_n"]

    for col in range(3):
        ax = fig.add_subplot(gs_grid[r, col])
        style_cell(ax)
        set_lim(ax, center, half, PROJ)

        if col == 0:
            sc_rgb(ax, before_xyz, before_rgb, alpha=0.75, ms=1.1)
        elif col == 1:
            sc(ax, input_pts, C_INPUT, alpha=0.60, ms=0.9)
            draw_box(ax, fill_c, fill_s, PROJ, C_FILL, lw=1.4)
        else:
            sc(ax, input_pts, C_INPUT, alpha=0.18, ms=0.7)
            sc(ax, occ_pts,   C_OBJ,   alpha=0.90, ms=1.8)

        if r == 0:
            ax.set_title(COL_HEADERS[col], fontsize=8.5, pad=5,
                         color=COL_COLORS[col], fontweight="bold")
        if col == 0:
            ax.set_ylabel(cat, fontsize=9.5, color="#333",
                          labelpad=4, rotation=0, va="center", ha="right")
        if col == 2:
            ax.text(0.97, 0.04, f"{occ_pct:.0f}% occ.",
                    transform=ax.transAxes, fontsize=7.5,
                    color="#C62828", ha="right", va="bottom",
                    bbox=dict(fc="white", ec="#FFCDD2",
                              boxstyle="round,pad=0.25", alpha=0.92, lw=0.7))

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=C_INPUT,       label="Input pts (after removal)"),
    mpatches.Patch(color=C_OBJ,         label="Occupied query pt (GT surface)"),
    mpatches.Patch(facecolor="none", edgecolor=C_FILL,
                   linestyle="--", linewidth=1.4, label="Fill-region boundary"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=9, framealpha=0.95, edgecolor="#ccc",
           bbox_to_anchor=(0.5, 0.0))

fig.suptitle(
    "ConvONet Fine-tuning Dataset  —  4 representative examples\n"
    "441 examples · 196 ScanNet scenes · 353 train / 44 val / 44 test",
    fontsize=10.5, fontweight="bold", color="#111", y=0.99,
)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_dataset_vis_c.{ext}"
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
plt.close()
