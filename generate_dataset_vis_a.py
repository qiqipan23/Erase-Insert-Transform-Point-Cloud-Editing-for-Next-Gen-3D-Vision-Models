"""
Option A — Per-example construction panel.

4 columns × 2 rows  (top-down X-Z | front X-Y)
  Col 0  Full scene with remove box
  Col 1  Before-removal crop  (ground-truth surface, RGB)
  Col 2  After-removal crop   (ConvONet input, coloured by fill-region distance)
  Col 3  Query points         (occupied=red, free=blue, subsampled)

Uses scene0000_00 / job015 (cabinet removal).
Output: figs/fig_dataset_vis_a.pdf / .png
"""
from __future__ import annotations
import json, re, struct
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

HYPO      = Path("/rds/general/user/qp23/home/Hypo3D")
PROC      = HYPO / "repo/scannet/processed/scene0000_00"
SCENE_ID  = "scene0000_00"
JOB_IDX   = 15
MODEL_ID  = f"{SCENE_ID}_job{JOB_IDX:03d}"
DS_DIR    = HYPO / "repo/scannet/completion/convonet_scannet_remove_train/scannet_remove" / MODEL_ID
OUT_DIR   = HYPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── Load full scene ───────────────────────────────────────────────────────────
xyz_all = np.load(PROC / "xyz.npy").astype(np.float32)
rgb_all = np.load(PROC / "rgb.npy").astype(np.uint8)
inst_all = np.load(PROC / "inst.npy")

manifest = json.loads((PROC / "edits" / f"job{JOB_IDX:03d}.json").read_text())
center   = np.array(manifest["remove_center"], dtype=np.float32)
size     = np.array(manifest["remove_size"],   dtype=np.float32)
label    = manifest.get("target_label", "object")
inst_id  = manifest.get("target_instance_id", -1)

mask_obj = inst_all == inst_id

# ── Load dataset example ─────────────────────────────────────────────────────
meta      = json.loads((DS_DIR / "metadata.json").read_text())
half      = np.array(meta["crop_half_extent"], dtype=np.float32)
fill_c    = np.array(meta.get("fill_region_center", center.tolist()), dtype=np.float32)
fill_s    = np.array(meta.get("fill_region_size",   size.tolist()),   dtype=np.float32)
fill_half = 0.5 * fill_s

pc   = np.load(DS_DIR / "pointcloud.npz")
qi   = np.load(DS_DIR / "points_iou.npz")

input_pts = pc["points"].astype(np.float32)          # after-removal crop, 100K pts
query_pts = qi["points"].astype(np.float32)
occ       = np.unpackbits(qi["occupancies"])[:query_pts.shape[0]].astype(bool)

# Build fill-region distance feature for colouring input points
offset   = np.abs(input_pts - fill_c) - fill_half
dist_raw = np.linalg.norm(np.maximum(offset, 0.0), axis=1)
scale    = max(float(fill_half.max()), 1e-6)
feat_dist = np.clip(dist_raw / scale, 0.0, 1.0)      # 0 = inside fill, 1 = far

# Crop the before-removal points from full scene
mn = center - half; mx = center + half
m_crop = np.all((xyz_all >= mn) & (xyz_all <= mx), axis=1)
before_xyz = xyz_all[m_crop]
before_rgb = rgb_all[m_crop]

# Subsample query points for display
RNG = np.random.default_rng(42)
MAX_Q = 20_000
MAX_I = 40_000
if query_pts.shape[0] > MAX_Q:
    idx = RNG.choice(query_pts.shape[0], MAX_Q, replace=False)
    query_pts_d, occ_d = query_pts[idx], occ[idx]
else:
    query_pts_d, occ_d = query_pts, occ
if input_pts.shape[0] > MAX_I:
    idx2 = RNG.choice(input_pts.shape[0], MAX_I, replace=False)
    input_pts_d = input_pts[idx2]; feat_d = feat_dist[idx2]
else:
    input_pts_d = input_pts; feat_d = feat_dist

# ── Helpers ───────────────────────────────────────────────────────────────────
def scatter(ax, xyz, rgb=None, c=None, cmap=None, vmin=0, vmax=1,
            alpha=0.5, proj=(0,2), ms=None):
    if xyz is None or xyz.shape[0] == 0: return
    n = xyz.shape[0]
    s = ms or (3 if n < 20_000 else 1.0 if n < 80_000 else 0.4)
    x, y = xyz[:, proj[0]], xyz[:, proj[1]]
    kw = dict(s=s, alpha=alpha, linewidths=0, rasterized=True)
    if c is not None and cmap:
        ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax, **kw)
    elif rgb is not None:
        ax.scatter(x, y, c=rgb.astype(np.float32)/255, **kw)
    else:
        ax.scatter(x, y, color="#888", **kw)

def draw_box(ax, ctr, sz, proj, color="#ff7f0e", lw=1.6, ls="--"):
    i, j = proj
    cx, cy = ctr[i], ctr[j]
    hx, hy = sz[i]/2, sz[j]/2
    ax.add_patch(plt.Rectangle((cx-hx, cy-hy), 2*hx, 2*hy,
                                fill=False, edgecolor=color,
                                linewidth=lw, linestyle=ls, zorder=5))

def style(ax, title, xlabel, ylabel, xlim, ylim):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=10, pad=6, color="#111", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8, color="#555", labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, color="#555", labelpad=2)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7, color="#aaa", labelcolor="#666")
    ax.grid(True, color="#eeeeee", linewidth=0.4, zorder=0)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("bottom","left"): ax.spines[sp].set_color("#cccccc")

# ── Axis limits ───────────────────────────────────────────────────────────────
PAD = 0.5
xlim_scene = (xyz_all[:,0].min()-PAD, xyz_all[:,0].max()+PAD)
zlim_scene = (xyz_all[:,2].min()-PAD, xyz_all[:,2].max()+PAD)
ylim_scene = (xyz_all[:,1].min()-PAD, xyz_all[:,1].max()+PAD)

crop_pad = 0.2
xlim_crop = (center[0]-half[0]-crop_pad, center[0]+half[0]+crop_pad)
zlim_crop = (center[2]-half[2]-crop_pad, center[2]+half[2]+crop_pad)
ylim_crop = (center[1]-half[1]-crop_pad, center[1]+half[1]+crop_pad)

PROJS  = [(0,2), (0,1)]
XLBLS  = ["X (m)", "X (m)"]
YLBLS  = ["Z (m)", "Y (m)"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8), facecolor="white")
gs  = GridSpec(2, 4, figure=fig,
               hspace=0.42, wspace=0.30,
               top=0.88, bottom=0.10, left=0.06, right=0.97)

COL_TITLES = [
    f"(a) Full scene\n({label} highlighted)",
    "(b) Before-removal crop\n(ground-truth surface)",
    "(c) After-removal crop\n(ConvONet input)",
    "(d) Query points\n(occupied / free)",
]

axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

for r, proj in enumerate(PROJS):
    lim_x = xlim_scene if r == 0 else xlim_scene
    lim_yz_scene = zlim_scene if r == 0 else ylim_scene
    lim_yz_crop  = zlim_crop  if r == 0 else ylim_crop

    for c in range(4):
        ax = axes[r][c]
        if r == 0:
            ax.set_title(COL_TITLES[c], fontsize=9.5, pad=6,
                         color="#111", fontweight="bold")

        if c == 0:
            # Full scene
            scatter(ax, xyz_all[~mask_obj], rgb_all[~mask_obj],
                    alpha=0.40, proj=proj)
            scatter(ax, xyz_all[mask_obj], c=np.ones(mask_obj.sum()),
                    cmap="Reds", vmin=0, vmax=1,
                    alpha=0.95, proj=proj, ms=4)
            draw_box(ax, center, size, proj, "#555555", lw=1.3)
            style(ax, COL_TITLES[c] if r==0 else "",
                  XLBLS[r], YLBLS[r] if c==0 else "",
                  xlim_scene, lim_yz_scene)

        elif c == 1:
            # Before-removal crop
            scatter(ax, before_xyz, before_rgb, alpha=0.70, proj=proj)
            draw_box(ax, center, size, proj, "#555555", lw=1.3)
            style(ax, COL_TITLES[c] if r==0 else "",
                  XLBLS[r], YLBLS[r] if c==0 else "",
                  xlim_crop, lim_yz_crop)

        elif c == 2:
            # After-removal crop coloured by fill-region distance
            scatter(ax, input_pts_d, c=feat_d,
                    cmap="RdYlGn_r", vmin=0, vmax=1,
                    alpha=0.65, proj=proj)
            draw_box(ax, fill_c, fill_s, proj, "#2196F3", lw=1.5, ls="-")
            style(ax, COL_TITLES[c] if r==0 else "",
                  XLBLS[r], YLBLS[r] if c==0 else "",
                  xlim_crop, lim_yz_crop)

        else:
            # Query points: occupied vs free
            free_m = ~occ_d; occ_m = occ_d
            scatter(ax, query_pts_d[free_m], c=np.zeros(free_m.sum()),
                    cmap="Blues", vmin=-1, vmax=1,
                    alpha=0.25, proj=proj, ms=0.8)
            scatter(ax, query_pts_d[occ_m],  c=np.ones(occ_m.sum()),
                    cmap="Reds",  vmin=0,  vmax=1,
                    alpha=0.85, proj=proj, ms=2)
            draw_box(ax, center, size, proj, "#555555", lw=1.2, ls="--")
            style(ax, COL_TITLES[c] if r==0 else "",
                  XLBLS[r], YLBLS[r] if c==0 else "",
                  xlim_crop, lim_yz_crop)

# ── Row labels ────────────────────────────────────────────────────────────────
for r, lbl in enumerate(["Top-down (X – Z)", "Front (X – Y)"]):
    pos = axes[r][0].get_position()
    fig.text(0.008, pos.y0 + pos.height/2, lbl,
             va="center", ha="left", fontsize=9,
             color="#444", rotation=90, fontweight="bold")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color="#d62728",  label=f"{label.capitalize()} (target)"),
    mpatches.Patch(color="#4CAF50",  label="Inside fill region (input feat)"),
    mpatches.Patch(color="#F44336",  label="Occupied query point (GT)"),
    mpatches.Patch(color="#90CAF9",  label="Free query point (GT)"),
    mpatches.Patch(facecolor="none", edgecolor="#2196F3",
                   linewidth=1.5, label="Fill region boundary"),
]
fig.legend(handles=handles, loc="lower center", ncol=5,
           fontsize=8.5, framealpha=0.95, edgecolor="#ccc",
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    f"ConvONet Dataset Construction — {SCENE_ID} / job{JOB_IDX:03d}  "
    f"({label} removal)\n"
    f"Input: {meta['crop_points_after']:,} pts  |  "
    f"Query: {meta['points_iou_n']:,}  |  "
    f"Occupied: {meta['occupancy_positive']:,} "
    f"({100*meta['occupancy_positive']/meta['points_iou_n']:.1f}%)",
    fontsize=11, color="#111", y=0.96, fontweight="bold",
)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_dataset_vis_a.{ext}"
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
plt.close()
