"""
Object completion failure figure — stool (inst 029, scene0000_00).

Shows a case where PBIC completion fails: the sparse one-sided scan
is completed as a rectangular solid mass rather than the correct stool shape.

Layout: 3 columns × 2 rows
  Col 0  Scene context   — full scene, stool highlighted
  Col 1  Partial input   — isolated partial scan cloud (~642 pts)
  Col 2  PBIC completion — incorrect dense prediction (rectangular mass)

  Row 0  Top-down  (X – Y)
  Row 1  Front     (X – Z)

Output: figs/fig_completion_failure_inst.pdf / .png
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree

HYPO      = Path("/rds/general/user/qp23/home/Hypo3D")
PROC      = HYPO / "repo/scannet/processed/scene0000_00"
COMP_DIR  = PROC / "completion/pbic_inst029_stool"
OUT_DIR   = HYPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

INST_ID   = 29
LABEL     = "stool"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

C_OBJECT   = "#E53935"
C_BOX      = "#555555"
C_COMPLETE = "#FF8F00"


def load_ply(path: Path):
    path = Path(path)
    if not path.exists():
        print(f"  [missing] {path.name}")
        return np.empty((0, 3), np.float32), None
    data = path.read_bytes()
    hi   = data.find(b"end_header")
    he   = hi + len(b"end_header") + 1
    if data[he - 1] != ord('\n'):
        he += 1
    hdr  = data[:he].decode("ascii", errors="ignore")
    n    = int(re.search(r"element vertex (\d+)", hdr).group(1))
    bin_ = "binary_little_endian" in hdr
    props = []; in_v = False
    for line in hdr.splitlines():
        line = line.strip()
        if   line.startswith("element vertex"): in_v = True
        elif line.startswith("element"):        in_v = False
        elif in_v and line.startswith("property") and "list" not in line:
            p = line.split()
            if len(p) >= 3: props.append((p[1], p[2]))
    TM = {"float":"f4","float32":"f4","double":"f8",
          "uchar":"u1","uint8":"u1","int":"i4","uint":"u4"}
    if bin_:
        dt  = np.dtype([(nm, TM.get(tp, "f4")) for tp, nm in props])
        arr = np.frombuffer(data[he:], dtype=dt, count=n)
    else:
        raw  = data[he:].decode("ascii", errors="ignore").splitlines()
        rows = [list(map(float, l.split())) for l in raw[:n] if l.strip()]
        arr  = np.zeros(n, dtype=np.dtype([(nm, TM.get(tp, "f4")) for tp, nm in props]))
        for i, row in enumerate(rows[:n]):
            for j, (_, nm) in enumerate(props):
                if j < len(row): arr[nm][i] = row[j]
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    rgb = None
    for names in (("red","green","blue"), ("r","g","b")):
        if all(nm in arr.dtype.names for nm in names):
            r_ = np.column_stack([arr[nm] for nm in names]).astype(np.float64)
            rgb = (r_/r_.max()*255 if r_.max() > 1.5 else r_*255).astype(np.uint8)
            break
    return xyz, rgb


print("Loading scene arrays …")
xyz_all  = np.load(PROC / "xyz.npy").astype(np.float32)
rgb_raw  = np.load(PROC / "rgb.npy")
rgb_all  = (rgb_raw * 255 if rgb_raw.max() <= 1.0 else rgb_raw).clip(0, 255).astype(np.uint8)
inst_all = np.load(PROC / "inst.npy")

mask_obj    = inst_all == INST_ID
partial_xyz = xyz_all[mask_obj]
partial_rgb = rgb_all[mask_obj]
print(f"  Partial ({LABEL}): {partial_xyz.shape[0]} pts")

print("Loading PBIC completion …")
completed_xyz, _ = load_ply(COMP_DIR / "completed.ply")
print(f"  Completed: {completed_xyz.shape[0]} pts")
print(f"  Completed range: {completed_xyz.min(axis=0).round(3)} -> {completed_xyz.max(axis=0).round(3)}")

# Colour-transfer from partial
_, nn_idx      = cKDTree(partial_xyz).query(completed_xyz, k=1)
completed_rgb  = partial_rgb[nn_idx]

obj_min  = partial_xyz.min(axis=0)
obj_max  = partial_xyz.max(axis=0)
obj_ctr  = (obj_min + obj_max) / 2
obj_size = obj_max - obj_min

RNG = np.random.default_rng(42)

def sub(arr, n):
    if arr.shape[0] > n:
        return arr[RNG.choice(arr.shape[0], n, replace=False)]
    return arr

def sc(ax, xyz, color, alpha=0.55, ms=None, proj=(0,1)):
    if xyz is None or xyz.shape[0] == 0: return
    n = xyz.shape[0]
    s = ms if ms else (4 if n < 2000 else 1.5 if n < 20_000 else 0.5)
    ax.scatter(xyz[:,proj[0]], xyz[:,proj[1]], s=s, c=color,
               alpha=alpha, linewidths=0, rasterized=True)

def sc_rgb(ax, xyz, rgb, alpha=0.50, ms=None, proj=(0,1)):
    if xyz is None or xyz.shape[0] == 0: return
    n = xyz.shape[0]
    s = ms if ms else (1.5 if n < 20_000 else 0.5)
    c = rgb.astype(np.float32) / 255.0
    ax.scatter(xyz[:,proj[0]], xyz[:,proj[1]], s=s, c=c,
               alpha=alpha, linewidths=0, rasterized=True)

def draw_box(ax, ctr, size, proj, color="#555", lw=1.5, ls="--"):
    i, j = proj
    cx, cy = ctr[i], ctr[j]
    hx, hy = size[i]/2, size[j]/2
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


PAD_SCENE = 0.3
PAD_OBJ   = 0.25

xlim_scene = (xyz_all[:,0].min()-PAD_SCENE, xyz_all[:,0].max()+PAD_SCENE)
ylim_scene = (xyz_all[:,1].min()-PAD_SCENE, xyz_all[:,1].max()+PAD_SCENE)
zlim_scene = (xyz_all[:,2].min()-PAD_SCENE, xyz_all[:,2].max()+PAD_SCENE)

all_obj  = np.concatenate([partial_xyz, completed_xyz], axis=0)
xlim_obj = (all_obj[:,0].min()-PAD_OBJ, all_obj[:,0].max()+PAD_OBJ)
ylim_obj = (all_obj[:,1].min()-PAD_OBJ, all_obj[:,1].max()+PAD_OBJ)
zlim_obj = (all_obj[:,2].min()-PAD_OBJ, all_obj[:,2].max()+PAD_OBJ)

# Row 0: top-down X–Y, Row 1: front X–Z
PROJS    = [(0,1), (0,2)]
XLBLS    = ["X (m)", "X (m)"]
YLBLS    = ["Y (m)", "Z (m)"]
ROW_NAMES = ["Top-down (X – Y)", "Front (X – Z)"]

COL_TITLES = [
    f"(a) Scene context\n({LABEL.capitalize()} highlighted)",
    f"(b) Partial input\n(raw scan extract, {partial_xyz.shape[0]} pts)",
    f"(c) PBIC completion\n(dense prediction — rectangular solid)",
]

fig = plt.figure(figsize=(13, 8.5), facecolor="white")
gs  = GridSpec(2, 3, figure=fig,
               hspace=0.38, wspace=0.22,
               top=0.88, bottom=0.09, left=0.07, right=0.97)

not_mask = ~mask_obj
idx_sub  = RNG.choice(xyz_all[not_mask].shape[0],
                      min(40_000, xyz_all[not_mask].shape[0]), replace=False)
scene_sub_xyz = xyz_all[not_mask][idx_sub]
scene_sub_rgb = rgb_all[not_mask][idx_sub]

for r, proj in enumerate(PROJS):
    lim_x      = xlim_scene
    lim_scene2 = ylim_scene if r == 0 else zlim_scene
    lim_obj2   = ylim_obj   if r == 0 else zlim_obj

    for c in range(3):
        ax = fig.add_subplot(gs[r, c])
        t  = COL_TITLES[c] if r == 0 else ""
        xl = XLBLS[r]
        yl = YLBLS[r] if c == 0 else ""

        if c == 0:
            style(ax, t, xl, yl, lim_x, lim_scene2)
            sc_rgb(ax, scene_sub_xyz, scene_sub_rgb, alpha=0.40, proj=proj)
            sc(ax, partial_xyz, C_OBJECT, alpha=0.95, ms=5, proj=proj)
            draw_box(ax, obj_ctr, obj_size + 0.15, proj, C_BOX, lw=1.3)
            if r == 0:
                ax.text(obj_ctr[0], obj_ctr[1] + obj_size[1]/2 + 0.35,
                        LABEL, ha="center", va="bottom", fontsize=8,
                        color=C_OBJECT, fontweight="bold",
                        bbox=dict(fc="white", ec=C_OBJECT,
                                  boxstyle="round,pad=0.25", lw=0.8, alpha=0.9))

        elif c == 1:
            style(ax, t, xl, yl, xlim_obj, lim_obj2)
            sc_rgb(ax, partial_xyz, partial_rgb, alpha=0.95, ms=8, proj=proj)
            ax.text(0.05, 0.05, f"{partial_xyz.shape[0]} pts",
                    transform=ax.transAxes, fontsize=8, color="#555",
                    bbox=dict(fc="white", ec="#ccc",
                              boxstyle="round,pad=0.25", alpha=0.9, lw=0.7))

        else:
            style(ax, t, xl, yl, xlim_obj, lim_obj2)
            # completed points in orange to contrast with partial
            sc(ax, completed_xyz, C_COMPLETE, alpha=0.70, ms=3, proj=proj)
            sc_rgb(ax, partial_xyz, partial_rgb, alpha=0.95, ms=8, proj=proj)
            ax.text(0.05, 0.05,
                    f"{partial_xyz.shape[0]} partial\n+ {completed_xyz.shape[0]} predicted",
                    transform=ax.transAxes, fontsize=7.5, color="#555",
                    bbox=dict(fc="white", ec="#ccc",
                              boxstyle="round,pad=0.25", alpha=0.9, lw=0.7))
            # annotate the failure
            ax.text(0.97, 0.95, "fills bounding\nbox — no\nlegs recovered",
                    transform=ax.transAxes, fontsize=7.5, color="#B71C1C",
                    ha="right", va="top",
                    bbox=dict(fc="#FFEBEE", ec="#E53935",
                              boxstyle="round,pad=0.3", lw=0.8, alpha=0.9))

for r, lbl in enumerate(ROW_NAMES):
    pos = fig.get_axes()[r*3].get_position()
    fig.text(0.005, pos.y0 + pos.height/2, lbl,
             va="center", ha="left", fontsize=9,
             color="#444", rotation=90, fontweight="bold")

handles = [
    mpatches.Patch(color="#90A4AE", label="Scene geometry (RGB)"),
    mpatches.Patch(color=C_OBJECT,  label="Partial scan input (RGB, highlighted)"),
    mpatches.Patch(color=C_COMPLETE,label="PBIC completion (predicted — incorrect)"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=9, framealpha=0.95, edgecolor="#ccc",
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    f"Instance Completion Failure — {LABEL.capitalize()} "
    f"(scene0000\\_00, inst {INST_ID})\n"
    f"Partial: {partial_xyz.shape[0]} pts  →  "
    f"PBIC prediction: {completed_xyz.shape[0]} pts  "
    f"(rectangular solid, stool structure not recovered)",
    fontsize=11, fontweight="bold", color="#111", y=0.96,
)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_completion_failure_inst.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
plt.close(fig)
