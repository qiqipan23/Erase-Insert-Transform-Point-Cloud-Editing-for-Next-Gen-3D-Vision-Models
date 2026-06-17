"""
Full-scene background-completion comparison — job015 cabinet removal,
in the clean style of figs/fig_remove.png.

2 columns x 2 rows:
  Col 0  ConvONet raw completion   (red, over-filled volume)
  Col 1  Structural surface fill   (green, planar — used)
  Row 0  Top-down (X-Z)
  Row 1  Front    (X-Y)

Full scene shown in original RGB with the cabinet removed; completion
points overlaid on top.  Mirrors generate_ops_mini.py styling.

Output: figs/fig_completion_fullscene.pdf / .png
"""
from __future__ import annotations
import re, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HYPO = Path("/rds/general/user/qp23/home/Hypo3D")
PROC = HYPO / "repo/scannet/processed/scene0000_00"
COMP = PROC / "completion/scene0000_00_job015"
OUT  = HYPO / "figs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9})

C_CONV   = "#d62728"   # ConvONet over-completion
C_STRUCT = "#2ca02c"   # structural fill
C_VOID   = "#ff7f0e"

INST_CABINET = 9


# ── PLY loader ────────────────────────────────────────────────────────────────
def load_ply(path: Path):
    data = path.read_bytes()
    hi = data.find(b"end_header"); he = hi + len(b"end_header") + 1
    if data[he - 1] != ord('\n'): he += 1
    hdr = data[:he].decode("ascii", "ignore")
    n = int(re.search(r"element vertex (\d+)", hdr).group(1))
    binary = "binary_little_endian" in hdr
    props = []; in_v = False
    for line in hdr.splitlines():
        line = line.strip()
        if line.startswith("element vertex"): in_v = True
        elif line.startswith("element"): in_v = False
        elif in_v and line.startswith("property") and "list" not in line:
            p = line.split()
            if len(p) >= 3: props.append((p[1], p[2]))
    TM = {"float":"f4","float32":"f4","uchar":"u1","uint8":"u1",
          "int":"i4","uint":"u4","double":"f8"}
    if binary:
        dt = np.dtype([(nm, TM.get(tp,"f4")) for tp,nm in props])
        arr = np.frombuffer(data[he:], dtype=dt, count=n)
        xyz = np.column_stack([arr["x"],arr["y"],arr["z"]]).astype(np.float32)
    else:
        rows = [list(map(float, l.split()))
                for l in data[he:].decode("ascii","ignore").splitlines()[:n] if l.strip()]
        xyz = np.array(rows)[:, :3].astype(np.float32)
    return xyz


RNG = np.random.default_rng(0)
def sub(xyz, n):
    if xyz.shape[0] <= n: return xyz
    return xyz[RNG.choice(xyz.shape[0], n, replace=False)]


def sc(ax, xyz, rgb=None, color=None, alpha=0.55, proj=(0, 2), ms=None):
    if xyz is None or xyz.shape[0] == 0: return
    n = xyz.shape[0]
    s = ms if ms else (6 if n < 5000 else 1.5 if n < 50000 else 0.4)
    x, y = xyz[:, proj[0]], xyz[:, proj[1]]
    kw = dict(s=s, alpha=alpha, linewidths=0, rasterized=True)
    if color is not None:
        ax.scatter(x, y, c=color, **kw)
    elif rgb is not None:
        ax.scatter(x, y, c=rgb.astype(np.float32)/255, **kw)


def style_ax(ax, title=None, col="#222", xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=10, pad=7, color=col, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color="#666", labelpad=2)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color="#666", labelpad=2)


def set_lim(ax, xyz_all, proj, pad=0.15):
    ax.set_xlim(xyz_all[:, proj[0]].min()-pad, xyz_all[:, proj[0]].max()+pad)
    ax.set_ylim(xyz_all[:, proj[1]].min()-pad, xyz_all[:, proj[1]].max()+pad)
    ax.set_aspect("equal")


# ── Load full scene (cabinet removed) ─────────────────────────────────────────
xyz_all = np.load(PROC / "xyz.npy").astype(np.float32)
rgb_all = (np.load(PROC / "rgb.npy") * 255).clip(0, 255).astype(np.uint8)
inst    = np.load(PROC / "inst.npy")

keep = inst != INST_CABINET
scene_xyz = xyz_all[keep]
scene_rgb = rgb_all[keep]

# completion outputs (world coords)
conv_xyz   = sub(load_ply(COMP / "convonet_completed_mesh_colored.ply"), 60000)
struct_xyz = load_ply(COMP / "structural_surface_completed_points.ply")

scene_sub = sub(np.arange(scene_xyz.shape[0]), 60000)
scene_xyz_s = scene_xyz[scene_sub]; scene_rgb_s = scene_rgb[scene_sub]

print(f"scene={scene_xyz_s.shape[0]}  conv={conv_xyz.shape[0]}  struct={struct_xyz.shape[0]}")

PROJS   = [(0, 2), (0, 1)]
XLABELS = ["X (m)", "X (m)"]
YLABELS = ["Z (m)", "Y (m)"]
ROW_LBL = ["Top-down\n(X – Z)", "Front\n(X – Y)"]
PAD = 0.15

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), facecolor="white")
plt.subplots_adjust(hspace=0.12, wspace=0.06,
                    top=0.88, bottom=0.10, left=0.07, right=0.97)

titles = [("(a)  ConvONet raw completion", "#922B21"),
          ("(b)  Structural surface fill", "#1E6B22")]

for c in range(2):
    for r, proj in enumerate(PROJS):
        ax = axes[r][c]
        style_ax(ax,
                 title=titles[c][0] if r == 0 else None,
                 col=titles[c][1],
                 xlabel=XLABELS[r],
                 ylabel=YLABELS[r] if c == 0 else None)
        # scene backdrop in RGB
        sc(ax, scene_xyz_s, scene_rgb_s, alpha=0.45, proj=proj)
        if c == 0:
            sc(ax, conv_xyz, color=C_CONV, alpha=0.40, proj=proj, ms=1.2)
        else:
            sc(ax, struct_xyz, color=C_STRUCT, alpha=0.85, proj=proj, ms=3)
        set_lim(ax, scene_xyz, proj, PAD)

# row labels
for r, lbl in enumerate(ROW_LBL):
    pos = axes[r][0].get_position()
    fig.text(0.018, pos.y0 + pos.height/2, lbl,
             va="center", ha="center", fontsize=9,
             color="#444", rotation=90, fontweight="bold")

fig.suptitle("Background Completion — scene0000\\_00 / job015  (cabinet removal)",
             fontsize=12, fontweight="bold", color="#444", y=0.94)

legend = [
    mpatches.Patch(color="#aaaaaa",  label="Scene geometry (RGB, cabinet removed)"),
    mpatches.Patch(color=C_CONV,     label="ConvONet raw completion (over-filled)"),
    mpatches.Patch(color=C_STRUCT,   label="Structural surface fill (planar, used)"),
]
fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.01),
           ncol=3, fontsize=9, framealpha=0.95, edgecolor="#ccc",
           facecolor="white", handlelength=1.3)

for ext in ("pdf", "png"):
    out = OUT / f"fig_completion_fullscene.{ext}"
    fig.savefig(out, facecolor="white", dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)
