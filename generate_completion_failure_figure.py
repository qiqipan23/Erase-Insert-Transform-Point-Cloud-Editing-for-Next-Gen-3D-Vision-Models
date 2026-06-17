"""
Background-completion failure analysis figure — job015 cabinet removal.

Contrasts the raw ConvONet neural completion (which over-fills the void
with dense, geometrically implausible volume) against the structural
surface fill (clean planar reconstruction actually used downstream).

Layout: 3 columns x 2 rows
  Col 0  After removal      (void region exposed — ConvONet input)
  Col 1  ConvONet raw mesh  (dense over-completion — poor geometry)
  Col 2  Structural fill    (planar surfaces — used in final scene)
  Row 0  Top-down (X-Z)
  Row 1  Front    (X-Y)

Output: figs/fig_completion_failure.pdf / .png
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

HYPO     = Path("/rds/general/user/qp23/home/Hypo3D")
COMP     = HYPO / "repo/scannet/processed/scene0000_00/completion/scene0000_00_job015"
OUT_DIR  = HYPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

C_VOID   = "#9E9E9E"
C_CONV   = "#E53935"   # ConvONet over-completion (failure)
C_STRUCT = "#2E7D32"   # structural fill (good)


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
        rgb = None
        if all(k in arr.dtype.names for k in ("red","green","blue")):
            rgb = np.column_stack([arr["red"],arr["green"],arr["blue"]]).astype(np.uint8)
    else:
        rows = [list(map(float, l.split()))
                for l in data[he:].decode("ascii","ignore").splitlines()[:n] if l.strip()]
        a = np.array(rows); xyz = a[:, :3].astype(np.float32)
        rgb = a[:, 3:6].astype(np.uint8) if a.shape[1] >= 6 else None
    return xyz, rgb


RNG = np.random.default_rng(0)
def sub(xyz, rgb, n):
    if xyz.shape[0] <= n: return xyz, rgb
    idx = RNG.choice(xyz.shape[0], n, replace=False)
    return xyz[idx], (rgb[idx] if rgb is not None else None)


def sc(ax, xyz, color, alpha=0.55, ms=None, proj=(0, 2)):
    if xyz is None or xyz.shape[0] == 0: return
    n = xyz.shape[0]
    s = ms if ms else (3 if n < 5000 else 1.2 if n < 40000 else 0.5)
    ax.scatter(xyz[:, proj[0]], xyz[:, proj[1]], s=s, c=color,
               alpha=alpha, linewidths=0, rasterized=True)


def style(ax, title, xl, yl, xlim, ylim):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=9.5, pad=6, color="#111", fontweight="bold")
    ax.set_xlabel(xl, fontsize=8, color="#555", labelpad=2)
    ax.set_ylabel(yl, fontsize=8, color="#555", labelpad=2)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.tick_params(labelsize=7, color="#aaa", labelcolor="#666")
    ax.grid(True, color="#eeeeee", linewidth=0.4, zorder=0)
    for s_ in ("top","right"): ax.spines[s_].set_visible(False)
    for s_ in ("bottom","left"): ax.spines[s_].set_color("#cccccc")


# ── Load ──────────────────────────────────────────────────────────────────────
void_xyz, _   = load_ply(COMP / "crop_after_remove.ply")
conv_xyz, _   = load_ply(COMP / "convonet_completed_mesh_colored.ply")
struct_xyz, _ = load_ply(COMP / "structural_surface_completed_points.ply")

void_xyz, _   = sub(void_xyz, None, 30000)
conv_xyz, _   = sub(conv_xyz, None, 50000)

print(f"void={void_xyz.shape[0]}  convonet={conv_xyz.shape[0]}  struct={struct_xyz.shape[0]}")

# Shared limits from the union of all three
allp = np.concatenate([void_xyz, conv_xyz, struct_xyz], axis=0)
PAD = 0.2
xlim = (allp[:,0].min()-PAD, allp[:,0].max()+PAD)
zlim = (allp[:,2].min()-PAD, allp[:,2].max()+PAD)
ylim = (allp[:,1].min()-PAD, allp[:,1].max()+PAD)

PROJS = [(0, 2), (0, 1)]
XL = ["X (m)", "X (m)"]
YL = ["Z (m)", "Y (m)"]
ROW = ["Top-down (X – Z)", "Front (X – Y)"]
COL_TITLES = [
    "(a) After removal\n(void — ConvONet input)",
    "(b) ConvONet raw mesh\n(dense over-completion)",
    "(c) Structural fill\n(planar surfaces, used)",
]

fig = plt.figure(figsize=(13, 8.5), facecolor="white")
gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.22,
              top=0.87, bottom=0.10, left=0.07, right=0.97)

for r, proj in enumerate(PROJS):
    lim_yz = zlim if r == 0 else ylim
    for c in range(3):
        ax = fig.add_subplot(gs[r, c])
        t = COL_TITLES[c] if r == 0 else ""
        yl = YL[r] if c == 0 else ""
        style(ax, t, XL[r], yl, xlim, lim_yz)
        if c == 0:
            sc(ax, void_xyz, C_VOID, alpha=0.55, proj=proj)
        elif c == 1:
            sc(ax, void_xyz, C_VOID, alpha=0.25, proj=proj)
            sc(ax, conv_xyz, C_CONV, alpha=0.35, proj=proj)
        else:
            sc(ax, void_xyz, C_VOID, alpha=0.25, proj=proj)
            sc(ax, struct_xyz, C_STRUCT, alpha=0.75, proj=proj)

# row labels
for r, lbl in enumerate(ROW):
    pos = fig.get_axes()[r*3].get_position()
    fig.text(0.005, pos.y0 + pos.height/2, lbl, va="center", ha="left",
             fontsize=9, color="#444", rotation=90, fontweight="bold")

handles = [
    mpatches.Patch(color=C_VOID,   label="Scene after removal (void)"),
    mpatches.Patch(color=C_CONV,   label="ConvONet raw completion (over-filled)"),
    mpatches.Patch(color=C_STRUCT, label="Structural surface fill (planar)"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
           framealpha=0.95, edgecolor="#ccc", bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "Background Completion Failure Analysis — Cabinet Removal "
    "(scene0000\\_00, job015)\n"
    f"ConvONet emits {conv_xyz.shape[0]:,}+ dense vertices filling the void volume; "
    f"structural fill recovers {struct_xyz.shape[0]:,} planar surface points",
    fontsize=11, fontweight="bold", color="#111", y=0.95,
)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_completion_failure.{ext}"
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
plt.close()
