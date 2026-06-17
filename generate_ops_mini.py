"""
Two separate figures:
  figs/fig_remove.pdf/png  — REMOVE operation (2 cols × 2 rows)
  figs/fig_move.pdf/png    — MOVE   operation (2 cols × 2 rows)

Columns: Before | After
Rows:    Top-down (X–Z) | Landscape (X–Y)
Full scene shown in original RGB, target/anchor highlighted on top.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

HYPO    = Path("/rds/general/user/qp23/home/Hypo3D")
PROC    = HYPO / "repo/scannet/processed/scene0000_00"
OUT_DIR = HYPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9})

COL_CABINET = "#d62728"
COL_BED    = "#d62728"
COL_DESK   = "#1f77b4"
COL_FRIDGE = "#ff7f0e"
COL_VOID   = "#ff7f0e"

# ── PLY loader ────────────────────────────────────────────────────────────────
def load_ply(path: Path):
    if not path.exists():
        print(f"  [missing] {path.name}")
        return np.empty((0, 3), np.float32), None
    data = path.read_bytes()
    hi   = data.find(b"end_header")
    nl   = data[hi + len("end_header")]
    he   = hi + len("end_header") + (2 if nl == ord('\r') else 1)
    hdr  = data[:he].decode("ascii", errors="ignore")
    n    = int(re.search(r"element vertex (\d+)", hdr).group(1))
    bin_ = "binary_little_endian" in hdr
    props = []
    in_v  = False
    for line in hdr.splitlines():
        line = line.strip()
        if   line.startswith("element vertex"): in_v = True
        elif line.startswith("element"):        in_v = False
        elif in_v and line.startswith("property") and "list" not in line:
            p = line.split()
            if len(p) >= 3: props.append((p[1], p[2]))
    TM = {"float":"f4","float32":"f4","double":"f8","uchar":"u1","uint8":"u1",
          "char":"i1","short":"i2","ushort":"u2","int":"i4","uint":"u4","int32":"i4"}
    if bin_:
        dt  = np.dtype([(nm, TM.get(tp, "f4")) for tp, nm in props])
        arr = np.frombuffer(data[he:], dtype=dt, count=n)
    else:
        lines = data[he:].decode("ascii", errors="ignore").splitlines()
        rows  = [list(map(float, l.split())) for l in lines[:n] if l.strip()]
        arr   = np.array(rows)
        arr   = np.rec.fromarrays(arr.T, names=[nm for _, nm in props])
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    rgb = None
    for names in (("red","green","blue"), ("r","g","b")):
        if all(nm in arr.dtype.names for nm in names):
            r_ = np.column_stack([arr[nm] for nm in names]).astype(np.float64)
            rgb = (r_/r_.max()*255 if r_.max()>1.5 else r_*255).astype(np.uint8)
            break
    return xyz, rgb

def sc(ax, xyz, rgb=None, color=None, alpha=0.55, proj=(0,2), ms=None):
    if xyz is None or xyz.shape[0] == 0: return
    n = xyz.shape[0]
    s = ms if ms else (6 if n < 5_000 else 1.5 if n < 50_000 else 0.4)
    x, y = xyz[:, proj[0]], xyz[:, proj[1]]
    kw = dict(s=s, alpha=alpha, linewidths=0, rasterized=True)
    if   color is not None: ax.scatter(x, y, c=color, **kw)
    elif rgb   is not None: ax.scatter(x, y, c=rgb.astype(np.float32)/255, **kw)
    else:                   ax.scatter(x, y, color="#888", **kw)

def bbox(ax, center, size, proj, color, lw=1.6, label=None):
    i, j = proj
    cx, cy = center[i], center[j]
    hx, hy = size[i]/2, size[j]/2
    rect = plt.Rectangle((cx-hx, cy-hy), 2*hx, 2*hy,
                          fill=False, edgecolor=color,
                          linewidth=lw, linestyle="--", zorder=5)
    ax.add_patch(rect)
    if label:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=8, color=color, style="italic",
                bbox=dict(fc="white", ec=color, boxstyle="round,pad=0.2",
                          alpha=0.9, lw=0.8), zorder=6)

def annotate(ax, xyz, text, color, proj, dy=0.22, anchor="bottom"):
    if xyz is None or xyz.shape[0] == 0: return
    c = xyz.mean(axis=0)
    va = "bottom" if anchor == "bottom" else "top"
    offset = dy if anchor == "bottom" else -dy
    ax.text(c[proj[0]], c[proj[1]]+offset, text,
            ha="center", va=va, fontsize=8, color=color, fontweight="bold",
            bbox=dict(fc="white", ec=color, boxstyle="round,pad=0.18",
                      alpha=0.92, lw=0.8), zorder=8)

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
    ax.set_xlim(xyz_all[:,proj[0]].min()-pad, xyz_all[:,proj[0]].max()+pad)
    ax.set_ylim(xyz_all[:,proj[1]].min()-pad, xyz_all[:,proj[1]].max()+pad)
    ax.set_aspect("equal")

def make_fig(axes_2x2, titles_top, row_labels,
             projs, xlabels, ylabels, draw_fn, legend_handles, suptitle):
    """Fill a 2×2 grid of axes using draw_fn(ax, col, proj)."""
    for c in range(2):
        for r, proj in enumerate(projs):
            ax = axes_2x2[r][c]
            col_title = titles_top[c][0] if r == 0 else None
            col_color = titles_top[c][1] if r == 0 else "#222"
            style_ax(ax, title=col_title, col=col_color,
                     xlabel=xlabels[r], ylabel=ylabels[r] if c == 0 else None)
            draw_fn(ax, c, proj)
    # row labels
    for r, lbl in enumerate(row_labels):
        pos = axes_2x2[r][0].get_position()
        axes_2x2[r][0].figure.text(
            0.025, pos.y0 + pos.height/2, lbl,
            va="center", ha="center", fontsize=9,
            color="#444", rotation=90, fontweight="bold")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
xyz_all = np.load(PROC/"xyz.npy").astype(np.float32)
rgb_all = np.load(PROC/"rgb.npy").astype(np.uint8)
inst    = np.load(PROC/"inst.npy")

mask_cabinet = inst == 9
mask_bed    = inst == 38
mask_desk   = inst == 8
mask_fridge = inst == 34

rm_mf = json.loads((PROC/"edits"/"job015.json").read_text())
rm_c  = np.array(rm_mf["remove_center"])
rm_s  = np.array(rm_mf["remove_size"])
rm_xyz, rm_rgb = load_ply(HYPO / rm_mf["canonical_edit_file"])

mv_mf  = json.loads((PROC/"edits"/"job000.json").read_text())
mv_bk_xyz, mv_bk_rgb = load_ply(Path(mv_mf["background_edit_file"]))
mv_pl_xyz, _         = load_ply(Path(mv_mf["placed_object_file"]))

desk_c   = xyz_all[mask_desk].mean(axis=0)   if mask_desk.sum()   > 0 else None
fridge_c = xyz_all[mask_fridge].mean(axis=0) if mask_fridge.sum() > 0 else None
placed_c = mv_pl_xyz.mean(axis=0)            if mv_pl_xyz.shape[0]> 0 else None
desk_sz  = (xyz_all[mask_desk].max(axis=0) - xyz_all[mask_desk].min(axis=0) + 0.1) \
            if mask_desk.sum() > 0 else None

PROJS   = [(0, 2), (0, 1)]
XLABELS = ["X (m)", "X (m)"]
YLABELS = ["Z (m)", "Y (m)"]
ROW_LBL = ["Top-down\n(X – Z)", "Landscape\n(X – Y)"]

PAD = 0.15

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — REMOVE
# ══════════════════════════════════════════════════════════════════════════════
fig_rm, axes_rm = plt.subplots(2, 2, figsize=(10, 7.5), facecolor="white")
plt.subplots_adjust(hspace=0.12, wspace=0.06,
                    top=0.88, bottom=0.10, left=0.07, right=0.97)

def draw_rm(ax, col, proj):
    if col == 0:   # Before
        sc(ax, xyz_all[~mask_cabinet], rgb_all[~mask_cabinet], alpha=0.50, proj=proj)
        sc(ax, xyz_all[mask_cabinet],  color=COL_CABINET, alpha=0.95, proj=proj, ms=4)
        bbox(ax, rm_c, rm_s, proj, "#555555", lw=1.2)
        if proj == (0, 2):
            annotate(ax, xyz_all[mask_cabinet], "cabinet  (target)", COL_CABINET, proj)
    else:          # After
        sc(ax, rm_xyz, rm_rgb, alpha=0.55, proj=proj)
        bbox(ax, rm_c, rm_s, proj, COL_VOID, lw=1.8, label="void")
    set_lim(ax, xyz_all, proj, PAD)

for c in range(2):
    for r, proj in enumerate(PROJS):
        ax = axes_rm[r][c]
        titles = [("(a)  Before removal", "#922B21"),
                  ("(b)  After removal",  "#922B21")]
        style_ax(ax,
                 title=titles[c][0] if r == 0 else None,
                 col=titles[c][1],
                 xlabel=XLABELS[r],
                 ylabel=YLABELS[r] if c == 0 else None)
        draw_rm(ax, c, proj)

for r, lbl in enumerate(ROW_LBL):
    pos = axes_rm[r][0].get_position()
    fig_rm.text(0.018, pos.y0 + pos.height/2, lbl,
                va="center", ha="center", fontsize=9,
                color="#444", rotation=90, fontweight="bold")

fig_rm.suptitle("REMOVE Operation  —  scene0000\\_00 / job015  (cabinet removal)",
                fontsize=12, fontweight="bold", color="#922B21", y=0.94)

rm_legend = [
    mpatches.Patch(color="#aaaaaa",  label="Scene geometry (RGB)"),
    mpatches.Patch(color=COL_CABINET, label="Removed object (cabinet)"),
    mpatches.Patch(fc="none", ec=COL_VOID, ls="--", lw=1.4, label="Void region"),
]
fig_rm.legend(handles=rm_legend, loc="lower center",
              bbox_to_anchor=(0.5, 0.01), ncol=3,
              fontsize=9, framealpha=0.95, edgecolor="#ccc",
              facecolor="white", handlelength=1.3)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_remove.{ext}"
    fig_rm.savefig(out, facecolor="white", dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig_rm)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — MOVE
# ══════════════════════════════════════════════════════════════════════════════
fig_mv, axes_mv = plt.subplots(2, 2, figsize=(10, 7.5), facecolor="white")
plt.subplots_adjust(hspace=0.12, wspace=0.06,
                    top=0.88, bottom=0.10, left=0.07, right=0.97)

def draw_mv(ax, col, proj):
    if col == 0:   # Before
        sc(ax, xyz_all[~mask_desk & ~mask_fridge],
               rgb_all[~mask_desk & ~mask_fridge], alpha=0.45, proj=proj)
        sc(ax, xyz_all[mask_fridge], color=COL_FRIDGE, alpha=0.95, proj=proj, ms=4)
        sc(ax, xyz_all[mask_desk],   color=COL_DESK,   alpha=0.95, proj=proj, ms=4)
        if proj == (0, 2):
            annotate(ax, xyz_all[mask_desk],   "desk\n(target)",       COL_DESK,   proj, dy=0.22)
            annotate(ax, xyz_all[mask_fridge], "refrigerator\n(anchor)",COL_FRIDGE, proj, dy=0.22, anchor="top")
            # move direction arrow
            if desk_c is not None and placed_c is not None:
                ax.annotate("",
                    xy=(placed_c[0], placed_c[2]),
                    xytext=(desk_c[0], desk_c[2]),
                    arrowprops=dict(arrowstyle="-|>", color="#333",
                                    lw=1.5, mutation_scale=13,
                                    connectionstyle="arc3,rad=0.25"), zorder=6)
                mid = ((desk_c[0]+placed_c[0])/2, (desk_c[2]+placed_c[2])/2)
                ax.text(mid[0]+0.1, mid[1], "RIGHT_OF",
                        fontsize=7.5, color="#444", style="italic",
                        ha="left", va="center", zorder=7)
    else:          # After
        sc(ax, mv_bk_xyz, mv_bk_rgb, alpha=0.50, proj=proj)
        sc(ax, mv_pl_xyz, color=COL_DESK, alpha=0.95, proj=proj, ms=4)
        if proj == (0, 2):
            annotate(ax, mv_pl_xyz, "desk\n(new position)", COL_DESK, proj, dy=0.22)
        # ghost box: original desk position
        if desk_c is not None and desk_sz is not None:
            bbox(ax, desk_c, desk_sz, proj, "#aaaaaa", lw=1.0, label="original")
    set_lim(ax, xyz_all, proj, PAD)

for c in range(2):
    for r, proj in enumerate(PROJS):
        ax = axes_mv[r][c]
        titles = [("(a)  Before move", "#1f77b4"),
                  ("(b)  After move",  "#1f77b4")]
        style_ax(ax,
                 title=titles[c][0] if r == 0 else None,
                 col=titles[c][1],
                 xlabel=XLABELS[r],
                 ylabel=YLABELS[r] if c == 0 else None)
        draw_mv(ax, c, proj)

for r, lbl in enumerate(ROW_LBL):
    pos = axes_mv[r][0].get_position()
    fig_mv.text(0.018, pos.y0 + pos.height/2, lbl,
                va="center", ha="center", fontsize=9,
                color="#444", rotation=90, fontweight="bold")

fig_mv.suptitle(
    "MOVE Operation  —  scene0000\\_00 / job000  (desk → right of refrigerator)",
    fontsize=12, fontweight="bold", color="#1f77b4", y=0.94)

mv_legend = [
    mpatches.Patch(color="#aaaaaa",  label="Scene geometry (RGB)"),
    mpatches.Patch(color=COL_DESK,   label="Moved object (desk)"),
    mpatches.Patch(color=COL_FRIDGE, label="Anchor object (refrigerator)"),
    mpatches.Patch(fc="none", ec="#aaaaaa", ls="--", lw=1.0, label="Original position"),
]
fig_mv.legend(handles=mv_legend, loc="lower center",
              bbox_to_anchor=(0.5, 0.01), ncol=4,
              fontsize=9, framealpha=0.95, edgecolor="#ccc",
              facecolor="white", handlelength=1.3)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_move.{ext}"
    fig_mv.savefig(out, facecolor="white", dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig_mv)

print("Done.")
