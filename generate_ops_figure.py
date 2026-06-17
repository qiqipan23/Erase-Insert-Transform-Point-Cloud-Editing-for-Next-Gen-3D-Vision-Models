"""
Two-row figure showing a REMOVE and a MOVE operation (top-down view).

Row 1 — REMOVE (job014: bed):
  (a) Original scene  — bed highlighted in red
  (b) After removal   — void exposed

Row 2 — MOVE (job000: desk → right of refrigerator):
  (c) Original scene  — desk in blue, refrigerator (anchor) in orange
  (d) After move      — background + desk at new position (blue)

Output: figs/fig_operations.pdf / .png
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
from scipy.spatial import cKDTree

# ── Paths ─────────────────────────────────────────────────────────────────────
HYPO    = Path("/rds/general/user/qp23/home/Hypo3D")
PROC    = HYPO / "repo/scannet/processed/scene0000_00"
OUT_DIR = HYPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Colours
COL_SCENE  = "#aec7e8"   # light blue   – background scene points
COL_BED    = "#d62728"   # red          – removed object (bed)
COL_DESK   = "#1f77b4"   # blue         – moved object (desk)
COL_FRIDGE = "#ff7f0e"   # orange       – anchor object (refrigerator)
COL_VOID   = "#ff7f0e"   # orange dashed – removal bounding box
COL_FILL   = "#2ca02c"   # green        – fill indicator (not used here)

# ── PLY loader ────────────────────────────────────────────────────────────────
def load_ply(path: Path):
    if not path.exists():
        print(f"  [missing] {path.name}")
        return np.empty((0,3),np.float32), None
    data = path.read_bytes()
    hi   = data.find(b"end_header")
    nl   = data[hi + len("end_header")]
    he   = hi + len("end_header") + (2 if nl == ord('\r') else 1)
    hdr  = data[:he].decode("ascii", errors="ignore")
    n    = int(re.search(r"element vertex (\d+)", hdr).group(1))
    bin_ = "binary_little_endian" in hdr
    props = []
    in_v = False
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
        dt  = np.dtype([(nm, TM.get(tp,"f4")) for tp, nm in props])
        arr = np.frombuffer(data[he:], dtype=dt, count=n)
    else:
        lines = data[he:].decode("ascii", errors="ignore").splitlines()
        rows  = [list(map(float, l.split())) for l in lines[:n] if l.strip()]
        arr   = np.array(rows)
        arr   = np.rec.fromarrays(arr.T, names=[nm for _,nm in props])
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    rgb = None
    for names in (("red","green","blue"),("r","g","b")):
        if all(nm in arr.dtype.names for nm in names):
            r_ = np.column_stack([arr[nm] for nm in names]).astype(np.float64)
            rgb = (r_/r_.max()*255 if r_.max()>1.5 else r_*255).astype(np.uint8)
            break
    return xyz, rgb

# ── Scatter helper ────────────────────────────────────────────────────────────
def scatter(ax, xyz, rgb=None, color=None, alpha=0.6, proj=(0,2), ms=None):
    if xyz is None or xyz.shape[0] == 0: return
    n = xyz.shape[0]
    s = ms if ms else (5 if n < 5_000 else 2 if n < 30_000 else 0.6)
    x, y = xyz[:, proj[0]], xyz[:, proj[1]]
    kw = dict(s=s, alpha=alpha, linewidths=0, rasterized=True)
    if color is not None:
        ax.scatter(x, y, c=color, **kw)
    elif rgb is not None:
        ax.scatter(x, y, c=rgb.astype(np.float32)/255, **kw)
    else:
        ax.scatter(x, y, color="#888", **kw)

def style(ax, title, xlabel="X (m)", ylabel="Z (m)"):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=10.5, pad=7, color="#111", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8, color="#666")
    ax.set_ylabel(ylabel, fontsize=8, color="#666")
    ax.tick_params(labelsize=7, color="#bbb", labelcolor="#777")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("bottom","left"): ax.spines[sp].set_color("#dddddd")
    ax.grid(True, color="#f0f0f0", lw=0.5, zorder=0)

def bbox_rect(ax, center, size, proj=(0,2), color="#888", lw=1.5, label=None):
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
                bbox=dict(fc="white", ec=color, boxstyle="round,pad=0.25",
                          alpha=0.85, lw=0.8), zorder=6)

def set_lim(ax, center, size, pad, proj=(0,2)):
    i, j = proj
    ax.set_xlim(center[i]-size[i]/2-pad, center[i]+size[i]/2+pad)
    ax.set_ylim(center[j]-size[j]/2-pad, center[j]+size[j]/2+pad)
    ax.set_aspect("equal")

# ── Load shared data: xyz, rgb, inst arrays ───────────────────────────────────
print("Loading scene arrays …")
xyz_all = np.load(PROC / "xyz.npy").astype(np.float32)   # (81369, 3)
rgb_all = np.load(PROC / "rgb.npy").astype(np.uint8)     # (81369, 3)
inst    = np.load(PROC / "inst.npy")                     # (81369,)

# ── Instance masks ────────────────────────────────────────────────────────────
BED_ID    = 38
DESK_ID   = 8
FRIDGE_ID = 34

mask_bed    = inst == BED_ID
mask_desk   = inst == DESK_ID
mask_fridge = inst == FRIDGE_ID
mask_scene  = ~(mask_bed | mask_desk | mask_fridge)

print(f"  Bed:          {mask_bed.sum():,} pts")
print(f"  Desk:         {mask_desk.sum():,} pts")
print(f"  Refrigerator: {mask_fridge.sum():,} pts")
print(f"  Rest:         {mask_scene.sum():,} pts")

# ── Load edit files ───────────────────────────────────────────────────────────
# REMOVE
rm_manifest = json.loads((PROC/"edits"/"job014.json").read_text())
rm_center   = np.array(rm_manifest["remove_center"])
rm_size     = np.array(rm_manifest["remove_size"])
rm_after_xyz, rm_after_rgb = load_ply(HYPO / rm_manifest["canonical_edit_file"])
print(f"  REMOVE after: {rm_after_xyz.shape[0]:,} pts")

# MOVE
mv_manifest  = json.loads((PROC/"edits"/"job000.json").read_text())
mv_bkgd_xyz, mv_bkgd_rgb   = load_ply(Path(mv_manifest["background_edit_file"]))
mv_placed_xyz, mv_placed_rgb = load_ply(Path(mv_manifest["placed_object_file"]))
print(f"  MOVE bkgd:    {mv_bkgd_xyz.shape[0]:,} pts")
print(f"  MOVE placed:  {mv_placed_xyz.shape[0]:,} pts")

# ── Crop helpers ──────────────────────────────────────────────────────────────
def crop(xyz, rgb, center, size, pad=2.5):
    mn = center - size/2 * pad
    mx = center + size/2 * pad
    m  = np.all((xyz >= mn) & (xyz <= mx), axis=1)
    return xyz[m], (rgb[m] if rgb is not None else None)

# REMOVE crop region
rm_pad = 2.5
rm_xyz_c,  rm_rgb_c   = crop(xyz_all,      rgb_all,       rm_center, rm_size, rm_pad)
rm_bed_c,  _          = crop(xyz_all[mask_bed], rgb_all[mask_bed], rm_center, rm_size, rm_pad)
rm_aft_c,  rm_aft_rgb = crop(rm_after_xyz, rm_after_rgb,  rm_center, rm_size, rm_pad)

# MOVE crop region — centre between desk original and new position
desk_center  = xyz_all[mask_desk].mean(axis=0)
fridge_center= xyz_all[mask_fridge].mean(axis=0)
placed_center= mv_placed_xyz.mean(axis=0) if mv_placed_xyz.shape[0] > 0 else desk_center
mv_center    = (np.minimum(desk_center, placed_center) +
                np.maximum(desk_center, placed_center)) / 2
mv_size      = np.abs(placed_center - desk_center) + np.array([2.5, 2.5, 2.5])

mv_pad = 1.0
mv_all_c,   mv_all_rgb   = crop(xyz_all,        rgb_all,          mv_center, mv_size, mv_pad)
mv_desk_c,  _            = crop(xyz_all[mask_desk],   None,        mv_center, mv_size, mv_pad)
mv_fridge_c,_            = crop(xyz_all[mask_fridge], None,        mv_center, mv_size, mv_pad)
mv_rest_c,  mv_rest_rgb  = crop(xyz_all[~mask_desk],  rgb_all[~mask_desk], mv_center, mv_size, mv_pad)
mv_bkgd_c,  mv_bkgd_rgb2 = crop(mv_bkgd_xyz,   mv_bkgd_rgb,      mv_center, mv_size, mv_pad)
mv_placed_c, _           = crop(mv_placed_xyz,  None,              mv_center, mv_size, mv_pad)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 9), facecolor="white")
gs  = GridSpec(2, 2, figure=fig,
               hspace=0.45, wspace=0.25,
               top=0.91, bottom=0.09, left=0.08, right=0.97)

# ── Row labels ────────────────────────────────────────────────────────────────
fig.text(0.01, 0.72, "REMOVE\noperation", va="center", ha="left",
         fontsize=9.5, color="#922B21", fontweight="bold", rotation=90)
fig.text(0.01, 0.30, "MOVE\noperation",  va="center", ha="left",
         fontsize=9.5, color="#1f77b4",  fontweight="bold", rotation=90)

PROJ = (0, 2)   # top-down: X → , Z ↑

# ══════════════════════════════════════════════════════════════════════════════
# Row 0 — REMOVE
# ══════════════════════════════════════════════════════════════════════════════
ax_rm0 = fig.add_subplot(gs[0, 0])
ax_rm1 = fig.add_subplot(gs[0, 1])

style(ax_rm0, "(a)  Original scene — bed highlighted")
style(ax_rm1, "(b)  After removal — void exposed")

PAD_RM = 0.9

# (a) original: rest in grey, bed in red
scatter(ax_rm0, rm_xyz_c, color=COL_SCENE, alpha=0.40, proj=PROJ)
scatter(ax_rm0, rm_bed_c, color=COL_BED,   alpha=0.90, proj=PROJ, ms=4)
bbox_rect(ax_rm0, rm_center, rm_size, proj=PROJ, color="#555", lw=1.3)
set_lim(ax_rm0, rm_center, rm_size, PAD_RM, PROJ)

# (b) after removal: scene minus bed, bounding box shows void
scatter(ax_rm1, rm_aft_c,  rm_aft_rgb, alpha=0.55, proj=PROJ)
bbox_rect(ax_rm1, rm_center, rm_size, proj=PROJ,
          color=COL_VOID, lw=1.8, label="void")
set_lim(ax_rm1, rm_center, rm_size, PAD_RM, PROJ)

# ══════════════════════════════════════════════════════════════════════════════
# Row 1 — MOVE
# ══════════════════════════════════════════════════════════════════════════════
ax_mv0 = fig.add_subplot(gs[1, 0])
ax_mv1 = fig.add_subplot(gs[1, 1])

style(ax_mv0, "(c)  Original scene — desk & anchor highlighted")
style(ax_mv1, "(d)  After move — desk placed right of refrigerator")

PAD_MV = 0.6

# (c) original: rest grey, desk blue, fridge orange
scatter(ax_mv0, mv_rest_c,   color=COL_SCENE,  alpha=0.35, proj=PROJ)
scatter(ax_mv0, mv_fridge_c, color=COL_FRIDGE, alpha=0.95, proj=PROJ, ms=5)
scatter(ax_mv0, mv_desk_c,   color=COL_DESK,   alpha=0.95, proj=PROJ, ms=5)

# Annotate desk and fridge
if mv_desk_c.shape[0] > 0:
    dc = mv_desk_c.mean(axis=0)
    ax_mv0.text(dc[PROJ[0]], dc[PROJ[1]]+0.18, "desk\n(target)",
                ha="center", va="bottom", fontsize=7.5,
                color=COL_DESK, fontweight="bold",
                bbox=dict(fc="white", ec=COL_DESK, boxstyle="round,pad=0.2",
                          alpha=0.9, lw=0.8), zorder=7)

if mv_fridge_c.shape[0] > 0:
    fc = mv_fridge_c.mean(axis=0)
    ax_mv0.text(fc[PROJ[0]], fc[PROJ[1]]-0.22, "refrigerator\n(anchor)",
                ha="center", va="top", fontsize=7.5,
                color=COL_FRIDGE, fontweight="bold",
                bbox=dict(fc="white", ec=COL_FRIDGE, boxstyle="round,pad=0.2",
                          alpha=0.9, lw=0.8), zorder=7)

# Draw relation arrow: desk → right of fridge
if mv_desk_c.shape[0] > 0 and mv_placed_c.shape[0] > 0:
    dc  = mv_desk_c.mean(axis=0)
    npc = mv_placed_c.mean(axis=0)
    ax_mv0.annotate("",
        xy   =(npc[PROJ[0]], npc[PROJ[1]]),
        xytext=(dc[PROJ[0]],  dc[PROJ[1]]),
        arrowprops=dict(arrowstyle="-|>", color="#555",
                        lw=1.4, mutation_scale=12,
                        connectionstyle="arc3,rad=0.25"),
        zorder=6)
    ax_mv0.text((dc[PROJ[0]]+npc[PROJ[0]])/2 + 0.15,
                (dc[PROJ[1]]+npc[PROJ[1]])/2,
                "RIGHT_OF", fontsize=7, color="#555",
                style="italic", ha="left", va="center", zorder=7)

# Shared limits for both MOVE panels
all_pts = np.concatenate([mv_rest_c, mv_desk_c, mv_fridge_c,
                           mv_placed_c], axis=0) if mv_placed_c.shape[0] > 0 \
          else np.concatenate([mv_rest_c, mv_desk_c, mv_fridge_c], axis=0)
xmin, xmax = all_pts[:,PROJ[0]].min()-PAD_MV, all_pts[:,PROJ[0]].max()+PAD_MV
ymin, ymax = all_pts[:,PROJ[1]].min()-PAD_MV, all_pts[:,PROJ[1]].max()+PAD_MV
for ax_ in (ax_mv0, ax_mv1):
    ax_.set_xlim(xmin, xmax)
    ax_.set_ylim(ymin, ymax)
    ax_.set_aspect("equal")

# (d) after move: background (grey) + placed desk (blue)
scatter(ax_mv1, mv_bkgd_c,   mv_bkgd_rgb2, alpha=0.45, proj=PROJ)
scatter(ax_mv1, mv_placed_c, color=COL_DESK, alpha=0.95, proj=PROJ, ms=6)

# Annotate new desk position
if mv_placed_c.shape[0] > 0:
    npc = mv_placed_c.mean(axis=0)
    ax_mv1.text(npc[PROJ[0]], npc[PROJ[1]]+0.18, "desk\n(new position)",
                ha="center", va="bottom", fontsize=7.5,
                color=COL_DESK, fontweight="bold",
                bbox=dict(fc="white", ec=COL_DESK, boxstyle="round,pad=0.2",
                          alpha=0.9, lw=0.8), zorder=7)
# Show original desk position (void)
if mv_desk_c.shape[0] > 0:
    dc = mv_desk_c.mean(axis=0)
    dsize = (mv_desk_c.max(axis=0) - mv_desk_c.min(axis=0))
    bbox_rect(ax_mv1, dc, dsize+0.1, proj=PROJ,
              color="#aaaaaa", lw=1.2, label="original\nposition")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=COL_SCENE,  label="Scene geometry"),
    mpatches.Patch(color=COL_BED,    label="Removed object (bed)"),
    mpatches.Patch(color=COL_DESK,   label="Moved object (desk)"),
    mpatches.Patch(color=COL_FRIDGE, label="Anchor object (refrigerator)"),
    mpatches.Patch(facecolor="none", edgecolor=COL_VOID,
                   linestyle="--", linewidth=1.5, label="Removal bounding box / original position"),
]
fig.legend(handles=handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.01), ncol=5,
           fontsize=8.2, framealpha=0.95, edgecolor="#ccc",
           facecolor="white", handlelength=1.4, columnspacing=1.0)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle("3D Scene Editing Operations: REMOVE and MOVE  (Top-down view)",
             fontsize=13, fontweight="bold", color="#1C2833", y=0.97)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_operations.{ext}"
    plt.savefig(out, facecolor="white", dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close()
