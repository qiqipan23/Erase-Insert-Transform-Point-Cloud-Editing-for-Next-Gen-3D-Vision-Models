"""
Generate a clean, dissertation-quality figure for the removal pipeline.

Layout: 3 columns × 2 rows
  Col 1 — Original scene  (bed highlighted in red)
  Col 2 — After removal   (void visible)
  Col 3 — Reconstructed   (ConvONet + structural fill merged)

  Row 1 — Top-down view  (X → , Z ↑)
  Row 2 — Front view     (X → , Y ↑)

Output: figs/fig_removal_clean.pdf / .png
"""

from __future__ import annotations
import json, struct, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree

# ── Paths ─────────────────────────────────────────────────────────────────────
HYPO        = Path("/rds/general/user/qp23/home/Hypo3D")
PROCESSED   = HYPO / "repo/scannet/processed"
SCENE_ID    = "scene0000_00"
JOB_IDX     = 15
COMP_DIR    = PROCESSED / SCENE_ID / "completion" / f"{SCENE_ID}_job{JOB_IDX:03d}"
EDIT_DIR    = PROCESSED / SCENE_ID / "edits"
OUT_DIR     = HYPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

BED_RED   = "#d62728"
VOID_COL  = "#ff7f0e"
FILL_COL  = "#2ca02c"
SCENE_COL = "#aec7e8"

# ── PLY loader ────────────────────────────────────────────────────────────────
def load_ply(path: Path):
    """Returns (xyz [N,3] float32, rgb [N,3] uint8 or None)."""
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
            if len(p) >= 3:
                props.append((p[1], p[2]))

    TM = {"float":"f4","float32":"f4","double":"f8","uchar":"u1","uint8":"u1",
          "char":"i1","short":"i2","ushort":"u2","int":"i4","uint":"u4","int32":"i4"}
    if bin_:
        dt  = np.dtype([(nm, TM.get(tp, "f4")) for tp, nm in props])
        arr = np.frombuffer(data[he:], dtype=dt, count=n)
    else:
        lines = data[he:].decode("ascii", errors="ignore").splitlines()
        rows  = [list(map(float, l.split())) for l in lines[:n] if l.strip()]
        if not rows:
            return np.empty((0, 3), np.float32), None
        arr = np.array(rows)
        arr = np.rec.fromarrays(arr.T, names=[nm for _, nm in props])

    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    rgb = None
    for names in (("red","green","blue"), ("r","g","b")):
        if all(nm in arr.dtype.names for nm in names):
            r_ = np.column_stack([arr[nm] for nm in names]).astype(np.float64)
            rgb = (r_ / r_.max() * 255 if r_.max() > 1.5 else r_ * 255).astype(np.uint8)
            break
    return xyz, rgb

# ── Helpers ───────────────────────────────────────────────────────────────────
def crop(xyz, rgb, center, size, pad=2.0):
    if xyz is None or xyz.shape[0] == 0:
        return xyz, rgb
    mn = np.array(center) - np.array(size) / 2 * pad
    mx = np.array(center) + np.array(size) / 2 * pad
    m  = np.all((xyz >= mn) & (xyz <= mx), axis=1)
    return xyz[m], (rgb[m] if rgb is not None else None)

def scatter(ax, xyz, rgb=None, color=None, alpha=0.7, proj=(0,2), ms=None):
    if xyz is None or xyz.shape[0] == 0:
        ax.text(0.5, 0.5, "no data", color="#999", ha="center",
                va="center", transform=ax.transAxes, fontsize=9)
        return
    n = xyz.shape[0]
    s = ms if ms else (4 if n < 20_000 else (1.5 if n < 80_000 else 0.4))
    x, y = xyz[:, proj[0]], xyz[:, proj[1]]
    if color is not None:
        ax.scatter(x, y, s=s, c=color, alpha=alpha, linewidths=0, rasterized=True)
    elif rgb is not None:
        ax.scatter(x, y, s=s, c=rgb.astype(np.float32)/255,
                   alpha=alpha, linewidths=0, rasterized=True)
    else:
        ax.scatter(x, y, s=s, color="#666", alpha=alpha, linewidths=0, rasterized=True)

def draw_aabb(ax, center, size, proj, color="#ff7f0e", lw=1.5):
    i, j  = proj
    cx, cy = center[i], center[j]
    hx, hy = size[i]/2, size[j]/2
    rect = plt.Rectangle((cx-hx, cy-hy), 2*hx, 2*hy,
                          fill=False, edgecolor=color,
                          linewidth=lw, linestyle="--", zorder=5)
    ax.add_patch(rect)

def style_ax(ax, xlabel, ylabel):
    ax.set_facecolor("white")
    ax.set_xlabel(xlabel, fontsize=8, color="#555")
    ax.set_ylabel(ylabel, fontsize=8, color="#555")
    ax.tick_params(labelsize=7, color="#aaa", labelcolor="#666")
    for sp in ("top","right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom","left"):
        ax.spines[sp].set_color("#cccccc")
    ax.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)

# ── Load data ─────────────────────────────────────────────────────────────────
manifest = json.loads((EDIT_DIR / f"job{JOB_IDX:03d}.json").read_text())
center   = np.array(manifest["remove_center"])
size     = np.array(manifest["remove_size"])
label    = manifest.get("target_label", "object")

print(f"Scene: {SCENE_ID}  |  Job {JOB_IDX:03d}  |  Removing: {label}")
print(f"  Center: {center}  |  Size: {size}")

# Original scan
orig_xyz, orig_rgb = load_ply(PROCESSED / SCENE_ID / f"original_{SCENE_ID}.ply")
print(f"  Original: {orig_xyz.shape[0]:,} pts")

# After removal — canonical edited scene (void exposed), from manifest
removed_xyz, removed_rgb = load_ply(HYPO / manifest["canonical_edit_file"])
print(f"  After removal: {removed_xyz.shape[0]:,} pts")

# Reconstructed — final pipeline output (bed removed + background filled)
recon_xyz, recon_rgb = load_ply(COMP_DIR / "occluded_surface_full_scene_merged.ply")
print(f"  Reconstructed: {recon_xyz.shape[0]:,} pts")

# Fill-only points (both wall fill + floor fill combined)
fill_parts = [
    COMP_DIR / "nn_floor_fill_completed_points.ply",        # wall fill
    COMP_DIR / "structural_surface_completed_points.ply",   # floor fill
    COMP_DIR / "occluded_surface_completed_points.ply",     # hidden surface
]
fill_xyz_list = []
for p in fill_parts:
    if p.exists() and p.stat().st_size > 200:
        xyz_, _ = load_ply(p)
        if xyz_.shape[0] > 0:
            fill_xyz_list.append(xyz_)
            print(f"  Fill part ({p.name}): {xyz_.shape[0]:,} pts")
fill_xyz = np.concatenate(fill_xyz_list, axis=0) if fill_xyz_list else np.empty((0,3), np.float32)
print(f"  Total fill: {fill_xyz.shape[0]:,} pts")

# ── Identify removed points (present in orig, missing after removal) ──────────
PAD = 2.2
orig_c,    orig_c_rgb    = crop(orig_xyz,    orig_rgb,    center, size, PAD)
removed_c, removed_c_rgb = crop(removed_xyz, removed_rgb, center, size, PAD)
recon_c,   recon_c_rgb   = crop(recon_xyz,   recon_rgb,   center, size, PAD)
fill_c,    _             = crop(fill_xyz,    None,        center, size, PAD)

removed_mask = np.zeros(orig_c.shape[0], dtype=bool)
if orig_c.shape[0] > 0 and removed_c.shape[0] > 0:
    dists, _ = cKDTree(removed_c).query(orig_c, k=1)
    removed_mask = dists > 0.03
    print(f"  Detected {removed_mask.sum():,} removed pts in crop")

keep_xyz = orig_c[~removed_mask]
keep_rgb = orig_c_rgb[~removed_mask] if orig_c_rgb is not None else None
bed_xyz  = orig_c[removed_mask]

# ── Axis limits ───────────────────────────────────────────────────────────────
PAD_LIM = 0.7
cx, cy, cz = center
sx, sy, sz = size
xlim = (cx - sx/2 - PAD_LIM, cx + sx/2 + PAD_LIM)
ylim = (cy - sy/2 - PAD_LIM, cy + sy/2 + PAD_LIM)
zlim = (cz - sz/2 - PAD_LIM, cz + sz/2 + PAD_LIM)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7.5), facecolor="white")
gs  = GridSpec(2, 3, figure=fig,
               hspace=0.38, wspace=0.22,
               top=0.88, bottom=0.09, left=0.07, right=0.97)

col_titles = [
    f"(a) Original scene\n({label.capitalize()} highlighted)",
    "(b) After removal\n(void region exposed)",
    "(c) Background reconstructed\n(final pipeline output)",
]
row_labels = [
    ("X (m)", "Z (m)"),   # top-down
    ("X (m)", "Y (m)"),   # front
]
PROJS = [(0, 2), (0, 1)]
LIMS  = [(xlim, zlim), (xlim, ylim)]

axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

for r in range(2):
    proj  = PROJS[r]
    lim_x, lim_y = LIMS[r]

    for c in range(3):
        ax = axes[r][c]
        style_ax(ax, row_labels[r][0], row_labels[r][1])
        if r == 0:
            ax.set_title(col_titles[c], fontsize=10, pad=6, color="#222")

        # ── Col 0: original with bed highlighted ──────────────────────────────
        if c == 0:
            scatter(ax, keep_xyz,    keep_rgb,  proj=proj, alpha=0.55)
            scatter(ax, bed_xyz,     color=BED_RED, proj=proj, alpha=0.90)
            draw_aabb(ax, center, size, proj, color="#555555", lw=1.4)

        # ── Col 1: after removal ──────────────────────────────────────────────
        elif c == 1:
            scatter(ax, removed_c, removed_c_rgb, proj=proj, alpha=0.55)
            draw_aabb(ax, center, size, proj, color=VOID_COL, lw=1.6)
            ax.text(cx, cz if r==0 else cy, "void", ha="center", va="center",
                    fontsize=8, color=VOID_COL, style="italic",
                    bbox=dict(fc="white", ec=VOID_COL, boxstyle="round,pad=0.3",
                              alpha=0.85, lw=0.8))

        # ── Col 2: reconstructed scene ────────────────────────────────────────
        else:
            # Render the final pipeline output with its own RGB colours.
            # The fill is already merged in — no overlay needed.
            scatter(ax, recon_c, recon_c_rgb, proj=proj, alpha=0.70)
            draw_aabb(ax, center, size, proj, color="#aaaaaa", lw=1.0)

        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_aspect("equal")

# ── Row labels ────────────────────────────────────────────────────────────────
for r, label_txt in enumerate(["Top-down view", "Front view"]):
    fig.text(0.005, axes[r][0].get_position().y0 +
             axes[r][0].get_position().height / 2,
             label_txt, va="center", ha="left", fontsize=9,
             color="#444", rotation=90, fontweight="bold")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=SCENE_COL, label="Scene geometry"),
    mpatches.Patch(color=BED_RED,   label=f"Removed object ({label})"),
    mpatches.Patch(facecolor="none", edgecolor=VOID_COL,
                   linestyle="--", linewidth=1.5, label="Removal bounding box"),
]
fig.legend(handles=handles, loc="lower center", ncol=4,
           framealpha=0.9, edgecolor="#cccccc",
           fontsize=8.5, bbox_to_anchor=(0.5, 0.01))

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    f"Object Removal Pipeline — {SCENE_ID}  (removing {label})",
    fontsize=13, color="#111", y=0.95, fontweight="bold"
)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_removal_clean.{ext}"
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")

plt.close()
