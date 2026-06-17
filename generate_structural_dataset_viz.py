"""
Structural surface dataset visualisation — real ScanNet colours.
Loads xyz.npy/rgb.npy from the processed scene; overlays void bbox and
target surface voxel positions from the training npz.

3 rows (floor / wall / on-top) × 3 columns (before / void / target).
Output: figs/fig_structural_dataset_examples.png/.pdf
"""
import numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TRAIN_BASE = Path("/rds/general/user/qp23/home/Hypo3D/repo/scannet/completion/"
                  "structural_surface_scannet_wallfix_train/scannet_remove_structural_surface")
PROC       = Path("/rds/general/user/qp23/home/Hypo3D/repo/scannet/processed")
OUT        = Path("/rds/general/user/qp23/home/Hypo3D/figs/fig_structural_dataset_examples")

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "savefig.dpi": 300, "figure.dpi": 130})

# ── Pick one clear example per support type ─────────────────────────────────
WANT   = {"floor": None, "against_wall": None, "on_top_of": None}
MINPTS = {"floor": 150,  "against_wall": 100,  "on_top_of": 80}
for d in sorted(TRAIN_BASE.iterdir()):
    if not (d / "metadata.json").exists(): continue
    if all(v is not None for v in WANT.values()): break
    tgt  = np.load(d / "target_surface.npz")
    meta = json.loads((d / "metadata.json").read_text())
    st   = meta.get("synthetic_support_type", "")
    if st in WANT and WANT[st] is None and int(tgt["surface_mask"].sum()) >= MINPTS.get(st, 80):
        WANT[st] = (d, meta)

ORDER  = ["floor", "against_wall", "on_top_of"]
RLABEL = ["(a) Floor support", "(b) Wall support", "(c) On-top support"]
RDESC  = {
    "floor":        "Removed object rested on floor — hidden surface is the floor beneath it",
    "against_wall": "Removed object stood against wall — hidden surface is the wall face behind it",
    "on_top_of":    "Removed object sat on furniture — hidden surface is the top face below it",
}

# Which 2D projection per support type: (axis_h, axis_v)
# ScanNet: X=col(horiz), Y=depth, Z=up
# floor/on_top: top-down X-Y;  wall: front view X-Z
VIEW_PROJ = {
    "floor":        (0, 1, "X (m)", "Y (m)"),
    "against_wall": (0, 2, "X (m)", "Z (m)"),
    "on_top_of":    (0, 1, "X (m)", "Y (m)"),
}

def quantized_bounds_min(center, half, vox):
    return (np.floor((center - half) / vox) * vox).astype(np.float32)

def target_surface_xyz(meta, surface_mask):
    """Convert (D,H,W) surface mask to world-space XYZ point array."""
    vox  = float(meta["voxel_size"])
    ctr  = np.array(meta["remove_center"], dtype=np.float32)
    half = np.array(meta["crop_half_extent"], dtype=np.float32)
    bmin = quantized_bounds_min(ctr, half, vox)
    # mask shape (1,D,H,W) or (D,H,W); D=Z,H=Y,W=X
    m = surface_mask[0] if surface_mask.ndim == 4 else surface_mask
    iz, iy, ix = np.where(m > 0.5)
    xs = bmin[0] + (ix + 0.5) * vox
    ys = bmin[1] + (iy + 0.5) * vox
    zs = bmin[2] + (iz + 0.5) * vox
    return np.column_stack([xs, ys, zs])

def load_real_scene(meta):
    """Load original ScanNet npy and crop around the removal centre."""
    sid  = meta["scene_id"]
    ctr  = np.array(meta["remove_center"], dtype=np.float32)
    half = np.array(meta["crop_half_extent"], dtype=np.float32)
    xyz_all = np.load(PROC / sid / "xyz.npy").astype(np.float32)
    rgb_raw = np.load(PROC / sid / "rgb.npy")
    rgb_all = (rgb_raw * 255 if rgb_raw.max() <= 1.0 else rgb_raw).astype(np.uint8)
    m = np.all((xyz_all >= ctr - half) & (xyz_all <= ctr + half), axis=1)
    return xyz_all[m], rgb_all[m]

def scatter2d(ax, pts, col, ah, av, ms=1.2, alpha=0.7, zorder=2):
    if pts.shape[0] == 0: return
    ax.scatter(pts[:, ah], pts[:, av], s=ms, c=col,
               alpha=alpha, linewidths=0, rasterized=True, zorder=zorder)

def style_ax(ax, xl, yl, lims):
    ax.set_facecolor("#f8f8f8")
    ax.set_xlabel(xl, fontsize=7.5, color="#555")
    ax.set_ylabel(yl, fontsize=7.5, color="#555")
    ax.tick_params(labelsize=6.5, color="#bbb", labelcolor="#666")
    ax.grid(True, color="#e8e8e8", lw=0.5, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"): ax.spines[sp].set_color("#ccc")
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_aspect("equal")

# ── Figure ────────────────────────────────────────────────────────────────────
COL_TITLES = [
    "Input: original scene\n(real ScanNet RGB)",
    "Network input\n(orange = void to reconstruct)",
    "Ground truth target\n(red = hidden surface to predict)",
]
EDGE_COL = ["#333333", "#B84F00", "#B71C1C"]

fig, axes = plt.subplots(3, 3, figsize=(11, 9.5), facecolor="white",
                         gridspec_kw=dict(hspace=0.52, wspace=0.28))
fig.subplots_adjust(top=0.87, bottom=0.11, left=0.09, right=0.98)

for c, t in enumerate(COL_TITLES):
    axes[0, c].set_title(t, fontsize=9, fontweight="bold", color="#222", pad=5)

for row, st in enumerate(ORDER):
    d, meta = WANT[st]
    tgt  = np.load(d / "target_surface.npz")
    surf_xyz = target_surface_xyz(meta, tgt["surface_mask"])

    xyz, rgb = load_real_scene(meta)
    rgb_f = rgb.astype(np.float32) / 255.0

    ah, av, xl, yl = VIEW_PROJ[st]
    ctr  = np.array(meta["remove_center"])
    half = np.array(meta["crop_half_extent"])
    pad  = 0.12
    lims = ((ctr[ah] - half[ah] - pad, ctr[ah] + half[ah] + pad),
            (ctr[av] - half[av] - pad, ctr[av] + half[av] + pad))

    fill = meta["fill_region"]
    fmin = np.array(fill["bounds_min"]); fmax = np.array(fill["bounds_max"])

    # -- Col 0: original scene (all points, real colours) ----------------------
    ax = axes[row, 0]
    scatter2d(ax, xyz, rgb_f, ah, av, ms=1.5, alpha=0.75)
    style_ax(ax, xl, yl, lims)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE_COL[0]); sp.set_linewidth(1.3)

    # -- Col 1: scene after removal (background only) + empty void box ----------
    ax = axes[row, 1]
    outside_void = ~np.all((xyz >= fmin) & (xyz <= fmax), axis=1)
    scatter2d(ax, xyz[outside_void], rgb_f[outside_void], ah, av, ms=1.5, alpha=0.75)
    # Empty orange box — represents the void left after object removal
    rx = fmin[ah]; ry = fmin[av]
    rw = fmax[ah] - fmin[ah]; rh = fmax[av] - fmin[av]
    ax.add_patch(plt.Rectangle((rx, ry), rw, rh,
                 facecolor="#E65100", alpha=0.15,
                 edgecolor="#E65100", linewidth=2.0, linestyle="-", zorder=4))
    style_ax(ax, xl, yl, lims)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE_COL[1]); sp.set_linewidth(1.3)

    # -- Col 2: original scene with hidden surface points highlighted red --------
    ax = axes[row, 2]
    in_fill = np.all((xyz >= fmin) & (xyz <= fmax), axis=1)
    # background points in natural colour
    scatter2d(ax, xyz[~in_fill], rgb_f[~in_fill], ah, av, ms=1.5, alpha=0.75)
    # hidden surface points (inside void region in original scan) in red
    if in_fill.sum() > 0:
        scatter2d(ax, xyz[in_fill], "#C62828", ah, av, ms=3, alpha=0.9, zorder=5)
    style_ax(ax, xl, yl, lims)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE_COL[2]); sp.set_linewidth(1.3)

    # Row label
    pos = axes[row, 0].get_position()
    fig.text(0.01, pos.y0 + pos.height / 2,
             RLABEL[row], ha="left", va="center",
             fontsize=9, fontweight="bold", color="#111", rotation=90)


# ── Title & subtitle ──────────────────────────────────────────────────────────
fig.text(0.54, 0.96,
         "Structural Surface Training Dataset — ScanNet Scenes with Synthetically Removed Objects",
         ha="center", va="top", fontsize=11, fontweight="bold", color="#111")
fig.text(0.54, 0.925,
         "Real scan point colours from ScanNet.  Each crop spans ≈ 1.6 m × 1.8 m around the removed object.  "
         "205 scenes, 615 training crops.",
         ha="center", va="top", fontsize=8, color="#555")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor="#8FA8C8", edgecolor="#333",
                   label="Scene geometry — real ScanNet RGB point cloud"),
    mpatches.Patch(facecolor="#E65100", alpha=0.85,
                   label="Void region — object removed, surface hidden (network input)"),
    mpatches.Patch(facecolor="#C62828", alpha=0.85,
                   label="Target surface — hidden geometry the network must predict"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
           framealpha=0.97, edgecolor="#ccc",
           bbox_to_anchor=(0.54, 0.00))

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", facecolor="white", bbox_inches="tight")
    print(f"Saved {OUT}.{ext}")
plt.close()
