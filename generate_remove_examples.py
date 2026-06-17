"""
Multi-example REMOVE operation visualisation.
3 examples × 5 panels (oblique object view + top-down before/after + elevation before/after).
Output: figs/fig_remove_examples.pdf / .png
"""
from __future__ import annotations
import json, struct, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

SCENE_DIR = Path("/rds/general/user/qp23/home/Hypo3D/repo/scannet/processed/scene0000_00")
FIGS      = Path("/rds/general/user/qp23/home/Hypo3D/figs")

PLY_TO_STRUCT = {
    'char':'b','int8':'b','uchar':'B','uint8':'B',
    'short':'h','int16':'h','ushort':'H','uint16':'H',
    'int':'i','int32':'i','uint':'I','uint32':'I',
    'float':'f','float32':'f','double':'d','float64':'d',
}

def read_ply(path):
    data = path.read_bytes()
    hdr_end = data.find(b'end_header')
    nl = data[hdr_end + len('end_header')]
    body = hdr_end + len('end_header') + (2 if nl == ord('\r') else 1)
    header = data[:body].decode('ascii', errors='ignore')
    n = int(re.search(r'element vertex (\d+)', header).group(1))
    props, in_v = [], False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith('element vertex'):   in_v = True
        elif line.startswith('element'):        in_v = False
        elif in_v and line.startswith('property') and 'list' not in line:
            p = line.split()
            if len(p) >= 3: props.append((p[1], p[2]))
    is_bin = 'binary_little_endian' in header
    if is_bin:
        fmt = '<' + ''.join(PLY_TO_STRUCT.get(t,'f') for t,_ in props)
        sz  = struct.calcsize(fmt)
        rows = [struct.unpack_from(fmt, data, body + i*sz) for i in range(n)]
        arr  = np.array(rows, dtype=np.float64)
    else:
        lines = data[body:].decode('ascii', errors='ignore').splitlines()
        arr   = np.array([list(map(float, l.split())) for l in lines[:n] if l.strip()])
    n2c = {nm: i for i,(_, nm) in enumerate(props)}
    xyz = arr[:, [n2c['x'], n2c['y'], n2c['z']]].astype(np.float32)
    rgb = np.zeros((len(xyz),3), dtype=np.uint8)
    for ch, col in enumerate(('red','green','blue')):
        if col in n2c:
            rgb[:,ch] = np.clip(arr[:,n2c[col]],0,255).astype(np.uint8)
    return xyz, rgb

# ── Load scene arrays (used for instance lookup) ───────────────────────────────
print("Loading scene arrays...")
xyz_scene = np.load(SCENE_DIR / "xyz.npy").astype(np.float32)
rgb_scene = (np.load(SCENE_DIR / "rgb.npy") * 255).astype(np.uint8)
inst      = np.load(SCENE_DIR / "inst.npy").astype(np.int32)

# ── Examples (avoid job014/bed, used elsewhere in dissertation) ────────────────
EXAMPLES = [
    dict(job="job015", label="Cabinet", ply="job015_remove_cabinet.ply", accent="#C62828",
         description='Cabinet (job015) — target highlighted in red (left); '
                     'exposed void region after removal (right).'),
    dict(job="job017", label="Desk", ply="job017_remove_desk.ply", accent="#1565C0",
         description='Desk (job017) — target highlighted in blue (left); '
                     'exposed void region after removal (right).'),
    dict(job="job016", label="Nightstand", ply="job016_remove_nightstand.ply", accent="#2E7D32",
         description='Nightstand (job016) — target highlighted in green (left); '
                     'exposed void region after removal (right).'),
]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
})
PT = 1.0
PAD = 0.85   # metres of context around each object

# ── Figure: 3 rows × 5 columns ────────────────────────────────────────────────
# Col 0: oblique object view (ScanNet RGB, narrow)
# Col 1-2: top-down (X–Y) before, after
# Col 3-4: elevation (X–Z) before, after
NROWS = len(EXAMPLES)

fig = plt.figure(figsize=(17, 4.0 * NROWS), facecolor="white")

outer_gs = gridspec.GridSpec(NROWS, 1, figure=fig,
                              hspace=0.45,
                              left=0.05, right=0.97, top=0.97, bottom=0.09)

# Column headers (positions tuned for width_ratios=[0.75, 1, 1, 1, 1])
fig.text(0.40, 0.988, "Top-down view (X–Y plane)",
         ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333333")
fig.text(0.77, 0.988, "Elevation view (X–Z plane)",
         ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333333")


def scatter2d(ax, x, y, depth, rgb, pt=PT, alpha=0.85):
    """Scatter with painter's sort along `depth` axis."""
    idx = np.argsort(depth)
    ax.scatter(x[idx], y[idx], s=pt, c=rgb[idx]/255.0,
               linewidths=0, alpha=alpha, rasterized=True, zorder=2)


def render_object_oblique(ax, obj_xyz, obj_rgb, accent):
    """Oblique 3D projection of an isolated object point cloud."""
    if len(obj_xyz) == 0:
        ax.set_facecolor('#F5F5F5')
        ax.set_xticks([]); ax.set_yticks([])
        return

    pts = obj_xyz - obj_xyz.mean(axis=0)

    # Rotate: azimuth 210° around Z-up, then 25° elevation tilt
    az = np.radians(210)
    el = np.radians(25)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0, 0, 1]], dtype=np.float32)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el),  np.cos(el)]], dtype=np.float32)
    pts_r = ((Rx @ Rz) @ pts.T).T  # Nx3

    # Back-to-front painter's sort on projected depth
    idx = np.argsort(pts_r[:, 2])
    ax.scatter(pts_r[idx, 0], pts_r[idx, 1],
               s=10, c=obj_rgb[idx] / 255.0,
               linewidths=0, alpha=0.95, rasterized=True)

    span = max(pts_r[:, 0].ptp(), pts_r[:, 1].ptp())
    pad  = span * 0.12
    ax.set_xlim(pts_r[:, 0].min() - pad, pts_r[:, 0].max() + pad)
    ax.set_ylim(pts_r[:, 1].min() - pad, pts_r[:, 1].max() + pad)
    ax.set_aspect('equal')
    ax.set_facecolor('#F5F5F5')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(accent)
        sp.set_linewidth(1.8)


def void_rect(ax, cx, cy, hw, hh, color):
    r = mpatches.Rectangle((cx-hw, cy-hh), 2*hw, 2*hh,
        linewidth=1.6, edgecolor=color, facecolor=color, alpha=0.10,
        linestyle="--", zorder=5)
    ax.add_patch(r)
    ax.text(cx, cy, "void", ha="center", va="center", fontsize=8.5,
            color=color, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=color,
                      boxstyle="round,pad=0.25", linewidth=0.9), zorder=6)


def aabb_outline(ax, cx, cy, hw, hh, color):
    r = mpatches.Rectangle((cx-hw, cy-hh), 2*hw, 2*hh,
        linewidth=1.6, edgecolor=color, facecolor="none",
        linestyle="--", zorder=5)
    ax.add_patch(r)


for row, ex in enumerate(EXAMPLES):
    print(f"Processing {ex['label']}...")
    job = json.loads((SCENE_DIR / "edits" / f"{ex['job']}.json").read_text())
    after_xyz, after_rgb = read_ply(SCENE_DIR / "edits" / ex["ply"])

    tid   = int(job["target_instance_id"])
    omask = inst == tid
    obj_xyz = xyz_scene[omask]
    obj_rgb = rgb_scene[omask]

    c  = np.array(job["remove_center"], dtype=np.float32)
    sz = np.array(job["remove_size"],   dtype=np.float32)

    # Crop masks
    xlo = c[0]-sz[0]/2-PAD;  xhi = c[0]+sz[0]/2+PAD
    ylo = c[1]-sz[1]/2-PAD;  yhi = c[1]+sz[1]/2+PAD
    zlo = max(c[2]-sz[2]/2-PAD*0.6, xyz_scene[:,2].min())
    zhi = c[2]+sz[2]/2+PAD*0.6

    def cm_xy(xyz):
        return ((xyz[:,0]>=xlo)&(xyz[:,0]<=xhi)&
                (xyz[:,1]>=ylo)&(xyz[:,1]<=yhi))
    def cm_xz(xyz):
        return ((xyz[:,0]>=xlo)&(xyz[:,0]<=xhi)&
                (xyz[:,2]>=zlo)&(xyz[:,2]<=zhi))

    bef_xy_m  = cm_xy(xyz_scene)
    bef_xz_m  = cm_xz(xyz_scene)
    aft_xy_m  = cm_xy(after_xyz)
    aft_xz_m  = cm_xz(after_xyz)
    obj_xy_m  = cm_xy(obj_xyz)
    obj_xz_m  = cm_xz(obj_xyz)

    accent_uint = np.array([int(ex["accent"][1:3],16),
                             int(ex["accent"][3:5],16),
                             int(ex["accent"][5:7],16)], dtype=np.uint8)

    # ── 5-column inner gridspec ────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer_gs[row],
        width_ratios=[0.75, 1, 1, 1, 1],
        wspace=0.06)
    ax_photo  = fig.add_subplot(inner[0])
    ax_td_bef = fig.add_subplot(inner[1])
    ax_td_aft = fig.add_subplot(inner[2])
    ax_el_bef = fig.add_subplot(inner[3])
    ax_el_aft = fig.add_subplot(inner[4])

    for ax in (ax_td_bef, ax_td_aft, ax_el_bef, ax_el_aft):
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_facecolor("white")

    # ── Oblique object view ────────────────────────────────────────────────────
    render_object_oblique(ax_photo, obj_xyz, obj_rgb, ex["accent"])
    if row == 0:
        ax_photo.set_title("Object", fontsize=9.5, fontweight="bold",
                           pad=4, color="#333333")

    # ── Top-down BEFORE ────────────────────────────────────────────────────────
    ax = ax_td_bef
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    scatter2d(ax, xyz_scene[bef_xy_m,0], xyz_scene[bef_xy_m,1],
              xyz_scene[bef_xy_m,2], rgb_scene[bef_xy_m])
    if obj_xy_m.sum() > 0:
        obj_v = obj_xyz[obj_xy_m]
        obj_c = np.tile(accent_uint, (len(obj_v),1))
        scatter2d(ax, obj_v[:,0], obj_v[:,1], obj_v[:,2], obj_c, pt=PT*3.5)
    aabb_outline(ax, c[0], c[1], sz[0]/2, sz[1]/2, ex["accent"])
    ax.text(c[0], c[1]-sz[1]/2-0.05, ex["label"],
            ha="center", va="top", fontsize=8, color=ex["accent"],
            fontweight="bold", zorder=6)
    if row == 0:
        ax.set_title("Before", fontsize=9.5, fontweight="bold",
                     pad=4, color="#333333")

    # ── Top-down AFTER ─────────────────────────────────────────────────────────
    ax = ax_td_aft
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    scatter2d(ax, after_xyz[aft_xy_m,0], after_xyz[aft_xy_m,1],
              after_xyz[aft_xy_m,2], after_rgb[aft_xy_m])
    void_rect(ax, c[0], c[1], sz[0]/2, sz[1]/2, ex["accent"])
    if row == 0:
        ax.set_title("After", fontsize=9.5, fontweight="bold",
                     pad=4, color="#333333")

    # ── Elevation BEFORE ───────────────────────────────────────────────────────
    ax = ax_el_bef
    ax.set_xlim(xlo, xhi); ax.set_ylim(zlo, zhi)
    scatter2d(ax, xyz_scene[bef_xz_m,0], xyz_scene[bef_xz_m,2],
              xyz_scene[bef_xz_m,1], rgb_scene[bef_xz_m])
    if obj_xz_m.sum() > 0:
        obj_v = obj_xyz[obj_xz_m]
        obj_c = np.tile(accent_uint, (len(obj_v),1))
        scatter2d(ax, obj_v[:,0], obj_v[:,2], obj_v[:,1], obj_c, pt=PT*3.5)
    aabb_outline(ax, c[0], c[2], sz[0]/2, sz[2]/2, ex["accent"])
    if row == 0:
        ax.set_title("Before", fontsize=9.5, fontweight="bold",
                     pad=4, color="#333333")

    # ── Elevation AFTER ────────────────────────────────────────────────────────
    ax = ax_el_aft
    ax.set_xlim(xlo, xhi); ax.set_ylim(zlo, zhi)
    scatter2d(ax, after_xyz[aft_xz_m,0], after_xyz[aft_xz_m,2],
              after_xyz[aft_xz_m,1], after_rgb[aft_xz_m])
    void_rect(ax, c[0], c[2], sz[0]/2, sz[2]/2, ex["accent"])
    if row == 0:
        ax.set_title("After", fontsize=9.5, fontweight="bold",
                     pad=4, color="#333333")

    # Row description below axes
    pos = ax_td_bef.get_position()
    desc_y = pos.y0 - 0.012
    accent_rgb_norm = tuple(v/255 for v in [
        int(ex["accent"][1:3],16), int(ex["accent"][3:5],16), int(ex["accent"][5:7],16)])
    fig.text(0.58, desc_y,
             f"{ex['description']}",
             ha="center", va="top", fontsize=7.8,
             color="#222222",
             bbox=dict(facecolor=(*accent_rgb_norm, 0.08),
                       edgecolor=(*accent_rgb_norm, 0.35),
                       boxstyle="round,pad=0.3", linewidth=0.8))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor="#888888", label="Scene geometry (original RGB)"),
    mpatches.Patch(facecolor="#C62828", alpha=0.9, label="Ex. 1 — Cabinet (target)"),
    mpatches.Patch(facecolor="#1565C0", alpha=0.9, label="Ex. 2 — Desk (target)"),
    mpatches.Patch(facecolor="#2E7D32", alpha=0.9, label="Ex. 3 — Nightstand (target)"),
    mpatches.Patch(facecolor="white", edgecolor="#666666",
                   linestyle="--", linewidth=1.3, label="AABB / void region"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=8.5, framealpha=0.95,
           bbox_to_anchor=(0.58, 0.0))

for ext in ("pdf", "png"):
    out = FIGS / f"fig_remove_examples.{ext}"
    plt.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"Saved {out}")
plt.close()
print("Done.")
