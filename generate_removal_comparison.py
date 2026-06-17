"""
Visualise three removal strategies for job017 (remove desk, scene0000_00).

Four panels — top-down (XZ) view, original RGB colours throughout:
  (a) Original scene
  (b) After instance removal  (pipeline default)
  (c) After AABB removal
  (d) After OBB removal

Bounding box outlines shown as dashed lines on (c) and (d).

Output: figs/fig_removal_comparison.pdf / .png
"""
from __future__ import annotations
import json, struct, re, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.patches as mpatches

HYPO  = Path("/rds/general/user/qp23/home/Hypo3D")
SCENE = HYPO / "repo/scannet/processed/scene0000_00"
FIGS  = HYPO / "figs"; FIGS.mkdir(exist_ok=True)

sys.path.insert(0, str(HYPO / "repo/scannet"))
from edit_job_utils import compute_obb, points_in_obb
from scipy.spatial import cKDTree

# ── PLY reader ─────────────────────────────────────────────────────────────────
PLY_TO_STRUCT = {
    'char':'b','int8':'b','uchar':'B','uint8':'B',
    'short':'h','int16':'h','ushort':'H','uint16':'H',
    'int':'i','int32':'i','uint':'I','uint32':'I',
    'float':'f','float32':'f','double':'d','float64':'d',
}

def read_ply(path: Path):
    data = path.read_bytes()
    hdr_end = data.find(b'end_header')
    nl = data[hdr_end + len('end_header')]
    body_start = hdr_end + len('end_header') + (2 if nl == ord('\r') else 1)
    header = data[:body_start].decode('ascii', errors='ignore')
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
        fmt = '<' + ''.join(PLY_TO_STRUCT.get(t, 'f') for t, _ in props)
        sz  = struct.calcsize(fmt)
        rows = [struct.unpack_from(fmt, data, body_start + i*sz) for i in range(n)]
        arr  = np.array(rows, dtype=np.float64)
    else:
        lines = data[body_start:].decode('ascii', errors='ignore').splitlines()
        arr   = np.array([list(map(float, l.split())) for l in lines[:n] if l.strip()])
    n2c = {nm: i for i, (_, nm) in enumerate(props)}
    xyz = arr[:, [n2c['x'], n2c['y'], n2c['z']]].astype(np.float32)
    rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    for ch, col in enumerate(('red', 'green', 'blue')):
        if col in n2c:
            rgb[:, ch] = np.clip(arr[:, n2c[col]], 0, 255).astype(np.uint8)
    return xyz, rgb

# ── Load ───────────────────────────────────────────────────────────────────────
job      = json.loads((SCENE / "edits/job017.json").read_text())
orig_xyz, orig_rgb = read_ply(SCENE / "original_scene0000_00.ply")
inst_xyz, inst_rgb = read_ply(SCENE / "edits/job017_remove_desk.ply")
aabb_xyz, aabb_rgb = read_ply(SCENE / "edits/job017_remove_aabb.ply")

center     = np.array(job["remove_center"], dtype=np.float32)
size       = np.array(job["remove_size"],   dtype=np.float32)
obb_center = np.array(job["remove_obb"]["center"],       dtype=np.float32)
obb_axes   = np.array(job["remove_obb"]["axes"],         dtype=np.float32)
obb_half   = np.array(job["remove_obb"]["half_extents"], dtype=np.float32)

# OBB removal — compute from desk points (instance result used to identify desk)
tree3d = cKDTree(inst_xyz)
dists,  _ = tree3d.query(orig_xyz, k=1)
desk_mask = dists > 0.02
desk_pts  = orig_xyz[desk_mask]
obb_comp  = compute_obb(desk_pts, padding=0.05)
keep_obb  = ~points_in_obb(orig_xyz, obb_comp)
obb_xyz   = orig_xyz[keep_obb]
obb_rgb   = orig_rgb[keep_obb]

print(f"Original : {len(orig_xyz):,} pts")
print(f"Instance : {len(inst_xyz):,} pts  (−{len(orig_xyz)-len(inst_xyz):,})")
print(f"AABB     : {len(aabb_xyz):,} pts  (−{len(orig_xyz)-len(aabb_xyz):,})")
print(f"OBB      : {len(obb_xyz):,} pts  (−{len(orig_xyz)-len(obb_xyz):,})")

# ── Crop window around the desk (top-down = XY plane, Z is up) ────────────────
PAD  = 0.9
xmin = center[0] - size[0]/2 - PAD;  xmax = center[0] + size[0]/2 + PAD
ymin = center[1] - size[1]/2 - PAD;  ymax = center[1] + size[1]/2 + PAD

def crop(xyz, rgb):
    m = ((xyz[:,0] >= xmin) & (xyz[:,0] <= xmax) &
         (xyz[:,1] >= ymin) & (xyz[:,1] <= ymax))
    # sort by Z (height) so higher points render on top
    idx = np.argsort(xyz[m, 2])
    return xyz[m][idx], rgb[m][idx]

orig_c, orig_c_rgb = crop(orig_xyz, orig_rgb)
inst_c, inst_c_rgb = crop(inst_xyz, inst_rgb)
aabb_c, aabb_c_rgb = crop(aabb_xyz, aabb_rgb)
obb_c,  obb_c_rgb  = crop(obb_xyz,  obb_rgb)

# ── Bounding box corners projected onto XY (top-down) ─────────────────────────
def aabb_corners_xy(c, s):
    hx, hy = s[0]/2, s[1]/2
    return np.array([[c[0]-hx, c[1]-hy],[c[0]+hx, c[1]-hy],
                     [c[0]+hx, c[1]+hy],[c[0]-hx, c[1]+hy]])

def obb_corners_xy(c, axes, half):
    # Project all 8 3D corners onto XY, take convex hull ordering
    from itertools import product
    corners_3d = [c + sx*axes[:,0]*half[0]
                    + sy*axes[:,1]*half[1]
                    + sz*axes[:,2]*half[2]
                  for sx, sy, sz in product((-1,1),(-1,1),(-1,1))]
    pts = np.array([[p[0], p[1]] for p in corners_3d])
    # sort by angle from centroid for polygon
    cen = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1]-cen[1], pts[:,0]-cen[0])
    return pts[np.argsort(angles)]

aabb_box = aabb_corners_xy(center, size)
obb_box  = obb_corners_xy(obb_center, obb_axes, obb_half)

# ── Figure ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

C_AABB = "#C62828"
C_OBB  = "#1565C0"
PT     = 1.8

fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), facecolor="white")

panels = [
    (orig_c, orig_c_rgb, "(a)  Original scene",              "#333333", None,     None),
    (inst_c, inst_c_rgb, "(b)  Instance removal\n(pipeline default)", "#2E7D32", None, None),
    (aabb_c, aabb_c_rgb, "(c)  AABB removal",                C_AABB,    aabb_box, C_AABB),
    (obb_c,  obb_c_rgb,  "(d)  OBB removal",                 C_OBB,     obb_box,  C_OBB),
]

for ax, (xyz, rgb, title, tcol, box, bcol) in zip(axes, panels):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=10, fontweight="bold", color=tcol, pad=6)

    # scatter with original RGB colours (X=horizontal, Y=depth, top-down XY view)
    ax.scatter(xyz[:,0], xyz[:,1], s=PT,
               c=rgb/255.0, linewidths=0, alpha=0.85,
               zorder=2, rasterized=True)

    # bounding box outline
    if box is not None:
        ax.add_patch(MplPolygon(box, closed=True, fill=False,
                                edgecolor=bcol, linewidth=2.0,
                                linestyle="--", zorder=5))

    # point count
    n_removed = len(orig_c) - len(xyz)
    if n_removed > 0:
        ax.text(0.5, 0.03, f"−{n_removed:,} pts removed",
                transform=ax.transAxes, ha="center",
                fontsize=8.5, color=tcol, fontweight="bold")

# ── Suptitle + legend ──────────────────────────────────────────────────────────
fig.suptitle(
    "Object removal comparison — scene0000_00, job017 (desk)\n"
    "Top-down view (XY plane)  ·  original point-cloud colours  ·"
    "  dashed outline = bounding box used for removal",
    fontsize=9.5, fontweight="bold", y=1.03)

handles = [
    mpatches.Patch(edgecolor=C_AABB, facecolor="none",
                   linestyle="--", linewidth=1.5, label="AABB outline"),
    mpatches.Patch(edgecolor=C_OBB,  facecolor="none",
                   linestyle="--", linewidth=1.5, label="OBB outline"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
for ext in ("pdf", "png"):
    out = FIGS / f"fig_removal_comparison.{ext}"
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
plt.close()
