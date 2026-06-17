"""
Generate 7 pipeline-diagram thumbnails for scene0000_00 / job015 (cabinet removal).
Isometric projection, real point-cloud data at each stage.
Output: figs/pipeline_bkgd_thumbs/stage{N}_*.png
"""
from __future__ import annotations
import json, struct, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SCENE  = Path("/rds/general/user/qp23/home/Hypo3D/repo/scannet/processed/scene0000_00")
COMP   = SCENE / "completion/scene0000_00_job017"
COMPB  = SCENE / "completion/scene0000_00_job017_plane_fill_no_nnwall"
OUT    = Path("/rds/general/user/qp23/home/Hypo3D/figs/pipeline_bkgd_thumbs")
OUT.mkdir(parents=True, exist_ok=True)

JOB = json.loads((SCENE / "edits/job017.json").read_text())
C    = np.array(JOB["remove_center"], dtype=np.float32)  # [x,y,z]
BBOX = np.array(JOB["remove_size"],   dtype=np.float32)

PLY_STRUCT = {
    'char':'b','int8':'b','uchar':'B','uint8':'B',
    'short':'h','int16':'h','ushort':'H','uint16':'H',
    'int':'i','int32':'i','uint':'I','uint32':'I',
    'float':'f','float32':'f','double':'d','float64':'d',
}

def read_ply(path, stride=1):
    data = path.read_bytes()
    hdr_end = data.find(b'end_header')
    nl   = data[hdr_end + len('end_header')]
    body = hdr_end + len('end_header') + (2 if nl == ord('\r') else 1)
    hdr  = data[:body].decode('ascii', errors='ignore')
    n    = int(re.search(r'element vertex (\d+)', hdr).group(1))
    props, in_v = [], False
    for line in hdr.splitlines():
        line = line.strip()
        if line.startswith('element vertex'):   in_v = True
        elif line.startswith('element'):        in_v = False
        elif in_v and line.startswith('property') and 'list' not in line:
            p = line.split()
            if len(p) >= 3: props.append((p[1], p[2]))
    is_bin = 'binary_little_endian' in hdr
    if is_bin:
        fmt  = '<' + ''.join(PLY_STRUCT.get(t,'f') for t,_ in props)
        sz   = struct.calcsize(fmt)
        rows = [struct.unpack_from(fmt, data, body + i*sz)
                for i in range(0, n, stride)]
        arr  = np.array(rows, dtype=np.float64)
    else:
        lines = data[body:].decode('ascii', errors='ignore').splitlines()
        arr   = np.array([list(map(float, l.split()))
                          for l in lines[:n:stride] if l.strip()])
    n2c = {nm: i for i,(_, nm) in enumerate(props)}
    xyz = arr[:, [n2c['x'], n2c['y'], n2c['z']]].astype(np.float32)
    rgb = np.zeros((len(xyz),3), dtype=np.uint8)
    for ch, col in enumerate(('red','green','blue')):
        if col in n2c:
            rgb[:,ch] = np.clip(arr[:,n2c[col]],0,255).astype(np.uint8)
    return xyz, rgb

# ── Isometric projection ───────────────────────────────────────────────────────
AZ = np.radians(215)   # azimuth: looking from front-right
EL = np.radians(55)    # elevation: more top-down

def iso(xyz):
    """Return (x_screen, y_screen, depth) for painter's algorithm."""
    ca, sa = np.cos(AZ), np.sin(AZ)
    xr = xyz[:,0]*ca - xyz[:,1]*sa
    yr = xyz[:,0]*sa + xyz[:,1]*ca
    zr = xyz[:,2]
    xs = xr
    ys = zr * np.cos(EL) + yr * np.sin(EL)
    d  = -yr * np.cos(EL) + zr * np.sin(EL)
    return xs, ys, d

# ── Box corners projected to screen ───────────────────────────────────────────
def box_edges_iso(c, bbox):
    """Return list of (p1, p2) screen-coord pairs for 12 AABB edges."""
    hx, hy, hz = bbox[0]/2, bbox[1]/2, bbox[2]/2
    corners_3d = np.array([
        [c[0]+dx*hx, c[1]+dy*hy, c[2]+dz*hz]
        for dx in (-1,1) for dy in (-1,1) for dz in (-1,1)
    ], dtype=np.float32)
    xs, ys, _ = iso(corners_3d)
    # Indices for 12 edges
    edges = [
        (0,1),(2,3),(4,5),(6,7),   # z edges
        (0,2),(1,3),(4,6),(5,7),   # y edges
        (0,4),(1,5),(2,6),(3,7),   # x edges
    ]
    return [(xs[a],ys[a],xs[b],ys[b]) for a,b in edges]

# ── Load scene data ────────────────────────────────────────────────────────────
print("Loading base scene (npy)...")
_xyz_full  = np.load(SCENE/"xyz.npy").astype(np.float32)
_rgb_raw   = np.load(SCENE/"rgb.npy")
_rgb_full  = ((_rgb_raw * 255) if _rgb_raw.max() <= 1.0 else _rgb_raw).astype(np.uint8)
_inst_full = np.load(SCENE/"inst.npy").astype(np.int32)

# Desk instance: extract at full density for solid stage-1 coverage
cab_id = int(JOB["target_instance_id"])
_desk_m        = _inst_full == cab_id
desk_xyz_full  = _xyz_full[_desk_m]
desk_rgb_full  = _rgb_full[_desk_m]

# Subsample full scene for speed (every 4th point)
S = 4
xyz_s = _xyz_full[::S]
rgb_s = _rgb_full[::S]
inst  = _inst_full[::S]
cab_m   = inst == cab_id
others  = ~cab_m

# Load fill component PLYs
print("Loading fill PLYs...")
xyz_bkgd, rgb_bkgd   = read_ply(SCENE/"edits/job017_remove_desk.ply", stride=3)
xyz_plane, rgb_plane  = read_ply(COMPB/"nn_floor_fill_completed_points.ply")
xyz_struct, rgb_struct= read_ply(COMPB/"structural_surface_completed_points.ply")
xyz_conv, rgb_conv    = read_ply(COMPB/"convonet_completed_points.ply", stride=2)
xyz_merged, rgb_merged= read_ply(COMP/"occluded_surface_full_scene_merged.ply", stride=3)
xyz_done, rgb_done    = read_ply(COMPB/"completed_scene_merged.ply", stride=3)
print("All PLYs loaded.")

# Clip ConvONet to void AABB (avoid wild outliers far outside the cabinet region)
MARGIN = 0.25
conv_mask = ((xyz_conv[:,0] >= C[0]-BBOX[0]/2-MARGIN) &
             (xyz_conv[:,0] <= C[0]+BBOX[0]/2+MARGIN) &
             (xyz_conv[:,1] >= C[1]-BBOX[1]/2-MARGIN) &
             (xyz_conv[:,1] <= C[1]+BBOX[1]/2+MARGIN) &
             (xyz_conv[:,2] >= C[2]-BBOX[2]/2-MARGIN) &
             (xyz_conv[:,2] <= C[2]+BBOX[2]/2+MARGIN))
xyz_conv, rgb_conv = xyz_conv[conv_mask], rgb_conv[conv_mask]

# Clip completed scene to original scene XYZ bounding box (removes scan artefacts)
scene_xlo, scene_xhi = xyz_s[:,0].min()-0.1, xyz_s[:,0].max()+0.1
scene_ylo, scene_yhi = xyz_s[:,1].min()-0.1, xyz_s[:,1].max()+0.1
scene_zlo, scene_zhi = xyz_s[:,2].min()-0.1, xyz_s[:,2].max()+0.1
def clip_to_scene(xyz, rgb):
    m = ((xyz[:,0]>=scene_xlo)&(xyz[:,0]<=scene_xhi)&
         (xyz[:,1]>=scene_ylo)&(xyz[:,1]<=scene_yhi)&
         (xyz[:,2]>=scene_zlo)&(xyz[:,2]<=scene_zhi))
    return xyz[m], rgb[m]
xyz_merged, rgb_merged = clip_to_scene(xyz_merged, rgb_merged)
xyz_done,   rgb_done   = clip_to_scene(xyz_done,   rgb_done)

# ── Scene extents for axis limits (from original scene npy) ───────────────────
xs_all, ys_all, _ = iso(xyz_s)
PAD = 0.1
XL  = (xs_all.min()-PAD, xs_all.max()+PAD)
YL  = (ys_all.min()-PAD, ys_all.max()+PAD)

# ── Thumbnail helper ──────────────────────────────────────────────────────────
SZ_IN   = 2.8   # inches per thumbnail
DPI     = 120
PT_BASE = 0.4
BG_COL  = "#F8F5F0"   # warm off-white background (matches hand-drawn style)

def make_thumb(name, layers, void_box=False, rect_box=False, stride_extra=1, box_lw=2.2, box_ls="--"):
    """
    layers: list of (xyz, rgb_or_None, colour_override, pt_size, alpha)
    void_box: draw 3-D isometric AABB edges
    rect_box: draw a simple 2-D screen-space rectangle around the AABB projection
    """
    fig, ax = plt.subplots(figsize=(SZ_IN, SZ_IN), facecolor=BG_COL)
    ax.set_facecolor(BG_COL)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(*XL); ax.set_ylim(*YL)

    for xyz, rgb, col_ov, pt, alpha in layers:
        if len(xyz) == 0:
            continue
        xs, ys, depth = iso(xyz)
        idx = np.argsort(depth)          # painter: back → front
        xs, ys, depth = xs[idx], ys[idx], depth[idx]
        if col_ov is not None:
            c = np.tile(np.array(col_ov, dtype=np.uint8), (len(xs),1))
        else:
            c = rgb[idx]
        ax.scatter(xs[::stride_extra], ys[::stride_extra],
                   s=pt, c=c[::stride_extra]/255.0,
                   linewidths=0, alpha=alpha, rasterized=True, zorder=2)

    if void_box:
        for x0,y0,x1,y1 in box_edges_iso(C, BBOX):
            ax.plot([x0,x1],[y0,y1], color="#CC2222", lw=box_lw,
                    linestyle=box_ls, zorder=6, solid_capstyle="round")

    if rect_box:
        hx, hy, hz = BBOX[0]/2, BBOX[1]/2, BBOX[2]/2
        corners = np.array([[C[0]+dx*hx, C[1]+dy*hy, C[2]+dz*hz]
                             for dx in (-1,1) for dy in (-1,1) for dz in (-1,1)],
                            dtype=np.float32)
        xs_c, ys_c, _ = iso(corners)
        pad = 0.03
        rx, ry = xs_c.min()-pad, ys_c.min()-pad
        rw, rh = xs_c.max()-xs_c.min()+2*pad, ys_c.max()-ys_c.min()+2*pad
        ax.add_patch(mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=box_lw, edgecolor="#CC2222",
            facecolor="none", linestyle=box_ls, zorder=6))

    out = OUT / f"{name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight",
                facecolor=BG_COL, pad_inches=0.04)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── Load all voxel layers (uniform 1.5 cm spacing, used for all stages) ───────
THUMB_DIR = Path("/rds/general/user/qp23/home/Hypo3D/figs/pipeline_bkgd_thumbs")
xyz_bkgd_v, rgb_bkgd_v = clip_to_scene(*read_ply(THUMB_DIR/"bkgd_voxel.ply"))
xyz_desk_v, rgb_desk_v = read_ply(THUMB_DIR/"desk_voxel.ply")
xyz_pl_v,   rgb_pl_v   = read_ply(THUMB_DIR/"plane_voxel.ply")
xyz_st_v,   rgb_st_v   = read_ply(THUMB_DIR/"struct_voxel.ply")
xyz_cv_v,   rgb_cv_v   = clip_to_scene(*read_ply(THUMB_DIR/"conv_voxel.ply"))

PT_M = 2.2   # point size: 1.5cm-spaced dots overlap into a solid surface

GREEN  = [60,  200, 100]
TEAL   = [0,   160, 145]
PURPLE = [130,  60, 200]
RED    = [210,  50,  50]

# ── Stage 1: full scene — bkgd voxel + desk in natural npy colour + red box ──
print("Stage 1: input scene")
xyz_full_v, rgb_full_v = read_ply(THUMB_DIR/"full_scene_voxel.ply")
make_thumb("stage1_input_scene", [
    (xyz_full_v, rgb_full_v, None, PT_M, 0.90),
], rect_box=True, stride_extra=1, box_lw=1.0, box_ls="-")

# ── Stage 2: edited scene — desk removed, no annotation ──────────────────────
print("Stage 2: excision")
make_thumb("stage2_excision", [
    (xyz_bkgd_v, rgb_bkgd_v, None, PT_M, 0.90),
])

# ── Stage 3: plane fill — bkgd voxel (faded) + plane fill green ──────────────
print("Stage 3: plane fill")
make_thumb("stage3_plane_fill", [
    (xyz_bkgd_v, rgb_bkgd_v, None,  PT_M,   0.70),
    (xyz_pl_v,   None,       GREEN, PT_M,   0.95),
], void_box=False)

# ── Stage 4: structural surface — bkgd voxel (faded) + struct teal ────────────
print("Stage 4: structural surface")
make_thumb("stage4_struct_surface", [
    (xyz_bkgd_v, rgb_bkgd_v, None, PT_M,   0.70),
    (xyz_st_v,   None,       TEAL, PT_M,   0.95),
], void_box=False)

# ── Stage 5: ConvONet — bkgd voxel (faded) + conv purple ─────────────────────
print("Stage 5: ConvONet")
make_thumb("stage5_convonet", [
    (xyz_bkgd_v, rgb_bkgd_v, None,   PT_M, 0.70),
    (xyz_cv_v,   None,       PURPLE, PT_M, 0.95),
], void_box=False)

# ── Stage 6: merge — all voxel layers in natural RGB ─────────────────────────
print("Stage 6: merge")
make_thumb("stage6_merged", [
    (xyz_bkgd_v, rgb_bkgd_v, None, PT_M, 0.90),
    (xyz_pl_v,   rgb_pl_v,   None, PT_M, 0.95),
    (xyz_st_v,   rgb_st_v,   None, PT_M, 0.95),
    (xyz_cv_v,   rgb_cv_v,   None, PT_M, 0.95),
], stride_extra=1)


print(f"\nAll thumbnails saved to {OUT}")
