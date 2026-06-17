"""
Generate per-stage thumbnail images for job180 (chair to right of desk, scene0039_00).
Outputs one square PNG per pipeline stage into figs/pipeline_thumbs/.
"""
from __future__ import annotations
import json, struct, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCENE_DIR = Path("/rds/general/user/qp23/home/Hypo3D/repo/scannet/processed/scene0039_00")
FIGS      = Path("/rds/general/user/qp23/home/Hypo3D/figs/pipeline_thumbs")
FIGS.mkdir(parents=True, exist_ok=True)

JOB = json.loads((SCENE_DIR / "edits/job180.json").read_text())

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

# ── Load arrays ────────────────────────────────────────────────────────────────
xyz_orig = np.load(SCENE_DIR / "xyz.npy").astype(np.float32)
rgb_orig = np.load(SCENE_DIR / "rgb.npy").astype(np.uint8)
inst     = np.load(SCENE_DIR / "inst.npy").astype(np.int32)

CHAIR_ID = int(JOB["target_instance_id"])   # 7
DESK_ID  = int(JOB["anchor_instance_id"])   # 11

orig_ply, orig_rgb_ply = read_ply(SCENE_DIR / "original_scene0039_00.ply")
bkgd_xyz, bkgd_rgb    = read_ply(SCENE_DIR / "edits/job180_chair_move_bkgd.ply")
final_xyz, final_rgb   = read_ply(SCENE_DIR / "edits/job180_chair_to_right_of_desk_v2.ply")

# ── Scene bounds ───────────────────────────────────────────────────────────────
xmin, xmax = float(xyz_orig[:,0].min()), float(xyz_orig[:,0].max())
ymin, ymax = float(xyz_orig[:,1].min()), float(xyz_orig[:,1].max())

SZ = 4.0   # thumbnail size inches
DPI = 150
PT = 0.6

def make_fig():
    fig, ax = plt.subplots(figsize=(SZ, SZ), facecolor='white')
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    return fig, ax

def scatter(ax, xyz, rgb, pt=PT, alpha=0.85):
    idx = np.argsort(xyz[:,2])
    ax.scatter(xyz[idx,0], xyz[idx,1], s=pt,
               c=rgb[idx]/255.0, linewidths=0, alpha=alpha,
               zorder=2, rasterized=True)

def save(fig, name):
    out = FIGS / f"{name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {out}")

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Instance Lookup: scene with chair(blue) + desk(amber) highlighted
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = make_fig()
# Grey for everything else
others = ~((inst == CHAIR_ID) | (inst == DESK_ID))
xyz_o  = xyz_orig[others]; rgb_o = np.full((others.sum(),3),200,dtype=np.uint8)
scatter(ax, xyz_o, rgb_o, alpha=0.5)

# Desk — amber
desk_xyz = xyz_orig[inst == DESK_ID]
desk_rgb = np.tile(np.array([[230,140,30]],dtype=np.uint8),(len(desk_xyz),1))
scatter(ax, desk_xyz, desk_rgb, pt=1.5)

# Chair — blue
chair_xyz = xyz_orig[inst == CHAIR_ID]
chair_rgb = np.tile(np.array([[30,100,220]],dtype=np.uint8),(len(chair_xyz),1))
scatter(ax, chair_xyz, chair_rgb, pt=1.5)

# Bounding boxes
def bbox_rect(xyz, color, label, above=True):
    x0,x1 = xyz[:,0].min(), xyz[:,0].max()
    y0,y1 = xyz[:,1].min(), xyz[:,1].max()
    rect = mpatches.Rectangle((x0,y0), x1-x0, y1-y0,
        linewidth=1.5, edgecolor=color, facecolor='none',
        linestyle='--', zorder=5)
    ax.add_patch(rect)
    ty = y1+0.04 if above else y0-0.1
    ax.text((x0+x1)/2, ty, label, ha='center', va='bottom',
            fontsize=8, color=color, fontweight='bold', zorder=6)

bbox_rect(chair_xyz, '#1E64DC', 'chair', above=True)
bbox_rect(desk_xyz,  '#E68C1E', 'desk',  above=False)
save(fig, 'stage2_instance_lookup')

# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Object Removal: background with void shown
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = make_fig()
scatter(ax, bkgd_xyz, bkgd_rgb)

# Void rectangle
cx = float(xyz_orig[inst==CHAIR_ID, 0].mean())
cy_v = float(xyz_orig[inst==CHAIR_ID, 1].mean())
hx = (xyz_orig[inst==CHAIR_ID,0].max() - xyz_orig[inst==CHAIR_ID,0].min())/2 + 0.04
hy = (xyz_orig[inst==CHAIR_ID,1].max() - xyz_orig[inst==CHAIR_ID,1].min())/2 + 0.04
void_rect = mpatches.Rectangle((cx-hx, cy_v-hy), 2*hx, 2*hy,
    linewidth=1.8, edgecolor='#CC2222', facecolor='#FFE5E5',
    linestyle='--', alpha=0.7, zorder=5)
ax.add_patch(void_rect)
ax.text(cx, cy_v, 'void', ha='center', va='center',
        fontsize=9, color='#CC2222', fontweight='bold', zorder=6)
save(fig, 'stage3_object_removal')

# ══════════════════════════════════════════════════════════════════════════════
# Panel 4 — Background Reconstruction: original RGB, floor filled
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = make_fig()
scatter(ax, bkgd_xyz, bkgd_rgb, alpha=0.7)

# Highlight fill region
fill_xyz = xyz_orig[inst==CHAIR_ID]
fill_col  = np.tile(np.array([[60,200,120]],dtype=np.uint8),(len(fill_xyz),1))
ax.scatter(fill_xyz[:,0], fill_xyz[:,1], s=2.0,
           c=fill_col/255.0, linewidths=0, alpha=0.9, zorder=4)

ax.annotate('filled', xy=(cx, cy_v), xytext=(cx+0.45, cy_v+0.35),
            fontsize=8, color='#228855', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#228855', lw=1.2), zorder=7)
save(fig, 'stage4_reconstruction')

# ══════════════════════════════════════════════════════════════════════════════
# Panel 5a — Spatial Relation Resolver: desk + 8 directions + RIGHT_OF
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = make_fig()
scatter(ax, bkgd_xyz, bkgd_rgb, alpha=0.25)   # faint scene

# Desk footprint (filled amber)
dx0 = desk_xyz[:,0].min(); dx1 = desk_xyz[:,0].max()
dy0 = desk_xyz[:,1].min(); dy1 = desk_xyz[:,1].max()
desk_rect = mpatches.Rectangle((dx0,dy0), dx1-dx0, dy1-dy0,
    linewidth=1.5, edgecolor='#E68C1E', facecolor='#FFF3DC',
    alpha=0.85, zorder=4)
ax.add_patch(desk_rect)
dcx, dcy = (dx0+dx1)/2, (dy0+dy1)/2

# 8 direction arrows
L = 0.55
dirs = [(1,0,'RIGHT_OF'), (-1,0,''), (0,1,''), (0,-1,''),
        (0.707,0.707,''), (-0.707,0.707,''), (0.707,-0.707,''), (-0.707,-0.707,'')]
for dx, dy, lbl in dirs:
    is_right = lbl == 'RIGHT_OF'
    col = '#1E64DC' if is_right else '#AAAAAA'
    lw  = 2.2 if is_right else 0.9
    ax.annotate('', xy=(dcx+dx*L, dcy+dy*L), xytext=(dcx, dcy),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=lw,
                                mutation_scale=10), zorder=6)

# New position dot
tx = JOB['translation']
new_cx = cx + tx[0]; new_cy = cy_v + tx[1]
ax.plot(new_cx, new_cy, 'o', color='#1E64DC', markersize=7, zorder=8)
ax.text(new_cx + 0.08, new_cy + 0.12, 'new\nposition',
        fontsize=7.5, color='#1E64DC', fontweight='bold', zorder=9)

# Old position dashed circle
circ = plt.Circle((cx, cy_v), max(hx,hy)*0.85,
    fill=False, edgecolor='#888888', linestyle='--', linewidth=1.2, zorder=5)
ax.add_patch(circ)

ax.text(dcx, dy0-0.12, 'desk', ha='center', va='top',
        fontsize=8, color='#E68C1E', fontweight='bold', zorder=7)
save(fig, 'stage5a_spatial_resolver')

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Original scene (for reference / Language Parser card)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = make_fig()
scatter(ax, orig_ply, orig_rgb_ply)
save(fig, 'stage1_original_scene')

# ══════════════════════════════════════════════════════════════════════════════
# Panel 5b — Final edited scene: chair in new position
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = make_fig()
scatter(ax, final_xyz, final_rgb)

# Arrow from old to new chair position
ax.annotate('', xy=(new_cx, new_cy), xytext=(cx, cy_v),
            arrowprops=dict(arrowstyle='-|>', color='#1E64DC', lw=1.8,
                            mutation_scale=12, linestyle='dashed'), zorder=7)
ax.plot(cx, cy_v, 'x', color='#888888', markersize=7, markeredgewidth=1.5, zorder=7)
ax.plot(new_cx, new_cy, 'o', color='#1E64DC', markersize=5,
        markerfacecolor='none', markeredgewidth=1.5, zorder=7)
save(fig, 'stage5b_final_scene')

print("\nAll thumbnails saved to", FIGS)
