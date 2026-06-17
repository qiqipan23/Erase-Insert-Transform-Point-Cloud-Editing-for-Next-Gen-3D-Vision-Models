#!/usr/bin/env python3
"""
Rebuild bkgd pointcloud.npz using the FULL original scene scan (minus removed region)
plus structural fill, with 2-dim features for the retrain model.

Features: [0, 0] = scan point, [1, 0] = fill point
This matches the retrain model's pointcloud_feature_dim: 2 training setup.
"""
import json
import re
import struct
import numpy as np
from pathlib import Path

HYPO       = Path('/rds/general/user/qp23/home/Hypo3D')
PROC_BASE  = HYPO / 'repo/scannet/processed/scene0000_00'
COMP_BASE  = PROC_BASE / 'completion'
DATA_BASE  = HYPO / 'repo/scannet/completion/convonet_data'
SCENE_PLY  = PROC_BASE / 'original_scene0000_00.ply'


def load_ply_xyz_normals(path: Path):
    data = path.read_bytes()
    hdr_end_idx = data.find(b'end_header')
    nl = data[hdr_end_idx + len('end_header')]
    hdr_end = hdr_end_idx + len('end_header') + (2 if nl == ord('\r') else 1)
    header = data[:hdr_end].decode('ascii', errors='ignore')
    n = int(re.search(r'element vertex (\d+)', header).group(1))
    is_bin = 'binary_little_endian' in header

    props = []
    in_v = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith('element vertex'):   in_v = True
        elif line.startswith('element'):        in_v = False
        elif in_v and line.startswith('property') and 'list' not in line:
            parts = line.split()
            if len(parts) >= 3:
                props.append((parts[1], parts[2]))

    type_map = {'float': ('f', 4), 'float32': ('f', 4), 'double': ('d', 8),
                'uchar': ('B', 1), 'uint8': ('B', 1), 'int': ('i', 4)}
    if is_bin:
        fmt = '<' + ''.join(type_map.get(t, ('f', 4))[0] for t, _ in props)
        sz = struct.calcsize(fmt)
        body = data[hdr_end:]
        rows = [struct.unpack_from(fmt, body, i * sz) for i in range(n)]
        arr = np.array(rows, dtype=np.float64)
    else:
        lines = data[hdr_end:].decode('ascii', errors='ignore').splitlines()
        arr = np.array([list(map(float, l.split())) for l in lines[:n] if l.strip()],
                       dtype=np.float64)

    n2c = {name: i for i, (_, name) in enumerate(props)}
    xyz = arr[:, [n2c['x'], n2c['y'], n2c['z']]].astype(np.float32)
    normals = np.zeros((n, 3), np.float32)
    for i, k in enumerate(('nx', 'ny', 'nz')):
        if k in n2c:
            normals[:, i] = arr[:, n2c[k]].astype(np.float32)
    return xyz, normals


print(f'Loading full scene scan: {SCENE_PLY.name}')
scene_xyz, scene_nrm = load_ply_xyz_normals(SCENE_PLY)
print(f'  {len(scene_xyz):,} pts')

for j in range(5):
    job_id  = f'scene0000_00_job00{j}_bkgd'
    comp_dir = COMP_BASE / job_id
    data_dir = DATA_BASE / job_id / 'scannet_remove' / job_id
    out_npz  = data_dir / 'pointcloud.npz'

    meta   = json.loads((comp_dir / 'metadata.json').read_text())
    center = np.array(meta['remove_center'], dtype=np.float32)
    size   = np.array(meta['remove_size'],   dtype=np.float32)
    half   = size / 2.0 * 1.05          # tiny extra margin to erase the object cleanly

    # Remove the object's bounding box from the full scene scan
    mn, mx = center - half, center + half
    outside = ~np.all((scene_xyz >= mn) & (scene_xyz <= mx), axis=1)
    scan_xyz = scene_xyz[outside]
    scan_nrm = scene_nrm[outside]

    # Load both structural fill sources
    fill_parts_xyz, fill_parts_nrm = [], []
    for fname in ('nn_floor_fill_completed_points.ply',
                  'structural_surface_completed_points.ply'):
        p = comp_dir / fname
        if p.exists():
            fxyz, fnrm = load_ply_xyz_normals(p)
            if len(fxyz) > 0:
                fill_parts_xyz.append(fxyz)
                fill_parts_nrm.append(fnrm)
                print(f'  {fname}: {len(fxyz):,} pts')

    if fill_parts_xyz:
        fill_xyz = np.vstack(fill_parts_xyz)
        fill_nrm = np.vstack(fill_parts_nrm)
    else:
        fill_xyz = np.empty((0, 3), np.float32)
        fill_nrm = np.empty((0, 3), np.float32)

    # Combine: scan [0,0] + fill [1,0]  (feature_dim=2, matching retrain model)
    pts = np.vstack([scan_xyz, fill_xyz])
    nrm = np.vstack([scan_nrm, fill_nrm])
    feat = np.zeros((len(pts), 2), dtype=np.float32)
    feat[len(scan_xyz):, 0] = 1.0       # fill points get feature[0]=1

    np.savez(out_npz, points=pts, normals=nrm, features=feat)
    print(f'{job_id}: scan={len(scan_xyz):,} + fill={len(fill_xyz):,} = {len(pts):,} total → {out_npz.name}')

print('\nDone. Now update configs and re-run ConvONet inference.')
