#!/usr/bin/env python3
"""
Filter convonet_completed_points.ply by proximity to structural fill reference.

Keeps only ConvONet points within --threshold metres of any nn_floor_fill
or structural_surface point. Discards everything else as floating artifacts.

Clears downstream stale outputs so postprocessing regenerates them.

Usage:
  python filter_by_structural_fill.py --job 0 [--threshold 0.20]
  python filter_by_structural_fill.py --all  [--threshold 0.20]
"""
import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

COMP_BASE = Path('/rds/general/user/qp23/home/Hypo3D/repo/scannet/processed/scene0000_00/completion')


def load_ply_xyz(path: Path) -> np.ndarray:
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
    return arr[:, [n2c['x'], n2c['y'], n2c['z']]].astype(np.float32)


def filter_job(job_index: int, threshold: float, no_proximity_filter: bool = False):
    job_id = f'scene0000_00_job00{job_index}_bkgd'
    comp_dir = COMP_BASE / job_id

    conv_ply = comp_dir / 'convonet_completed_points.ply'
    if not conv_ply.exists():
        print(f'[{job_id}] convonet_completed_points.ply missing — skipping')
        return

    conv_xyz = load_ply_xyz(conv_ply)
    n_total = len(conv_xyz)

    # Hard clip to fill_region box — never allow ConvONet outside the room walls
    valid = np.ones(n_total, dtype=bool)
    fill_region_path = comp_dir / 'fill_region.json'
    if fill_region_path.exists():
        import json
        fr = json.loads(fill_region_path.read_text())
        fr_center = np.array(fr['center'], dtype=np.float32)
        fr_half   = 0.5 * np.array(fr['size'], dtype=np.float32)
        in_box = np.all((conv_xyz >= fr_center - fr_half) &
                        (conv_xyz <= fr_center + fr_half), axis=1)
        clipped = int((~in_box).sum())
        valid &= in_box
        print(f'  fill_region clip: removed {clipped} pts outside box')

    if no_proximity_filter:
        keep = valid.copy()
        print(f'[{job_id}] fill_region clip only: {n_total} → {keep.sum()} pts')
    else:
        # Build reference point cloud from structural fill + existing scan boundary
        ref_parts = []
        for fname in ('nn_floor_fill_completed_points.ply',
                      'structural_surface_completed_points.ply',
                      'crop_after_remove.ply'):
            p = comp_dir / fname
            if p.exists():
                pts = load_ply_xyz(p)
                if len(pts) > 0:
                    ref_parts.append(pts)
                    print(f'  reference {fname}: {len(pts)} pts')

        if not ref_parts:
            print(f'[{job_id}] no structural fill reference found — skipping filter')
            return

        ref_xyz = np.vstack(ref_parts)

        # KDTree proximity filter on the clipped subset
        kdtree = cKDTree(ref_xyz)
        dists, _ = kdtree.query(conv_xyz[valid], k=1, workers=-1)
        prox_keep = dists <= threshold

        # Combine: valid indices that also pass proximity filter
        valid_idx = np.where(valid)[0]
        keep = np.zeros(n_total, dtype=bool)
        keep[valid_idx[prox_keep]] = True

        before = n_total
        after = keep.sum()
        print(f'[{job_id}] proximity filter (threshold={threshold}m): '
              f'{before} → {after} pts (removed {before - after})')

    # Write filtered point cloud back
    pcd_in = o3d.io.read_point_cloud(str(conv_ply))
    pts_all = np.asarray(pcd_in.points)
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts_all[keep])
    if pcd_in.has_colors():
        pcd_out.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd_in.colors)[keep])
    o3d.io.write_point_cloud(str(conv_ply), pcd_out)

    # Clear stale merge + Poisson outputs so they regenerate from filtered points
    for f in ['completed_scene_merged.ply', 'poisson_completion_mesh.ply',
              'poisson_scene_mesh.ply', 'completed_scene_pointcloud.ply',
              'bkgd_completion_comparison.png', 'merge_stats.json']:
        p = comp_dir / f
        if p.exists():
            p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--threshold', type=float, default=0.20)
    ap.add_argument('--no-proximity-filter', action='store_true',
                    help='Skip proximity filter; apply fill_region box clip only')
    args = ap.parse_args()

    jobs = list(range(5)) if args.all else ([args.job] if args.job is not None else [])
    if not jobs:
        ap.print_help()
        sys.exit(1)

    for j in jobs:
        filter_job(j, args.threshold, no_proximity_filter=args.no_proximity_filter)

    print('\nDone — re-submit postprocessing PBS (steps 3+ will regenerate)')


if __name__ == '__main__':
    main()
