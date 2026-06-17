#!/usr/bin/env python3
"""
Combine completed bkgd scene with placed object to produce final move result.

Usage:
  python combine_move_result.py            # all 5 jobs
  python combine_move_result.py --job 0    # single job
  python combine_move_result.py --force    # overwrite existing
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HYPO = Path('/rds/general/user/qp23/home/Hypo3D')
COMP_BASE = HYPO / 'repo/scannet/processed/scene0000_00/completion'
EDITS = HYPO / 'repo/scannet/processed/scene0000_00/edits'

# Placed-object file for each job index
PLACED_FILES = {
    0: EDITS / 'job000_desk_to_right_of_refrigerator_pointsea_test2_placed.ply',
    1: EDITS / 'job001_bicycle_to_back_of_bed_placed.ply',
    2: EDITS / 'job002_backpack_to_next_to_laundry_basket_placed.ply',
    3: EDITS / 'job003_stool_to_next_to_couch_placed.ply',
    4: EDITS / 'job004_stool_to_back_of_backpack_placed.ply',
}


def read_ply(path: Path):
    import re, struct
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
    rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    for ch, col in enumerate(('red', 'green', 'blue')):
        if col in n2c:
            rgb[:, ch] = arr[:, n2c[col]].astype(np.uint8)
    return xyz, rgb


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    with path.open('w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(xyz)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(xyz, rgb):
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n')


def combine_job(job_index: int, force: bool):
    job_id = f'scene0000_00_job{job_index:03d}_bkgd'
    comp_dir = COMP_BASE / job_id
    out_path = comp_dir / 'completed_move_scene.ply'

    if out_path.exists() and not force:
        print(f'[job{job_index:03d}] already exists — skipping (use --force to overwrite)')
        return

    bkgd_ply = comp_dir / 'completed_scene_pointcloud.ply'
    if not bkgd_ply.exists():
        print(f'[job{job_index:03d}] completed_scene_pointcloud.ply missing — skipping')
        return

    placed_ply = PLACED_FILES.get(job_index)
    if placed_ply is None or not placed_ply.exists():
        print(f'[job{job_index:03d}] placed object file missing: {placed_ply}')
        return

    bkgd_xyz, bkgd_rgb = read_ply(bkgd_ply)
    placed_xyz, placed_rgb = read_ply(placed_ply)

    final_xyz = np.concatenate([bkgd_xyz, placed_xyz], axis=0)
    final_rgb = np.concatenate([bkgd_rgb, placed_rgb], axis=0)

    write_ply(out_path, final_xyz, final_rgb)
    print(f'[job{job_index:03d}] {len(bkgd_xyz)} bkgd + {len(placed_xyz)} placed = {len(final_xyz)} pts → {out_path.name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    jobs = [args.job] if args.job is not None else list(range(5))
    for j in jobs:
        combine_job(j, args.force)

    print('Done.')


if __name__ == '__main__':
    main()
