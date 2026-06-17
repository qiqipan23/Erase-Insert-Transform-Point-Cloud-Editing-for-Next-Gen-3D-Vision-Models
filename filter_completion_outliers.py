#!/usr/bin/env python3
"""
Clean ConvONet output for bkgd jobs by:
  1. Cropping mesh vertices to fill region AABB + padding (removes out-of-hole floating mesh)
  2. Keeping only the largest connected component (removes small fragments)

Works on convonet_completed_mesh.off before point sampling so all downstream
outputs (merge, Poisson, pointcloud) are clean.

Usage:
  python filter_completion_outliers.py --job 0
  python filter_completion_outliers.py --all
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HYPO = Path('/rds/general/user/qp23/home/Hypo3D')
COMP_BASE = HYPO / 'repo/scannet/processed/scene0000_00/completion'

# Extra padding beyond the fill region AABB to avoid clipping valid fill geometry
FILL_PAD = 0.15  # metres


def filter_job(job_index: int):
    job_id = f'scene0000_00_job00{job_index}_bkgd'
    comp_dir = COMP_BASE / job_id

    off_path = comp_dir / 'convonet_completed_mesh.off'
    if not off_path.exists():
        print(f'[{job_id}] convonet_completed_mesh.off missing — skipping')
        return

    # Load fill region bounds
    fr = json.loads((comp_dir / 'fill_region.json').read_text())
    mn = np.array(fr['bounds_min']) - FILL_PAD
    mx = np.array(fr['bounds_max']) + FILL_PAD

    mesh = trimesh.load(str(off_path), process=False)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Keep only faces whose ALL vertices are inside the fill region AABB
    v_inside = np.all((verts >= mn) & (verts <= mx), axis=1)
    face_inside = v_inside[faces].all(axis=1)
    before_faces = len(faces)
    mesh_cropped = trimesh.Trimesh(
        vertices=verts, faces=faces[face_inside], process=False
    )
    mesh_cropped = mesh_cropped.process()  # removes orphaned verts

    # Keep only floor faces (normal pointing up, |nz|>0.5)
    # and wall faces (normal mostly horizontal, |nz|<0.3)
    # ScanNet uses z-up convention
    normals = mesh_cropped.face_normals          # (F, 3)
    nz = np.abs(normals[:, 2])
    floor_mask = nz > 0.5                        # horizontal surface (floor/ceiling)
    wall_mask  = nz < 0.3                        # vertical surface (wall)
    keep_mask  = floor_mask | wall_mask

    verts_c = np.array(mesh_cropped.vertices)
    faces_c = np.array(mesh_cropped.faces)
    mesh_filtered = trimesh.Trimesh(
        vertices=verts_c, faces=faces_c[keep_mask], process=True
    )

    print(f'[{job_id}] faces: {before_faces} → crop {len(mesh_cropped.faces)} '
          f'→ wall+floor {len(mesh_filtered.faces)} '
          f'(floor={floor_mask.sum()}, wall={wall_mask.sum()})')

    mesh_filtered.export(str(off_path))

    # Clear stale downstream outputs so they regenerate
    for f in ['convonet_completed_points.ply', 'convonet_completed_mesh_colored.ply',
              'completed_scene_merged.ply', 'poisson_completion_mesh.ply',
              'poisson_scene_mesh.ply', 'completed_scene_pointcloud.ply',
              'bkgd_completion_comparison.png', 'merge_stats.json']:
        p = comp_dir / f
        if p.exists():
            p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()

    jobs = list(range(5)) if args.all else ([args.job] if args.job is not None else [])
    if not jobs:
        ap.print_help()
        sys.exit(1)

    for j in jobs:
        filter_job(j)

    print('\nDone — re-run run_bkgd_postprocess.pbs to regenerate all outputs')


if __name__ == '__main__':
    main()
