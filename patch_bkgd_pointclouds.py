"""
Replace structural fill in _bkgd pointcloud.npz with nn_floor_fill,
which is denser and matches job014's fill quality.
"""
import numpy as np
from pathlib import Path
from plyfile import PlyData

SCANNET = Path('repo/scannet')
DATA_ROOT = SCANNET / 'completion/convonet_data'
COMP_ROOT = SCANNET / 'processed/scene0000_00/completion'

def read_ply_xyz(path):
    d = PlyData.read(str(path))['vertex']
    return np.stack([d['x'], d['y'], d['z']], axis=1).astype(np.float32)

for j in range(5):
    job_id = f'scene0000_00_job00{j}_bkgd'
    comp_dir = COMP_ROOT / job_id
    npz_path = DATA_ROOT / job_id / 'scannet_remove' / job_id / 'pointcloud.npz'

    if not npz_path.exists():
        print(f'{job_id}: pointcloud.npz not found, skipping')
        continue

    # Load existing (has scan + sparse structural fill)
    old = np.load(npz_path)
    old_pts = old['points']
    old_feat = old['features']

    # Identify real scan points (feature[:,0] == 0)
    scan_mask = old_feat[:, 0] == 0.0
    scan_pts = old_pts[scan_mask]
    scan_normals = old['normals'][scan_mask]

    # Load both fill sources: nn_floor_fill (wall) + structural_surface (floor)
    nn_pts  = read_ply_xyz(comp_dir / 'nn_floor_fill_completed_points.ply')
    st_pts  = read_ply_xyz(comp_dir / 'structural_surface_completed_points.ply')
    fill_pts = np.concatenate([nn_pts, st_pts], axis=0)
    fill_normals = np.zeros((len(fill_pts), 3), dtype=np.float32)

    # Combine: scan (feat=0) + wall+floor fill (feat=1)
    new_pts = np.concatenate([scan_pts, fill_pts], axis=0)
    new_normals = np.concatenate([scan_normals, fill_normals], axis=0)
    new_feat = np.concatenate([
        np.zeros((len(scan_pts), 2), dtype=np.float32),
        np.ones((len(fill_pts), 2), dtype=np.float32),
    ], axis=0)

    np.savez(npz_path, points=new_pts, normals=new_normals, features=new_feat)
    print(f'{job_id}: {len(scan_pts)} scan + {len(nn_pts)} wall + {len(st_pts)} floor = {len(new_pts)} total  (was {len(old_pts)})')
