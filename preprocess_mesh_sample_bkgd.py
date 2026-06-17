"""
Voxel-resample ALL layers (background + fills) at the same 1.5 cm grid
so every point in the merged thumbnail has identical spatial density.
Run with:  conda activate o3d && python preprocess_mesh_sample_bkgd.py
"""
from pathlib import Path
import json, struct, re
import numpy as np
import open3d as o3d

SCENE = Path("/rds/general/user/qp23/home/Hypo3D/repo/scannet/processed/scene0000_00")
COMPB = SCENE / "completion/scene0000_00_job017_plane_fill_no_nnwall"
OUT   = Path("/rds/general/user/qp23/home/Hypo3D/figs/pipeline_bkgd_thumbs")
OUT.mkdir(parents=True, exist_ok=True)

VOXEL = 0.015

# ── PLY files to resample ──────────────────────────────────────────────────────
files = {
    "bkgd":   SCENE / "edits/job017_remove_desk.ply",
    "plane":  COMPB  / "nn_floor_fill_completed_points.ply",
    "struct": COMPB  / "structural_surface_completed_points.ply",
    "conv":   COMPB  / "convonet_completed_points.ply",
}

for tag, src in files.items():
    print(f"Processing {tag}...")
    pcd = o3d.io.read_point_cloud(str(src))
    print(f"  {len(pcd.points)} pts in")
    pcd_ds = pcd.voxel_down_sample(voxel_size=VOXEL)
    print(f"  {len(pcd_ds.points)} pts after {VOXEL*100:.0f} cm voxel")
    out = OUT / f"{tag}_voxel.ply"
    o3d.io.write_point_cloud(str(out), pcd_ds)
    print(f"  Saved {out.name}")

# ── Full original scene: all npy points voxel-resampled ───────────────────────
print("Processing full original scene...")
xyz_all  = np.load(SCENE / "xyz.npy").astype(np.float32)
_rgb_raw = np.load(SCENE / "rgb.npy")
rgb_all  = ((_rgb_raw * 255) if _rgb_raw.max() <= 1.0 else _rgb_raw).astype(np.uint8)
inst     = np.load(SCENE / "inst.npy").astype(np.int32)

pcd_full = o3d.geometry.PointCloud()
pcd_full.points = o3d.utility.Vector3dVector(xyz_all.astype(np.float64))
pcd_full.colors = o3d.utility.Vector3dVector(rgb_all.astype(np.float64) / 255.0)
pcd_full_ds = pcd_full.voxel_down_sample(voxel_size=VOXEL)
print(f"  {len(pcd_full.points)} → {len(pcd_full_ds.points)} pts after voxel")
o3d.io.write_point_cloud(str(OUT / "full_scene_voxel.ply"), pcd_full_ds)
print("  Saved full_scene_voxel.ply")

# ── Desk instance: extract from npy, save as PLY, then voxel-resample ─────────
print("Processing desk instance...")
JOB  = json.loads((SCENE / "edits/job017.json").read_text())
desk_id = int(JOB["target_instance_id"])

xyz_all  = np.load(SCENE / "xyz.npy").astype(np.float32)
_rgb_raw = np.load(SCENE / "rgb.npy")
rgb_all  = ((_rgb_raw * 255) if _rgb_raw.max() <= 1.0 else _rgb_raw).astype(np.uint8)
inst     = np.load(SCENE / "inst.npy").astype(np.int32)

desk_m   = inst == desk_id
desk_xyz = xyz_all[desk_m]
desk_rgb = rgb_all[desk_m]
print(f"  {desk_m.sum()} desk points")

pcd_desk = o3d.geometry.PointCloud()
pcd_desk.points = o3d.utility.Vector3dVector(desk_xyz.astype(np.float64))
pcd_desk.colors = o3d.utility.Vector3dVector(desk_rgb.astype(np.float64) / 255.0)
pcd_desk_ds = pcd_desk.voxel_down_sample(voxel_size=VOXEL)
print(f"  {len(pcd_desk_ds.points)} pts after voxel")
o3d.io.write_point_cloud(str(OUT / "desk_voxel.ply"), pcd_desk_ds)
print("  Saved desk_voxel.ply")
