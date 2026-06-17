#!/usr/bin/env python3
"""
Run pbic (SCCNet) object completion on the desk from job 000 (scene0000_00 MOVE).

Input:  processed/scene0000_00/completion/scene0000_00_job000_inst/crop_removed_region.ply
Output: processed/scene0000_00/completion/scene0000_00_job000_inst/pbic_completed_desk.ply
        processed/scene0000_00/completion/scene0000_00_job000_inst/pbic_completed_desk_merged.ply
"""
import sys
import os
import numpy as np
import torch
import open3d as o3d
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PBIC = ROOT / "repo/point_based_instance_completion"
sys.path.insert(0, str(PBIC))
sys.path.insert(0, str(PBIC / "main"))
sys.path.insert(0, str(PBIC / "models"))
sys.path.insert(0, str(PBIC / "utils"))
sys.path.insert(0, str(PBIC / "datasets"))

from utils.pc_utils import normalize_point_cloud, batch_downsample

INST_DIR = ROOT / "repo/scannet/processed/scene0000_00/completion/scene0000_00_job000_inst"
CKPT = PBIC / "experiments/Train_SCCNet_ScanWCF_2025-02-17_21-24/ckpts/ckpt_best.pth"
N_INPUT = 1024        # top-level input points for pbic
N_FREE  = 8192        # must match num_free_space_points in scanwcf.yaml
N_OCC   = 8192        # must match num_occupied_space_points in scanwcf.yaml
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


def read_ply_xyz(path):
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float32)


def estimate_normals(xyz, knn=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(knn)
    return np.asarray(pcd.normals, dtype=np.float32)


def write_ply(path, xyz, rgb=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, pt in enumerate(xyz):
            line = f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}"
            if rgb is not None:
                line += f" {int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}"
            f.write(line + "\n")


def sample(pts, n, rng):
    if pts.shape[0] == 0:
        return np.zeros((n, 3), dtype=np.float32)
    idx = rng.choice(pts.shape[0], size=n, replace=pts.shape[0] < n)
    return pts[idx].astype(np.float32)


def compute_raycasted_free_space(scene_xyz, pose_dir, n_target, rng, pts_per_cam=3000, n_steps=8):
    """
    Compute raycasted free space from camera poses.
    For each camera pose, sample rays to scene surface points and collect
    points along those rays (confirmed empty space before the surface hit).
    """
    from scipy.spatial import cKDTree

    pose_dir = Path(pose_dir)
    pose_files = sorted(pose_dir.glob("frame-*.pose.txt"))
    if not pose_files:
        return None

    cam_positions = []
    for pf in pose_files:
        T = np.loadtxt(str(pf)).reshape(4, 4)
        cam_positions.append(T[:3, 3].astype(np.float32))  # camera world position

    print(f"[free space] {len(cam_positions)} camera positions")

    # Subsample scene surface points for raycasting
    scene_sub_idx = rng.choice(len(scene_xyz), min(20000, len(scene_xyz)), replace=False)
    scene_sub = scene_xyz[scene_sub_idx]

    all_free = []
    for cam_pos in cam_positions:
        # Pick a random subset of surface points visible from this camera
        ray_idx = rng.choice(len(scene_sub), min(pts_per_cam, len(scene_sub)), replace=False)
        surf_pts = scene_sub[ray_idx]

        # For each surface point, sample free space along the ray cam_pos → surf_pt
        ray_dirs = surf_pts - cam_pos[None, :]          # (N, 3)
        ray_lens = np.linalg.norm(ray_dirs, axis=1)     # (N,)
        ray_dirs_norm = ray_dirs / (ray_lens[:, None] + 1e-8)

        # Sample at fractions [1/(n+1), ..., n/(n+1)] of the ray length
        fracs = np.linspace(0.1, 0.85, n_steps)         # stay well short of surface
        for frac in fracs:
            free_pts = cam_pos[None, :] + frac * ray_dirs
            all_free.append(free_pts)

    free_candidates = np.concatenate(all_free, axis=0).astype(np.float32)
    print(f"[free space] {len(free_candidates)} raycast candidates")

    # Remove any candidates that ended up too close to a surface
    tree = cKDTree(scene_xyz[rng.choice(len(scene_xyz), 10000, replace=False)])
    dists, _ = tree.query(free_candidates, k=1)
    free_candidates = free_candidates[dists >= 0.05]
    print(f"[free space] {len(free_candidates)} after 0.05m surface filter")

    if len(free_candidates) < n_target:
        print(f"[free space] warning: only {len(free_candidates)} candidates, needed {n_target}")

    return sample(free_candidates, n_target, rng)


def main():
    print(f"[device] {DEVICE}")
    rng = np.random.default_rng(42)

    # ── Load desk partial scan ────────────────────────────────────────────
    desk_pts = read_ply_xyz(INST_DIR / "crop_removed_region.ply")
    print(f"[input] desk partial: {len(desk_pts)} pts")

    # ── Estimate normals on full cloud, then downsample to 1024 ──────────
    # Order matches ScanWCF dataset: normals computed first, then FPS subsample,
    # then normalize xyz. Normals are directional so they don't need scaling.
    desk_normals = estimate_normals(desk_pts)

    # Downsample to num_input_points=1024 (matching ScanWCF training config)
    from utils.pc_utils import sample_point_cloud
    desk_pts_1k, desk_normals_1k = sample_point_cloud(desk_pts.copy(), N_INPUT, desk_normals.copy())

    # ── Normalize to unit cube (must match ScanWCF training normalization) ───
    desk_norm, centroid, scale = normalize_point_cloud(desk_pts_1k.copy(), method="unit_cube", return_stats=True)
    object_transforms = np.concatenate([centroid[None, :], np.array([[scale]])], axis=1)
    object_transforms = torch.tensor(object_transforms, dtype=torch.float32).to(DEVICE)  # (1,4)

    # ── Build multi-scale inputs [1024, 512, 256, 128] ───────────────────
    partial_t = torch.tensor(desk_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    normal_t  = torch.tensor(desk_normals_1k, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        downsampled_pts, downsampled_nrm = batch_downsample(
            partial_t, n_samples=[1024, 512, 256, 128], normals=normal_t
        )
    partial_sets = [partial_t] + downsampled_pts
    normal_sets  = [normal_t]  + downsampled_nrm

    # ── Build scene context ───────────────────────────────────────────────
    # Strategy A: full-scene occupied (8192 sampled from all scan pts) +
    #             raycasted free space from 6 camera poses.
    # Strategy B (self-constraint): use the desk partial itself as both
    #             free and occupied — normalizes to same positions as the
    #             input, so cross-attention sees the desk's own surface.
    full_scene = np.load(ROOT / "repo/scannet/processed/scene0000_00/xyz.npy").astype(np.float32)

    # Strategy B: self-constraint using the desk partial
    # upsample desk_pts to N_OCC / N_FREE by repeating with small jitter
    def jitter_upsample(pts, n, rng, sigma=0.005):
        idx = rng.choice(len(pts), size=n, replace=True)
        jitter = sigma * rng.standard_normal((n, 3)).astype(np.float32)
        return pts[idx] + jitter

    occ_pts  = jitter_upsample(desk_pts, N_OCC, rng)   # occupied = desk surface
    free_pts = jitter_upsample(desk_pts, N_FREE, rng)   # free = also desk surface (semantically wrong but neutral)
    print(f"[constraints] self-constraint: occ/free = {len(occ_pts)} / {len(free_pts)} (desk surface pts)")

    occ_t  = torch.tensor(occ_pts,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
    free_t = torch.tensor(free_pts, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # ── Load SCCNet model ─────────────────────────────────────────────────
    print(f"[model] loading {CKPT}")
    os.chdir(str(PBIC))
    import datetime
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("index", lambda x, idx: x[idx])
    OmegaConf.register_new_resolver("datetime", lambda x: datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    from main.builder import build_model
    import hydra
    with hydra.initialize_config_dir(config_dir=str(PBIC / "configs"), version_base=None):
        cfg = hydra.compose(config_name="test")
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

    experiment_dir = str(PBIC / "experiments" / cfg.experiment_name)
    import logging
    logger = logging.getLogger("pbic_infer")

    model, _ = build_model(cfg, experiment_dir, logger)
    model = model.to(DEVICE)
    model.eval()
    print(f"[model] loaded SCCNet OK")

    # ── Run inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        completion_set, normals, obj_center = model(
            partial_sets,
            normal_sets,
            free_t,
            occ_t,
            object_transforms,
        )

    # ── Debug outputs ────────────────────────────────────────────────────
    print(f"[debug] num completion levels: {len(completion_set)}")
    for i, c in enumerate(completion_set):
        ci = c[0].cpu().numpy()
        print(f"[debug]   level {i}: {ci.shape}, range [{ci.min():.3f},{ci.max():.3f}]")
    print(f"[debug] obj_center (normalized): {obj_center[0].cpu().numpy()}")
    print(f"[debug] object_transforms: centroid={centroid}, scale={scale:.4f}")

    # completion_set is list of coarse→fine; take finest (last)
    pred_norm = completion_set[-1][0].cpu().numpy()  # (N, 3) normalized
    print(f"[debug] pred_norm extent: {pred_norm.max(0) - pred_norm.min(0)}")
    print(f"[debug] pred_norm x hist: {np.histogram(pred_norm[:,0], bins=8)[0]}")
    print(f"[debug] pred_norm y hist: {np.histogram(pred_norm[:,1], bins=8)[0]}")
    print(f"[debug] pred_norm z hist: {np.histogram(pred_norm[:,2], bins=8)[0]}")

    pred_world = pred_norm * scale + centroid

    print(f"[output] completed desk: {len(pred_world)} pts")

    # ── Save outputs ───────────────────────────────────────────────────────
    out_pts  = INST_DIR / "pbic_completed_desk.ply"
    out_mrg  = INST_DIR / "pbic_completed_desk_merged.ply"

    write_ply(out_pts, pred_world)
    print(f"[saved] {out_pts}")

    # Merged: original partial (grey) + completed (orange)
    merged = np.concatenate([desk_pts, pred_world], axis=0)
    rgb = np.zeros((len(merged), 3), dtype=np.uint8)
    rgb[:len(desk_pts)] = [160, 160, 160]
    rgb[len(desk_pts):] = [255, 140, 0]
    write_ply(out_mrg, merged, rgb)
    print(f"[saved] {out_mrg}")
    print("Done.")


if __name__ == "__main__":
    main()
