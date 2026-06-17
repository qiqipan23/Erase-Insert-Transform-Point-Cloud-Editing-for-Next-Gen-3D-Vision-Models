#!/usr/bin/env python3
"""
Run pbic (SCCNet) object completion on any instance from scene0000_00.
Usage: python run_pbic_inst.py --inst <instance_id>
"""
import sys
import os
import argparse
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

SCENE_DIR = ROOT / "repo/scannet/processed/scene0000_00"
CKPT = PBIC / "experiments/Train_SCCNet_ScanWCF_2025-02-17_21-24/ckpts/ckpt_best.pth"
N_INPUT = 1024
N_FREE  = 8192
N_OCC   = 8192
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


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


def jitter_upsample(pts, n, rng, sigma=0.005):
    idx = rng.choice(len(pts), size=n, replace=True)
    jitter = sigma * rng.standard_normal((n, 3)).astype(np.float32)
    return pts[idx] + jitter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", type=int, required=True, help="instance ID")
    args = parser.parse_args()

    inst_id = args.inst
    print(f"[device] {DEVICE}")
    print(f"[inst] processing instance {inst_id}")
    rng = np.random.default_rng(42)

    # ── Load instance points from scene ──────────────────────────────────
    xyz_all = np.load(SCENE_DIR / "xyz.npy").astype(np.float32)
    inst_all = np.load(SCENE_DIR / "inst.npy")
    import json
    with open(SCENE_DIR / "inst_to_label.json") as f:
        lbl = json.load(f)

    mask = inst_all == inst_id
    obj_pts = xyz_all[mask]
    label = lbl.get(str(inst_id), "unknown")
    print(f"[input] inst {inst_id} ({label}): {len(obj_pts)} pts, extent {(obj_pts.max(0)-obj_pts.min(0)).round(3)}")

    if len(obj_pts) < 50:
        print("ERROR: too few points"); sys.exit(1)

    # ── Estimate normals ─────────────────────────────────────────────────
    obj_normals = estimate_normals(obj_pts)

    from utils.pc_utils import sample_point_cloud
    obj_pts_1k, obj_normals_1k = sample_point_cloud(obj_pts.copy(), N_INPUT, obj_normals.copy())

    # ── Normalize to unit cube ───────────────────────────────────────────
    obj_norm, centroid, scale = normalize_point_cloud(obj_pts_1k.copy(), method="unit_cube", return_stats=True)
    object_transforms = np.concatenate([centroid[None, :], np.array([[scale]])], axis=1)
    object_transforms = torch.tensor(object_transforms, dtype=torch.float32).to(DEVICE)

    # ── Multi-scale input ────────────────────────────────────────────────
    partial_t = torch.tensor(obj_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    normal_t  = torch.tensor(obj_normals_1k, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        downsampled_pts, downsampled_nrm = batch_downsample(
            partial_t, n_samples=[1024, 512, 256, 128], normals=normal_t
        )
    partial_sets = [partial_t] + downsampled_pts
    normal_sets  = [normal_t]  + downsampled_nrm

    # ── Constraints: self-constraint (object surface as both free+occ) ───
    occ_pts  = jitter_upsample(obj_pts, N_OCC, rng)
    free_pts = jitter_upsample(obj_pts, N_FREE, rng)

    occ_t  = torch.tensor(occ_pts,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
    free_t = torch.tensor(free_pts, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # ── Load SCCNet model ─────────────────────────────────────────────────
    print(f"[model] loading checkpoint")
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
    print(f"[model] loaded OK")

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        completion_set, normals, obj_center = model(
            partial_sets, normal_sets, free_t, occ_t, object_transforms,
        )

    # Debug
    pred_norm = completion_set[-1][0].cpu().numpy()
    pred_world = pred_norm * scale + centroid
    print(f"[debug] pred_norm extent: {(pred_norm.max(0)-pred_norm.min(0)).round(3)}")
    print(f"[debug] obj_center (norm): {obj_center[0].cpu().numpy()}")
    print(f"[output] completed: {len(pred_world)} pts")

    # ── Save ────────────────────────────────────────────────────────────
    out_dir = SCENE_DIR / "completion" / f"pbic_inst{inst_id:03d}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ply(out_dir / "completed.ply", pred_world)
    merged = np.concatenate([obj_pts, pred_world], axis=0)
    rgb = np.zeros((len(merged), 3), dtype=np.uint8)
    rgb[:len(obj_pts)] = [160, 160, 160]
    rgb[len(obj_pts):] = [255, 140, 0]
    write_ply(out_dir / "merged.ply", merged, rgb)
    print(f"[saved] {out_dir}/merged.ply")
    print("Done.")


if __name__ == "__main__":
    main()
