#!/usr/bin/env python3
"""Extract original coloured point clouds for given instance IDs."""
import numpy as np, json, sys
from pathlib import Path

SCENE_DIR = Path("/rds/general/user/qp23/home/Hypo3D/repo/scannet/processed/scene0000_00")
OUT_DIR = Path("/rds/general/user/qp23/home/Hypo3D/inst_orig")
OUT_DIR.mkdir(exist_ok=True)

xyz = np.load(SCENE_DIR / "xyz.npy").astype(np.float32)
rgb = (np.load(SCENE_DIR / "rgb.npy") * 255).clip(0, 255).astype(np.uint8)
inst = np.load(SCENE_DIR / "inst.npy")
with open(SCENE_DIR / "inst_to_label.json") as f:
    lbl = json.load(f)

inst_ids = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [29, 30, 35]

def write_ply(path, xyz, rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(xyz)):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n")

for iid in inst_ids:
    mask = inst == iid
    label = lbl.get(str(iid), "unknown")
    pts = xyz[mask]; colors = rgb[mask]
    print(f"inst {iid} ({label}): {mask.sum()} pts")
    out = OUT_DIR / f"inst{iid:03d}_{label}_orig.ply"
    write_ply(out, pts, colors)
    print(f"  -> {out}")

print("Done.")
