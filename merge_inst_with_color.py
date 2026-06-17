#!/usr/bin/env python3
"""
Merge original coloured instance points with pbic completed points.
Original partial: real scan RGB. Completed: orange (255, 140, 0).
"""
import numpy as np, json, sys
from pathlib import Path

SCENE_DIR = Path("/rds/general/user/qp23/home/Hypo3D/repo/scannet/processed/scene0000_00")
COMP_DIR  = SCENE_DIR / "completion"
OUT_DIR   = Path("/rds/general/user/qp23/home/Hypo3D/inst_orig")
OUT_DIR.mkdir(exist_ok=True)

xyz  = np.load(SCENE_DIR / "xyz.npy").astype(np.float32)
rgb  = (np.load(SCENE_DIR / "rgb.npy") * 255).clip(0, 255).astype(np.uint8)
inst = np.load(SCENE_DIR / "inst.npy")
with open(SCENE_DIR / "inst_to_label.json") as f:
    lbl = json.load(f)


def read_ply_xyz(path):
    pts = []
    with open(path) as f:
        in_header = True
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            vals = line.split()
            pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(pts, dtype=np.float32)


def write_ply(path, xyz, rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(xyz)):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                    f"{int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n")


inst_ids = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [8, 29, 30, 35]

for iid in inst_ids:
    label = lbl.get(str(iid), "unknown")
    mask  = inst == iid
    orig_xyz = xyz[mask]
    orig_rgb = rgb[mask]

    # find pbic completed.ply for this instance
    comp_dirs = list(COMP_DIR.glob(f"pbic_inst{iid:03d}*"))
    if not comp_dirs:
        print(f"inst {iid} ({label}): no pbic completion found, skipping")
        continue
    comp_file = comp_dirs[0] / "completed.ply"
    if not comp_file.exists():
        print(f"inst {iid} ({label}): {comp_file} not found, skipping")
        continue

    pred_xyz = read_ply_xyz(comp_file)
    pred_rgb = np.full((len(pred_xyz), 3), [255, 140, 0], dtype=np.uint8)

    merged_xyz = np.concatenate([orig_xyz, pred_xyz], axis=0)
    merged_rgb = np.concatenate([orig_rgb, pred_rgb], axis=0)

    out = OUT_DIR / f"inst{iid:03d}_{label}_pbic_colored.ply"
    write_ply(out, merged_xyz, merged_rgb)
    print(f"inst {iid} ({label}): {len(orig_xyz)} orig + {len(pred_xyz)} pred -> {out}")

print("Done.")
