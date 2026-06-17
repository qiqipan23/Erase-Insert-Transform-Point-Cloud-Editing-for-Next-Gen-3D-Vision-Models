#!/usr/bin/env python3
"""
Pre-compute per-scene orientation contexts for GPT-4o evaluation.

For each scene in axis_definition.xlsx:
  1. Locate the landmark object's centroid in world space
  2. Determine which image quadrant it falls in (top-of-image = high-Y, right = high-X)
  3. Generate a human-readable orientation sentence mapping scene terms to image positions

Output: dataset/orientation_contexts.json  { scene_id: orientation_string }

The rendering convention (from rerender_and_build_eval_v2.py):
  image-right  → world +X
  image-top    → world +Y (high Y = top of image)
  image-bottom → world -Y (low Y = bottom of image)
  image-left   → world -X

Usage:
  source /rds/general/user/qp23/home/miniconda3/etc/profile.d/conda.sh && conda activate placeit3d
  python compute_scene_orientation_contexts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HYPO_ROOT      = Path(__file__).resolve().parent
AXIS_DEF       = HYPO_ROOT / "dataset" / "axis_definition.xlsx"
PROC_ROOT      = HYPO_ROOT / "repo" / "scannet" / "processed"
OUT_JSON       = HYPO_ROOT / "dataset" / "orientation_contexts.json"

UP = np.array([0.0, 1.0, 0.0])

# In the top-down image: X+ → right, Y+ → top (high-Y → top pixel).
# Each world direction maps to an image label:
WORLD_TO_IMAGE = {
    ( 1, 0): "right side",   # +X
    (-1, 0): "left side",    # -X
    ( 0, 1): "top",          # +Y → top of image
    ( 0,-1): "bottom",       # -Y → bottom of image
}

DIRECTION_TO_IMAGE_LABEL = {
    "front": "top",      # will be overridden by per-scene computation if possible
    "back":  "bottom",
    "left":  "left side",
    "right": "right side",
}


def dominant_image_direction(vec_xz: np.ndarray) -> str:
    """Map a 2D world vector [x, y] (horizontal plane) to an image position label.

    In the rendering, world X → image X, world Y (horizontal) → image Y-inverted.
    So high world-Y → top of image; high world-X → right of image.
    """
    x, y = float(vec_xz[0]), float(vec_xz[1])
    # Determine primary axis
    if abs(x) >= abs(y):
        return "right side" if x > 0 else "left side"
    else:
        return "top" if y > 0 else "bottom"


def compass_from_vec(vec_xz: np.ndarray) -> str:
    x, y = float(vec_xz[0]), float(vec_xz[1])
    angle = np.degrees(np.arctan2(x, y))  # 0=+Y, 90=+X, -90=-X, 180/-180=-Y
    # Quantise to 8 compass directions
    sectors = [
        (-22.5,  22.5, "top"),
        ( 22.5,  67.5, "top-right corner"),
        ( 67.5, 112.5, "right side"),
        (112.5, 157.5, "bottom-right corner"),
        (157.5, 180,   "bottom"),
        (-180, -157.5, "bottom"),
        (-157.5,-112.5, "bottom-left corner"),
        (-112.5, -67.5, "left side"),
        (-67.5, -22.5, "top-left corner"),
    ]
    for lo, hi, label in sectors:
        if lo <= angle < hi:
            return label
    return "top"


def build_context(scene_id: str, row: pd.Series,
                  proc_root: Path) -> str:
    scene_dir = proc_root / scene_id
    xyz_path  = scene_dir / "xyz.npy"
    inst_path = scene_dir / "inst.npy"
    label_path = scene_dir / "inst_to_label.json"

    # Base line: always state the image axis convention.
    base = (
        "In this top-down (bird's-eye) view: "
        "the right side of the image = positive X (east), "
        "the top of the image = positive Y (north). "
        "Direction terms (front/back/left/right) are relative to the room's orientation."
    )

    if not (xyz_path.exists() and inst_path.exists() and label_path.exists()):
        return base

    xyz  = np.load(xyz_path)
    inst = np.load(inst_path)
    label_map: dict = json.loads(label_path.read_text())
    label_to_insts: dict[str, list[int]] = {}
    for iid, lbl in label_map.items():
        label_to_insts.setdefault(lbl.lower(), []).append(int(iid))

    scene_center = xyz.mean(axis=0)
    scene_center[2] = 0  # ignore height (Z = up in this coordinate system)

    landmark_hints: list[str] = []
    direction_map: dict[str, str] = {}  # scene_direction → image_label

    for direction in ["Front", "Back", "Left", "Right"]:
        val = row.get(direction)
        if pd.isna(val) or not str(val).strip():
            continue
        landmark = str(val).strip()
        insts = label_to_insts.get(landmark.lower(), [])
        if not insts:
            continue

        # Find instance with centroid furthest from scene center
        best_pts, best_d = None, -1.0
        for iid in insts:
            pts = xyz[inst == iid]
            if len(pts) == 0:
                continue
            c = pts.mean(axis=0)
            c_flat = c.copy(); c_flat[2] = 0  # zero out Z (height)
            d = float(np.linalg.norm(c_flat - scene_center))
            if d > best_d:
                best_d, best_pts = d, pts

        if best_pts is None:
            continue

        centroid = best_pts.mean(axis=0)
        # The rendering (render_dense) uses xyz[:,0] as image-X and xyz[:,1] as image-Y.
        # Z is the vertical height (up direction) and is only used for depth sorting.
        # So the horizontal plane is (X, Y), matching the image axes directly.
        vec_xz = np.array([centroid[0] - scene_center[0],
                           centroid[1] - scene_center[1]])
        norm = np.linalg.norm(vec_xz)
        if norm < 0.1:
            continue
        vec_xz /= norm
        img_label = compass_from_vec(vec_xz)
        direction_map[direction.lower()] = img_label
        landmark_hints.append(
            f"the {landmark} (scene {direction.lower()}) is toward the {img_label} of this image"
        )

    if landmark_hints:
        hint_str = "; ".join(landmark_hints) + "."
        return base + " Specifically: " + hint_str

    return base


def main() -> None:
    df = pd.read_excel(AXIS_DEF, engine="openpyxl")
    contexts: dict[str, str] = {}
    n_with_hints = 0

    for _, row in df.iterrows():
        scene_id = str(row["scene_id"])
        ctx = build_context(scene_id, row, PROC_ROOT)
        contexts[scene_id] = ctx
        if "Specifically:" in ctx:
            n_with_hints += 1

    OUT_JSON.write_text(json.dumps(contexts, indent=2, ensure_ascii=False))
    print(f"Written: {OUT_JSON}")
    print(f"  Total scenes: {len(contexts)}")
    print(f"  Scenes with landmark hints: {n_with_hints}")
    print(f"\nSample (scene0000_00):")
    print(f"  {contexts.get('scene0000_00', 'NOT FOUND')}")


if __name__ == "__main__":
    main()
