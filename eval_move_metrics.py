#!/usr/bin/env python3
"""
Geometric evaluation metrics for the MOVE pipeline.

For each job manifest in processed/<scene_id>/edits/job*.json, computes:

  1. relation_correct    bool   — moved object centroid is in the correct
                                  spatial direction from the anchor centroid
  2. placement_error_m  float  — Euclidean distance (metres) between the
                                  expected centroid (original + translation)
                                  and the actual centroid in the edited cloud
  3. chamfer_dist       float  — one-sided Chamfer Distance from the expected
                                  moved cloud to the actual edited cloud
                                  (mean nearest-neighbour distance in metres)
  4. intersection_frac  float  — fraction of moved-object points that are
                                  within 0.05 m of any non-target point
  5. completion_ratio   float  — completion_completed_points /
                                  completion_input_points  (0 if skipped)

Outputs:
  dataset/move_eval_geometric.json   full per-job results
  (summary printed to stdout)

Usage:
  source miniconda3/etc/profile.d/conda.sh && conda activate placeit3d
  python eval_move_metrics.py
  python eval_move_metrics.py --scene scene0000_00   # single scene
  python eval_move_metrics.py --limit 50             # first N jobs
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

try:
    import open3d as o3d
    def load_ply_xyz(path: Path) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points, dtype=np.float32)
except ImportError:
    # Fallback: read PLY with numpy (handles binary and ascii)
    def load_ply_xyz(path: Path) -> np.ndarray:  # type: ignore[misc]
        import struct, re
        data = path.read_bytes()
        # Find end_header
        header_end = data.find(b"end_header\n") + len(b"end_header\n")
        header = data[:header_end].decode("ascii", errors="ignore")
        n_verts = int(re.search(r"element vertex (\d+)", header).group(1))
        is_binary_le = "binary_little_endian" in header
        props = re.findall(r"property (\w+) (\w+)", header)
        if is_binary_le:
            fmts = {"float": "f", "double": "d", "uchar": "B",
                    "int": "i", "uint": "I", "short": "h", "ushort": "H"}
            fmt = "<" + "".join(fmts.get(p[0], "f") for p in props)
            sz  = struct.calcsize(fmt)
            rows = [struct.unpack_from(fmt, data, header_end + i * sz)
                    for i in range(n_verts)]
            arr = np.array(rows, dtype=np.float32)
        else:
            arr = np.loadtxt(
                data[header_end:].decode("ascii").splitlines()[:n_verts],
                dtype=np.float32,
            )
        xyz_cols = [i for i,(t,n) in enumerate(props) if n in ("x","y","z")]
        return arr[:, xyz_cols[:3]]

HYPO_ROOT     = Path(__file__).resolve().parent
SCANNET_ROOT  = HYPO_ROOT / "repo" / "scannet"
PROCESSED_ROOT = SCANNET_ROOT / "processed"
OUT_PATH      = HYPO_ROOT / "dataset" / "move_eval_geometric.json"
AXIS_DEF      = HYPO_ROOT / "dataset" / "axis_definition.xlsx"

INTERSECTION_THRESHOLD = 0.05  # metres
UP = np.array([0.0, 1.0, 0.0])  # Y = up in ScanNet


# ── per-scene orientation ────────────────────────────────────────────────────

def load_scene_axes(scene_dir: Path,
                    axis_df: pd.DataFrame) -> dict[str, np.ndarray] | None:
    """
    Derive per-scene direction vectors from axis_definition.xlsx landmark objects.
    Returns {front, back, left, right, up} as unit 3D vectors, or None if unavailable.

    Strategy: find which object label is designated Front (or any other direction),
    locate its centroid in the original scan, then define:
        front_vec = normalize( front_centroid - scene_center )   [XZ plane only]
        right_vec = normalize( cross(up, front_vec) )
    """
    scene_id = scene_dir.name
    rows = axis_df[axis_df["scene_id"] == scene_id]
    if rows.empty:
        return None

    row = rows.iloc[0]
    xyz  = np.load(scene_dir / "xyz.npy")
    inst = np.load(scene_dir / "inst.npy")
    label_map: dict[str, str] = json.loads(
        (scene_dir / "inst_to_label.json").read_text()
    )
    # reverse: label → list of instance ids
    label_to_insts: dict[str, list[int]] = {}
    for inst_id, label in label_map.items():
        label_to_insts.setdefault(label.lower(), []).append(int(inst_id))

    scene_center = xyz.mean(axis=0)
    scene_center[1] = 0  # ignore Y for horizontal axes

    direction_vecs: dict[str, np.ndarray] = {}

    for direction in ["Front", "Back", "Left", "Right"]:
        label_val = row.get(direction)
        if pd.isna(label_val) or not str(label_val).strip():
            continue
        label_key = str(label_val).strip().lower()
        inst_ids = label_to_insts.get(label_key, [])
        if not inst_ids:
            continue
        # Pick the instance whose centroid is furthest from scene center
        # (most "at the edge" — best landmark)
        best_pts, best_dist = None, -1.0
        for iid in inst_ids:
            pts = xyz[inst == iid]
            if len(pts) == 0:
                continue
            c = pts.mean(axis=0)
            c_flat = c.copy(); c_flat[1] = 0
            d = float(np.linalg.norm(c_flat - scene_center))
            if d > best_dist:
                best_dist, best_pts = d, pts
        if best_pts is None:
            continue
        c = best_pts.mean(axis=0)
        vec = c - scene_center
        vec[1] = 0  # project to horizontal plane
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            continue
        direction_vecs[direction] = vec / norm

    if not direction_vecs:
        return None

    # Build full set from whatever we have, deriving missing ones
    # Priority: use explicit definitions, fill gaps with derived ones
    axes: dict[str, np.ndarray] = {}

    if "Front" in direction_vecs:
        axes["front"] = direction_vecs["Front"]
        axes["back"]  = -direction_vecs["Front"]
        axes["right"] = np.cross(UP, direction_vecs["Front"])
        axes["right"] /= np.linalg.norm(axes["right"])
        axes["left"]  = -axes["right"]
    elif "Back" in direction_vecs:
        axes["back"]  = direction_vecs["Back"]
        axes["front"] = -direction_vecs["Back"]
        axes["right"] = np.cross(UP, axes["front"])
        axes["right"] /= np.linalg.norm(axes["right"])
        axes["left"]  = -axes["right"]
    elif "Right" in direction_vecs:
        axes["right"] = direction_vecs["Right"]
        axes["left"]  = -direction_vecs["Right"]
        axes["front"] = np.cross(direction_vecs["Right"], UP)
        axes["front"] /= np.linalg.norm(axes["front"])
        axes["back"]  = -axes["front"]
    elif "Left" in direction_vecs:
        axes["left"]  = direction_vecs["Left"]
        axes["right"] = -direction_vecs["Left"]
        axes["front"] = np.cross(UP, axes["right"])
        axes["front"] /= np.linalg.norm(axes["front"])
        axes["back"]  = -axes["front"]

    # Override with explicit definitions where available
    for d, key in [("Front","front"),("Back","back"),("Left","left"),("Right","right")]:
        if d in direction_vecs:
            axes[key] = direction_vecs[d]

    axes["up"]   = UP.copy()
    axes["down"] = -UP.copy()
    return axes


RELATION_TO_AXIS_KEY = {
    "RIGHT_OF":  ("right",  1),
    "LEFT_OF":   ("left",   1),
    "FRONT_OF":  ("front",  1),
    "BACK_OF":   ("back",   1),
    "ON_TOP_OF": ("up",     1),
    "UNDER":     ("down",   1),
    "NEXT_TO":   None,
}


# ── point cloud helpers ──────────────────────────────────────────────────────

def load_original(scene_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz [N,3], inst [N]) for the original scene."""
    xyz  = np.load(scene_dir / "xyz.npy")
    inst = np.load(scene_dir / "inst.npy")
    return xyz, inst


def centroid(pts: np.ndarray) -> np.ndarray:
    return pts.mean(axis=0)


# ── metric functions ─────────────────────────────────────────────────────────

def relation_correct(moved_centroid: np.ndarray,
                     anchor_centroid: np.ndarray,
                     relation: str,
                     scene_axes: dict[str, np.ndarray] | None = None) -> bool | None:
    """True if the moved object is in the expected direction from anchor."""
    spec = RELATION_TO_AXIS_KEY.get(relation)
    if spec is None:
        return None  # NEXT_TO — skip

    axis_key, _ = spec
    delta = moved_centroid - anchor_centroid

    if scene_axes and axis_key in scene_axes:
        # Project delta onto the scene's actual direction vector
        axis_vec = scene_axes[axis_key]
        return bool(float(np.dot(delta, axis_vec)) > 0)
    else:
        # Fallback: global axes (Y=up consistent; X/Z best-guess for horizontal)
        fallback = {"right": 0, "left": 0, "front": 2, "back": 2, "up": 1, "down": 1}
        fallback_sign = {"right": 1, "left": -1, "front": 1, "back": -1, "up": 1, "down": -1}
        idx  = fallback[axis_key]
        sign = fallback_sign[axis_key]
        return bool((delta[idx] * sign) > 0)


def placement_error(expected_centroid: np.ndarray,
                    actual_centroid: np.ndarray) -> float:
    """Euclidean distance between expected and actual centroid (metres)."""
    return float(np.linalg.norm(expected_centroid - actual_centroid))


def one_sided_chamfer(source: np.ndarray, target: np.ndarray) -> float:
    """Mean nearest-neighbour distance from source to target (metres)."""
    if len(source) == 0 or len(target) == 0:
        return float("nan")
    tree = KDTree(target)
    dists, _ = tree.query(source, k=1)
    return float(dists.mean())


def intersection_fraction(moved_pts: np.ndarray,
                          other_pts: np.ndarray,
                          threshold: float = INTERSECTION_THRESHOLD) -> float:
    """Fraction of moved-object points within `threshold` of any other point."""
    if len(moved_pts) == 0 or len(other_pts) == 0:
        return 0.0
    tree = KDTree(other_pts)
    dists, _ = tree.query(moved_pts, k=1)
    return float((dists < threshold).mean())


def completion_ratio(manifest: dict) -> float:
    """completed_points / input_points; 0 if completion was skipped."""
    if manifest.get("completion_skipped", True):
        return 0.0
    inp = manifest.get("completion_input_points") or 0
    out = manifest.get("completion_completed_points") or 0
    return float(out / inp) if inp > 0 else 0.0


# ── per-job evaluation ───────────────────────────────────────────────────────

def evaluate_job(manifest: dict, scene_dir: Path,
                 scene_axes: dict | None = None) -> dict | None:
    """Compute all metrics for one job. Returns None if data missing."""
    if manifest.get("op") != "MOVE":
        return None

    canonical = manifest.get("canonical_edit_file", "")
    edit_ply  = Path(canonical) if canonical else None

    if not edit_ply or not edit_ply.exists():
        return {"error": f"edited PLY not found: {edit_ply}"}

    target_id  = manifest["target_instance_id"]
    anchor_id  = manifest["anchor_instance_id"]
    translation = np.array(manifest["translation"], dtype=np.float32)
    relation    = manifest.get("relation", "")

    try:
        orig_xyz, orig_inst = load_original(scene_dir)
    except Exception as e:
        return {"error": f"load original failed: {e}"}

    target_pts = orig_xyz[orig_inst == target_id]
    anchor_pts = orig_xyz[orig_inst == anchor_id]
    other_pts  = orig_xyz[orig_inst != target_id]

    if len(target_pts) == 0:
        return {"error": f"target instance {target_id} not found"}
    if len(anchor_pts) == 0:
        return {"error": f"anchor instance {anchor_id} not found"}

    # Expected moved position
    expected_pts      = target_pts + translation
    expected_centroid = centroid(expected_pts)
    anchor_centroid_v = centroid(anchor_pts)

    # Load edited scene and extract actual moved object
    try:
        edit_pts = load_ply_xyz(edit_ply)
    except Exception as e:
        return {"error": f"load edit PLY failed: {e}"}

    # Actual moved object: points in edited scene near the expected location
    # (within 2× the object's bounding box radius)
    obj_radius = float(np.linalg.norm(
        target_pts.max(axis=0) - target_pts.min(axis=0)
    )) / 2 + 0.3  # add 30 cm margin

    tree_edit = KDTree(edit_pts)
    idx = tree_edit.query_ball_point(expected_centroid, r=obj_radius)
    actual_moved_pts = edit_pts[idx] if idx else edit_pts[:0]

    if len(actual_moved_pts) < 10:
        actual_centroid = expected_centroid  # fallback
        cd = float("nan")
    else:
        actual_centroid = centroid(actual_moved_pts)
        cd = one_sided_chamfer(expected_pts, actual_moved_pts)

    # Other-object points in edited scene (everything except the moved region)
    all_idx   = set(range(len(edit_pts)))
    moved_idx = set(idx) if idx else set()
    other_edit_pts = edit_pts[list(all_idx - moved_idx)] if moved_idx else edit_pts

    return {
        "scene_id":          manifest["scene_id"],
        "job_stem":          Path(canonical).stem if canonical else "",
        "target_label":      manifest.get("target_label", ""),
        "anchor_label":      manifest.get("anchor_label", ""),
        "relation":          relation,
        "relation_correct":  relation_correct(actual_centroid, anchor_centroid_v, relation, scene_axes),
        "used_scene_axes":   scene_axes is not None,
        "placement_error_m": placement_error(expected_centroid, actual_centroid),
        "chamfer_dist":      cd,
        "intersection_frac": intersection_fraction(
            actual_moved_pts, other_edit_pts
        ) if len(actual_moved_pts) >= 10 else float("nan"),
        "completion_ratio":  completion_ratio(manifest),
        "n_target_pts":      int(len(target_pts)),
        "n_actual_moved":    int(len(actual_moved_pts)),
        "completion_backend": manifest.get("completion_backend", "none"),
    }


# ── summary printing ─────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"\n{'='*60}")
    print(f"MOVE Geometric Evaluation  —  {len(valid)} jobs, {len(errors)} errors")
    print(f"{'='*60}")

    def safe_mean(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return np.mean(v) if v else float("nan")

    # Relation accuracy
    rel_vals = [r["relation_correct"] for r in valid if r["relation_correct"] is not None]
    rel_acc  = np.mean(rel_vals) * 100 if rel_vals else float("nan")

    pe    = safe_mean([r["placement_error_m"] for r in valid])
    cd    = safe_mean([r["chamfer_dist"] for r in valid])
    inter = safe_mean([r["intersection_frac"] for r in valid]) * 100
    comp  = safe_mean([r["completion_ratio"] for r in valid])

    print(f"\n{'Metric':<30}{'Value'}")
    print(f"{'-'*45}")
    with_axes = sum(1 for r in valid if r.get("used_scene_axes"))
    print(f"{'Relation Accuracy':<30}{rel_acc:.1f}%  ({len(rel_vals)} checkable, {with_axes} used per-scene axes)")
    print(f"{'Placement Error':<30}{pe:.4f} m")
    print(f"{'Chamfer Distance':<30}{cd:.4f} m")
    print(f"{'Intersection Fraction':<30}{inter:.1f}%")
    print(f"{'Completion Ratio':<30}{comp:.3f}  (completed/input pts)")

    # Per-relation breakdown
    by_rel: dict[str, list] = defaultdict(list)
    for r in valid:
        by_rel[r["relation"]].append(r)

    print(f"\n{'Relation':<15}{'N':>5}{'Rel.Acc%':>10}{'PlaceErr(m)':>13}{'CD(m)':>10}{'Inter%':>10}")
    print("-" * 65)
    for rel, items in sorted(by_rel.items()):
        ra  = [i["relation_correct"] for i in items if i["relation_correct"] is not None]
        ra_pct = np.mean(ra) * 100 if ra else float("nan")
        pe_ = safe_mean([i["placement_error_m"] for i in items])
        cd_ = safe_mean([i["chamfer_dist"] for i in items])
        it_ = safe_mean([i["intersection_frac"] for i in items]) * 100
        print(f"{rel:<15}{len(items):>5}{ra_pct:>10.1f}{pe_:>13.4f}{cd_:>10.4f}{it_:>10.1f}")

    if errors:
        print(f"\n{len(errors)} jobs had errors (first 3):")
        for e in errors[:3]:
            print(f"  {e}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None, help="Evaluate only this scene")
    parser.add_argument("--limit", type=int, default=None, help="Max jobs to evaluate")
    args = parser.parse_args()

    axis_df = pd.read_excel(AXIS_DEF, engine="openpyxl") if AXIS_DEF.exists() else None
    if axis_df is None:
        print("[warn] axis_definition.xlsx not found — falling back to global axes")

    scene_dirs = sorted(PROCESSED_ROOT.iterdir())
    if args.scene:
        scene_dirs = [d for d in scene_dirs if d.name == args.scene]

    results = []
    n_jobs  = 0

    # Cache scene axes (expensive to recompute per job)
    scene_axes_cache: dict[str, dict | None] = {}

    for scene_dir in scene_dirs:
        edits_dir = scene_dir / "edits"
        if not edits_dir.exists():
            continue

        # Compute per-scene axes once per scene
        if scene_dir.name not in scene_axes_cache:
            scene_axes_cache[scene_dir.name] = (
                load_scene_axes(scene_dir, axis_df) if axis_df is not None else None
            )
        scene_axes = scene_axes_cache[scene_dir.name]

        for manifest_path in sorted(edits_dir.glob("job*.json")):
            stem = manifest_path.stem
            if stem.endswith("_no_pointr") or stem.endswith("_with_pointr"):
                continue

            manifest = json.loads(manifest_path.read_text())
            if manifest.get("op") != "MOVE":
                continue

            if args.limit and n_jobs >= args.limit:
                break

            result = evaluate_job(manifest, scene_dir, scene_axes)
            if result:
                results.append(result)
            n_jobs += 1
            print(f"  [{n_jobs}] {scene_dir.name} {stem} → "
                  f"rel={'?' if not result or 'error' in result else result['relation_correct']}  "
                  f"pe={result.get('placement_error_m', 'err'):.3f}m"
                  if result and "error" not in result else
                  f"  [{n_jobs}] {scene_dir.name} {stem} → ERROR: {result}",
                  end="\r", flush=True)

        if args.limit and n_jobs >= args.limit:
            break

    print()
    print_summary(results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nFull results written: {OUT_PATH}")


if __name__ == "__main__":
    main()
