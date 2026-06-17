#!/usr/bin/env python3
"""
Generate reverse job manifests for round-trip evaluation.

For each processed forward MOVE job:
  1. Load the original scene (xyz.npy, inst.npy)
  2. Compute centroid_A of the target object
  3. Score every other instance as a candidate anchor:
       v = centroid_A - anchor_centroid  (horizontal plane)
       confidence = dominant_axis_score / ||v||
  4. Pick the anchor + relation with highest confidence
  5. Write a reverse job manifest in the same format as forward jobs

Output: dataset/reverse_jobs.jsonl  (one JSON object per line)

Usage:
  python generate_reverse_jobs.py [--limit N] [--scene scene0000_00]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

HYPO_ROOT     = Path(__file__).resolve().parent
SCANNET_ROOT  = HYPO_ROOT / "repo" / "scannet"
PROCESSED_ROOT = SCANNET_ROOT / "processed"
AXIS_DEF      = HYPO_ROOT / "dataset" / "axis_definition.xlsx"
OUT_PATH      = HYPO_ROOT / "dataset" / "reverse_jobs.jsonl"

UP = np.array([0.0, 1.0, 0.0])

HORIZONTAL_RELATIONS = ["RIGHT_OF", "LEFT_OF", "FRONT_OF", "BACK_OF"]
VERTICAL_RELATIONS   = ["ON_TOP_OF", "UNDER"]
MIN_ANCHOR_PTS = 30       # ignore tiny/sparse instances
MIN_ANCHOR_DIST = 0.3     # ignore anchors very close to the target


# ── Scene axes (reused from eval_move_metrics.py) ────────────────────────────

def fallback_axes():
    return {
        "RIGHT_OF":  np.array([1., 0., 0.], dtype=np.float32),
        "LEFT_OF":   np.array([-1., 0., 0.], dtype=np.float32),
        "FRONT_OF":  np.array([0., 1., 0.], dtype=np.float32),
        "BACK_OF":   np.array([0., -1., 0.], dtype=np.float32),
        "ON_TOP_OF": np.array([0., 0., 1.], dtype=np.float32),
        "UNDER":     np.array([0., 0., -1.], dtype=np.float32),
    }


def load_scene_axes(scene_dir: Path, axis_df: pd.DataFrame | None) -> dict:
    """Return relation→unit_vector dict for the scene."""
    if axis_df is None:
        return fallback_axes()

    scene_id = scene_dir.name
    rows = axis_df[axis_df["scene_id"] == scene_id]
    if rows.empty:
        return fallback_axes()

    row = rows.iloc[0]
    xyz  = np.load(scene_dir / "xyz.npy")
    inst = np.load(scene_dir / "inst.npy")
    label_map = json.loads((scene_dir / "inst_to_label.json").read_text())
    label_to_insts: dict[str, list[int]] = defaultdict(list)
    for iid, lbl in label_map.items():
        label_to_insts[lbl.lower()].append(int(iid))

    scene_center = xyz.mean(axis=0).copy()
    scene_center[1] = 0

    def dir_to(label_val):
        if not label_val or (isinstance(label_val, float) and np.isnan(label_val)):
            return None
        key = str(label_val).strip().lower()
        best_v, best_d = None, -1.0
        for iid in label_to_insts.get(key, []):
            pts = xyz[inst == iid]
            if len(pts) == 0:
                continue
            c = pts.mean(axis=0).copy(); c[1] = 0
            d = float(np.linalg.norm(c - scene_center))
            if d > best_d:
                best_d, best_v = d, c
        if best_v is None:
            return None
        v = best_v - scene_center
        v[1] = 0
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 1e-6 else None

    front = dir_to(row.get("Front"))
    right = dir_to(row.get("Right"))

    if front is None and right is None:
        return fallback_axes()
    if front is None:
        front = np.array([-right[1], right[0], 0.], dtype=np.float32)
    if right is None:
        right = np.array([front[1], -front[0], 0.], dtype=np.float32)

    front[1] = 0; front = front / (np.linalg.norm(front) + 1e-8)
    right = right - (right @ front) * front
    right[1] = 0; right = right / (np.linalg.norm(right) + 1e-8)

    return {
        "RIGHT_OF":  right,
        "LEFT_OF":  -right,
        "FRONT_OF":  front,
        "BACK_OF":  -front,
        "ON_TOP_OF": np.array([0., 1., 0.], dtype=np.float32),
        "UNDER":     np.array([0., -1., 0.], dtype=np.float32),
    }


# ── Core: find best anchor + relation ────────────────────────────────────────

PLACEMENT_MARGIN = 0.10   # same clearance as the forward pipeline


def derive_best_anchor(
    centroid_A: np.ndarray,
    target_pts: np.ndarray,
    target_id: int,
    xyz: np.ndarray,
    inst: np.ndarray,
    label_map: dict,
    axes: dict,
) -> dict | None:
    """
    Score every candidate anchor by:

        combined = directional_confidence × adjacency_score

    directional_confidence = how cleanly v aligns to one axis  (0–1)
    adjacency_score        = 1 / (1 + gap)
        gap = |A_proj − (anchor_outer_edge + object_half + margin)|
            = how far A is from where the placement formula would put it

    Anchors that are both well-aligned AND adjacent to A score highest,
    ensuring the reverse placement formula naturally lands near A.
    """
    best = None
    unique_ids = np.unique(inst)

    for anchor_id in unique_ids:
        if int(anchor_id) == int(target_id) or int(anchor_id) == 0:
            continue

        anchor_pts = xyz[inst == anchor_id]
        if len(anchor_pts) < MIN_ANCHOR_PTS:
            continue

        anchor_centroid = anchor_pts.mean(axis=0)
        v    = centroid_A - anchor_centroid
        dist = float(np.linalg.norm(v))
        if dist < MIN_ANCHOR_DIST:
            continue

        # ── Horizontal relations ──────────────────────────────────────────
        v_horiz    = v.copy(); v_horiz[1] = 0
        horiz_norm = float(np.linalg.norm(v_horiz))

        for rel in HORIZONTAL_RELATIONS:
            axis  = axes[rel]
            score = float(v_horiz @ axis)
            if score <= 0:
                continue

            dir_conf = score / (horiz_norm + 1e-8)   # 0–1

            # Extent of anchor in this direction (furthest surface toward A)
            anchor_outer = float((anchor_pts @ axis).max())
            # Half-extent of target along this axis
            obj_half     = 0.5 * float((target_pts @ axis).max()
                                       - (target_pts @ axis).min())
            # Where the formula would place centroid_A
            formula_pos  = anchor_outer + obj_half + PLACEMENT_MARGIN
            # Gap between formula position and actual A
            a_proj       = float(centroid_A @ axis)
            gap          = abs(a_proj - formula_pos)

            adjacency    = 1.0 / (1.0 + gap)
            combined     = dir_conf * adjacency

            if best is None or combined > best["combined_score"]:
                best = {
                    "anchor_id":       int(anchor_id),
                    "anchor_label":    label_map.get(str(anchor_id), "unknown"),
                    "relation":        rel,
                    "dir_conf":        float(dir_conf),
                    "adjacency_score": float(adjacency),
                    "combined_score":  float(combined),
                    "gap_m":           float(gap),
                    "dist_m":          float(dist),
                    "anchor_centroid": anchor_centroid.tolist(),
                }

        # ── Vertical relations ────────────────────────────────────────────
        v_vert = float(v[1])
        if abs(v_vert) < 1e-6:
            continue
        rel_v    = "ON_TOP_OF" if v_vert > 0 else "UNDER"
        axis_v   = axes[rel_v]
        dir_conf_v = abs(v_vert) / (dist + 1e-8)

        anchor_outer_v = float((anchor_pts @ axis_v).max())
        obj_half_v     = 0.5 * float((target_pts @ axis_v).max()
                                     - (target_pts @ axis_v).min())
        formula_pos_v  = anchor_outer_v + obj_half_v + PLACEMENT_MARGIN
        a_proj_v       = float(centroid_A @ axis_v)
        gap_v          = abs(a_proj_v - formula_pos_v)
        adjacency_v    = 1.0 / (1.0 + gap_v)
        combined_v     = dir_conf_v * adjacency_v

        if best is None or combined_v > best["combined_score"]:
            best = {
                "anchor_id":       int(anchor_id),
                "anchor_label":    label_map.get(str(anchor_id), "unknown"),
                "relation":        rel_v,
                "dir_conf":        float(dir_conf_v),
                "adjacency_score": float(adjacency_v),
                "combined_score":  float(combined_v),
                "gap_m":           float(gap_v),
                "dist_m":          float(dist),
                "anchor_centroid": anchor_centroid.tolist(),
            }

    return best


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scene", default=None)
    args = ap.parse_args()

    axis_df = pd.read_excel(AXIS_DEF, engine="openpyxl") if AXIS_DEF.exists() else None
    if axis_df is None:
        print("[warn] axis_definition.xlsx not found — using fallback axes")

    scene_dirs = sorted(PROCESSED_ROOT.iterdir())
    if args.scene:
        scene_dirs = [d for d in scene_dirs if d.name == args.scene]

    records = []
    axes_cache: dict[str, dict] = {}
    n_jobs = 0

    for scene_dir in scene_dirs:
        edits_dir = scene_dir / "edits"
        if not edits_dir.exists():
            continue

        # load scene arrays once per scene
        try:
            xyz  = np.load(scene_dir / "xyz.npy").astype(np.float32)
            inst = np.load(scene_dir / "inst.npy")
            label_map = json.loads((scene_dir / "inst_to_label.json").read_text())
        except Exception as e:
            print(f"[skip] {scene_dir.name}: {e}")
            continue

        if scene_dir.name not in axes_cache:
            axes_cache[scene_dir.name] = load_scene_axes(scene_dir, axis_df)
        axes = axes_cache[scene_dir.name]

        for manifest_path in sorted(edits_dir.glob("job*.json")):
            stem = manifest_path.stem
            if stem.endswith(("_no_pointr", "_with_pointr")):
                continue

            manifest = json.loads(manifest_path.read_text())
            if manifest.get("op") != "MOVE":
                continue

            # forward job must have a valid edited PLY
            fwd_ply = Path(manifest.get("canonical_edit_file", ""))
            if not fwd_ply.exists():
                continue

            target_id = int(manifest["target_instance_id"])
            target_pts = xyz[inst == target_id]
            if len(target_pts) == 0:
                continue

            centroid_A = target_pts.mean(axis=0)

            best = derive_best_anchor(
                centroid_A, target_pts, target_id, xyz, inst, label_map, axes
            )
            if best is None:
                print(f"[skip] {scene_dir.name} {stem}: no valid anchor found")
                continue

            # ideal reverse translation: from B back to A
            fwd_translation = np.array(manifest["translation"], dtype=np.float32)
            ideal_rev_translation = (-fwd_translation).tolist()

            record = {
                # ── identity ──────────────────────────────────────────────
                "op":                  "MOVE",
                "scene_id":            manifest["scene_id"],
                "forward_job_stem":    stem,
                "forward_job_index":   manifest.get("job_index"),
                # ── target (same as forward) ──────────────────────────────
                "target_instance_id":  target_id,
                "target_label":        manifest.get("target_label", ""),
                "original_centroid_A": centroid_A.tolist(),
                # ── reverse anchor (derived from scene geometry) ──────────
                "anchor_instance_id":  best["anchor_id"],
                "anchor_label":        best["anchor_label"],
                "relation":            best["relation"],
                "reverse_confidence":  best["combined_score"],
                "dir_conf":            best["dir_conf"],
                "adjacency_score":     best["adjacency_score"],
                "gap_m":               best["gap_m"],
                "anchor_dist_m":       best["dist_m"],
                # ── file references ───────────────────────────────────────
                "start_scene_ply":     str(fwd_ply),   # object at B
                # ideal translation for reference (not used by pipeline)
                "ideal_rev_translation": ideal_rev_translation,
            }

            records.append(record)
            n_jobs += 1
            print(f"[{n_jobs:4d}] {scene_dir.name}/{stem}  "
                  f"→ reverse: {best['relation']} {best['anchor_label']}  "
                  f"combined={best['combined_score']:.3f}  "
                  f"dir={best['dir_conf']:.2f}  adj={best['adjacency_score']:.2f}  "
                  f"gap={best['gap_m']:.2f}m")

            if args.limit and n_jobs >= args.limit:
                break

        if args.limit and n_jobs >= args.limit:
            break

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n[done] {n_jobs} reverse jobs written to {OUT_PATH}")

    # summary of relation distribution
    from collections import Counter
    rels = Counter(r["relation"] for r in records)
    confs = [r["reverse_confidence"] for r in records]
    print(f"Relation distribution: {dict(rels)}")
    print(f"Confidence: mean={np.mean(confs):.3f}  min={np.min(confs):.3f}  max={np.max(confs):.3f}")


if __name__ == "__main__":
    main()
