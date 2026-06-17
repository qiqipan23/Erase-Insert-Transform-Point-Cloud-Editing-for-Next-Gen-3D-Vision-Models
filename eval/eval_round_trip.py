#!/usr/bin/env python3
"""
Round-trip evaluation for the MOVE pipeline.

For each reverse job in dataset/reverse_jobs.jsonl:
  1. Load start_scene_ply  (forward-edited scene, object at B)
  2. Extract moved object at B using KDTree radius search
  3. Run compute_xy_translation(moved_pts_B, anchor_pts, relation, axes)
     → reverse_trans  (what the pipeline would compute)
  4. recovered_centroid_C = centroid(moved_pts_B) + reverse_trans
  5. round_trip_error = ||centroid_A − centroid_C||

Compares against:
  - ideal_error = 0 (if pipeline perfectly placed object at A)
  - forward placement_error for same jobs (from move_eval_geometric.json)

Output: dataset/round_trip_eval.json

Usage:
  python eval_round_trip.py [--limit N]
"""
from __future__ import annotations
import argparse, json, re, struct, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent / "repo/scannet"))
from placement import compute_xy_translation

HYPO_ROOT      = Path(__file__).resolve().parent
PROCESSED_ROOT = HYPO_ROOT / "repo/scannet/processed"
REVERSE_JOBS   = HYPO_ROOT / "dataset/reverse_jobs.jsonl"
OUT_PATH       = HYPO_ROOT / "dataset/round_trip_eval.json"
AXIS_DEF       = HYPO_ROOT / "dataset/axis_definition.xlsx"

PLACEMENT_MARGIN = 0.10    # same clearance as forward pass


# ── PLY loader ────────────────────────────────────────────────────────────────

def load_ply_xyz(path: Path) -> np.ndarray:
    data = path.read_bytes()
    hi   = data.find(b"end_header")
    he   = hi + len(b"end_header") + 1
    if data[he - 1] != ord('\n'):
        he += 1
    hdr  = data[:he].decode("ascii", errors="ignore")
    n    = int(re.search(r"element vertex (\d+)", hdr).group(1))
    bin_ = "binary_little_endian" in hdr
    props = []
    in_v  = False
    for line in hdr.splitlines():
        line = line.strip()
        if   line.startswith("element vertex"): in_v = True
        elif line.startswith("element"):        in_v = False
        elif in_v and line.startswith("property") and "list" not in line:
            p = line.split()
            if len(p) >= 3: props.append((p[1], p[2]))
    TM = {"float":"f4","float32":"f4","uchar":"u1","uint8":"u1",
          "int":"i4","uint":"u4","double":"f8"}
    if bin_:
        dt  = np.dtype([(nm, TM.get(tp, "f4")) for tp, nm in props])
        arr = np.frombuffer(data[he:], dtype=dt, count=n)
    else:
        raw  = data[he:].decode("ascii", errors="ignore").splitlines()
        rows = [list(map(float, l.split())) for l in raw[:n] if l.strip()]
        arr  = np.zeros(n, dtype=np.dtype([(nm, TM.get(tp,"f4")) for tp,nm in props]))
        for i, row in enumerate(rows[:n]):
            for j, (_, nm) in enumerate(props):
                if j < len(row): arr[nm][i] = row[j]
    return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)


# ── Scene axes ────────────────────────────────────────────────────────────────

from collections import defaultdict as _dd

def fallback_axes():
    return {
        "RIGHT_OF":  np.array([1., 0., 0.], np.float32),
        "LEFT_OF":   np.array([-1., 0., 0.], np.float32),
        "FRONT_OF":  np.array([0., 1., 0.], np.float32),
        "BACK_OF":   np.array([0., -1., 0.], np.float32),
        "ON_TOP_OF": np.array([0., 1., 0.], np.float32),
        "UNDER":     np.array([0., -1., 0.], np.float32),
    }


def load_scene_axes(scene_dir: Path, axis_df) -> dict:
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
    l2i: dict = _dd(list)
    for iid, lbl in label_map.items():
        l2i[lbl.lower()].append(int(iid))
    scene_center = xyz.mean(axis=0).copy(); scene_center[1] = 0

    def dir_to(lv):
        if not lv or (isinstance(lv, float) and np.isnan(lv)): return None
        key = str(lv).strip().lower()
        best_v, best_d = None, -1.0
        for iid in l2i.get(key, []):
            pts = xyz[inst == iid]
            if not len(pts): continue
            c = pts.mean(axis=0).copy(); c[1] = 0
            d = float(np.linalg.norm(c - scene_center))
            if d > best_d: best_d, best_v = d, c
        if best_v is None: return None
        v = best_v - scene_center; v[1] = 0
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 1e-6 else None

    front = dir_to(row.get("Front"))
    right = dir_to(row.get("Right"))
    if front is None and right is None: return fallback_axes()
    if front is None: front = np.array([-right[1], right[0], 0.], np.float32)
    if right is None: right = np.array([front[1], -front[0], 0.], np.float32)
    front[1] = 0; front /= (np.linalg.norm(front) + 1e-8)
    right = right - (right @ front) * front
    right[1] = 0; right /= (np.linalg.norm(right) + 1e-8)
    return {
        "RIGHT_OF": right, "LEFT_OF": -right,
        "FRONT_OF": front, "BACK_OF": -front,
        "ON_TOP_OF": np.array([0.,1.,0.],np.float32),
        "UNDER":     np.array([0.,-1.,0.],np.float32),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_reverse_job(job: dict, scene_dir: Path, axes: dict) -> dict:
    centroid_A = np.array(job["original_centroid_A"], dtype=np.float32)
    target_id  = int(job["target_instance_id"])
    anchor_id  = int(job["anchor_instance_id"])
    relation   = job["relation"]

    # load original scene for anchor points
    try:
        xyz_orig  = np.load(scene_dir / "xyz.npy").astype(np.float32)
        inst_orig = np.load(scene_dir / "inst.npy")
    except Exception as e:
        return {"error": f"load xyz failed: {e}"}

    anchor_pts = xyz_orig[inst_orig == anchor_id]
    if len(anchor_pts) == 0:
        return {"error": f"anchor {anchor_id} not found in scene"}

    # load edited scene (object at B)
    start_ply = Path(job["start_scene_ply"])
    if not start_ply.exists():
        return {"error": f"start_scene_ply not found: {start_ply}"}
    try:
        edit_pts = load_ply_xyz(start_ply)
    except Exception as e:
        return {"error": f"load start_scene_ply failed: {e}"}

    # expected position B = centroid_A + forward_translation
    fwd_trans  = np.array(job["ideal_rev_translation"], dtype=np.float32)
    centroid_B = centroid_A + (-fwd_trans)   # = A + forward_translation

    # extract moved object at B using radius search
    orig_target_pts = xyz_orig[inst_orig == target_id]
    if len(orig_target_pts) == 0:
        return {"error": f"target {target_id} not found in original scene"}
    obj_radius = float(np.linalg.norm(
        orig_target_pts.max(axis=0) - orig_target_pts.min(axis=0)
    )) / 2 + 0.3

    tree = KDTree(edit_pts)
    idx  = tree.query_ball_point(centroid_B, r=obj_radius)
    if len(idx) < 10:
        return {"error": f"too few points found at B ({len(idx)} pts, radius={obj_radius:.2f}m)"}
    moved_pts_B = edit_pts[idx]

    # run spatial resolver in reverse
    try:
        reverse_trans = compute_xy_translation(
            target_xyz = moved_pts_B,
            anchor_xyz = anchor_pts,
            relation   = relation,
            axes       = axes,
            margin     = PLACEMENT_MARGIN,
        )
    except Exception as e:
        return {"error": f"compute_xy_translation failed: {e}"}

    centroid_C = moved_pts_B.mean(axis=0) + reverse_trans

    rte_3d = float(np.linalg.norm(centroid_A - centroid_C))
    # horizontal only (XZ plane, ignoring Y)
    delta_xz = centroid_A[[0,2]] - centroid_C[[0,2]]
    rte_xz   = float(np.linalg.norm(delta_xz))

    return {
        "scene_id":            job["scene_id"],
        "forward_job_stem":    job["forward_job_stem"],
        "target_label":        job["target_label"],
        "anchor_label":        job["anchor_label"],
        "relation":            relation,
        "reverse_confidence":  job["reverse_confidence"],
        "n_pts_at_B":          len(moved_pts_B),
        "round_trip_error_m":  rte_3d,
        "round_trip_error_xz_m": rte_xz,
        "centroid_A":          centroid_A.tolist(),
        "centroid_C":          centroid_C.tolist(),
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    valid  = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"\n{'='*60}")
    print(f"Round-Trip Evaluation  —  {len(valid)} valid, {len(errors)} errors")
    print(f"{'='*60}")

    rte    = [r["round_trip_error_m"]    for r in valid]
    rte_xz = [r["round_trip_error_xz_m"] for r in valid]
    confs  = [r["reverse_confidence"]    for r in valid]

    print(f"\n{'Metric':<35} Value")
    print("-" * 50)
    print(f"{'Round-trip error (3D)':<35} {np.mean(rte):.4f} m")
    print(f"{'Round-trip error (XZ, horiz.)':<35} {np.mean(rte_xz):.4f} m")
    print(f"{'Mean reverse confidence':<35} {np.mean(confs):.3f}")
    print(f"{'Jobs within 0.20 m (XZ)':<35} {np.mean(np.array(rte_xz) < 0.20)*100:.1f}%")
    print(f"{'Jobs within 0.50 m (XZ)':<35} {np.mean(np.array(rte_xz) < 0.50)*100:.1f}%")

    by_rel: dict = defaultdict(list)
    for r in valid: by_rel[r["relation"]].append(r)

    print(f"\n{'Relation':<14}{'N':>4}{'RTE_XZ(m)':>12}{'Conf':>8}")
    print("-" * 42)
    for rel, items in sorted(by_rel.items()):
        rte_r = [i["round_trip_error_xz_m"] for i in items]
        conf_r = [i["reverse_confidence"]   for i in items]
        print(f"{rel:<14}{len(items):>4}{np.mean(rte_r):>12.4f}{np.mean(conf_r):>8.3f}")

    if errors:
        print(f"\nFirst 5 errors:")
        for e in errors[:5]:
            print(f"  {e.get('forward_job_stem','?')}: {e['error']}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not REVERSE_JOBS.exists():
        print(f"[error] {REVERSE_JOBS} not found — run generate_reverse_jobs.py first")
        sys.exit(1)

    jobs = [json.loads(l) for l in REVERSE_JOBS.read_text().splitlines() if l.strip()]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"Loaded {len(jobs)} reverse jobs")

    axis_df = pd.read_excel(AXIS_DEF, engine="openpyxl") if AXIS_DEF.exists() else None
    axes_cache: dict = {}
    results = []

    for i, job in enumerate(jobs):
        scene_id  = job["scene_id"]
        scene_dir = PROCESSED_ROOT / scene_id
        if scene_id not in axes_cache:
            axes_cache[scene_id] = load_scene_axes(scene_dir, axis_df)
        axes = axes_cache[scene_id]

        r = evaluate_reverse_job(job, scene_dir, axes)
        results.append(r)

        if "error" not in r:
            print(f"[{i+1:4d}] {scene_id}/{job['forward_job_stem']}"
                  f"  rel={r['relation']:<12}"
                  f"  RTE_XZ={r['round_trip_error_xz_m']:.3f}m"
                  f"  conf={r['reverse_confidence']:.2f}",
                  end="\r", flush=True)
        else:
            print(f"[{i+1:4d}] {scene_id}/{job['forward_job_stem']}  ERROR: {r['error']}")

    print()
    print_summary(results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults written: {OUT_PATH}")


if __name__ == "__main__":
    main()
