#!/usr/bin/env python3
"""
Post-pipeline step for the v2 rerun.

For each scene in the move eval, finds the *_v2.ply files produced by
run_edit_jobs.py --out_suffix _v2, renders them with rerender_denser.py,
then builds dataset/contextvqa_move_eval_dense_v2.json referencing
dataset/2D_VLM_data/move_edits_top_view_dense_v2/.

Usage (after all PBS jobs finish):
  cd /rds/general/user/qp23/home/Hypo3D
  source miniconda3/etc/profile.d/conda.sh && conda activate placeit3d
  python rerender_and_build_eval_v2.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HYPO         = Path(__file__).resolve().parent
PROC_ROOT    = HYPO / "repo/scannet/processed"
EVAL_JSON_V1 = HYPO / "dataset/contextvqa_move_eval_dense.json"
OUT_IMG_DIR  = HYPO / "dataset/2D_VLM_data/move_edits_top_view_dense_v2"
OUT_JSON     = HYPO / "dataset/contextvqa_move_eval_dense_v2.json"
RADIUS       = 3   # disk-splat radius (same as original dense render)
IMG_SIZE     = 800


# ── PLY loader (identical to rerender_denser.py) ──────────────────────────────
def load_ply(path: Path):
    data = path.read_bytes()
    hi   = data.find(b"end_header")
    nl   = data[hi + len("end_header")]
    he   = hi + len("end_header") + (2 if nl == ord('\r') else 1)
    hdr  = data[:he].decode("ascii", "ignore")
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
    TM = {"float": "f4", "float32": "f4", "double": "f8",
          "uchar": "u1", "uint8": "u1", "int": "i4", "uint": "u4"}
    if bin_:
        dt  = np.dtype([(nm, TM.get(tp, "f4")) for tp, nm in props])
        arr = np.frombuffer(data[he:], dtype=dt, count=n)
    else:
        rows = [list(map(float, l.split()))
                for l in data[he:].decode("ascii", "ignore").splitlines()[:n]
                if l.strip()]
        arr = np.rec.fromarrays(
            np.array(rows).T,
            names=[nm for _, nm in props],
        )
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    rgb = None
    for names in (("red", "green", "blue"), ("r", "g", "b")):
        if all(nm in arr.dtype.names for nm in names):
            r_ = np.column_stack([arr[nm] for nm in names]).astype(np.float64)
            rgb = (r_ / r_.max() * 255 if r_.max() > 1.5 else r_ * 255).astype(np.uint8)
            break
    if rgb is None:
        rgb = np.full((len(xyz), 3), 150, np.uint8)
    return xyz, rgb


def render_dense(xyz, rgb, size=IMG_SIZE, pad=0.05, radius=RADIUS, bg=(240, 240, 240)):
    x, y = xyz[:, 0], xyz[:, 1]
    xmn, xmx = x.min(), x.max()
    ymn, ymx = y.min(), y.max()
    span  = max(xmx - xmn, ymx - ymn, 1e-3)
    p     = span * pad
    xmn -= p; xmx += p; ymn -= p; ymx += p
    spanp = max(xmx - xmn, ymx - ymn)
    order = np.argsort(xyz[:, 2])
    x, y, rgb = x[order], y[order], rgb[order]
    px = ((x - xmn) / spanp * (size - 1)).astype(np.int32).clip(0, size - 1)
    py = ((1.0 - (y - ymn) / spanp) * (size - 1)).astype(np.int32).clip(0, size - 1)
    img = np.full((size, size, 3), bg, np.uint8)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            qx = (px + dx).clip(0, size - 1)
            qy = (py + dy).clip(0, size - 1)
            img[qy, qx] = rgb
    return img


def main():
    with open(EVAL_JSON_V1) as f:
        v1_data = json.load(f)

    rendered  = 0
    missing   = 0
    v2_data: dict = {}

    for scene_id, entries in v1_data.items():
        edits_dir    = PROC_ROOT / scene_id / "edits"
        scene_out    = OUT_IMG_DIR / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            job_stem = entry["job_stem"]
            # v2 PLY has same stem but with _v2 suffix
            ply_path = edits_dir / f"{job_stem}_v2.ply"
            if not ply_path.exists():
                print(f"  [missing] {scene_id}/{job_stem}_v2.ply")
                missing += 1
                continue

            out_png = scene_out / f"{job_stem}.png"
            xyz, rgb = load_ply(ply_path)
            img = render_dense(xyz, rgb)
            Image.fromarray(img).save(out_png)
            rendered += 1

            new_entry = dict(entry)
            new_entry["image_path"] = str(out_png.relative_to(HYPO))
            v2_data.setdefault(scene_id, []).append(new_entry)

        if rendered % 50 == 0:
            print(f"  rendered {rendered} so far ...", flush=True)

    OUT_JSON.write_text(json.dumps(v2_data, indent=2, ensure_ascii=False))
    print(f"\nDone: rendered={rendered}  missing={missing}")
    print(f"Written: {OUT_JSON}")
    total_qa = sum(len(e["questions_answers"]) for es in v2_data.values() for e in es)
    print(f"Scenes: {len(v2_data)}  QA pairs: {total_qa}")


if __name__ == "__main__":
    main()
