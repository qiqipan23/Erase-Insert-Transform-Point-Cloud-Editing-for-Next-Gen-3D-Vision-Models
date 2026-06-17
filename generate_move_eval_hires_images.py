#!/usr/bin/env python3
"""
Re-render MOVE eval labelled images at 1600×1600 with larger disk radius (5).
Reads:  dataset/contextvqa_move_eval_labelled.json
Writes: dataset/2D_VLM_data/move_edits_top_view_labelled_hi/<scene>/<job>.png
        dataset/contextvqa_move_eval_labelled_hi.json  (same structure, updated image_path)

Usage:
  source miniconda3/etc/profile.d/conda.sh && conda activate placeit3d
  python generate_move_eval_hires_images.py [--limit N]
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HYPO      = Path(__file__).resolve().parent
PROC_ROOT = HYPO / "repo/scannet/processed"
EVAL_JSON = HYPO / "dataset" / "contextvqa_move_eval_labelled.json"
OUT_DIR   = HYPO / "dataset" / "2D_VLM_data" / "move_edits_top_view_labelled_hi"
OUT_JSON  = HYPO / "dataset" / "contextvqa_move_eval_labelled_hi.json"

IMAGE_SIZE = 1600
RADIUS     = 5
PAD_FRAC   = 0.05
BG_COLOR   = (240, 240, 240)


def load_ply(path: Path):
    data = path.read_bytes()
    hi   = data.find(b"end_header")
    nl   = data[hi + len("end_header")]
    he   = hi + len("end_header") + (2 if nl == ord('\r') else 1)
    hdr  = data[:he].decode("ascii", errors="ignore")
    n    = int(re.search(r"element vertex (\d+)", hdr).group(1))
    bin_ = "binary_little_endian" in hdr
    props = []; in_v = False
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
        dt  = np.dtype([(nm, TM.get(tp,"f4")) for tp,nm in props])
        arr = np.frombuffer(data[he:], dtype=dt, count=n)
    else:
        raw  = data[he:].decode("ascii","ignore").splitlines()
        rows = [list(map(float, l.split())) for l in raw[:n] if l.strip()]
        arr  = np.zeros(n, dtype=np.dtype([(nm, TM.get(tp,"f4")) for tp,nm in props]))
        for i, row in enumerate(rows[:n]):
            for j, (_, nm) in enumerate(props):
                if j < len(row): arr[nm][i] = row[j]
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    rgb = None
    for names in (("red","green","blue"), ("r","g","b")):
        if all(nm in arr.dtype.names for nm in names):
            r_ = np.column_stack([arr[nm] for nm in names]).astype(np.float64)
            rgb = (r_/r_.max()*255 if r_.max()>1.5 else r_*255).astype(np.uint8)
            break
    return xyz, rgb


def render_disk_splat(xyz, rgb, size=IMAGE_SIZE, pad=PAD_FRAC, radius=RADIUS):
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    xmn, xmx = x.min(), x.max()
    ymn, ymx = y.min(), y.max()
    span = max(xmx - xmn, ymx - ymn, 1e-3)
    p = span * pad
    xmn -= p; xmx += p; ymn -= p; ymx += p
    spanp = max(xmx - xmn, ymx - ymn)
    # draw higher-z points on top
    order = np.argsort(z)
    x, y, rgb = x[order], y[order], rgb[order]
    px = ((x - xmn) / spanp * (size - 1)).astype(np.int32).clip(0, size-1)
    py = ((1.0 - (y - ymn) / spanp) * (size - 1)).astype(np.int32).clip(0, size-1)
    img = np.full((size, size, 3), BG_COLOR, np.uint8)
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            if dx*dx + dy*dy > radius*radius: continue
            qx = (px + dx).clip(0, size-1)
            qy = (py + dy).clip(0, size-1)
            img[qy, qx] = rgb
    return img, (xmn, ymn, spanp)


def project_xy(world_xy, xmn, ymn, spanp, size=IMAGE_SIZE):
    px = ((world_xy[:,0] - xmn) / spanp * (size - 1)).astype(np.int32).clip(0, size-1)
    py = ((1.0 - (world_xy[:,1] - ymn) / spanp) * (size - 1)).astype(np.int32).clip(0, size-1)
    return px, py


def render_with_labels(ply_path: Path, inst_path: Path | None, entry: dict) -> np.ndarray:
    xyz, rgb = load_ply(ply_path)
    if rgb is None:
        rgb = np.full((len(xyz), 3), 150, np.uint8)

    img_arr, (xmn, ymn, spanp) = render_disk_splat(xyz, rgb)
    pil = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil)

    # Load instance info for label overlay
    inst_npy = PROC_ROOT / entry.get("scene_id", "") / "inst.npy"
    xyz_npy  = PROC_ROOT / entry.get("scene_id", "") / "xyz.npy"
    inst_to_label_json = PROC_ROOT / entry.get("scene_id", "") / "inst_to_label.json"

    if inst_npy.exists() and xyz_npy.exists() and inst_to_label_json.exists():
        inst_all = np.load(inst_npy).astype(np.int32)
        xyz_all  = np.load(xyz_npy).astype(np.float32)
        with open(inst_to_label_json) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except Exception:
            font = ImageFont.load_default()

        for inst_id, label in label_map.items():
            m = inst_all == inst_id
            if m.sum() < 10: continue
            cx = float(xyz_all[m, 0].mean())
            cy = float(xyz_all[m, 1].mean())
            world_xy = np.array([[cx, cy]])
            px, py = project_xy(world_xy, xmn, ymn, spanp)
            draw.text((int(px[0]), int(py[0])), label,
                      fill=(255, 50, 50), font=font,
                      stroke_width=2, stroke_fill=(0,0,0))

    return np.array(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = json.loads(EVAL_JSON.read_text())
    out_data = {}
    count = 0

    for scene_id, entries in data.items():
        out_entries = []
        for entry in entries:
            if args.limit and count >= args.limit:
                break

            canonical = entry.get("canonical_edit_file", "")
            ply_path  = Path(canonical) if canonical else None

            if not ply_path or not ply_path.exists():
                # fall back to image_path-derived PLY name
                job_stem = entry.get("job_stem", "")
                ply_path = PROC_ROOT / scene_id / "edits" / f"{job_stem}.ply"

            if not ply_path or not ply_path.exists():
                print(f"  [skip] {scene_id} — PLY not found")
                out_entries.append(entry)
                continue

            out_scene_dir = OUT_DIR / scene_id
            out_scene_dir.mkdir(parents=True, exist_ok=True)
            job_stem = entry.get("job_stem", Path(ply_path).stem)
            out_img  = out_scene_dir / f"{job_stem}.png"

            if not out_img.exists():
                img_arr = render_with_labels(ply_path, None, {**entry, "scene_id": scene_id})
                Image.fromarray(img_arr).save(out_img)
                print(f"  Rendered {scene_id}/{job_stem}.png  ({IMAGE_SIZE}×{IMAGE_SIZE})")
            else:
                print(f"  [exists] {scene_id}/{job_stem}.png")

            new_entry = dict(entry)
            new_entry["image_path"] = str(out_img.relative_to(HYPO))
            out_entries.append(new_entry)
            count += 1

        out_data[scene_id] = out_entries
        if args.limit and count >= args.limit:
            break

    OUT_JSON.write_text(json.dumps(out_data, indent=2))
    print(f"\nDone. {count} images rendered.")
    print(f"Eval JSON written to: {OUT_JSON}")


if __name__ == "__main__":
    main()
