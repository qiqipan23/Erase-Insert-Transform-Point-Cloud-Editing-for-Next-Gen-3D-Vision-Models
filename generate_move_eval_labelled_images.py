#!/usr/bin/env python3
"""
Re-render MOVE eval images with instance label overlays.

For each entry in contextvqa_move_eval.json:
  1. Load the canonical_edit_file PLY (edited scene)
  2. Render top-down view using the same projection as render_top_view.py
  3. Project each instance centroid to pixel space
  4. Draw label text at each centroid
  5. Save to dataset/2D_VLM_data/move_edits_top_view_labelled/

Usage:
  python generate_move_eval_labelled_images.py [--limit N]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HYPO      = Path(__file__).resolve().parent
PROC_ROOT = HYPO / "repo/scannet/processed"
EVAL_JSON = HYPO / "dataset/contextvqa_move_eval.json"
OUT_DIR   = HYPO / "dataset/2D_VLM_data/move_edits_top_view_labelled"
IMAGE_SIZE = 800
PAD_FRAC   = 0.05
BG_COLOR   = (240, 240, 240)


# ── PLY loader ────────────────────────────────────────────────────────────────
def load_ply(path: Path):
    data = path.read_bytes()
    hi   = data.find(b"end_header")
    he   = hi + len(b"end_header") + 1
    if data[he - 1] != ord('\n'): he += 1
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
        dt  = np.dtype([(nm, TM.get(tp,"f4")) for tp,nm in props])
        arr = np.frombuffer(data[he:], dtype=dt, count=n)
    else:
        raw  = data[he:].decode("ascii","ignore").splitlines()
        rows = [list(map(float, l.split())) for l in raw[:n] if l.strip()]
        arr  = np.zeros(n, dtype=np.dtype([(nm, TM.get(tp,"f4")) for tp,nm in props]))
        for i, row in enumerate(rows[:n]):
            for j,(_, nm) in enumerate(props):
                if j < len(row): arr[nm][i] = row[j]
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    rgb = None
    for names in (("red","green","blue"), ("r","g","b")):
        if all(nm in arr.dtype.names for nm in names):
            r_ = np.column_stack([arr[nm] for nm in names]).astype(np.float64)
            rgb = (r_/r_.max()*255 if r_.max() > 1.5 else r_*255).astype(np.uint8)
            break
    return xyz, rgb


# ── Rendering ─────────────────────────────────────────────────────────────────
def world_to_pixel(xy_world, x_min, y_min, span_padded):
    """Return (px, py) arrays for world XY coordinates."""
    px = ((xy_world[:, 0] - x_min) / span_padded * (IMAGE_SIZE - 1)).astype(np.int32)
    py = ((1.0 - (xy_world[:, 1] - y_min) / span_padded) * (IMAGE_SIZE - 1)).astype(np.int32)
    return np.clip(px, 0, IMAGE_SIZE-1), np.clip(py, 0, IMAGE_SIZE-1)


def render_with_labels(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: list[tuple[np.ndarray, str]],   # [(centroid_xyz, label_text), ...]
) -> np.ndarray:
    x, y = xyz[:, 0], xyz[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    span    = max(x_max - x_min, y_max - y_min, 1e-3)
    pad     = span * PAD_FRAC
    x_min  -= pad; x_max += pad
    y_min  -= pad; y_max += pad
    span_p  = max(x_max - x_min, y_max - y_min)

    # Sort by Z for depth
    z = xyz[:, 2]; order = np.argsort(z)
    xs, ys, rgb_s = x[order], y[order], rgb[order]

    # Map to pixels
    px = ((xs - x_min) / span_p * (IMAGE_SIZE - 1)).astype(np.int32).clip(0, IMAGE_SIZE - 1)
    py = ((1.0 - (ys - y_min) / span_p) * (IMAGE_SIZE - 1)).astype(np.int32).clip(0, IMAGE_SIZE - 1)

    img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), BG_COLOR, dtype=np.uint8)
    img[py, px] = rgb_s
    for dy in range(2):
        for dx in range(2):
            py2 = (py + dy).clip(0, IMAGE_SIZE - 1)
            px2 = (px + dx).clip(0, IMAGE_SIZE - 1)
            mask = img[py2, px2, 0] == BG_COLOR[0]
            img[py2[mask], px2[mask]] = rgb_s[mask]

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = font_sm = ImageFont.load_default()

    for centroid, label_text in labels:
        cx = float(centroid[0]); cy = float(centroid[1])
        lpx = int((cx - x_min) / span_p * (IMAGE_SIZE - 1))
        lpy = int((1.0 - (cy - y_min) / span_p) * (IMAGE_SIZE - 1))
        lpx = max(5, min(IMAGE_SIZE - 5, lpx))
        lpy = max(5, min(IMAGE_SIZE - 5, lpy))

        # White background box + label
        bbox = draw.textbbox((lpx, lpy), label_text, font=font_sm)
        pad2 = 2
        draw.rectangle([bbox[0]-pad2, bbox[1]-pad2, bbox[2]+pad2, bbox[3]+pad2],
                        fill=(255, 255, 255, 200))
        draw.text((lpx, lpy), label_text, fill=(30, 30, 30), font=font_sm)
        # Small dot at centroid
        r = 3
        draw.ellipse([lpx-r, lpy-r, lpx+r, lpy+r], fill=(255, 80, 0))

    return np.array(pil)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    eval_data = json.loads(EVAL_JSON.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    for scene_id, changes in eval_data.items():
        scene_dir  = PROC_ROOT / scene_id
        out_scene  = OUT_DIR / scene_id
        out_scene.mkdir(parents=True, exist_ok=True)

        # Load original scene instance data once per scene
        try:
            xyz_orig  = np.load(scene_dir / "xyz.npy").astype(np.float32)
            rgb_orig  = (np.load(scene_dir / "rgb.npy") * 255).clip(0,255).astype(np.uint8)
            inst_orig = np.load(scene_dir / "inst.npy")
            label_map = json.loads((scene_dir / "inst_to_label.json").read_text())
        except Exception as e:
            print(f"[skip] {scene_id}: {e}")
            skipped += len(changes)
            continue

        for ch in changes:
            job_stem  = ch.get("job_stem", "")
            edit_ply  = Path(ch.get("image_path", "").replace(
                "move_edits_top_view", "").strip("/"))
            # actual PLY is in canonical_edit_file
            # image_path points to the rendered PNG; derive PLY from job manifest
            target_id = int(ch.get("target_instance_id", -1))
            anchor_id = int(ch.get("anchor_instance_id", -1))
            target_lbl = ch.get("target_label", "")
            anchor_lbl = ch.get("anchor_label", "")

            # output image path mirrors input
            in_img_path = HYPO / ch["image_path"]
            out_img_path = OUT_DIR / scene_id / in_img_path.name
            if out_img_path.exists():
                done += 1
                continue

            if not in_img_path.exists():
                skipped += 1
                continue

            # Load the existing rendered image (already has correct colours)
            img_arr = np.array(Image.open(in_img_path).convert("RGB"))

            # Build label list: all instances with >= 50 points
            labels = []
            unique_ids = np.unique(inst_orig)
            for iid in unique_ids:
                if iid == 0: continue
                pts = xyz_orig[inst_orig == iid]
                if len(pts) < 50: continue
                lbl = label_map.get(str(int(iid)), "")
                if not lbl: continue
                centroid = pts.mean(axis=0)
                # mark moved object distinctly
                marker = f"★{lbl}" if int(iid) == target_id else lbl
                labels.append((centroid, marker))

            # Project labels onto existing image
            pil = Image.fromarray(img_arr)
            draw = ImageDraw.Draw(pil)

            # Compute projection parameters from original scene extent
            x = xyz_orig[:, 0]; y = xyz_orig[:, 1]
            x_min, x_max = float(x.min()), float(x.max())
            y_min, y_max = float(y.min()), float(y.max())
            span  = max(x_max - x_min, y_max - y_min, 1e-3)
            pad   = span * PAD_FRAC
            x_min -= pad; x_max += pad
            y_min -= pad; y_max += pad
            span_p = max(x_max - x_min, y_max - y_min)

            try:
                font_sm   = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16)
                font_bold = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 17)
            except Exception:
                font_sm = font_bold = ImageFont.load_default()

            for centroid, label_text in labels:
                cx = float(centroid[0]); cy = float(centroid[1])
                lpx = int((cx - x_min) / span_p * (IMAGE_SIZE - 1))
                lpy = int((1.0 - (cy - y_min) / span_p) * (IMAGE_SIZE - 1))
                lpx = max(5, min(IMAGE_SIZE - 60, lpx))
                lpy = max(5, min(IMAGE_SIZE - 20, lpy))
                is_target = label_text.startswith("★")
                font_use  = font_bold if is_target else font_sm
                col       = (180, 30, 0) if is_target else (20, 20, 20)
                dot_col   = (255, 60, 0) if is_target else (60, 60, 220)
                p2 = 3
                bbox = draw.textbbox((lpx, lpy), label_text, font=font_use)
                draw.rectangle([bbox[0]-p2, bbox[1]-p2, bbox[2]+p2, bbox[3]+p2],
                                fill=(255, 255, 255))
                draw.text((lpx, lpy), label_text, fill=col, font=font_use)
                r = 4 if is_target else 3
                draw.ellipse([lpx-r, lpy-r, lpx+r, lpy+r], fill=dot_col)

            pil.save(out_img_path)
            done += 1
            print(f"[{done}] {scene_id}/{in_img_path.name}", end="\r", flush=True)

            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break

    print(f"\n[done] {done} images saved to {OUT_DIR}  ({skipped} skipped)")


if __name__ == "__main__":
    main()
