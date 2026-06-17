#!/usr/bin/env python3
"""
Visualize MOVE pipeline side-by-side: original vs geometric vs VLM placement.

For each job, renders three top-down panels:
  1. Original scene  (target at original position, highlighted orange)
  2. Geometric MOVE  (target moved by rule-based translation)
  3. VLM MOVE        (target moved by GPT-4o predicted position)

Usage:
  # Compare job 0 and job 1, auto-detect PLY suffixes
  python visualize_move_comparison.py --jobs 0 1

  # Explicit geo/vlm suffixes
  python visualize_move_comparison.py --jobs 0 1 --geo-suffix "" --vlm-suffix "_vlm_test"

  # Save to custom path
  python visualize_move_comparison.py --jobs 0 1 --output my_comparison.png
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

HYPO_ROOT      = Path(__file__).resolve().parent
PROCESSED_ROOT = HYPO_ROOT / "repo" / "scannet" / "processed"

TARGET_COLOR = np.array([255, 140,   0], dtype=np.uint8)
ANCHOR_COLOR = np.array([  0, 200, 220], dtype=np.uint8)


# ── PLY reader ────────────────────────────────────────────────────────────────

def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xyz float32 [N,3], rgb uint8 [N,3])."""
    with path.open("rb") as f:
        # parse header
        props = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))  # (type, name)
            elif line == "end_header":
                break
        # build structured dtype
        type_map = {"float": "f4", "float32": "f4", "uchar": "u1", "uint8": "u1",
                    "double": "f8", "int": "i4", "int32": "i4",
                    "short": "i2", "ushort": "u2"}
        dtype = np.dtype([(name, type_map.get(tp, "f4")) for tp, name in props])
        data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype)

    xyz = np.stack([data["x"].astype(np.float32),
                    data["y"].astype(np.float32),
                    data["z"].astype(np.float32)], axis=1)
    names = [name for _, name in props]
    r = data["red"]   if "red"   in names else (data["r"] if "r" in names else None)
    g = data["green"] if "green" in names else (data["g"] if "g" in names else None)
    b = data["blue"]  if "blue"  in names else (data["b"] if "b" in names else None)
    if r is None:
        rgb = np.full((n, 3), 180, dtype=np.uint8)
    else:
        rgb = np.stack([np.clip(r, 0, 255).astype(np.uint8),
                        np.clip(g, 0, 255).astype(np.uint8),
                        np.clip(b, 0, 255).astype(np.uint8)], axis=1)
    return xyz, rgb


# ── top-down renderer ─────────────────────────────────────────────────────────

def render_topdown(xyz: np.ndarray, rgb: np.ndarray,
                   image_size: int = 512,
                   pad_frac: float = 0.05) -> np.ndarray:
    """Render a top-down view; returns HxWx3 uint8 array."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    span = max(x_max - x_min, y_max - y_min, 1e-3)
    pad = span * pad_frac
    x_min -= pad; y_min -= pad; span += 2 * pad

    order = np.argsort(z)
    x, y, rgb_s = x[order], y[order], rgb[order]

    px = ((x - x_min) / span * (image_size - 1)).astype(np.int32).clip(0, image_size - 1)
    py = ((1.0 - (y - y_min) / span) * (image_size - 1)).astype(np.int32).clip(0, image_size - 1)

    img = np.full((image_size, image_size, 3), 240, dtype=np.uint8)
    img[py, px] = rgb_s
    for dy in range(2):
        for dx in range(2):
            img[(py + dy).clip(0, image_size - 1),
                (px + dx).clip(0, image_size - 1)] = rgb_s
    return img


# ── scene-array loader ────────────────────────────────────────────────────────

def load_scene_arrays(scene_dir: Path):
    xyz  = np.load(scene_dir / "xyz.npy").astype(np.float32)
    rgb  = np.load(scene_dir / "rgb.npy")
    inst = np.load(scene_dir / "inst.npy").astype(np.int32)
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.0 + 1e-6 \
              else rgb.clip(0, 255).astype(np.uint8)
    return xyz, rgb.astype(np.uint8), inst


def render_original(scene_dir: Path, manifest: dict, image_size: int = 512) -> np.ndarray:
    """Original scene with target (orange) at its original position, anchor (cyan)."""
    xyz, rgb, inst = load_scene_arrays(scene_dir)
    target_id = int(manifest["target_instance_id"])
    anchor_id = int(manifest["anchor_instance_id"])

    mask_t = inst == target_id
    mask_a = inst == anchor_id

    rgb = rgb.copy()
    # Tint anchor cyan
    rgb[mask_a] = (0.3 * rgb[mask_a].astype(np.float32) +
                   0.7 * ANCHOR_COLOR.astype(np.float32)).clip(0, 255).astype(np.uint8)
    # Tint target orange
    rgb[mask_t] = (0.2 * rgb[mask_t].astype(np.float32) +
                   0.8 * TARGET_COLOR.astype(np.float32)).clip(0, 255).astype(np.uint8)

    return render_topdown(xyz, rgb, image_size)


# ── find PLY files ─────────────────────────────────────────────────────────────

def find_ply(edits_dir: Path, job_index: int, suffix: str) -> Path | None:
    """Find the final merged scene PLY (not _placed or _bkgd or _HL)."""
    pattern = f"job{job_index:03d}_*{suffix}.ply"
    candidates = [
        p for p in edits_dir.glob(pattern)
        if not any(p.name.endswith(s) for s in ("_bkgd.ply", "_placed.ply", "_HL.ply",
                                                  "_lifted.ply", "_completed.ply",
                                                  "_merged.ply", "_relaxed.ply"))
    ]
    return candidates[0] if candidates else None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", type=int, default=[0, 1],
                    help="Job indices to compare")
    ap.add_argument("--jobs-file", default="repo/scannet/derived/generated_jobs_clean.jsonl")
    ap.add_argument("--geo-suffix", default="",
                    help="Filename suffix for geometric PLY (default: empty)")
    ap.add_argument("--vlm-suffix", default="_vlm_test",
                    help="Filename suffix for VLM PLY (default: _vlm_test)")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    # Load all jobs
    all_jobs = [json.loads(l) for l in open(HYPO_ROOT / args.jobs_file) if l.strip()]

    n_jobs = len(args.jobs)
    fig, axes = plt.subplots(n_jobs, 3, figsize=(15, 5 * n_jobs))
    if n_jobs == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Geometric placement", "VLM placement (GPT-4o)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=8)

    for row, job_idx in enumerate(args.jobs):
        manifest = all_jobs[job_idx]
        scene_id = manifest["scene_id"]
        scene_dir = PROCESSED_ROOT / scene_id
        edits_dir = scene_dir / "edits"

        target_lbl = manifest.get("target_label", f"inst{manifest['target_instance_id']}")
        anchor_lbl = manifest.get("anchor_label", "anchor")
        relation   = manifest.get("relation", "near")

        job_label = f"Job {job_idx:03d} | {scene_id}\n{target_lbl} → {relation} → {anchor_lbl}"

        # ── Panel 1: Original ──────────────────────────────────────────────
        try:
            img_orig = render_original(scene_dir, manifest, args.image_size)
        except Exception as e:
            img_orig = np.full((args.image_size, args.image_size, 3), 200, dtype=np.uint8)
            print(f"[warn] original render failed for job {job_idx}: {e}")

        axes[row, 0].imshow(img_orig)
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(job_label, fontsize=9, rotation=0, labelpad=120,
                                 va="center", ha="right")

        # ── Panels 2 & 3: Geo and VLM from PLY files ──────────────────────
        for col, suffix in enumerate([args.geo_suffix, args.vlm_suffix], start=1):
            ply_path = find_ply(edits_dir, job_idx, suffix)
            if ply_path is None:
                axes[row, col].imshow(np.full((args.image_size, args.image_size, 3), 200, dtype=np.uint8))
                axes[row, col].axis("off")
                axes[row, col].text(0.5, 0.5, f"PLY not found\n(suffix: {repr(suffix)})",
                                    ha="center", va="center", transform=axes[row, col].transAxes,
                                    fontsize=10, color="red")
                print(f"[warn] PLY not found: job{job_idx:03d}*{suffix}.ply in {edits_dir}")
                continue

            try:
                xyz_ply, rgb_ply = read_ply(ply_path)
                img_ply = render_topdown(xyz_ply, rgb_ply, args.image_size)
            except Exception as e:
                img_ply = np.full((args.image_size, args.image_size, 3), 200, dtype=np.uint8)
                print(f"[warn] PLY render failed for {ply_path.name}: {e}")

            axes[row, col].imshow(img_ply)
            axes[row, col].axis("off")
            axes[row, col].text(0.01, 0.01, ply_path.name, transform=axes[row, col].transAxes,
                                fontsize=6, color="white",
                                bbox=dict(facecolor="black", alpha=0.5, pad=2))

    # Legend
    legend_handles = [
        mpatches.Patch(color=(1.0, 0.55, 0.0), label="Target object"),
        mpatches.Patch(color=(0.0, 0.78, 0.86), label="Anchor object"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("MOVE Pipeline Comparison: Geometric vs VLM Placement", fontsize=14, y=1.01)
    fig.tight_layout(pad=2.0)

    out_path = Path(args.output) if args.output else HYPO_ROOT / "move_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
