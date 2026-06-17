#!/usr/bin/env python3
"""
Visualize MOVE pipeline evaluation results.

Generates a multi-page PDF showing top-view renders with questions,
reference answers, and GPT-4o predictions. Correct answers are green,
wrong answers are red.

Usage:
  python visualize_move_eval.py
  python visualize_move_eval.py --file dataset/contextvqa_move_eval_gpt4o_v2.json
  python visualize_move_eval.py --type Direction --n 40   # only Direction questions
  python visualize_move_eval.py --wrong-only              # only show failures
  python visualize_move_eval.py --correct-only            # only show successes
"""
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw
import numpy as np
import json

try:
    from word2number import w2n
    _W2N = True
except ImportError:
    _W2N = False

HYPO_ROOT      = Path(__file__).resolve().parent
PROCESSED_ROOT = HYPO_ROOT / "repo" / "scannet" / "processed"

# Highlight colours (RGB uint8)
TARGET_COLOR = np.array([255, 140,   0], dtype=np.uint8)   # orange
ANCHOR_COLOR = np.array([  0, 210, 210], dtype=np.uint8)   # cyan


def _load_scene_arrays(scene_dir: Path):
    xyz  = np.load(scene_dir / "xyz.npy").astype(np.float32)
    rgb  = np.load(scene_dir / "rgb.npy")
    inst = np.load(scene_dir / "inst.npy").astype(np.int32)
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.0 + 1e-6 else rgb.clip(0, 255).astype(np.uint8)
    return xyz, rgb.astype(np.uint8), inst


def render_highlighted_scene(scene_id: str, manifest: dict, image_size: int = 512) -> Image.Image:
    """
    Render a top-down view of the MOVE scene WITHOUT PoinTr completion.
    The moved object (target) is highlighted orange; the anchor is highlighted cyan.
    Falls back to a gray placeholder on any error.
    """
    scene_dir = PROCESSED_ROOT / scene_id
    if not scene_dir.exists():
        return Image.new("RGB", (image_size, image_size), (200, 200, 200))

    try:
        xyz, rgb, inst = _load_scene_arrays(scene_dir)
    except Exception:
        return Image.new("RGB", (image_size, image_size), (200, 200, 200))

    target_id = int(manifest.get("target_instance_id", -1))
    anchor_id = int(manifest.get("anchor_instance_id", -1))
    translation = np.array(manifest.get("translation", [0, 0, 0]), dtype=np.float32)

    mask_t = inst == target_id
    mask_a = inst == anchor_id

    # Build combined point cloud: scene (no target) + moved target + anchor override
    mask_scene = ~mask_t
    scene_xyz = xyz[mask_scene]
    scene_rgb = rgb[mask_scene].copy()

    # Tint anchor points cyan (in place on the scene copy)
    anchor_in_scene = mask_a[mask_scene]
    scene_rgb[anchor_in_scene] = (
        0.35 * scene_rgb[anchor_in_scene].astype(np.float32) +
        0.65 * ANCHOR_COLOR.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # Target at new position, highlighted orange
    target_xyz = xyz[mask_t] + translation
    target_rgb = np.broadcast_to(TARGET_COLOR, (mask_t.sum(), 3)).copy()

    all_xyz = np.concatenate([scene_xyz, target_xyz], axis=0)
    all_rgb = np.concatenate([scene_rgb, target_rgb], axis=0)

    # ── top-down render (X → pixel-x, Y → pixel-y, Z for depth sort) ──────
    pad = 0.05
    x, y, z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    span = max(x_max - x_min, y_max - y_min, 1e-3)
    p = span * pad
    x_min -= p; y_min -= p; span += 2 * p

    order = np.argsort(z)
    x, y, rgb_s = x[order], y[order], all_rgb[order]

    px = ((x - x_min) / span * (image_size - 1)).astype(np.int32).clip(0, image_size - 1)
    py = ((1.0 - (y - y_min) / span) * (image_size - 1)).astype(np.int32).clip(0, image_size - 1)

    img_arr = np.full((image_size, image_size, 3), 240, dtype=np.uint8)
    img_arr[py, px] = rgb_s
    for dy in range(2):
        for dx in range(2):
            py2 = (py + dy).clip(0, image_size - 1)
            px2 = (px + dx).clip(0, image_size - 1)
            mask = img_arr[py2, px2, 0] == 240
            img_arr[py2[mask], px2[mask]] = rgb_s[mask]

    img = Image.fromarray(img_arr)

    # ── draw bounding box around moved target ──────────────────────────────
    if mask_t.any():
        tx = target_xyz[:, 0]
        ty = target_xyz[:, 1]
        bx0 = int(((tx.min() - x_min) / span * (image_size - 1)).clip(0, image_size - 1)) - 3
        bx1 = int(((tx.max() - x_min) / span * (image_size - 1)).clip(0, image_size - 1)) + 3
        by0 = int(((1.0 - (ty.max() - y_min) / span) * (image_size - 1)).clip(0, image_size - 1)) - 3
        by1 = int(((1.0 - (ty.min() - y_min) / span) * (image_size - 1)).clip(0, image_size - 1)) + 3
        draw = ImageDraw.Draw(img)
        for lw in range(3, 0, -1):
            draw.rectangle([bx0 - lw, by0 - lw, bx1 + lw, by1 + lw],
                           outline=(255, 140, 0) if lw == 1 else (0, 0, 0))

    return img


# ── normalization (mirrors metric_compute.py) ────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    replacements = {
        'back and right': 'back right', 'back and left': 'back left',
        'front and right': 'front right', 'front and left': 'front left',
        'behind and to the right': 'back right', 'behind and to the left': 'back left',
        'in front and to the right': 'front right',
        'to the': '', 'by the': '', 'on the': '', 'near': '', 'next': '', 'corner': '',
        'behind': 'back', 'bottom': 'back', 'top': 'front',
        'right side': 'right', 'left side': 'left', 'front side': 'front', 'back side': 'back',
        'in front of': 'front', 'on the left of': 'left', 'on the right of': 'right',
        'on the left': 'left', 'on the right': 'right',
        'north': 'front', 'south': 'back', 'east': 'right', 'west': 'left',
        'northwest': 'front left', 'northeast': 'front right',
        'southwest': 'back left', 'southeast': 'back right',
        'forward': 'front', 'backward': 'back',
        'bottom of': 'back', 'left of': 'left', 'right of': 'right',
        'front of': 'front', 'back of': 'back',
    }
    sorted_keys = sorted(replacements, key=len, reverse=True)
    pat = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_keys)) + r')\b')
    text = pat.sub(lambda m: replacements[m.group(0)], text)
    text = re.sub(r'\b(?:a|an|the)\b', '', text).strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    if _W2N:
        words, out, is_dig = text.split(), [], False
        for w in words:
            try:
                out.append(str(w2n.word_to_num(w))); is_dig = True
            except (ValueError, AttributeError):
                if not is_dig:
                    out.append(w)
        text = ' '.join(out)
    return text.strip()


def is_correct(pred: str, ref: str) -> bool:
    return _normalize(pred) == _normalize(ref)


# ── layout helpers ───────────────────────────────────────────────────────────

COLS = 3
ROWS = 3
PER_PAGE = COLS * ROWS
FIG_W, FIG_H = 18, 18

CORRECT_COLOR = "#2ecc71"
WRONG_COLOR   = "#e74c3c"
UNKNOWN_COLOR = "#f39c12"


def wrap(text: str, width: int = 38) -> str:
    return "\n".join(textwrap.wrap(str(text), width))


def make_page(samples: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(ROWS, COLS, figsize=(FIG_W, FIG_H))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, s in zip(axes, samples):
        # Render from point cloud with highlight; fall back to pre-rendered PNG
        job_idx = s.get("job_index")
        if job_idx is None:
            m = re.match(r'job(\d+)', s.get("job_stem", ""))
            job_idx = int(m.group(1)) if m else None
        manifest_path = (PROCESSED_ROOT / s["scene_id"] / "edits" / f"job{job_idx:03d}.json"
                         if job_idx is not None else Path("/dev/null"))
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest:
            img = render_highlighted_scene(s["scene_id"], manifest)
        else:
            try:
                img = Image.open(HYPO_ROOT / s["image_path"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (512, 512), color=(200, 200, 200))

        ax.imshow(img)
        ax.axis("off")

        correct = is_correct(s["predicted"], s["reference"])
        unknown = "unknown" in s["predicted"].lower() or s["predicted"].strip() == ""
        color = UNKNOWN_COLOR if unknown else (CORRECT_COLOR if correct else WRONG_COLOR)
        mark  = "?" if unknown else ("✓" if correct else "✗")

        # Coloured border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(4)

        # Title: scene + change type
        ax.set_title(
            f"{s['scene_id']}  [{s['question_type']}]",
            fontsize=7, pad=2, color="#555555",
        )

        # Text box below image
        context_line = wrap(s["context_change"], 50)
        q_line       = wrap(s["question"], 50)
        ref_line     = s["reference"]
        pred_line    = s["predicted"]

        caption = (
            f"Context: {context_line}\n"
            f"Q: {q_line}\n"
            f"Ref:  {ref_line}\n"
            f"Pred: {pred_line}  {mark}"
        )
        ax.text(
            0.01, -0.02, caption,
            transform=ax.transAxes,
            fontsize=6.5, verticalalignment="top",
            color=color if not correct else "#1a7a42",
            fontfamily="monospace",
            clip_on=False,
        )

    fig.tight_layout(pad=1.2, h_pad=4.5, w_pad=1.5)
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="dataset/contextvqa_move_eval_gpt4o.json")
    parser.add_argument("--type", default=None, help="Filter by question type (Direction/Scale/Semantic)")
    parser.add_argument("-n", "--n", type=int, default=None, help="Max samples to show")
    parser.add_argument("--wrong-only",   action="store_true")
    parser.add_argument("--correct-only", action="store_true")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    data_path = HYPO_ROOT / args.file
    data = json.loads(data_path.read_text())

    # Collect all QA samples
    samples = []
    for scene_id, entries in data.items():
        for entry in entries:
            for qa in entry.get("questions_answers", []):
                if "predicted_answer" not in qa:
                    continue
                s = {
                    "scene_id":     scene_id,
                    "job_stem":     entry.get("job_stem", ""),
                    "job_index":    entry.get("job_index", None),
                    "image_path":   entry["image_path"],
                    "context_change": entry["context_change"],
                    "question":     qa["question"],
                    "question_type": qa.get("question_type", ""),
                    "reference":    qa["answer"],
                    "predicted":    qa["predicted_answer"],
                }
                samples.append(s)

    # Filter
    if args.type:
        samples = [s for s in samples if args.type in s["question_type"]]
    if args.wrong_only:
        samples = [s for s in samples if not is_correct(s["predicted"], s["reference"])]
    if args.correct_only:
        samples = [s for s in samples if is_correct(s["predicted"], s["reference"])]

    # Balance: show up to n//3 per type for overview (unless filtered)
    if args.n:
        samples = samples[: args.n]
    else:
        # Default: 30 wrong + 15 correct, spread across types
        by_type: dict[str, list] = defaultdict(list)
        for s in samples:
            by_type[s["question_type"]].append(s)
        selected = []
        wrong_budget  = {qt: 10 for qt in ["Direction", "Scale", "Semantic", "Scale Direction"]}
        correct_budget = {qt: 5  for qt in ["Direction", "Scale", "Semantic", "Scale Direction"]}
        for qt, bucket in by_type.items():
            wb = wrong_budget.get(qt, 8)
            cb = correct_budget.get(qt, 4)
            wrong   = [s for s in bucket if not is_correct(s["predicted"], s["reference"])][:wb]
            correct = [s for s in bucket if is_correct(s["predicted"], s["reference"])][:cb]
            selected.extend(wrong + correct)
        samples = selected

    if not samples:
        print("No samples match the filters.")
        return

    out_name = args.output or (
        Path(args.file).stem + f"{'_' + args.type if args.type else ''}_viz.pdf"
    )
    out_path = HYPO_ROOT / out_name
    total_pages = (len(samples) + PER_PAGE - 1) // PER_PAGE

    print(f"Rendering {len(samples)} samples → {total_pages} pages → {out_path}")
    with PdfPages(out_path) as pdf:
        # Cover page
        fig_cover, ax_cover = plt.subplots(figsize=(12, 4))
        ax_cover.axis("off")

        # Summary stats
        correct_n = sum(1 for s in samples if is_correct(s["predicted"], s["reference"]))
        unknown_n = sum(1 for s in samples if "unknown" in s["predicted"].lower())
        total_n   = len(samples)

        summary = (
            f"GPT-4o MOVE Evaluation — {data_path.name}\n\n"
            f"Showing {total_n} QA pairs\n"
            f"Correct (green): {correct_n}  ({correct_n/total_n*100:.1f}%)\n"
            f"Wrong (red):     {total_n-correct_n-unknown_n}  ({(total_n-correct_n-unknown_n)/total_n*100:.1f}%)\n"
            f"Unknown (orange): {unknown_n}  ({unknown_n/total_n*100:.1f}%)\n\n"
        )
        if args.type:
            summary += f"Filtered to question type: {args.type}"

        ax_cover.text(0.5, 0.5, summary, ha="center", va="center",
                      fontsize=13, transform=ax_cover.transAxes, fontfamily="monospace")
        patches = [
            mpatches.Patch(color=CORRECT_COLOR, label="Correct"),
            mpatches.Patch(color=WRONG_COLOR,   label="Wrong"),
            mpatches.Patch(color=UNKNOWN_COLOR, label="Unknown / refused"),
        ]
        ax_cover.legend(handles=patches, loc="lower center", fontsize=11)
        pdf.savefig(fig_cover, bbox_inches="tight")
        plt.close(fig_cover)

        # Content pages
        for page_i in range(total_pages):
            chunk = samples[page_i * PER_PAGE : (page_i + 1) * PER_PAGE]
            fig = make_page(chunk)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  page {page_i+1}/{total_pages}", end="\r", flush=True)

    print(f"\nDone. Written: {out_path}")


if __name__ == "__main__":
    main()
