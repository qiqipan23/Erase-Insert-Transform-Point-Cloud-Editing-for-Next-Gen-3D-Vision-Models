#!/usr/bin/env python3
"""
Visualise background reconstruction for several MOVE jobs.

Grid: N rows × 3 columns
  Col 1 – original scan (target object highlighted in red)
  Col 2 – background after object removal and structural fill
  Col 3 – final edited scene (object placed at new position)

Output: figs/fig_bkgd_completion.png / .pdf
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

HYPO      = Path(__file__).resolve().parent
PROC      = HYPO / "repo/scannet/processed"
EVAL_JSON = HYPO / "dataset/contextvqa_move_eval_labelled.json"
OUT_DIR   = HYPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 800
RADIUS     = 3
PAD_FRAC   = 0.05
BG_COLOR   = (240, 240, 240)

# ── PLY loader ────────────────────────────────────────────────────────────────
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
            rgb = (r_ / r_.max() * 255 if r_.max() > 1.5 else r_ * 255).astype(np.uint8)
            break
    return xyz, rgb


def render_topdown(xyz, rgb, size=IMAGE_SIZE, radius=RADIUS, pad=PAD_FRAC,
                   highlight_mask=None):
    """Top-down disk-splat render. highlight_mask marks points to colour red."""
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    xmn, xmx = x.min(), x.max()
    ymn, ymx = y.min(), y.max()
    span = max(xmx - xmn, ymx - ymn, 1e-3)
    p = span * pad
    xmn -= p; xmx += p; ymn -= p; ymx += p
    spanp = max(xmx - xmn, ymx - ymn)

    # draw by z order so higher objects appear on top
    order = np.argsort(z)
    x, y, z = x[order], y[order], z[order]
    c = rgb[order].copy() if rgb is not None else np.full((len(order),3), 150, np.uint8)
    if highlight_mask is not None:
        hl = highlight_mask[order]
        c[hl] = [220, 50, 50]

    px = ((x - xmn) / spanp * (size - 1)).astype(np.int32).clip(0, size-1)
    py = ((1.0 - (y - ymn) / spanp) * (size - 1)).astype(np.int32).clip(0, size-1)

    img = np.full((size, size, 3), BG_COLOR, np.uint8)
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            if dx*dx + dy*dy > radius*radius: continue
            qx = (px + dx).clip(0, size-1)
            qy = (py + dy).clip(0, size-1)
            img[qy, qx] = c
    return img


# ── Job selection ─────────────────────────────────────────────────────────────
def find_manifest(scene_id: str, job_stem: str) -> Path | None:
    edit_dir = PROC / scene_id / "edits"
    prefix   = job_stem[:6]          # e.g. "job003"
    matches  = list(edit_dir.glob(f"{prefix}*.json"))
    # prefer exact short name, then any match
    short = edit_dir / f"{prefix}.json"
    return short if short.exists() else (matches[0] if matches else None)


SELECTED_EXAMPLES = [
    # (scene_id, job_stem) — diverse large furniture, uncluttered scenes, clear gap
    ("scene0000_00", "job009_bed_to_front_of_couch"),             # bed, clear floor fill
    ("scene0206_00", "job852_bookshelf_to_next_to_trash_can"),    # bookshelf 57k pts
    ("scene0172_00", "job766_bed_to_front_of_cabinet"),           # bed 29k pts
    ("scene0061_00", "job262_couch_to_next_to_bar"),              # couch 22k pts
    ("scene0287_00", "job1068_desk_to_next_to_window"),             # desk 13k pts
]


def collect_jobs(n: int = 5) -> list[dict]:
    data = json.loads(EVAL_JSON.read_text())
    jobs = []
    for scene_id, job_stem in SELECTED_EXAMPLES[:n]:
        mpath = find_manifest(scene_id, job_stem)
        if mpath is None:
            print(f"[skip] manifest not found: {scene_id}/{job_stem}")
            continue
        m = json.loads(mpath.read_text())
        bkgd  = m.get("background_edit_file", "")
        canon = m.get("canonical_edit_file", "")
        if not bkgd or not Path(bkgd).exists():
            print(f"[skip] missing bkgd PLY: {scene_id}/{job_stem}")
            continue
        if not canon or not Path(canon).exists():
            print(f"[skip] missing canon PLY: {scene_id}/{job_stem}")
            continue
        xyz_path  = PROC / scene_id / "xyz.npy"
        rgb_path  = PROC / scene_id / "rgb.npy"
        inst_path = PROC / scene_id / "inst.npy"
        if not xyz_path.exists():
            print(f"[skip] missing xyz.npy: {scene_id}")
            continue
        jobs.append({
            "scene_id":       scene_id,
            "job_stem":       job_stem,
            "target_label":   m.get("target_label", "object"),
            "anchor_label":   m.get("anchor_label", "anchor"),
            "relation":       m.get("relation", ""),
            "target_inst_id": m.get("target_instance_id"),
            "background_ply": Path(bkgd),
            "edited_ply":     Path(canon),
            "xyz_path":       xyz_path,
            "rgb_path":       rgb_path,
            "inst_path":      inst_path,
        })
    return jobs


# ── Main figure ───────────────────────────────────────────────────────────────
RELATION_PHRASE = {
    "RIGHT_OF": "right of",
    "LEFT_OF":  "left of",
    "FRONT_OF": "front of",
    "BACK_OF":  "behind",
    "NEXT_TO":  "next to",
    "ON_TOP_OF":"on top of",
    "UNDER":    "under",
}

def build_figure(n_rows: int = 5):
    jobs = collect_jobs(n_rows)
    if not jobs:
        sys.exit("No valid jobs found.")

    fig, axes = plt.subplots(len(jobs), 3,
                             figsize=(12, 3.2 * len(jobs)),
                             dpi=150)
    if len(jobs) == 1:
        axes = [axes]

    col_titles = ["Original scan", "After removal\n(background reconstructed)",
                  "Final edited scene"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row, job in enumerate(jobs):
        print(f"Rendering row {row+1}/{len(jobs)}: {job['scene_id']} / {job['job_stem']}")

        # ── Load original scan ────────────────────────────────────────────────
        xyz_all = np.load(job["xyz_path"]).astype(np.float32)
        rgb_raw = np.load(job["rgb_path"])
        rgb_all = (rgb_raw * 255 if rgb_raw.max() <= 1.0 else rgb_raw).astype(np.uint8)

        # highlight target instance
        hl_mask = np.zeros(len(xyz_all), bool)
        if job["inst_path"].exists() and job["target_inst_id"] is not None:
            inst_all = np.load(job["inst_path"]).astype(np.int32)
            hl_mask = inst_all == int(job["target_inst_id"])

        img_orig = render_topdown(xyz_all, rgb_all, highlight_mask=hl_mask)

        # ── Load background reconstruction ────────────────────────────────────
        xyz_bkgd, rgb_bkgd = load_ply(job["background_ply"])
        if rgb_bkgd is None:
            rgb_bkgd = np.full((len(xyz_bkgd), 3), 150, np.uint8)

        # Highlight points inside the removed object's bounding box in green —
        # these are the structural fill points added after removal.
        obj_xyz = xyz_all[hl_mask]
        if len(obj_xyz) > 0:
            pad = 0.05  # small padding to catch fill near boundary
            lo = obj_xyz.min(axis=0) - pad
            hi = obj_xyz.max(axis=0) + pad
            in_box = np.all((xyz_bkgd >= lo) & (xyz_bkgd <= hi), axis=1)
            rgb_bkgd_vis = rgb_bkgd.copy()
            # fade background to grey so fill pops out
            rgb_bkgd_vis[~in_box] = (rgb_bkgd[~in_box] * 0.45 + 130 * 0.55).astype(np.uint8)
            # colour fill region bright green
            rgb_bkgd_vis[in_box] = [50, 200, 80]
        else:
            rgb_bkgd_vis = rgb_bkgd
        img_bkgd = render_topdown(xyz_bkgd, rgb_bkgd_vis)

        # ── Load final edited scene ───────────────────────────────────────────
        xyz_edit, rgb_edit = load_ply(job["edited_ply"])
        if rgb_edit is None:
            rgb_edit = np.full((len(xyz_edit), 3), 150, np.uint8)
        img_edit = render_topdown(xyz_edit, rgb_edit)

        for col, img in enumerate([img_orig, img_bkgd, img_edit]):
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        # row label
        rel_phrase = RELATION_PHRASE.get(job["relation"], job["relation"].lower())
        lbl = f"{job['target_label'].replace('_',' ')} → {rel_phrase}\n{job['anchor_label'].replace('_',' ')}"
        axes[row][0].set_ylabel(lbl, fontsize=8.5, rotation=0, labelpad=90,
                                 va="center", ha="right")

    red_patch   = mpatches.Patch(color=(220/255, 50/255,  50/255), label="Removed object")
    green_patch = mpatches.Patch(color=(50/255,  200/255, 80/255),  label="Reconstructed background (fill region)")
    fig.legend(handles=[red_patch, green_patch], loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig_bkgd_completion.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    build_figure(n_rows=5)
