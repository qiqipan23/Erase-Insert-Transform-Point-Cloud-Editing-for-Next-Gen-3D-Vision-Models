#!/usr/bin/env python3
"""
Visualize PoinTr/completion output for MOVE pipeline jobs.

For a given scene + job, shows 4 views side-by-side:
  1. Original scene region (before removal)
  2. Gap after object removal
  3. Completion points only (what PoinTr added)
  4. Final merged scene

Each view is shown as top-down projection AND front-view projection.

Usage:
  python visualize_completion.py --scene scene0000_00 --job 14
  python visualize_completion.py --scene scene0000_00 --job 14 --output my_fig.png
  python visualize_completion.py --list              # list all available completions
  python visualize_completion.py --scene scene0000_00  # show first available job
"""
from __future__ import annotations

import argparse
import json
import struct
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HYPO_ROOT      = Path(__file__).resolve().parent
PROCESSED_ROOT = HYPO_ROOT / "repo" / "scannet" / "processed"


# ── PLY loader (no open3d dependency) ───────────────────────────────────────

def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load PLY file. Returns (xyz [N,3], rgb [N,3] uint8 or None)."""
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32), None

    data = path.read_bytes()
    # Handle both \n and \r\n line endings
    header_end_idx = data.find(b"end_header")
    newline = data[header_end_idx + len("end_header")]  # byte after "end_header"
    header_end = header_end_idx + len("end_header") + (2 if newline == ord('\r') else 1)
    header = data[:header_end].decode("ascii", errors="ignore")

    n_verts = int(re.search(r"element vertex (\d+)", header).group(1))
    is_binary = "binary_little_endian" in header

    # Parse only vertex-element properties (stop at next "element" line)
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            in_vertex = True
        elif line.startswith("element"):
            in_vertex = False
        elif in_vertex and line.startswith("property") and not line.startswith("property list"):
            parts = line.split()
            if len(parts) >= 3:
                props.append((parts[1], parts[2]))

    if is_binary:
        type_map = {
            "float": ("f", 4), "float32": ("f", 4),
            "double": ("d", 8), "float64": ("d", 8),
            "uchar": ("B", 1), "uint8": ("B", 1),
            "char": ("b", 1),  "int8": ("b", 1),
            "short": ("h", 2), "int16": ("h", 2),
            "ushort": ("H", 2), "uint16": ("H", 2),
            "int": ("i", 4),   "int32": ("i", 4),
            "uint": ("I", 4),  "uint32": ("I", 4),
        }
        fmt_str = "<" + "".join(type_map.get(t, ("f", 4))[0] for t, n in props)
        row_size = struct.calcsize(fmt_str)
        rows = []
        offset = header_end
        for _ in range(n_verts):
            rows.append(struct.unpack_from(fmt_str, data, offset))
            offset += row_size
        arr = np.array(rows, dtype=np.float64)
    else:
        lines = data[header_end:].decode("ascii", errors="ignore").splitlines()
        arr = np.array(
            [list(map(float, l.split()[:len(props)])) for l in lines[:n_verts] if l.strip()],
            dtype=np.float64,
        )

    prop_names = [n for _, n in props]
    xyz_idx = [prop_names.index(c) for c in ("x", "y", "z") if c in prop_names]
    xyz = arr[:, xyz_idx].astype(np.float32) if len(xyz_idx) == 3 else np.empty((0, 3), dtype=np.float32)

    rgb = None
    rgb_names = [c for c in ("red", "green", "blue") if c in prop_names]
    if len(rgb_names) == 3:
        rgb_idx = [prop_names.index(c) for c in rgb_names]
        rgb_raw = arr[:, rgb_idx]
        # Values may be 0-255 (uchar) or 0-1 (float) depending on PLY writer
        if rgb_raw.max() <= 1.0:
            rgb_raw = rgb_raw * 255.0
        rgb = rgb_raw.clip(0, 255).astype(np.uint8)

    return xyz, rgb


# ── projection helpers ───────────────────────────────────────────────────────

def project_2d(xyz: np.ndarray,
               rgb: np.ndarray | None,
               axis: str = "top",
               ax: plt.Axes = None,
               color: str | None = None,
               alpha: float = 0.6,
               s: float = 0.5,
               label: str = "") -> plt.Axes:
    """
    Scatter-plot a 2D projection of a point cloud.
    axis: 'top' (XZ), 'front' (XY), 'side' (ZY)
    """
    if ax is None:
        _, ax = plt.subplots()

    if len(xyz) == 0:
        ax.text(0.5, 0.5, "no points", ha="center", va="center", transform=ax.transAxes)
        return ax

    if axis == "top":    px, py = xyz[:, 0], xyz[:, 2]
    elif axis == "front": px, py = xyz[:, 0], xyz[:, 1]
    else:                 px, py = xyz[:, 2], xyz[:, 1]

    if color is not None:
        c = color
    elif rgb is not None:
        c = rgb / 255.0
    else:
        c = "#888888"

    ax.scatter(px, py, c=c, s=s, alpha=alpha, linewidths=0, rasterized=True)
    if label:
        ax.set_title(label, fontsize=8, pad=3)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    return ax


# ── find completion dirs ─────────────────────────────────────────────────────

def find_completion_dir(scene_dir: Path, job_idx: int) -> Path | None:
    comp_root = scene_dir / "completion"
    if not comp_root.exists():
        return None
    pattern = f"{scene_dir.name}_job{job_idx:03d}"
    candidates = list(comp_root.glob(f"{scene_dir.name}_job{job_idx:03d}"))
    if not candidates:
        # try without leading zero padding
        candidates = [d for d in comp_root.iterdir()
                      if d.name.startswith(f"{scene_dir.name}_job{job_idx}")]
    return candidates[0] if candidates else None


def list_completions(scene_dir: Path) -> list[tuple[int, Path]]:
    comp_root = scene_dir / "completion"
    if not comp_root.exists():
        return []
    result = []
    for d in sorted(comp_root.iterdir()):
        m = re.search(r"_job(\d+)$", d.name)
        if m:
            result.append((int(m.group(1)), d))
    return result


# ── main figure ─────────────────────────────────────────────────────────────

def make_figure(scene_dir: Path, job_idx: int, out_path: Path) -> None:
    comp_dir = find_completion_dir(scene_dir, job_idx)
    if comp_dir is None:
        print(f"No completion directory found for {scene_dir.name} job{job_idx}")
        return

    # Load manifest to get context
    edits_dir = scene_dir / "edits"
    manifest_path = edits_dir / f"job{job_idx:03d}.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    print(f"Loading PLY files from {comp_dir} ...")

    before_xyz, before_rgb = load_ply(comp_dir / "crop_before_remove.ply")
    after_xyz,  after_rgb  = load_ply(comp_dir / "crop_after_remove.ply")

    # Completion points: for REMOVE jobs prefer scene-completion backends (ConvONet);
    # for MOVE jobs prefer object-completion backends (PoinTr).
    op = str(manifest.get("op", "")).upper()
    if op == "REMOVE":
        comp_candidates = [
            ("NN floor fill",       "nn_floor_fill_completed_points.ply",      "#e67e22"),
            ("Structural surface",  "structural_surface_completed_points.ply", "#2ecc71"),
            ("Hidden surface",      "hidden_surface_completed_points.ply",     "#f39c12"),
            ("ConvONet",            "convonet_completed_points.ply",           "#3498db"),
            ("PoinTr (occluded)",   "occluded_surface_completed_points.ply",  "#e74c3c"),
        ]
    else:
        comp_candidates = [
            ("PoinTr (occluded)",   "occluded_surface_completed_points.ply",  "#e74c3c"),
            ("ConvONet",            "convonet_completed_points.ply",           "#3498db"),
            ("Structural",          "structural_surface_completed_points.ply", "#2ecc71"),
            ("Hidden surface",      "hidden_surface_completed_points.ply",     "#f39c12"),
        ]
    comp_xyz, comp_rgb, comp_label, comp_color = None, None, "Completion", "#e74c3c"
    for label, fname, color in comp_candidates:
        p = comp_dir / fname
        if p.exists() and p.stat().st_size > 100:
            comp_xyz, comp_rgb = load_ply(p)
            comp_label, comp_color = label, color
            break

    # For REMOVE jobs, crop completion points to the actual removal shape.
    # Use OBB from manifest if available (tighter fit for rotated objects),
    # otherwise fall back to the AABB fill_region bounds.
    if op == "REMOVE" and comp_xyz is not None and len(comp_xyz) > 0:
        obb_data = manifest.get("remove_obb")
        if obb_data is not None:
            obb_center = np.array(obb_data["center"], dtype=np.float32)
            obb_axes   = np.array(obb_data["axes"],   dtype=np.float32)
            obb_half   = np.array(obb_data["half_extents"], dtype=np.float32)
            local = (comp_xyz - obb_center) @ obb_axes
            in_gap = np.all(np.abs(local) <= obb_half, axis=1)
        else:
            fill_region_path = comp_dir / "fill_region.json"
            if fill_region_path.exists():
                fr = json.loads(fill_region_path.read_text())
                mn = np.array(fr["bounds_min"], dtype=np.float32)
                mx = np.array(fr["bounds_max"], dtype=np.float32)
                in_gap = np.all((comp_xyz >= mn) & (comp_xyz <= mx), axis=1)
            else:
                in_gap = np.ones(len(comp_xyz), dtype=bool)
        comp_xyz = comp_xyz[in_gap]
        comp_rgb = comp_rgb[in_gap] if comp_rgb is not None else None
        comp_label += " (OBB region)" if obb_data else " (gap region)"

    merged_xyz, merged_rgb = load_ply(comp_dir / "completed_scene_merged.ply")

    # Also load full original for context
    orig_xyz_path = scene_dir / f"original_{scene_dir.name}.ply"
    orig_xyz, orig_rgb = (load_ply(orig_xyz_path) if orig_xyz_path.exists()
                          else (np.empty((0,3)), None))

    print(f"  before_remove:  {len(before_xyz):,} pts")
    print(f"  after_remove:   {len(after_xyz):,} pts")
    print(f"  {comp_label}:  {len(comp_xyz) if comp_xyz is not None else 0:,} pts")
    print(f"  merged:         {len(merged_xyz):,} pts")

    # ── Build figure: 2 rows (top-down / front-view) × 5 cols ───────────────
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.25,
                   top=0.88, bottom=0.04, left=0.06, right=0.98)

    cols = [
        (before_xyz, before_rgb, "1. Before removal\n(original crop)", None),
        (after_xyz,  after_rgb,  "2. After removal\n(gap visible)",    None),
        (comp_xyz,   comp_rgb,   f"3. {comp_label}\n(added points)",   comp_color),
        (merged_xyz, merged_rgb, "4. Final merged\nscene",             None),
        (orig_xyz,   orig_rgb,   "5. Full original\nscene (context)",  None),
    ]

    for col_i, (xyz, rgb, title, fixed_color) in enumerate(cols):
        if xyz is None:
            xyz = np.empty((0, 3), dtype=np.float32)

        # Larger dots for sparse crops so colors are visible; smaller for dense full-scene
        n_pts = len(xyz)
        pt_size = 3.0 if n_pts < 20_000 else (1.0 if n_pts < 80_000 else 0.3)

        for row_i, (view, ylabel) in enumerate([("top", "Top-down (X-Z)"),
                                                 ("front", "Front view (X-Y)")]):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_facecolor("#0d0d1a")
            project_2d(xyz, rgb, axis=view, ax=ax,
                       color=fixed_color, alpha=0.8, s=pt_size,
                       label=title if row_i == 0 else "")
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=7, color="#cccccc")
            ax.tick_params(colors="#666666")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

    # ── Overlay: show completion points on top of "after" in col 1 ──────────
    if comp_xyz is not None and len(comp_xyz) > 0:
        for row_i, view in enumerate(["top", "front"]):
            ax_overlay = fig.add_subplot(gs[row_i, 1], label=f"overlay_{row_i}")
            ax_overlay.set_facecolor("#0d0d1a")
            project_2d(after_xyz, after_rgb, axis=view, ax=ax_overlay,
                       color="#888888", alpha=0.4, s=0.2)
            project_2d(comp_xyz, None, axis=view, ax=ax_overlay,
                       color=comp_color, alpha=0.9, s=1.5,
                       label="2+3. Gap + completion\n(overlaid)" if row_i == 0 else "")
            ax_overlay.tick_params(colors="#666666")
            for spine in ax_overlay.spines.values():
                spine.set_edgecolor("#333355")

    # Title
    target = manifest.get("target_label", "?")
    anchor = manifest.get("anchor_label", "?")
    relation = manifest.get("relation", "?")
    n_input  = manifest.get("completion_input_points", "?")
    n_output = manifest.get("completion_completed_points", "?")
    backend  = manifest.get("completion_backend", "?")

    title_str = (
        f"{scene_dir.name}  job{job_idx:03d}  —  "
        f"{target} → {relation} → {anchor}\n"
        f"Completion: {backend}  |  input={n_input} pts  →  completed={n_output} pts"
    )
    fig.suptitle(title_str, fontsize=11, color="white", y=0.97)

    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None)
    parser.add_argument("--job",   type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--list", action="store_true",
                        help="List all available completion jobs")
    args = parser.parse_args()

    if args.list:
        for scene_dir in sorted(PROCESSED_ROOT.iterdir()):
            comps = list_completions(scene_dir)
            if comps:
                print(f"{scene_dir.name}: jobs {[j for j,_ in comps]}")
        return

    if args.scene is None:
        # Pick first scene that has completions
        for scene_dir in sorted(PROCESSED_ROOT.iterdir()):
            comps = list_completions(scene_dir)
            if comps:
                args.scene = scene_dir.name
                args.job   = comps[0][0]
                print(f"Auto-selected: {args.scene} job{args.job}")
                break

    scene_dir = PROCESSED_ROOT / args.scene
    if not scene_dir.exists():
        raise SystemExit(f"Scene not found: {scene_dir}")

    if args.job is None:
        comps = list_completions(scene_dir)
        if not comps:
            raise SystemExit(f"No completion dirs found in {scene_dir}")
        args.job = comps[0][0]
        print(f"Auto-selected job {args.job}")

    out_name = args.output or f"{args.scene}_job{args.job:03d}_completion.png"
    out_path = HYPO_ROOT / out_name

    make_figure(scene_dir, args.job, out_path)


if __name__ == "__main__":
    main()
