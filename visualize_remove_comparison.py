#!/usr/bin/env python3
"""
Compare OBB vs AABB removal and show the completion pipeline for a REMOVE job.

Layout:
  Section 1 — Removal comparison (3 cols × 2 rows)
    Columns : Original scene | AABB removed | OBB removed
    Rows    : Top-down (XZ)  | Front (XY)

  Section 2 — Completion pipeline (N cols × 2 rows)
    Columns : After removal | NN floor fill | Structural | Hidden | ConvONet | Merged
    Rows    : Top-down (XZ) | Front (XY)

Usage:
  python visualize_remove_comparison.py --scene scene0000_00 --job 14
  python visualize_remove_comparison.py --scene scene0000_00 --job 14 --output out.png
  python visualize_remove_comparison.py --list
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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Polygon as MplPolygon

HYPO_ROOT      = Path(__file__).resolve().parent
PROCESSED_ROOT = HYPO_ROOT / "repo" / "scannet" / "processed"

BG = "#0d1117"


# ── PLY loaders ──────────────────────────────────────────────────────────────

def load_mesh_ply(path: Path):
    """Returns (verts [N,3] float32, rgb [N,3] uint8, faces [M,3] int32 or None)."""
    if not path.exists():
        return np.empty((0, 3), np.float32), None, None
    data = path.read_bytes()
    hdr_end_idx = data.find(b"end_header")
    nl = data[hdr_end_idx + len("end_header")]
    hdr_end = hdr_end_idx + len("end_header") + (2 if nl == ord('\r') else 1)
    header = data[:hdr_end].decode("ascii", errors="ignore")

    is_binary = "binary_little_endian" in header
    n_verts = int(re.search(r"element vertex (\d+)", header).group(1))
    m = re.search(r"element face (\d+)", header)
    n_faces = int(m.group(1)) if m else 0

    v_props: list[tuple[str, str]] = []
    in_v = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):   in_v = True
        elif line.startswith("element"):        in_v = False
        elif in_v and line.startswith("property") and "list" not in line:
            parts = line.split()
            if len(parts) >= 3:
                v_props.append((parts[1], parts[2]))

    type_map = {
        "float": ("f", 4), "float32": ("f", 4),
        "double": ("d", 8),
        "uchar": ("B", 1), "uint8": ("B", 1),
        "char": ("b", 1),
        "short": ("h", 2), "ushort": ("H", 2),
        "int": ("i", 4), "uint": ("I", 4), "int32": ("i", 4),
    }

    if is_binary:
        v_fmt = "<" + "".join(type_map.get(t, ("f", 4))[0] for t, _ in v_props)
        v_size = struct.calcsize(v_fmt)
        body = data[hdr_end:]
        rows = [struct.unpack_from(v_fmt, body, i * v_size) for i in range(n_verts)]
        v_arr = np.array(rows, dtype=np.float64)
        f_offset = n_verts * v_size
        faces_list = []
        ptr = f_offset
        for _ in range(n_faces):
            count = body[ptr]
            idxs = struct.unpack_from(f"<{count}i", body, ptr + 1)
            faces_list.append(idxs[:3])
            ptr += 1 + count * 4
        f_arr = np.array(faces_list, dtype=np.int32) if faces_list else None
    else:
        lines = data[hdr_end:].decode("ascii", errors="ignore").splitlines()
        v_arr = np.array([list(map(float, l.split())) for l in lines[:n_verts] if l.strip()],
                         dtype=np.float64)
        f_lines = [l for l in lines[n_verts:] if l.strip()]
        faces_list = []
        for l in f_lines[:n_faces]:
            parts = list(map(int, l.split()))
            faces_list.append(parts[1:4])
        f_arr = np.array(faces_list, dtype=np.int32) if faces_list else None

    n2c = {name: i for i, (_, name) in enumerate(v_props)}
    verts = v_arr[:, [n2c["x"], n2c["y"], n2c["z"]]].astype(np.float32)
    rgb = np.zeros((n_verts, 3), dtype=np.uint8)
    for out_i, name in enumerate(("red", "green", "blue")):
        if name in n2c:
            rgb[:, out_i] = np.clip(v_arr[:, n2c[name]], 0, 255).astype(np.uint8)
    return verts, rgb, f_arr


def render_mesh_2d(ax, verts, rgb, faces, proj=(0, 2), alpha=0.95):
    """Project mesh triangles onto a 2D plane and draw as filled polygons."""
    from matplotlib.collections import PolyCollection
    if faces is None or len(faces) == 0:
        scatter(ax, verts, rgb, proj=proj)
        return
    i, j = proj
    v2d = verts[:, [i, j]]           # [N, 2]
    tris = v2d[faces]                 # [M, 3, 2]
    face_rgb = rgb[faces].mean(axis=1).astype(np.float32) / 255.0
    coll = PolyCollection(tris, facecolors=face_rgb, edgecolors="none",
                          alpha=alpha, rasterized=True)
    ax.add_collection(coll)
    ax.autoscale_view()


def load_ply(path: Path):
    """Returns (xyz [N,3] float32, rgb [N,3] uint8 or None)."""
    if not path.exists():
        return np.empty((0, 3), dtype=np.float32), None
    data = path.read_bytes()
    header_end_idx = data.find(b"end_header")
    newline = data[header_end_idx + len("end_header")]
    header_end = header_end_idx + len("end_header") + (2 if newline == ord('\r') else 1)
    header = data[:header_end].decode("ascii", errors="ignore")
    n_verts = int(re.search(r"element vertex (\d+)", header).group(1))
    is_binary = "binary_little_endian" in header

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
        body = data[header_end:]
        rows = [struct.unpack_from(fmt_str, body, i * row_size) for i in range(n_verts)]
        arr = np.array(rows, dtype=np.float64)
    else:
        lines = data[header_end:].decode("ascii", errors="ignore").splitlines()
        arr = np.array([list(map(float, l.split())) for l in lines[:n_verts] if l.strip()],
                       dtype=np.float64)

    if arr.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), None

    name_to_col = {name: i for i, (_, name) in enumerate(props)}
    xyz_cols = [name_to_col[k] for k in ("x", "y", "z") if k in name_to_col]
    xyz = arr[:, xyz_cols].astype(np.float32)

    rgb = None
    for names in [("red", "green", "blue"), ("r", "g", "b")]:
        if all(n in name_to_col for n in names):
            raw = arr[:, [name_to_col[n] for n in names]]
            rgb = (raw / raw.max() * 255).astype(np.uint8) if raw.max() > 1.5 else (raw * 255).astype(np.uint8)
            break
    return xyz, rgb


# ── geometry helpers ─────────────────────────────────────────────────────────

def aabb_corners(center, size, proj):
    """4 corners of AABB projected onto two world axes (proj = (i, j))."""
    c = np.array(center); s = np.array(size)
    h = s / 2
    offsets = [[-1,-1],[ 1,-1],[ 1, 1],[-1, 1]]
    axes = list(range(3))
    i, j = proj
    k = [x for x in axes if x not in (i, j)][0]
    return np.array([[c[i] + o[0]*h[i], c[j] + o[1]*h[j]] for o in offsets])


def obb_corners(obb, proj):
    """4 corners of OBB projected onto two world axes."""
    center = np.array(obb["center"])
    axes   = np.array(obb["axes"])
    half   = np.array(obb["half_extents"])
    # Only expand along the two non-vertical axes (0 and 2); axis 1 = world Y (up)
    a0 = axes[:, 0] * half[0]
    a2 = axes[:, 2] * half[2]
    corners_3d = np.array([center+a0+a2, center-a0+a2, center-a0-a2, center+a0-a2])
    i, j = proj
    return corners_3d[:, [i, j]]


def obb_corners_xy(obb):
    """OBB corners for front (XY) view — uses horiz axis 0 + vertical axis 1."""
    center = np.array(obb["center"])
    axes   = np.array(obb["axes"])
    half   = np.array(obb["half_extents"])
    a0 = axes[:, 0] * half[0]
    a1 = axes[:, 1] * half[1]
    corners_3d = np.array([center+a0+a1, center-a0+a1, center-a0-a1, center+a0-a1])
    return corners_3d[:, [0, 1]]


# ── drawing helpers ──────────────────────────────────────────────────────────

def styled_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#555555", labelsize=5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.grid(True, color="#1a1a2a", linewidth=0.4)


def scatter(ax, xyz, rgb=None, color=None, alpha=0.6, proj=(0, 2)):
    if xyz is None or xyz.shape[0] == 0:
        ax.text(0.5, 0.5, "no data", color="#555", ha="center",
                va="center", transform=ax.transAxes, fontsize=7)
        return
    n = xyz.shape[0]
    s = 3.0 if n < 15_000 else (1.0 if n < 60_000 else 0.3)
    x, y = xyz[:, proj[0]], xyz[:, proj[1]]
    if color is not None:
        ax.scatter(x, y, s=s, c=color, alpha=alpha, linewidths=0, rasterized=True)
    elif rgb is not None:
        ax.scatter(x, y, s=s, c=rgb.astype(np.float32)/255, alpha=alpha, linewidths=0, rasterized=True)
    else:
        ax.scatter(x, y, s=s, color="#888", alpha=alpha, linewidths=0, rasterized=True)


def draw_box(ax, corners, color, lw=1.8, label=None):
    from matplotlib.patches import Polygon as P
    poly = P(corners, closed=True, fill=False, edgecolor=color, linewidth=lw, linestyle="--")
    ax.add_patch(poly)
    if label:
        cx, cy = corners.mean(axis=0)
        ax.text(cx, cy, label, color=color, fontsize=6, ha="center", va="center",
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.55, ec="none"))


def set_limits(ax, center, size, pad, proj):
    i, j = proj
    ax.set_xlim(center[i] - size[i]/2 - pad, center[i] + size[i]/2 + pad)
    ax.set_ylim(center[j] - size[j]/2 - pad, center[j] + size[j]/2 + pad)
    ax.set_aspect("equal")


# ── crop helper ──────────────────────────────────────────────────────────────

def crop(xyz, rgb, center, size, pad=1.5):
    if xyz is None or xyz.shape[0] == 0:
        return xyz, rgb
    half = np.array(size) / 2 * pad
    mn = np.array(center) - half
    mx = np.array(center) + half
    mask = np.all((xyz >= mn) & (xyz <= mx), axis=1)
    return xyz[mask], (rgb[mask] if rgb is not None else None)


# ── main ─────────────────────────────────────────────────────────────────────

def visualize(scene_id: str, job_index: int, out_path: Path):
    scene_dir = PROCESSED_ROOT / scene_id
    edit_dir  = scene_dir / "edits"
    comp_dir  = scene_dir / "completion" / f"{scene_id}_job{job_index:03d}"

    manifest_path = edit_dir / f"job{job_index:03d}.json"
    if not manifest_path.exists():
        print(f"[error] manifest not found: {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("op") != "REMOVE":
        print(f"[error] job {job_index} is not a REMOVE job")
        return

    label  = manifest.get("target_label", f"job{job_index:03d}")
    center = manifest["remove_center"]
    size   = manifest["remove_size"]
    obb    = manifest.get("remove_obb")

    # ── load PLYs ────────────────────────────────────────────────────────────
    print(f"Loading PLYs for {scene_id} job{job_index:03d} ({label}) ...")

    orig_xyz, orig_rgb = load_ply(scene_dir / f"original_{scene_id}.ply")
    obb_xyz,  obb_rgb  = load_ply(Path(manifest["canonical_edit_file"]))
    aabb_ply = manifest.get("aux_edit_file", "")
    aabb_xyz, aabb_rgb = load_ply(Path(aabb_ply)) if aabb_ply else (np.empty((0,3),np.float32), None)

    print(f"  original:     {orig_xyz.shape[0]:,} pts")
    print(f"  AABB removed: {obb_xyz.shape[0]:,} pts")

    # Crop to region of interest
    PAD = 2.0
    orig_c, orig_c_rgb = crop(orig_xyz, orig_rgb, center, size, PAD)
    aabb_c, aabb_c_rgb = crop(obb_xyz,  obb_rgb,  center, size, PAD)

    # Find removed points (present in orig but absent from AABB result)
    removed_mask = np.zeros(orig_c.shape[0], dtype=bool)
    if orig_c.shape[0] > 0 and aabb_c.shape[0] > 0:
        from scipy.spatial import cKDTree
        dists, _ = cKDTree(aabb_c).query(orig_c, k=1)
        removed_mask = dists > 0.03

    # ── load completion PLYs ─────────────────────────────────────────────────
    COMPLETION_LAYERS = [
        ("Structural surface", "structural_surface_completed_points.ply", "#2ecc71"),
        ("ConvONet",           "convonet_completed_points.ply",           "#3498db"),
        ("Poisson mesh",       "poisson_completion_mesh.ply",             "#e74c3c"),
    ]
    comp_layers = []
    if comp_dir.exists():
        for title, fname, color in COMPLETION_LAYERS:
            p = comp_dir / fname
            if p.exists() and p.stat().st_size > 200:
                xyz, rgb = load_ply(p)
                if xyz.shape[0] > 0:
                    comp_layers.append((title, xyz, rgb, color))
                    print(f"  {title}: {xyz.shape[0]:,} pts")

    # Final mesh: full-scene Poisson reconstruction (load with faces for proper 2D rendering)
    mesh_xyz, mesh_rgb, mesh_faces = None, None, None
    for mf in ["poisson_scene_mesh.ply", "scan_mesh_with_completion.ply", "poisson_scan_with_completion.ply"]:
        p = comp_dir / mf
        if p.exists() and p.stat().st_size > 1000:
            mesh_xyz, mesh_rgb, mesh_faces = load_mesh_ply(p)
            print(f"  Final mesh ({mf}): {mesh_xyz.shape[0]:,} verts, "
                  f"{len(mesh_faces) if mesh_faces is not None else 0:,} faces")
            break

    # Full-scene axis limits derived from the original scan cloud
    if orig_xyz.shape[0] > 0:
        scene_xlim = (float(orig_xyz[:, 0].min()) - 0.1, float(orig_xyz[:, 0].max()) + 0.1)
        scene_ylim = (float(orig_xyz[:, 1].min()) - 0.1, float(orig_xyz[:, 1].max()) + 0.1)
        scene_zlim = (float(orig_xyz[:, 2].min()) - 0.1, float(orig_xyz[:, 2].max()) + 0.1)
    else:
        scene_xlim, scene_ylim, scene_zlim = xlim, ylim, zlim

    has_completion = len(comp_layers) > 0 or mesh_xyz is not None

    # ── box geometry ─────────────────────────────────────────────────────────
    aabb_xz = aabb_corners(center, size, (0, 2))
    aabb_xy = aabb_corners(center, size, (0, 1))

    # Axis limits
    PAD_LIM = 0.6
    cx, cy, cz = center; sx, sy, sz = size
    xlim = (cx - sx/2 - PAD_LIM, cx + sx/2 + PAD_LIM)
    zlim = (cz - sz/2 - PAD_LIM, cz + sz/2 + PAD_LIM)
    ylim = (cy - sy/2 - PAD_LIM, cy + sy/2 + PAD_LIM)

    # ── figure layout ─────────────────────────────────────────────────────────
    # Section 1: 2 cols × 2 rows  (original | AABB removed)
    # Section 2: N cols × 2 rows  (after removal | structural | convonet | merged)
    n_sections = 2 if has_completion else 1
    fig_h = 8 * n_sections
    fig = plt.figure(figsize=(14, fig_h), facecolor=BG)

    master = GridSpec(n_sections, 1, figure=fig,
                      hspace=0.12, top=0.95, bottom=0.04,
                      left=0.04, right=0.98)

    fig.suptitle(
        f"{scene_id}  job{job_index:03d}  —  {label.upper()} removal  (AABB)",
        color="white", fontsize=13, y=0.98
    )

    # ── Section 1: Original | AABB removed ───────────────────────────────────
    gs1 = GridSpecFromSubplotSpec(2, 2, subplot_spec=master[0],
                                  hspace=0.30, wspace=0.12)

    col_titles_1 = [
        "Original scene\n(red = removed region)",
        "After AABB removal\n(axis-aligned box)",
    ]
    row_titles_1 = ["Top-down  (X → , Z ↑)", "Front view  (X → , Y ↑)"]

    axes1 = [[fig.add_subplot(gs1[r, c]) for c in range(2)] for r in range(2)]
    for r in range(2):
        for c in range(2):
            ax = axes1[r][c]
            styled_ax(ax)
            if r == 0:
                ax.set_title(col_titles_1[c], color="white", fontsize=9, pad=4)
            if c == 0:
                ax.set_ylabel(row_titles_1[r], color="#aaaaaa", fontsize=8)

    def render_removal_row(row, proj, xlim_, ylim_, box_corners):
        ax_o, ax_a = axes1[row]
        keep_xyz = orig_c[~removed_mask]
        keep_rgb = orig_c_rgb[~removed_mask] if orig_c_rgb is not None else None
        rem_xyz  = orig_c[removed_mask]
        scatter(ax_o, keep_xyz, keep_rgb, proj=proj)
        if rem_xyz.shape[0] > 0:
            scatter(ax_o, rem_xyz, color="#ff4444", alpha=0.9, proj=proj)
        draw_box(ax_o, box_corners, "#ff8c00", label="AABB")

        scatter(ax_a, aabb_c, aabb_c_rgb, proj=proj)
        draw_box(ax_a, box_corners, "#ff8c00", label="hole")

        for ax in (ax_o, ax_a):
            ax.set_xlim(*xlim_); ax.set_ylim(*ylim_); ax.set_aspect("equal")

    render_removal_row(0, (0, 2), xlim, zlim, aabb_xz)
    render_removal_row(1, (0, 1), xlim, ylim, aabb_xy)

    # ── Section 2: Completion pipeline ───────────────────────────────────────
    if has_completion:
        # Individual highlight columns + one whole-scene final mesh column
        # comp_cols entries: (title, xyz, rgb, color, faces_or_None)
        comp_cols = [(t, x, r, c, None) for t, x, r, c in comp_layers]
        if mesh_xyz is not None:
            comp_cols.append(("Final mesh\n(whole scene)", mesh_xyz, mesh_rgb, "white", mesh_faces))

        n_cc = len(comp_cols)
        gs2 = GridSpecFromSubplotSpec(2, n_cc, subplot_spec=master[1],
                                      hspace=0.30, wspace=0.12)
        axes2 = [[fig.add_subplot(gs2[r, c]) for c in range(n_cc)] for r in range(2)]

        for r in range(2):
            for c in range(n_cc):
                ax = axes2[r][c]
                styled_ax(ax)
                title, _, _, color, _ = comp_cols[c]
                if r == 0:
                    ax.set_title(title, color=color, fontsize=8, pad=4)
                if c == 0:
                    ax.set_ylabel(["Top-down  (X → , Z ↑)", "Front view  (X → , Y ↑)"][r],
                                  color="#aaaaaa", fontsize=8)

        def render_completion_row(row, proj, xlim_, ylim_, sxlim, szlim):
            box_corners = aabb_xz if proj == (0, 2) else aabb_xy
            for c, (title, xyz, rgb, color, faces) in enumerate(comp_cols):
                ax = axes2[row][c]
                if title.startswith("Final mesh"):
                    # Whole-scene mesh rendered as filled 2D triangles
                    render_mesh_2d(ax, xyz, rgb, faces, proj=proj)
                    draw_box(ax, box_corners, "#ff8c00", lw=1.0, label="fill")
                    ax.set_xlim(*sxlim); ax.set_ylim(*szlim); ax.set_aspect("equal")
                else:
                    # Individual method: highlight color over dimmed scene (zoomed in)
                    comp_c, comp_c_rgb = crop(xyz, rgb, center, size, PAD)
                    n_pts = comp_c.shape[0] if comp_c is not None else 0
                    scatter(ax, aabb_c, aabb_c_rgb, alpha=0.30, proj=proj)
                    scatter(ax, comp_c, color=color, alpha=0.9, proj=proj)
                    ax.set_title(f"{title}\n({n_pts:,} pts)", color=color, fontsize=8, pad=4)
                    ax.set_xlim(*xlim_); ax.set_ylim(*ylim_); ax.set_aspect("equal")

        render_completion_row(0, (0, 2), xlim, zlim, scene_xlim, scene_zlim)
        render_completion_row(1, (0, 1), xlim, ylim, scene_xlim, scene_ylim)

        fig.text(0.5, master[1].get_position(fig).y1 + 0.005,
                 "── Completion pipeline ──",
                 color="#aaaaaa", fontsize=10, ha="center", va="bottom")

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color="#ff4444", label="Removed points"),
        mpatches.Patch(color="#ff8c00", label="AABB removal box"),
    ]
    for title, _, _, color in comp_layers:  # comp_layers still 4-tuples
        handles.append(mpatches.Patch(color=color, label=title))

    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 6),
               facecolor="#1a1a2e", edgecolor="#333333",
               labelcolor="white", fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    # ── Stats ─────────────────────────────────────────────────────────────────
    diff_obb  = orig_xyz.shape[0] - obb_xyz.shape[0]
    diff_aabb = orig_xyz.shape[0] - aabb_xyz.shape[0]
    stats = (f"OBB removed {diff_obb:,} pts  |  AABB removed {diff_aabb:,} pts  |  "
             f"Extra removed by AABB: {max(diff_aabb - diff_obb, 0):,} pts")
    fig.text(0.5, 0.012, stats, ha="center", color="#666", fontsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",  default="scene0000_00")
    ap.add_argument("--job",    type=int, default=14)
    ap.add_argument("--output", default=None)
    ap.add_argument("--list",   action="store_true")
    args = ap.parse_args()

    if args.list:
        for scene_dir in sorted(PROCESSED_ROOT.iterdir()):
            for jf in sorted((scene_dir / "edits").glob("job*.json")):
                try:
                    d = json.loads(jf.read_text())
                    if d.get("op") == "REMOVE":
                        idx = d.get("job_index", "?")
                        comp = (scene_dir / "completion" /
                                f"{scene_dir.name}_job{idx:03d}").exists() if isinstance(idx, int) else False
                        print(f"  {scene_dir.name}  job{idx:03}  "
                              f"{d.get('target_label','?'):<20}  "
                              f"OBB={'yes' if 'remove_obb' in d else 'no '}  "
                              f"completion={'yes' if comp else 'no'}")
                except Exception:
                    pass
        return

    out = Path(args.output) if args.output else \
          PROCESSED_ROOT / args.scene / "completion" / \
          f"{args.scene}_job{args.job:03d}" / "remove_comparison.png"
    visualize(args.scene, args.job, out)


if __name__ == "__main__":
    main()
