#!/usr/bin/env python3
"""
Same-style comparison as remove_comparison.png but for bkgd MOVE jobs.

Layout:
  Section 1 — Removal (2 cols × 2 rows)
    Columns : Original scene (red = removed region) | After AABB removal
    Rows    : Top-down (XZ)  | Front (XY)

  Section 2 — Completion pipeline (N cols × 2 rows)
    Columns : Floor fill | Wall fill | ConvONet | Poisson fill-region | Final result
    Rows    : Top-down (XZ) | Front (XY)

Usage:
  python visualize_bkgd_comparison.py --job 0
  python visualize_bkgd_comparison.py --job 0 --output out.png
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.collections import PolyCollection

HYPO_ROOT      = Path(__file__).resolve().parent
PROCESSED_ROOT = HYPO_ROOT / "repo" / "scannet" / "processed"
BG = "#0d1117"


# ── PLY loaders ───────────────────────────────────────────────────────────────

def _parse_header(data: bytes):
    hdr_end_idx = data.find(b"end_header")
    nl = data[hdr_end_idx + len("end_header")]
    hdr_end = hdr_end_idx + len("end_header") + (2 if nl == ord('\r') else 1)
    header = data[:hdr_end].decode("ascii", errors="ignore")
    n_v = int(re.search(r"element vertex (\d+)", header).group(1))
    m = re.search(r"element face (\d+)", header)
    n_f = int(m.group(1)) if m else 0
    is_bin = "binary_little_endian" in header
    props = []
    in_v = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):   in_v = True
        elif line.startswith("element"):        in_v = False
        elif in_v and line.startswith("property") and "list" not in line:
            parts = line.split()
            if len(parts) >= 3:
                props.append((parts[1], parts[2]))
    return hdr_end, n_v, n_f, is_bin, props

_TYPE = {"float":("f",4),"float32":("f",4),"double":("d",8),"float64":("d",8),
         "uchar":("B",1),"uint8":("B",1),"char":("b",1),"int8":("b",1),
         "short":("h",2),"int16":("h",2),"ushort":("H",2),"uint16":("H",2),
         "int":("i",4),"int32":("i",4),"uint":("I",4),"uint32":("I",4)}


def load_ply(path: Path):
    if not path.exists():
        return np.empty((0,3),np.float32), None
    data = path.read_bytes()
    hdr_end, n_v, _, is_bin, props = _parse_header(data)
    if n_v == 0:
        return np.empty((0,3),np.float32), None
    if is_bin:
        fmt = "<" + "".join(_TYPE.get(t,("f",4))[0] for t,_ in props)
        sz = struct.calcsize(fmt)
        body = data[hdr_end:]
        arr = np.array([struct.unpack_from(fmt, body, i*sz) for i in range(n_v)], dtype=np.float64)
    else:
        lines = data[hdr_end:].decode("ascii", errors="ignore").splitlines()
        arr = np.array([list(map(float,l.split())) for l in lines[:n_v] if l.strip()], dtype=np.float64)
    n2c = {nm:i for i,(_,nm) in enumerate(props)}
    xyz = arr[:,[n2c["x"],n2c["y"],n2c["z"]]].astype(np.float32)
    rgb = None
    for names in [("red","green","blue"),("r","g","b")]:
        if all(n in n2c for n in names):
            raw = arr[:,[n2c[n] for n in names]]
            rgb = (raw/raw.max()*255).astype(np.uint8) if raw.max()>1.5 else (raw*255).astype(np.uint8)
            break
    return xyz, rgb


def load_mesh_ply(path: Path):
    """Returns (verts, rgb, faces) for mesh PLY files."""
    if not path.exists():
        return np.empty((0,3),np.float32), None, None
    data = path.read_bytes()
    hdr_end, n_v, n_f, is_bin, props = _parse_header(data)
    if n_v == 0:
        return np.empty((0,3),np.float32), None, None
    if is_bin:
        fmt = "<" + "".join(_TYPE.get(t,("f",4))[0] for t,_ in props)
        sz = struct.calcsize(fmt)
        body = data[hdr_end:]
        arr = np.array([struct.unpack_from(fmt, body, i*sz) for i in range(n_v)], dtype=np.float64)
        f_offset = n_v * sz
        faces_list = []
        ptr = f_offset
        for _ in range(n_f):
            count = body[ptr]
            idxs = struct.unpack_from(f"<{count}i", body, ptr+1)
            faces_list.append(idxs[:3])
            ptr += 1 + count * 4
        f_arr = np.array(faces_list, dtype=np.int32) if faces_list else None
    else:
        lines = data[hdr_end:].decode("ascii", errors="ignore").splitlines()
        arr = np.array([list(map(float,l.split())) for l in lines[:n_v] if l.strip()], dtype=np.float64)
        f_lines = [l for l in lines[n_v:] if l.strip()]
        faces_list = [list(map(int,l.split()))[1:4] for l in f_lines[:n_f]]
        f_arr = np.array(faces_list, dtype=np.int32) if faces_list else None
    n2c = {nm:i for i,(_,nm) in enumerate(props)}
    verts = arr[:,[n2c["x"],n2c["y"],n2c["z"]]].astype(np.float32)
    rgb = np.zeros((n_v,3),dtype=np.uint8)
    for oi, nm in enumerate(("red","green","blue")):
        if nm in n2c:
            rgb[:,oi] = np.clip(arr[:,n2c[nm]],0,255).astype(np.uint8)
    return verts, rgb, f_arr


# ── drawing helpers ───────────────────────────────────────────────────────────

def styled_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#555", labelsize=5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.grid(True, color="#1a1a2a", linewidth=0.4)


def scatter(ax, xyz, rgb=None, color=None, alpha=0.6, s=None, proj=(0,2)):
    if xyz is None or xyz.shape[0] == 0:
        ax.text(0.5,0.5,"no data",color="#555",ha="center",va="center",
                transform=ax.transAxes, fontsize=7)
        return
    n = xyz.shape[0]
    if s is None:
        s = 3.0 if n<15000 else (1.0 if n<60000 else 0.3)
    x, y = xyz[:,proj[0]], xyz[:,proj[1]]
    if color is not None:
        ax.scatter(x, y, s=s, c=color, alpha=alpha, linewidths=0, rasterized=True)
    elif rgb is not None:
        ax.scatter(x, y, s=s, c=rgb.astype(np.float32)/255, alpha=alpha, linewidths=0, rasterized=True)
    else:
        ax.scatter(x, y, s=s, color="#888", alpha=alpha, linewidths=0, rasterized=True)


def render_mesh_2d(ax, verts, rgb, faces, proj=(0,2), alpha=0.95):
    if faces is None or len(faces) == 0:
        scatter(ax, verts, rgb, proj=proj)
        return
    i, j = proj
    v2d = verts[:,[i,j]]
    tris = v2d[faces]
    face_rgb = rgb[faces].mean(axis=1).astype(np.float32)/255.0
    coll = PolyCollection(tris, facecolors=face_rgb, edgecolors="none",
                          alpha=alpha, rasterized=True)
    ax.add_collection(coll)
    ax.autoscale_view()


def draw_aabb(ax, center, size, proj, color="#ff8c00", lw=1.5, label=None, ls="--"):
    from matplotlib.patches import Rectangle
    i, j = proj
    c, s = np.array(center), np.array(size)
    ax.add_patch(Rectangle(
        (c[i]-s[i]/2, c[j]-s[j]/2), s[i], s[j],
        fill=False, edgecolor=color, linewidth=lw, linestyle=ls
    ))
    if label:
        ax.text(c[i]-s[i]/2, c[j]+s[j]/2, label, color=color,
                fontsize=5, ha="left", va="bottom", fontweight="bold")


def crop(xyz, rgb, center, size, pad=2.0):
    half = np.array(size)/2 * pad
    mn = np.array(center) - half
    mx = np.array(center) + half
    mask = np.all((xyz >= mn) & (xyz <= mx), axis=1)
    return xyz[mask], (rgb[mask] if rgb is not None else None)


# ── main ─────────────────────────────────────────────────────────────────────

def visualize(job_index: int, out_path: Path):
    from scipy.spatial import cKDTree

    scene_id = "scene0000_00"
    scene_dir = PROCESSED_ROOT / scene_id
    job_id    = f"{scene_id}_job{job_index:03d}_bkgd"
    comp_dir  = scene_dir / "completion" / job_id

    meta   = json.loads((comp_dir / "metadata.json").read_text())
    center = meta["remove_center"]
    size   = meta["remove_size"]
    label  = meta.get("model_id", job_id)

    fill_region = None
    fr_path = comp_dir / "fill_region.json"
    if fr_path.exists():
        fill_region = json.loads(fr_path.read_text())

    print(f"Loading data for {job_id} ...")

    # Original scene
    orig_xyz, orig_rgb = load_ply(scene_dir / f"original_{scene_id}.ply")
    # After removal (full scene, from edits)
    edit_file = Path(meta["edit_file"])
    after_xyz, after_rgb = load_ply(edit_file)

    print(f"  original:     {orig_xyz.shape[0]:,} pts")
    print(f"  after removal:{after_xyz.shape[0]:,} pts")

    # Crop to region of interest
    PAD = 2.0
    orig_c,  orig_c_rgb  = crop(orig_xyz,  orig_rgb,  center, size, PAD)
    after_c, after_c_rgb = crop(after_xyz, after_rgb, center, size, PAD)

    # Mark removed points: in orig crop but far from after-removal cloud
    removed_mask = np.zeros(orig_c.shape[0], dtype=bool)
    if orig_c.shape[0] > 0 and after_c.shape[0] > 0:
        dists, _ = cKDTree(after_c).query(orig_c, k=1)
        removed_mask = dists > 0.03

    # Completion layers
    LAYERS = [
        ("Floor fill\n(structural surface)", "structural_surface_completed_points.ply", "#2ecc71"),
        ("Wall fill\n(nn_floor_fill)",        "nn_floor_fill_completed_points.ply",       "#e67e22"),
        ("ConvONet",                           "convonet_completed_points.ply",             "#3498db"),
        ("Poisson\nfill-region",               "poisson_completion_mesh.ply",               "#e74c3c"),
    ]
    comp_layers = []
    for title, fname, color in LAYERS:
        p = comp_dir / fname
        if p.exists() and p.stat().st_size > 200:
            xyz, rgb = load_ply(p)
            if xyz.shape[0] > 0:
                comp_layers.append((title, xyz, rgb, color, None))
                print(f"  {title.split(chr(10))[0]}: {xyz.shape[0]:,} pts")

    # Final result (full scene point cloud)
    final_xyz, final_rgb, final_faces = None, None, None
    for fname in ("completed_scene_pointcloud.ply", "poisson_scene_mesh.ply"):
        p = comp_dir / fname
        if p.exists() and p.stat().st_size > 1000:
            if fname.endswith("_mesh.ply"):
                final_xyz, final_rgb, final_faces = load_mesh_ply(p)
            else:
                final_xyz, final_rgb = load_ply(p)
                final_faces = None
            print(f"  Final ({fname}): {final_xyz.shape[0]:,} pts")
            break
    if final_xyz is not None:
        comp_layers.append(("Final result\n(full scene)", final_xyz, final_rgb, "white", final_faces))

    # Axis limits
    PAD_LIM = 0.6
    cx, cy, cz = center
    sx, sy, sz = size
    xlim = (cx - sx/2 - PAD_LIM, cx + sx/2 + PAD_LIM)
    ylim = (cy - sy/2 - PAD_LIM, cy + sy/2 + PAD_LIM)
    zlim = (cz - sz/2 - PAD_LIM, cz + sz/2 + PAD_LIM)
    scene_xlim = (float(orig_xyz[:,0].min())-0.1, float(orig_xyz[:,0].max())+0.1)
    scene_ylim = (float(orig_xyz[:,1].min())-0.1, float(orig_xyz[:,1].max())+0.1)
    scene_zlim = (float(orig_xyz[:,2].min())-0.1, float(orig_xyz[:,2].max())+0.1)

    # Figure
    n_cc = len(comp_layers)
    fig = plt.figure(figsize=(max(10, 5*n_cc), 16), facecolor=BG)
    fig.suptitle(f"{scene_id}  job{job_index:03d}_bkgd  —  {label}  removal  (AABB)",
                 color="white", fontsize=13, y=0.99)

    master = GridSpec(2, 1, figure=fig, hspace=0.10,
                      top=0.96, bottom=0.04, left=0.04, right=0.98)

    # ── Section 1: Original | After removal ───────────────────────────────────
    gs1 = GridSpecFromSubplotSpec(2, 2, subplot_spec=master[0], hspace=0.28, wspace=0.10)
    col_titles_1 = ["Original scene\n(red = removed region)", "After AABB removal"]
    row_labels   = ["Top-down  (X→, Z↑)", "Front view  (X→, Y↑)"]
    axes1 = [[fig.add_subplot(gs1[r, c]) for c in range(2)] for r in range(2)]

    for r in range(2):
        for c in range(2):
            styled_ax(axes1[r][c])
            if r == 0:
                axes1[r][c].set_title(col_titles_1[c], color="white", fontsize=9, pad=4)
            if c == 0:
                axes1[r][c].set_ylabel(row_labels[r], color="#aaa", fontsize=8)

    def render_removal_row(row, proj, xlim_, ylim_):
        ax_o, ax_a = axes1[row]
        keep  = orig_c[~removed_mask]
        keep_rgb = orig_c_rgb[~removed_mask] if orig_c_rgb is not None else None
        rem   = orig_c[removed_mask]
        scatter(ax_o, keep, keep_rgb, proj=proj)
        if rem.shape[0] > 0:
            scatter(ax_o, rem, color="#ff4444", alpha=0.9, proj=proj)
        draw_aabb(ax_o, center, size, proj, color="#ff8c00", label="AABB")
        if fill_region:
            draw_aabb(ax_o, fill_region["center"], fill_region["size"],
                      proj, color="#00e5ff", ls="-", label="fill region")

        scatter(ax_a, after_c, after_c_rgb, proj=proj)
        draw_aabb(ax_a, center, size, proj, color="#ff8c00")
        if fill_region:
            draw_aabb(ax_a, fill_region["center"], fill_region["size"],
                      proj, color="#00e5ff", ls="-")

        for ax in (ax_o, ax_a):
            ax.set_xlim(*xlim_); ax.set_ylim(*ylim_); ax.set_aspect("equal")

    render_removal_row(0, (0,2), xlim, zlim)
    render_removal_row(1, (0,1), xlim, ylim)

    # ── Section 2: Completion pipeline ────────────────────────────────────────
    gs2 = GridSpecFromSubplotSpec(2, n_cc, subplot_spec=master[1], hspace=0.28, wspace=0.10)
    axes2 = [[fig.add_subplot(gs2[r, c]) for c in range(n_cc)] for r in range(2)]

    for r in range(2):
        for c in range(n_cc):
            styled_ax(axes2[r][c])
            title, _, _, color, _ = comp_layers[c]
            if r == 0:
                axes2[r][c].set_title(title, color=color, fontsize=8, pad=4)
            if c == 0:
                axes2[r][c].set_ylabel(row_labels[r], color="#aaa", fontsize=8)

    def render_completion_row(row, proj, xlim_, ylim_, sxlim, szlim):
        for c, (title, xyz, rgb, color, faces) in enumerate(comp_layers):
            ax = axes2[row][c]
            is_final = title.startswith("Final")
            if is_final:
                if faces is not None:
                    render_mesh_2d(ax, xyz, rgb, faces, proj=proj)
                else:
                    scatter(ax, xyz, rgb, proj=proj)
                draw_aabb(ax, center, size, proj, color="#ff8c00", lw=1.0)
                if fill_region:
                    draw_aabb(ax, fill_region["center"], fill_region["size"],
                              proj, color="#00e5ff", lw=1.0, ls="-")
                ax.set_xlim(*sxlim); ax.set_ylim(*szlim); ax.set_aspect("equal")
            else:
                n_pts = xyz.shape[0]
                comp_c, comp_rgb = crop(xyz, rgb, center, size, PAD)
                scatter(ax, after_c, after_c_rgb, alpha=0.25, proj=proj)
                scatter(ax, comp_c, color=color, alpha=0.9, proj=proj)
                ax.set_title(f"{title.split(chr(10))[0]}\n({n_pts:,} pts)",
                             color=color, fontsize=8, pad=4)
                draw_aabb(ax, center, size, proj, color="#ff8c00")
                if fill_region:
                    draw_aabb(ax, fill_region["center"], fill_region["size"],
                              proj, color="#00e5ff", ls="-")
                ax.set_xlim(*xlim_); ax.set_ylim(*ylim_); ax.set_aspect("equal")

    render_completion_row(0, (0,2), xlim, zlim, scene_xlim, scene_zlim)
    render_completion_row(1, (0,1), xlim, ylim, scene_xlim, scene_ylim)

    fig.text(0.5, master[1].get_position(fig).y1 + 0.005,
             "── Completion pipeline ──",
             color="#aaaaaa", fontsize=10, ha="center", va="bottom")

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(color="#ff4444", label="Removed points"),
        Line2D([0],[0], color="#ff8c00", lw=1.5, ls="--", label="Object AABB"),
        Line2D([0],[0], color="#00e5ff", lw=1.5, ls="-",  label="Fill region"),
    ]
    for title, _, _, color, _ in comp_layers:
        handles.append(mpatches.Patch(color=color, label=title.replace("\n"," ")))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles),5),
               facecolor="#1a1a2e", edgecolor="#333", labelcolor="white",
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job",    type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out = Path(args.output) if args.output else \
          PROCESSED_ROOT / "scene0000_00" / "completion" / \
          f"scene0000_00_job{args.job:03d}_bkgd" / "remove_comparison.png"
    visualize(args.job, out)


if __name__ == "__main__":
    main()
