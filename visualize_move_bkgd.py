#!/usr/bin/env python3
"""
Visualize the ConvONet input for a MOVE job background fill.
Shows: original scene | after removal | ConvONet input (scan/wall/floor coloured).

Usage:
  python visualize_move_bkgd.py --job 0
  python visualize_move_bkgd.py --job 0 --output out.png
"""
from __future__ import annotations
import argparse, json, re, struct
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

HYPO_ROOT      = Path(__file__).resolve().parent
PROCESSED_ROOT = HYPO_ROOT / "repo" / "scannet" / "processed"
DATA_ROOT      = HYPO_ROOT / "repo" / "scannet" / "completion" / "convonet_data"
SCENE_ID       = "scene0000_00"
BG             = "#0d1117"


# ── PLY loader ────────────────────────────────────────────────────────────────

def load_ply(path: Path):
    if not path.exists():
        return np.empty((0, 3), np.float32), None
    data = path.read_bytes()
    hdr_end_idx = data.find(b"end_header")
    nl = data[hdr_end_idx + len("end_header")]
    hdr_end = hdr_end_idx + len("end_header") + (2 if nl == ord('\r') else 1)
    header = data[:hdr_end].decode("ascii", errors="ignore")
    n_verts = int(re.search(r"element vertex (\d+)", header).group(1))
    is_binary = "binary_little_endian" in header
    props = []
    in_v = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):   in_v = True
        elif line.startswith("element"):        in_v = False
        elif in_v and line.startswith("property") and "list" not in line:
            parts = line.split()
            if len(parts) >= 3: props.append((parts[1], parts[2]))
    type_map = {"float":("f",4),"float32":("f",4),"double":("d",8),
                "uchar":("B",1),"uint8":("B",1),"int":("i",4),"uint":("I",4),
                "short":("h",2),"ushort":("H",2),"char":("b",1)}
    if is_binary:
        fmt = "<" + "".join(type_map.get(t,("f",4))[0] for t,_ in props)
        sz = struct.calcsize(fmt)
        body = data[hdr_end:]
        arr = np.array([struct.unpack_from(fmt, body, i*sz) for i in range(n_verts)], dtype=np.float64)
    else:
        lines = data[hdr_end:].decode("ascii", errors="ignore").splitlines()
        arr = np.array([list(map(float, l.split())) for l in lines[:n_verts] if l.strip()], dtype=np.float64)
    if arr.shape[0] == 0:
        return np.empty((0,3),np.float32), None
    n2c = {name:i for i,(_, name) in enumerate(props)}
    xyz = arr[:, [n2c["x"], n2c["y"], n2c["z"]]].astype(np.float32)
    rgb = None
    for names in [("red","green","blue"),("r","g","b")]:
        if all(n in n2c for n in names):
            raw = arr[:, [n2c[n] for n in names]]
            rgb = (raw/raw.max()*255).astype(np.uint8) if raw.max()>1.5 else (raw*255).astype(np.uint8)
            break
    return xyz, rgb


# ── drawing helpers ───────────────────────────────────────────────────────────

def styled_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#555", labelsize=5)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.grid(True, color="#1a1a2a", linewidth=0.4)

def scatter(ax, xyz, rgb=None, color=None, alpha=0.6, s=None, proj=(0,2)):
    if xyz is None or xyz.shape[0] == 0:
        ax.text(0.5,0.5,"no data",color="#555",ha="center",va="center",transform=ax.transAxes,fontsize=7)
        return
    n = xyz.shape[0]
    if s is None: s = 3.0 if n<15000 else (1.0 if n<60000 else 0.3)
    x, y = xyz[:,proj[0]], xyz[:,proj[1]]
    if color is not None:
        ax.scatter(x, y, s=s, c=color, alpha=alpha, linewidths=0, rasterized=True)
    elif rgb is not None:
        ax.scatter(x, y, s=s, c=rgb.astype(np.float32)/255, alpha=alpha, linewidths=0, rasterized=True)
    else:
        ax.scatter(x, y, s=s, color="#888", alpha=alpha, linewidths=0, rasterized=True)

def draw_box(ax, center, size, proj, color="#ff8c00", lw=1.8, label=None, linestyle="--"):
    from matplotlib.patches import Rectangle
    i, j = proj
    c, s = np.array(center), np.array(size)
    ax.add_patch(Rectangle(
        (c[i]-s[i]/2, c[j]-s[j]/2), s[i], s[j],
        fill=False, edgecolor=color, linewidth=lw, linestyle=linestyle
    ))
    if label:
        ax.text(c[i]-s[i]/2, c[j]+s[j]/2, label, color=color, fontsize=5,
                ha="left", va="bottom", fontweight="bold")

def set_lim(ax, center, size, pad, proj):
    i, j = proj
    c, s = np.array(center), np.array(size)
    ax.set_xlim(c[i]-s[i]/2-pad, c[i]+s[i]/2+pad)
    ax.set_ylim(c[j]-s[j]/2-pad, c[j]+s[j]/2+pad)
    ax.set_aspect("equal")


# ── main ──────────────────────────────────────────────────────────────────────

def visualize(job_index: int, out_path: Path):
    job_id   = f"{SCENE_ID}_job{job_index:03d}_bkgd"
    comp_dir = PROCESSED_ROOT / SCENE_ID / "completion" / job_id
    data_dir = DATA_ROOT / job_id / "scannet_remove" / job_id

    meta = json.loads((comp_dir / "metadata.json").read_text())
    label  = meta.get("model_id", job_id)
    center = meta["remove_center"]
    size   = meta["remove_size"]

    # Fill region (may differ from object AABB)
    fill_region = None
    fr_path = comp_dir / "fill_region.json"
    if fr_path.exists():
        fill_region = json.loads(fr_path.read_text())

    print(f"Loading data for {job_id} ...")

    # Scene original
    orig_xyz, orig_rgb = load_ply(PROCESSED_ROOT / SCENE_ID / f"original_{SCENE_ID}.ply")
    # After removal (scene crop, no fill)
    after_xyz, after_rgb = load_ply(comp_dir / "crop_after_remove.ply")
    # Fill components
    floor_xyz, _ = load_ply(comp_dir / "structural_surface_completed_points.ply")
    wall_xyz,  _ = load_ply(comp_dir / "nn_floor_fill_completed_points.ply")
    # Full ConvONet input
    pc = np.load(data_dir / "pointcloud.npz")
    input_xyz  = pc["points"]
    input_feat = pc.get("features")

    if input_feat is not None:
        scan_mask = input_feat[:,0] == 0.0
        fill_mask = input_feat[:,0] == 1.0
        scan_xyz  = input_xyz[scan_mask]
        fill_xyz  = input_xyz[fill_mask]
    else:
        scan_xyz, fill_xyz = input_xyz, np.empty((0,3), np.float32)

    print(f"  original:   {orig_xyz.shape[0]:,} pts")
    print(f"  after_rm:   {after_xyz.shape[0]:,} pts")
    print(f"  floor fill: {floor_xyz.shape[0]:,} pts")
    print(f"  wall fill:  {wall_xyz.shape[0]:,} pts")
    print(f"  total input:{input_xyz.shape[0]:,} pts")

    PAD = 0.6
    ROWS = [("Top-down  (X→, Z↑)", (0,2)), ("Front view  (X→, Y↑)", (0,1))]
    # Load completion outputs if they exist
    convonet_pts, _  = load_ply(comp_dir / "convonet_completed_points.ply")
    poisson_pts, _   = load_ply(comp_dir / "poisson_completion_mesh.ply")
    final_pts, final_rgb = load_ply(comp_dir / "completed_scene_pointcloud.ply")

    COLS = [
        "Original scene",
        "After removal\n(no fill)",
        "Floor fill\n(structural surface)",
        "Wall fill\n(nn_floor_fill)",
        "ConvONet input\n(combined)",
    ]
    if convonet_pts.shape[0] > 0:
        COLS.append("ConvONet\ncompletion")
    if poisson_pts.shape[0] > 0:
        COLS.append("Poisson\nmesh")
    if final_pts.shape[0] > 0:
        COLS.append("Final result\n(completed scene)")
    COLORS = {
        "floor": "#2ecc71",
        "wall":  "#e67e22",
        "scan":  "#3498db",
        "fill":  "#e74c3c",
    }

    fig = plt.figure(figsize=(5*len(COLS), 8), facecolor=BG)
    fig.suptitle(f"{SCENE_ID}  job{job_index:03d}_bkgd  —  {label}", color="white", fontsize=12, y=0.99)

    gs = GridSpec(2, len(COLS), figure=fig, hspace=0.25, wspace=0.08,
                  top=0.93, bottom=0.07, left=0.04, right=0.98)

    for row_i, (row_label, proj) in enumerate(ROWS):
        for col_i, col_title in enumerate(COLS):
            ax = fig.add_subplot(gs[row_i, col_i])
            styled_ax(ax)
            if row_i == 0:
                ax.set_title(col_title, color="white", fontsize=8, pad=4)
            if col_i == 0:
                ax.set_ylabel(row_label, color="#aaa", fontsize=7)

            def draw_both_boxes(ax, proj):
                """Draw object AABB (orange dashed) and fill_region (cyan solid)."""
                draw_box(ax, center, size, proj, color="#ff8c00", lw=1.2,
                         label="obj AABB", linestyle="--")
                if fill_region is not None:
                    fr_c = fill_region["center"]
                    fr_s = fill_region["size"]
                    draw_box(ax, fr_c, fr_s, proj, color="#00e5ff", lw=1.5,
                             label="fill region", linestyle="-")

            col_title_idx = COLS[col_i]

            if col_i == 0:  # Original scene (cropped to region)
                pad_xyz = np.array(center); pad_sz = np.array(size)
                mn = pad_xyz - pad_sz/2 - PAD*2; mx = pad_xyz + pad_sz/2 + PAD*2
                mask = np.all((orig_xyz>=mn)&(orig_xyz<=mx), axis=1)
                c_xyz = orig_xyz[mask]; c_rgb = orig_rgb[mask] if orig_rgb is not None else None
                scatter(ax, c_xyz, c_rgb, proj=proj)
                draw_both_boxes(ax, proj)

            elif col_i == 1:  # After removal
                scatter(ax, after_xyz, after_rgb, proj=proj)
                draw_both_boxes(ax, proj)

            elif col_i == 2:  # Floor fill only
                scatter(ax, after_xyz, alpha=0.25, proj=proj)
                scatter(ax, floor_xyz, color=COLORS["floor"], alpha=0.9, s=3, proj=proj)
                draw_both_boxes(ax, proj)
                ax.set_title(f"Floor fill\n({floor_xyz.shape[0]:,} pts)", color=COLORS["floor"], fontsize=8, pad=4)

            elif col_i == 3:  # Wall fill only
                scatter(ax, after_xyz, alpha=0.25, proj=proj)
                scatter(ax, wall_xyz, color=COLORS["wall"], alpha=0.9, s=3, proj=proj)
                draw_both_boxes(ax, proj)
                ax.set_title(f"Wall fill\n({wall_xyz.shape[0]:,} pts)", color=COLORS["wall"], fontsize=8, pad=4)

            elif col_i == 4:  # ConvONet input combined
                scatter(ax, scan_xyz, color=COLORS["scan"], alpha=0.4, proj=proj)
                scatter(ax, fill_xyz, color=COLORS["fill"], alpha=0.9, s=4, proj=proj)
                draw_both_boxes(ax, proj)
                ax.set_title(f"ConvONet input\n(scan={scan_xyz.shape[0]:,}, fill={fill_xyz.shape[0]:,})",
                             color="white", fontsize=8, pad=4)

            elif "ConvONet" in col_title_idx and convonet_pts.shape[0] > 0:
                scatter(ax, after_xyz, alpha=0.25, proj=proj)
                scatter(ax, convonet_pts, color="#3498db", alpha=0.9, s=3, proj=proj)
                draw_both_boxes(ax, proj)
                ax.set_title(f"ConvONet\n({convonet_pts.shape[0]:,} pts)", color="#3498db", fontsize=8, pad=4)

            elif "Poisson" in col_title_idx and poisson_pts.shape[0] > 0:
                scatter(ax, after_xyz, alpha=0.25, proj=proj)
                scatter(ax, poisson_pts, color="#e74c3c", alpha=0.9, s=2, proj=proj)
                draw_both_boxes(ax, proj)
                ax.set_title(f"Poisson mesh\n({poisson_pts.shape[0]:,} pts)", color="#e74c3c", fontsize=8, pad=4)

            elif "Final" in col_title_idx and final_pts.shape[0] > 0:
                scatter(ax, final_pts, final_rgb, alpha=0.7, proj=proj)
                draw_both_boxes(ax, proj)
                ax.set_title(f"Final result\n({final_pts.shape[0]:,} pts)", color="#2ecc71", fontsize=8, pad=4)

            set_lim(ax, center, size, PAD, proj)

    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(color=COLORS["floor"], label=f"Floor fill (structural surface, {floor_xyz.shape[0]:,} pts)"),
        mpatches.Patch(color=COLORS["wall"],  label=f"Wall fill (nn_floor_fill, {wall_xyz.shape[0]:,} pts)"),
        mpatches.Patch(color=COLORS["scan"],  label=f"Scan (after removal, {scan_xyz.shape[0]:,} pts)"),
        mpatches.Patch(color=COLORS["fill"],  label=f"Total fill input ({fill_xyz.shape[0]:,} pts)"),
        Line2D([0],[0], color="#ff8c00", lw=1.5, linestyle="--", label="Object AABB"),
        Line2D([0],[0], color="#00e5ff", lw=1.5, linestyle="-",  label="Fill region (aligned to walls)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
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
          PROCESSED_ROOT / SCENE_ID / "completion" / \
          f"{SCENE_ID}_job{args.job:03d}_bkgd" / "bkgd_fill_check.png"
    visualize(args.job, out)

if __name__ == "__main__":
    main()
