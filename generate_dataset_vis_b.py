"""
Option B — Dataset-level statistics panel.

2×2 figure:
  (a) Top-10 removed object categories (horizontal bar)
  (b) Distribution of occupancy positive ratio per example
  (c) Distribution of crop half-extent (max axis) per example
  (d) Train / Val / Test split summary with per-split category breakdown

Output: figs/fig_dataset_vis_b.pdf / .png
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

HYPO    = Path("/rds/general/user/qp23/home/Hypo3D")
DS_ROOT = HYPO / "repo/scannet/completion/convonet_scannet_remove_train"
DS_DIR  = DS_ROOT / "scannet_remove"
OUT_DIR = HYPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── Load metadata + labels ────────────────────────────────────────────────────
lines = (HYPO / "repo/scannet/derived/generated_jobs_clean.jsonl").read_text().splitlines()
jobs  = {i: json.loads(l) for i, l in enumerate(lines) if l.strip()}

train_ids = set((DS_DIR / "train.lst").read_text().splitlines())
val_ids   = set((DS_DIR / "val.lst").read_text().splitlines())
test_ids  = set((DS_DIR / "test.lst").read_text().splitlines())

metas = []
for d in sorted(DS_DIR.iterdir()):
    mf = d / "metadata.json"
    if not mf.exists(): continue
    m = json.loads(mf.read_text())
    m["label"] = jobs.get(m["job_index"], {}).get("target_label", "unknown")
    if   d.name in train_ids: m["split"] = "train"
    elif d.name in val_ids:   m["split"] = "val"
    elif d.name in test_ids:  m["split"] = "test"
    else:                     m["split"] = "unassigned"
    metas.append(m)

labels_all  = [m["label"] for m in metas]
occ_ratios  = [m["occupancy_positive"] / m["points_iou_n"] for m in metas]
max_extents = [max(m["crop_half_extent"]) * 2 for m in metas]   # full extent, metres
splits      = [m["split"] for m in metas]
label_cnt   = Counter(labels_all)

# ── Colours ───────────────────────────────────────────────────────────────────
SPLIT_COLS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}
BAR_COL    = "#455A64"

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor="white")
gs  = GridSpec(2, 2, figure=fig,
               hspace=0.45, wspace=0.35,
               top=0.91, bottom=0.08, left=0.09, right=0.97)

# ── (a) Category bar chart ────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
top_n = 12
top_labels = [l for l, _ in label_cnt.most_common(top_n)]
top_counts = [label_cnt[l] for l in top_labels]
bars = ax_a.barh(range(len(top_labels)), top_counts,
                 color=BAR_COL, edgecolor="white", linewidth=0.5)
ax_a.set_yticks(range(len(top_labels)))
ax_a.set_yticklabels([l.capitalize() for l in top_labels], fontsize=9)
ax_a.invert_yaxis()
ax_a.set_xlabel("Number of examples", fontsize=9)
ax_a.set_title(f"(a)  Removed object categories\n(top {top_n} of {len(label_cnt)})",
               fontsize=10, fontweight="bold", pad=6)
ax_a.axvline(x=0, color="#ccc", linewidth=0.5)
for bar, cnt in zip(bars, top_counts):
    ax_a.text(cnt + 0.3, bar.get_y() + bar.get_height()/2,
              str(cnt), va="center", fontsize=8, color="#333")
ax_a.set_xlim(0, max(top_counts) * 1.18)
ax_a.grid(axis="x", color="#eeeeee", linewidth=0.5, zorder=0)

# ── (b) Occupancy positive ratio histogram ────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
occ_pct = np.array(occ_ratios) * 100
ax_b.hist(occ_pct, bins=30, color="#E53935", edgecolor="white",
          linewidth=0.4, alpha=0.85)
ax_b.axvline(np.mean(occ_pct), color="#333", linewidth=1.4,
             linestyle="--", label=f"Mean: {np.mean(occ_pct):.1f}%")
ax_b.axvline(np.median(occ_pct), color="#888", linewidth=1.2,
             linestyle=":", label=f"Median: {np.median(occ_pct):.1f}%")
ax_b.set_xlabel("Occupied query points (%)", fontsize=9)
ax_b.set_ylabel("Examples", fontsize=9)
ax_b.set_title("(b)  Occupancy positive ratio\nper training example",
               fontsize=10, fontweight="bold", pad=6)
ax_b.legend(fontsize=8.5, framealpha=0.85)
ax_b.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

# ── (c) Crop size distribution ────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
extents = np.array(max_extents)
ax_c.hist(extents, bins=30, color="#1565C0", edgecolor="white",
          linewidth=0.4, alpha=0.85)
ax_c.axvline(np.mean(extents), color="#333", linewidth=1.4,
             linestyle="--", label=f"Mean: {np.mean(extents):.2f} m")
ax_c.axvline(np.median(extents), color="#888", linewidth=1.2,
             linestyle=":", label=f"Median: {np.median(extents):.2f} m")
ax_c.set_xlabel("Crop extent — longest axis (m)", fontsize=9)
ax_c.set_ylabel("Examples", fontsize=9)
ax_c.set_title("(c)  Crop bounding-box size\ndistribution",
               fontsize=10, fontweight="bold", pad=6)
ax_c.legend(fontsize=8.5, framealpha=0.85)
ax_c.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

# ── (d) Split summary stacked bar ────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
split_names  = ["train", "val", "test"]
split_totals = [splits.count(s) for s in split_names]

# top-6 categories per split as stacked bars
top6 = [l for l, _ in label_cnt.most_common(6)]
split_label_counts = defaultdict(lambda: defaultdict(int))
for m in metas:
    split_label_counts[m["split"]][m["label"]] += 1

CMAP = plt.get_cmap("tab10")
bottoms = np.zeros(3)
for i, lbl in enumerate(top6):
    vals = [split_label_counts[s][lbl] for s in split_names]
    ax_d.bar(split_names, vals, bottom=bottoms,
             color=CMAP(i), edgecolor="white", linewidth=0.4,
             label=lbl.capitalize(), alpha=0.88)
    bottoms += np.array(vals, dtype=float)

# add "other" on top
other_vals = [split_totals[i] - int(bottoms[i]) for i in range(3)]
ax_d.bar(split_names, other_vals, bottom=bottoms,
         color="#B0BEC5", edgecolor="white", linewidth=0.4,
         label="Other", alpha=0.85)

for i, (name, tot) in enumerate(zip(split_names, split_totals)):
    ax_d.text(i, tot + 2, str(tot), ha="center", fontsize=9,
              color="#222", fontweight="bold")

ax_d.set_ylabel("Examples", fontsize=9)
ax_d.set_title("(d)  Train / Val / Test split\nby object category",
               fontsize=10, fontweight="bold", pad=6)
ax_d.legend(fontsize=8, ncol=2, framealpha=0.85, loc="upper right")
ax_d.set_ylim(0, max(split_totals) * 1.18)
ax_d.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

# ── Title ─────────────────────────────────────────────────────────────────────
n_scenes = len(set(m["scene_id"] for m in metas))
fig.suptitle(
    f"ConvONet Fine-tuning Dataset Statistics\n"
    f"{len(metas)} examples  ·  {n_scenes} ScanNet scenes  ·  "
    f"{len(label_cnt)} object categories  ·  "
    f"80 / 10 / 10 train / val / test split",
    fontsize=11, fontweight="bold", color="#111", y=0.97,
)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_dataset_vis_b.{ext}"
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
plt.close()
