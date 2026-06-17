"""
Round-trip evaluation summary figure.

3-panel layout:
  (a) Histogram of round-trip errors (XZ)
  (b) Per-relation mean RTE_XZ with N labels
  (c) Cumulative distribution — % jobs within X metres

Output: figs/fig_roundtrip_eval.pdf / .png
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

HYPO    = Path("/rds/general/user/qp23/home/Hypo3D")
OUT_DIR = HYPO / "figs"
OUT_DIR.mkdir(exist_ok=True)

results  = json.loads((HYPO / "dataset/round_trip_eval.json").read_text())
rev_jobs = [json.loads(l) for l in
            (HYPO / "dataset/reverse_jobs.jsonl").read_text().splitlines() if l.strip()]

valid = [r for r in results if "error" not in r]
rev_dict = {r["forward_job_stem"]: r for r in rev_jobs}

rte_xz  = np.array([r["round_trip_error_xz_m"] for r in valid])
rte_3d  = np.array([r["round_trip_error_m"]     for r in valid])
confs   = np.array([r["reverse_confidence"]      for r in valid])

# also load forward placement errors for comparison
geo = json.loads((HYPO / "dataset/move_eval_geometric.json").read_text())
fwd_pe = np.array([r["placement_error_m"] for r in geo if "error" not in r])

RELS_ORDER = ["FRONT_OF", "BACK_OF", "RIGHT_OF", "LEFT_OF", "ON_TOP_OF", "UNDER"]
REL_COLOUR = {
    "FRONT_OF":  "#1565C0",
    "BACK_OF":   "#1E88E5",
    "RIGHT_OF":  "#E53935",
    "LEFT_OF":   "#EF9A9A",
    "ON_TOP_OF": "#2E7D32",
    "UNDER":     "#81C784",
}

by_rel: dict = defaultdict(list)
for r in valid:
    by_rel[r["relation"]].append(r["round_trip_error_xz_m"])

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig = plt.figure(figsize=(14, 5), facecolor="white")
gs  = GridSpec(1, 3, figure=fig, wspace=0.35,
               left=0.07, right=0.97, top=0.87, bottom=0.14)

C_RTE = "#1565C0"
C_FWD = "#E53935"

# ── (a) Histogram ─────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

bins = np.linspace(0, min(rte_xz.max(), 5.0), 25)
ax_a.hist(rte_xz, bins=bins, color=C_RTE, alpha=0.80,
          edgecolor="white", linewidth=0.4, label="Round-trip (XZ)")
ax_a.hist(fwd_pe, bins=np.linspace(0, min(fwd_pe.max(), 5.0), 25),
          color=C_FWD, alpha=0.55,
          edgecolor="white", linewidth=0.4, label="Forward placement")

ax_a.axvline(np.mean(rte_xz), color=C_RTE, linewidth=1.6, linestyle="--",
             label=f"RTE mean: {np.mean(rte_xz):.2f} m")
ax_a.axvline(np.mean(fwd_pe), color=C_FWD, linewidth=1.6, linestyle=":",
             label=f"Fwd mean: {np.mean(fwd_pe):.2f} m")

ax_a.set_xlabel("Error (m)")
ax_a.set_ylabel("Jobs")
ax_a.set_title("(a)  Error distribution", fontweight="bold")
ax_a.legend(fontsize=8, framealpha=0.9)
ax_a.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

# ── (b) Per-relation mean RTE ─────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

rel_means = [np.mean(by_rel.get(r, [np.nan])) for r in RELS_ORDER]
rel_ns    = [len(by_rel.get(r, [])) for r in RELS_ORDER]
colours_b = [REL_COLOUR[r] for r in RELS_ORDER]

bars = ax_b.bar(range(len(RELS_ORDER)), rel_means,
                color=colours_b, edgecolor="white", linewidth=0.4,
                width=0.6)

ax_b.axhline(np.mean(rte_xz), color="#333", linewidth=1.4,
             linestyle="--", label=f"Overall mean: {np.mean(rte_xz):.2f} m")

for bar, v, n in zip(bars, rel_means, rel_ns):
    if not np.isnan(v):
        ax_b.text(bar.get_x() + bar.get_width()/2, v + 0.03,
                  f"{v:.2f}\n(N={n})", ha="center", va="bottom",
                  fontsize=7.5, color="#111")

ax_b.set_xticks(range(len(RELS_ORDER)))
ax_b.set_xticklabels(RELS_ORDER, rotation=30, ha="right", fontsize=8.5)
ax_b.set_ylabel("Mean RTE XZ (m)")
ax_b.set_title("(b)  Round-trip error by relation", fontweight="bold")
ax_b.legend(fontsize=8.5, framealpha=0.9)
ax_b.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)
ax_b.set_ylim(0, max(rel_means) * 1.35)

# ── (c) Cumulative distribution ───────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])

thresholds = np.linspace(0, 4.0, 200)
cdf_rte = [np.mean(rte_xz <= t) * 100 for t in thresholds]
cdf_fwd = [np.mean(fwd_pe <= t) * 100 for t in thresholds]

ax_c.plot(thresholds, cdf_rte, color=C_RTE, linewidth=2.0,
          label="Round-trip error (XZ)")
ax_c.plot(thresholds, cdf_fwd, color=C_FWD, linewidth=2.0,
          linestyle="--", label="Forward placement error")

# threshold markers
for thresh, ls in [(0.20, ":"), (0.50, "--")]:
    pct_rte = np.mean(rte_xz <= thresh) * 100
    pct_fwd = np.mean(fwd_pe <= thresh) * 100
    ax_c.axvline(thresh, color="#aaa", linewidth=0.8, linestyle=ls)
    ax_c.text(thresh + 0.04, 8, f"{thresh:.2f} m",
              fontsize=7.5, color="#666", rotation=90, va="bottom")
    ax_c.annotate(f"{pct_rte:.0f}%", xy=(thresh, pct_rte),
                  xytext=(thresh + 0.15, pct_rte),
                  fontsize=8, color=C_RTE,
                  arrowprops=dict(arrowstyle="-", color=C_RTE, lw=0.8))
    ax_c.annotate(f"{pct_fwd:.0f}%", xy=(thresh, pct_fwd),
                  xytext=(thresh + 0.15, pct_fwd - 6),
                  fontsize=8, color=C_FWD,
                  arrowprops=dict(arrowstyle="-", color=C_FWD, lw=0.8))

ax_c.set_xlabel("Error threshold (m)")
ax_c.set_ylabel("Jobs within threshold (%)")
ax_c.set_title("(c)  Cumulative distribution", fontweight="bold")
ax_c.legend(fontsize=8.5, framealpha=0.9)
ax_c.set_xlim(0, 4.0); ax_c.set_ylim(0, 105)
ax_c.grid(color="#eeeeee", linewidth=0.5, zorder=0)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    f"Round-Trip Evaluation  ·  {len(valid)} MOVE jobs  ·  "
    f"Mean RTE XZ = {np.mean(rte_xz):.2f} m  ·  "
    f"{np.mean(rte_xz <= 0.50)*100:.0f}% within 0.50 m\n"
    f"Anchors selected by: directional confidence × adjacency score",
    fontsize=10, fontweight="bold", y=1.00,
)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"fig_roundtrip_eval.{ext}"
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
plt.close()
