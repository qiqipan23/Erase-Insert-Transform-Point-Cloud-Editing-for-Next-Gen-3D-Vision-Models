"""
Original Hypo3D vs. rendered-edit pipeline comparison (GPT-4o).

On the matched 1041-question Movement subset (identical scene+question+answer):
  - Original Hypo3D : original scene image + TEXT change description
  - Pipeline (no labels)  : rendered EDITED scene
  - Pipeline (with labels): rendered EDITED scene + instance labels

Output: figs/fig_orig_vs_pipeline.pdf / .png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HYPO = Path("/rds/general/user/qp23/home/Hypo3D")
FIGS = HYPO / "figs"; FIGS.mkdir(exist_ok=True)
d = json.load(open(HYPO / "dataset/_orig_vs_pipeline.json"))
N = d["n"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

QTYPES = ['Direction', 'Scale', 'Semantic']
labels = ['Original Hypo3D\n(text + original scene)',
          'Pipeline render\n(no labels)',
          'Pipeline render\n(with labels)',
          'Pipeline render\n(hi-res labels)']
keys   = ['orig', 'unlab', 'lab', 'lab_hires']
COLS   = ['#2E7D32', '#FF8F00', '#1565C0', '#6A1B9A']

overall = [d[k]['overall'] for k in keys]
byqt    = {k: d[k]['qt'] for k in keys}

fig = plt.figure(figsize=(15, 5), facecolor="white")
gs  = GridSpec(1, 2, figure=fig, wspace=0.30, left=0.07, right=0.97, top=0.85, bottom=0.16)

# ── (a) overall ───────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
bars = ax.bar(range(len(keys)), overall, color=COLS, width=0.6,
              edgecolor='white', linewidth=0.5, zorder=3)
for b, v in zip(bars, overall):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
# delta annotations vs original
for i in (1, 2):
    dlt = overall[i] - overall[0]
    ax.annotate(f'{dlt:+.1f} pp', xy=(i, overall[i]-3.0),
                ha='center', fontsize=9, color='#C62828', fontweight='bold')
ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('Exact Match (%)'); ax.set_ylim(0, 50)
ax.set_title('(a)  Overall EM (Movement questions)', fontweight='bold')
ax.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)

# ── (b) by question type ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
n_keys = len(keys); W = 0.18; group_gap = 0.08
total_w = n_keys * W + group_gap
x = np.arange(len(QTYPES)) * (total_w + 0.25)
offsets = [(i - (n_keys - 1) / 2) * W for i in range(n_keys)]
for i, k in enumerate(keys):
    vals = [byqt[k].get(q, 0) for q in QTYPES]
    bb = ax2.bar(x + offsets[i], vals, W, color=COLS[i],
                 edgecolor='white', linewidth=0.4, zorder=3,
                 label=labels[i].replace('\n', ' '))
    for b in bb:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 f'{b.get_height():.0f}', ha='center', va='bottom', fontsize=7)
ax2.set_xticks(x); ax2.set_xticklabels(QTYPES, fontsize=10)
ax2.set_ylabel('Exact Match (%)'); ax2.set_ylim(0, 60)
ax2.set_title('(b)  EM by question type', fontweight='bold')
ax2.legend(fontsize=7.5, framealpha=0.9, loc='upper right')
ax2.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)

fig.suptitle(
    f"Rendered-Edit Pipeline vs. Original Hypo3D  (GPT-4o, matched {N}-question Movement subset)\n"
    "Rendering the actual edited scene does not beat text-based hypothetical reasoning on the original scene",
    fontsize=9.5, fontweight='bold', y=1.0)

for ext in ('pdf', 'png'):
    out = FIGS / f'fig_orig_vs_pipeline.{ext}'
    plt.savefig(out, facecolor='white')
    print(f'Saved: {out}')
plt.close()
