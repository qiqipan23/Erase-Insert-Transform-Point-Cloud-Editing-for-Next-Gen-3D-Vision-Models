"""
Text-only vs visual comparison figure for GPT-4o.

3 conditions:
  - Text only        (context description, no image)
  - Image, no labels (rendered point cloud, no instance labels)
  - Image + labels   (rendered point cloud + instance labels)

3 panels:
  (a) Overall EM with gain annotations
  (b) By question type
  (c) By change type

Output: figs/fig_text_vs_visual.pdf / .png
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

HYPO = Path("/rds/general/user/qp23/home/Hypo3D")
EXP  = HYPO / "exp"
FIGS = HYPO / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── Normalisation ─────────────────────────────────────────────────────────────
REP = {'to the':'','by the':'','on the':'','near':'','next':'','corner':'',
       'behind':'back','right side':'right','left side':'left',
       'in front of':'front','on the left of':'left','on the right of':'right',
       'on the left':'left','on the right':'right',
       'left of':'left','right of':'right','front of':'front','back of':'back'}
SK  = sorted(REP.keys(), key=len, reverse=True)
PAT = re.compile(r'\b(' + '|'.join(map(re.escape, SK)) + r')\b')
def norm(t):
    t = t.lower(); t = PAT.sub(lambda m: REP[m.group(0)], t)
    t = re.sub(r'\b(?:a|an|the)\b', '', t).strip()
    t = re.sub(r'[^a-zA-Z0-9\s]', '', t); return t.strip()

CHANGE_TYPES = ['Movement', 'Removal', 'Attribute', 'Addition', 'Replacement']
QTYPES       = ['Direction', 'Scale', 'Semantic']

def compute(fname):
    p = EXP / fname
    if not p.exists(): return None
    d = json.load(open(p)); tot = 0; hit = 0
    ct_t: dict = defaultdict(int); ct_h: dict = defaultdict(int)
    qt_t: dict = defaultdict(int); qt_h: dict = defaultdict(int)
    for scene, changes in d.items():
        for ch in changes:
            raw = ch.get('change_type', '')
            ct  = raw.split()[1] if raw and len(raw.split()) > 1 else raw
            for qa in ch['questions_answers']:
                ref  = norm(str(qa.get('answer', '')))
                pred = norm(str(qa.get('predicted_answer', '')))
                h = int(ref == pred); tot += 1; hit += h
                ct_t[ct] += 1; ct_h[ct] += h
                for qt in str(qa.get('question_type', '')).split():
                    qt_t[qt] += 1; qt_h[qt] += h
    return {
        'overall': hit / tot * 100 if tot else 0,
        'ct': {k: ct_h[k] / ct_t[k] * 100 for k in CHANGE_TYPES if ct_t[k]},
        'qt': {k: qt_h[k] / qt_t[k] * 100 for k in QTYPES if qt_t[k]},
    }

CONDITIONS = [
    ('Text only\n(no image)',    'contextvqa_GPT4o_text.json',             '#607D8B'),
    ('Image only\n(no labels)',  'contextvqa_GPT4o_no_label_rotated.json', '#FF8F00'),
    ('Image + labels\n(full)',   'contextvqa_GPT4o_with_label_rotated.json','#1565C0'),
]

data = [(label, compute(fname), col) for label, fname, col in CONDITIONS]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5), facecolor="white")
gs  = GridSpec(1, 3, figure=fig, wspace=0.38,
               left=0.07, right=0.97, top=0.87, bottom=0.16)

# ── (a) Overall EM ────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

labels_a = [d[0] for d in data]
ems_a    = [d[1]['overall'] for d in data]
cols_a   = [d[2] for d in data]

bars = ax_a.bar(range(3), ems_a, color=cols_a, width=0.55,
                edgecolor='white', linewidth=0.5, zorder=3)

for i, (bar, v) in enumerate(zip(bars, ems_a)):
    ax_a.text(bar.get_x() + bar.get_width()/2, v + 0.5,
              f'{v:.1f}%', ha='center', va='bottom',
              fontsize=9.5, fontweight='bold')

# gain arrows: text→no_label and no_label→full
pairs = [(0, 1), (0, 2)]
y_arr = max(ems_a) + 4
for i, (src, tgt) in enumerate(pairs):
    gain = ems_a[tgt] - ems_a[src]
    col  = '#E53935' if gain < 0 else '#2E7D32'
    sign = '+' if gain >= 0 else ''
    ax_a.annotate('',
        xy=(tgt, ems_a[tgt] + 1.2),
        xytext=(src, ems_a[src] + 1.2),
        arrowprops=dict(arrowstyle='-|>', color=col, lw=1.4,
                        connectionstyle='arc3,rad=-0.25'))
    mid_x = (src + tgt) / 2
    ax_a.text(mid_x, y_arr + i * 3.5, f'{sign}{gain:.1f} pp',
              ha='center', fontsize=9, color=col, fontweight='bold')

ax_a.set_xticks(range(3))
ax_a.set_xticklabels(labels_a, fontsize=9)
ax_a.set_ylabel('Exact Match (%)')
ax_a.set_ylim(0, 52)
ax_a.set_title('(a)  Overall EM', fontweight='bold')
ax_a.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)

# ── (b) By question type ──────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

x  = np.arange(len(QTYPES))
W  = 0.26
for i, (label, m, col) in enumerate(data):
    vals = [m['qt'].get(qt, 0) for qt in QTYPES]
    offset = (i - 1) * W
    bars_b = ax_b.bar(x + offset, vals, W, color=col,
                      edgecolor='white', linewidth=0.4, zorder=3,
                      label=label.replace('\n', ' '))
    for bar, v in zip(bars_b, vals):
        ax_b.text(bar.get_x() + bar.get_width()/2, v + 0.4,
                  f'{v:.0f}', ha='center', va='bottom', fontsize=7.5)

# highlight the image-only < text-only finding
ax_b.annotate('Image alone\nhurts vs text!',
              xy=(0 - W, data[1][1]['qt']['Direction']),
              xytext=(-0.15, 25),
              fontsize=7.5, color='#E53935',
              arrowprops=dict(arrowstyle='->', color='#E53935', lw=0.8))

ax_b.set_xticks(x); ax_b.set_xticklabels(QTYPES, fontsize=10)
ax_b.set_ylabel('Exact Match (%)')
ax_b.set_title('(b)  EM by question type', fontweight='bold')
ax_b.set_ylim(0, 62)
ax_b.legend(fontsize=8, framealpha=0.9, loc='upper left')
ax_b.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)

# ── (c) By change type ────────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])

x  = np.arange(len(CHANGE_TYPES))
CT_SHORT = ['Move', 'Remove', 'Attrib.', 'Add', 'Replace']
W2 = 0.26
for i, (label, m, col) in enumerate(data):
    vals = [m['ct'].get(ct, 0) for ct in CHANGE_TYPES]
    offset = (i - 1) * W2
    ax_c.bar(x + offset, vals, W2, color=col,
             edgecolor='white', linewidth=0.4, zorder=3)

ax_c.set_xticks(x); ax_c.set_xticklabels(CT_SHORT, fontsize=9)
ax_c.set_ylabel('Exact Match (%)')
ax_c.set_title('(c)  EM by change type', fontweight='bold')
ax_c.set_ylim(0, 55)
ax_c.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)

# legend using colour patches
handles_c = [mpatches.Patch(color=col, label=lbl.replace('\n', ' '))
             for lbl, _, col in CONDITIONS]
ax_c.legend(handles=handles_c, fontsize=8, framealpha=0.9, loc='upper left')

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "GPT-4o: Text-only vs Visual Input  —  What does the pipeline's scene rendering contribute?\n"
    "Text only (context description, no image)  ·  "
    "Image only (rendered point cloud, no labels)  ·  "
    "Image + labels (full pipeline output)",
    fontsize=9.5, fontweight='bold', y=1.01,
)

for ext in ('pdf', 'png'):
    out = FIGS / f'fig_text_vs_visual.{ext}'
    plt.savefig(out, facecolor='white')
    print(f'Saved: {out}')
plt.close()
