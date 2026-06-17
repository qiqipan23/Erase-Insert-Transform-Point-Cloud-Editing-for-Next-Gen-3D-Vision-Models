"""
MOVE pipeline labelled-vs-unlabelled comparison figure.

Compares, on the matched 1182-question subset:
  - MOVE pipeline render, no labels   (contextvqa_move_eval_gpt4o_v2.json)
  - MOVE pipeline render, with labels (contextvqa_move_eval_gpt4o_labelled.json)
  - Full Hypo3D Movement benchmark    (reference upper line)

Two panels:
  (a) Overall EM bars + full-benchmark reference line
  (b) EM by question type (Direction / Scale / Semantic)

Output: figs/fig_labelled_comparison.pdf / .png
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
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

HYPO = Path("/rds/general/user/qp23/home/Hypo3D")
DS   = HYPO / "dataset"
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

REP = {'to the':'','by the':'','on the':'','near':'','next':'','corner':'',
       'behind':'back','right side':'right','left side':'left',
       'in front of':'front','on the left of':'left','on the right of':'right',
       'on the left':'left','on the right':'right',
       'left of':'left','right of':'right','front of':'front','back of':'back'}
SK  = sorted(REP, key=len, reverse=True)
PAT = re.compile(r'\b(' + '|'.join(map(re.escape, SK)) + r')\b')
def norm(t):
    t = t.lower(); t = PAT.sub(lambda m: REP[m.group(0)], t)
    t = re.sub(r'\b(?:a|an|the)\b', '', t).strip()
    t = re.sub(r'[^a-zA-Z0-9\s]', '', t); return t.strip()

QTYPES = ['Direction', 'Scale', 'Semantic']

def load_rows(path):
    d = json.load(open(path)); rows = {}
    for sc, changes in d.items():
        for ch in changes:
            for qa in ch.get('questions_answers', []):
                key = (sc, ch.get('job_stem',''), qa.get('question_id',''), qa.get('question',''))
                rows[key] = (norm(str(qa.get('answer',''))),
                             norm(str(qa.get('predicted_answer',''))),
                             str(qa.get('question_type','')))
    return rows

lab   = load_rows(DS / "contextvqa_move_eval_gpt4o_labelled.json")
unlab = load_rows(DS / "contextvqa_move_eval_gpt4o_v2.json")
common = set(lab) & set(unlab)

def em(rows, keys):
    tot = hit = 0; qt_t = defaultdict(int); qt_h = defaultdict(int)
    for k in keys:
        ref, pred, qt = rows[k]
        h = int(ref == pred and bool(pred)); tot += 1; hit += h
        for q in qt.split():
            qt_t[q] += 1; qt_h[q] += h
    overall = hit / tot * 100
    byqt = {q: qt_h[q] / qt_t[q] * 100 for q in qt_t}
    return overall, byqt

unlab_em, unlab_qt = em(unlab, common)
lab_em,   lab_qt   = em(lab,   common)

# Full-benchmark Movement reference (single-type questions, with labels)
def full_movement():
    d = json.load(open(EXP / "contextvqa_GPT4o_with_label_rotated.json"))
    tot = hit = 0; qt_t = defaultdict(int); qt_h = defaultdict(int)
    for sc, changes in d.items():
        for ch in changes:
            raw = ch.get('change_type','')
            ct = raw.split()[1] if raw and len(raw.split()) > 1 else raw
            if ct != 'Movement': continue
            for qa in ch['questions_answers']:
                ref = norm(str(qa.get('answer',''))); pred = norm(str(qa.get('predicted_answer','')))
                h = int(ref == pred); tot += 1; hit += h
                for q in str(qa.get('question_type','')).split():
                    qt_t[q] += 1; qt_h[q] += h
    return hit/tot*100, {q: qt_h[q]/qt_t[q]*100 for q in qt_t}

full_em, full_qt = full_movement()

print(f"matched={len(common)}  unlab={unlab_em:.1f}  lab={lab_em:.1f}  full={full_em:.1f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5), facecolor="white")
gs  = GridSpec(1, 2, figure=fig, wspace=0.30,
               left=0.07, right=0.97, top=0.86, bottom=0.13)

C_UNLAB = "#FF8F00"
C_LAB   = "#1565C0"
C_FULL  = "#9E9E9E"

# ── (a) overall ───────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
bars = ax.bar([0, 1], [unlab_em, lab_em],
              color=[C_UNLAB, C_LAB], width=0.5,
              edgecolor='white', linewidth=0.5, zorder=3)
for b, v in zip(bars, [unlab_em, lab_em]):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
# gain annotation
ax.annotate(f'{lab_em-unlab_em:+.1f} pp',
            xy=(0.5, max(unlab_em, lab_em)+3),
            ha='center', fontsize=10, color='#C62828', fontweight='bold')
# full-benchmark reference
ax.axhline(full_em, color=C_FULL, linewidth=2.0, linestyle='--', zorder=4,
           label=f'Full benchmark Movement ({full_em:.1f}%)')
ax.annotate('', xy=(1.35, lab_em), xytext=(1.35, full_em),
            arrowprops=dict(arrowstyle='<->', color='#777', lw=1.2))
ax.text(1.4, (lab_em+full_em)/2, f'gap\n{full_em-lab_em:.1f} pp',
        fontsize=8.5, color='#555', va='center')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pipeline render\n(no labels)', 'Pipeline render\n(with labels)'], fontsize=9.5)
ax.set_ylabel('Exact Match (%)')
ax.set_ylim(0, 50)
ax.set_xlim(-0.6, 1.9)
ax.set_title('(a)  Overall EM on MOVE pipeline outputs', fontweight='bold')
ax.legend(fontsize=8.5, framealpha=0.9, loc='upper left')
ax.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)

# ── (b) by question type ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(QTYPES)); W = 0.26
u = [unlab_qt.get(q,0) for q in QTYPES]
l = [lab_qt.get(q,0) for q in QTYPES]
f = [full_qt.get(q,0) for q in QTYPES]
b1 = ax2.bar(x-W, u, W, color=C_UNLAB, label='No labels', edgecolor='white', linewidth=0.4, zorder=3)
b2 = ax2.bar(x,   l, W, color=C_LAB,   label='With labels', edgecolor='white', linewidth=0.4, zorder=3)
b3 = ax2.bar(x+W, f, W, color=C_FULL,  label='Full benchmark', edgecolor='white', linewidth=0.4, zorder=3)
for bars in (b1, b2, b3):
    for b in bars:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 f'{b.get_height():.0f}', ha='center', va='bottom', fontsize=7)
ax2.set_xticks(x); ax2.set_xticklabels(QTYPES, fontsize=10)
ax2.set_ylabel('Exact Match (%)')
ax2.set_ylim(0, 70)
ax2.set_title('(b)  EM by question type', fontweight='bold')
ax2.legend(fontsize=8.5, framealpha=0.9)
ax2.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)

fig.suptitle(
    "Effect of Label Overlays on MOVE Pipeline Renders  "
    f"(matched {len(common)}-question subset, GPT-4o)\n"
    "Labels do not improve EM on edited-scene renders — the gap to the full "
    "benchmark reflects render quality, not missing labels",
    fontsize=9.5, fontweight='bold', y=1.0)

for ext in ('pdf', 'png'):
    out = FIGS / f'fig_labelled_comparison.{ext}'
    plt.savefig(out, facecolor='white')
    print(f'Saved: {out}')
plt.close()
