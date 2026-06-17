"""
Generate baseline comparison figure.

Two panels:
  (a) Overall EM: all models + majority baseline + GPT-4o text (no image) baseline
  (b) Per question-type: Direction / Scale / Semantic with baselines marked

Output: figs/fig_baseline_comparison.pdf / .png
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

HYPO    = Path("/rds/general/user/qp23/home/Hypo3D")
EXP     = HYPO / "exp"
DS      = HYPO / "dataset" / "hypo3d.json"
FIGS    = HYPO / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── Normalisation ─────────────────────────────────────────────────────────────
REP = {
    'back and right':'back right','back and left':'back left',
    'front and right':'front right','front and left':'front left',
    'to the':'','by the':'','on the':'','near':'','next':'','corner':'',
    'behind':'back','right side':'right','left side':'left',
    'in front of':'front','on the left of':'left','on the right of':'right',
    'on the left':'left','on the right':'right',
    'north':'front','south':'back','east':'right','west':'left',
    'forward':'front','backward':'back',
    'left of':'left','right of':'right','front of':'front','back of':'back',
}
SK  = sorted(REP.keys(), key=len, reverse=True)
PAT = re.compile(r'\b(' + '|'.join(map(re.escape, SK)) + r')\b')
def norm(t):
    t = t.lower()
    t = PAT.sub(lambda m: REP[m.group(0)], t)
    t = re.sub(r'\b(?:a|an|the)\b', '', t).strip()
    t = re.sub(r'[^a-zA-Z0-9\s]', '', t)
    return t.strip()

# ── Compute majority baseline from dataset ────────────────────────────────────
dataset = json.load(open(DS))
QTYPES  = ['Direction', 'Scale', 'Semantic']
by_qt: dict = defaultdict(list)
for scene, changes in dataset.items():
    for ch in changes:
        for qa in ch['questions_answers']:
            for qt in qa['question_type'].split():
                by_qt[qt].append(norm(qa['answer']))

qt_majority = {qt: Counter(ans).most_common(1)[0][0] for qt, ans in by_qt.items()}
qt_majority_em = {qt: Counter(ans).most_common(1)[0][1] / len(ans) * 100
                  for qt, ans in by_qt.items()}

# overall majority baseline
total = 0; hits = 0
qt_total: dict = defaultdict(int); qt_hits: dict = defaultdict(int)
for scene, changes in dataset.items():
    for ch in changes:
        for qa in ch['questions_answers']:
            ref  = norm(qa['answer'])
            qt   = qa['question_type'].split()[0]
            pred = qt_majority.get(qt, '')
            h    = int(ref == pred)
            total += 1; hits += h
            qt_total[qt] += 1; qt_hits[qt] += h
majority_overall = hits / total * 100
majority_by_qt   = {qt: qt_hits[qt] / qt_total[qt] * 100 for qt in QTYPES}

# ── Compute model metrics ─────────────────────────────────────────────────────
def compute(fname):
    p = EXP / fname
    if not p.exists(): return None
    d    = json.load(open(p))
    tot  = 0; hit = 0
    qt_t: dict = defaultdict(int); qt_h: dict = defaultdict(int)
    for scene, changes in d.items():
        for ch in changes:
            for qa in ch['questions_answers']:
                ref  = norm(str(qa.get('answer', '')))
                pred = norm(str(qa.get('predicted_answer', '')))
                h    = int(ref == pred)
                tot += 1; hit += h
                for qt in str(qa.get('question_type', '')).split():
                    qt_t[qt] += 1; qt_h[qt] += h
    return {
        'overall': hit / tot * 100 if tot else 0,
        'qtype':   {qt: qt_h[qt] / qt_t[qt] * 100 for qt in qt_t if qt_t[qt] > 0},
    }

MODELS = [
    # (label,          file,                                                      modality,  is_text_only)
    ('GPT-4o\n(Text)', 'contextvqa_GPT4o_text.json',                              'LLM',     True),
    ('LLaMA\n3.2-3B',  'contextvqa_llama-3.2-3B.json',                            'LLM',     False),
    ('GPT-4o\n(Vision)','contextvqa_GPT4o_with_label_rotated.json',               '2D VLM',  False),
    ('Claude\n3.5',    'contextvqa_Claude-3.5-Sonnet_with_label_rotated.json',     '2D VLM',  False),
    ('Qwen2-VL\n7B',   'contextvqa_Qwen2-VL-7B-Instruct_with_label_rotated.json', '2D VLM',  False),
    ('Qwen2-VL\n72B',  'contextvqa_Qwen2-VL-72B-Instruct-AWQ_with_label_rotated.json','2D VLM',False),
    ('LLaVA-OV\n7B',   'contextvqa_llava-onevision-qwen2-7b-ov-hf_with_label_rotated.json','2D VLM',False),
    ('LLaVA-OV\n72B',  'contextvqa_llava-onevision-qwen2-72b-ov-hf_with_label_rotated.json','2D VLM',False),
    ('LLaVA-3D\n7B',   'contextvqa_llava3d-7B.json',                              '3D VLM',  False),
    ('LEO',            'contextvqa_leo.json',                                     '3D VLM',  False),
]

MODALITY_COLOUR = {'LLM': '#4C72B0', '2D VLM': '#DD8452', '3D VLM': '#55A868'}

model_data = []
for label, fname, mod, text_only in MODELS:
    m = compute(fname)
    if m:
        model_data.append((label, m, mod, text_only))

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6), facecolor="white")
gs  = GridSpec(1, 2, figure=fig, wspace=0.35,
               left=0.07, right=0.97, top=0.88, bottom=0.18)

MAJORITY_COL = "#B0BEC5"
TEXT_COL     = "#7986CB"

# ── (a) Overall EM with baselines ─────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

labels  = [d[0] for d in model_data]
ems     = [d[1]['overall'] for d in model_data]
mods    = [d[2] for d in model_data]
colours = [MODALITY_COLOUR[m] for m in mods]
hatches = ['//' if d[3] else '' for d in model_data]   # hatch text-only

x = np.arange(len(labels))
bars = ax_a.bar(x, ems, color=colours, edgecolor='white',
                linewidth=0.5, width=0.65, zorder=3)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Majority baseline
ax_a.axhline(majority_overall, color=MAJORITY_COL, linewidth=2.0,
             linestyle='--', zorder=4,
             label=f'Majority-class baseline ({majority_overall:.1f}%)')

# Value labels
for bar, v in zip(bars, ems):
    ax_a.text(bar.get_x() + bar.get_width()/2, v + 0.4,
              f'{v:.0f}', ha='center', va='bottom', fontsize=7.5)

ax_a.set_xticks(x)
ax_a.set_xticklabels(labels, fontsize=8)
ax_a.set_ylabel('Exact Match (%)')
ax_a.set_title('(a)  Overall EM vs. majority baseline', fontweight='bold')
ax_a.set_ylim(0, 55)
ax_a.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)

# modality boundaries
prev = mods[0]
for i, m in enumerate(mods[1:], 1):
    if m != prev:
        ax_a.axvline(i - 0.5, color='#dddddd', linewidth=1.0, linestyle=':')
    prev = m

# legend
handles = [mpatches.Patch(color=c, label=m) for m, c in MODALITY_COLOUR.items()]
handles += [
    Line2D([0],[0], color=MAJORITY_COL, lw=2, linestyle='--',
           label=f'Majority baseline ({majority_overall:.1f}%)'),
    mpatches.Patch(facecolor='#aaa', hatch='//', edgecolor='white',
                   label='Text-only (no image)'),
]
ax_a.legend(handles=handles, fontsize=7.5, framealpha=0.9,
            loc='upper left', ncol=1)

# ── (b) Per question type with baselines ──────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

n_models = len(model_data)
n_qt     = len(QTYPES)
W        = 0.22
xb       = np.arange(len(QTYPES))

QT_COLOURS = {'Direction': '#1565C0', 'Scale': '#E53935', 'Semantic': '#2E7D32'}

for i, (label, m, mod, text_only) in enumerate(model_data):
    vals   = [m['qtype'].get(qt, 0) for qt in QTYPES]
    offset = (i - n_models/2 + 0.5) * (W * 0.85)
    alpha  = 0.5 if text_only else 0.75
    col    = MODALITY_COLOUR[mod]
    ax_b.bar(xb + offset, vals, W * 0.82,
             color=col, alpha=alpha, edgecolor='none', zorder=3)

# Majority baselines per question type
for j, qt in enumerate(QTYPES):
    mb = majority_by_qt[qt]
    ax_b.plot([j - 0.5, j + 0.5], [mb, mb],
              color=MAJORITY_COL, linewidth=2.0, linestyle='--', zorder=5)
    ax_b.text(j + 0.52, mb, f'{mb:.0f}%',
              fontsize=8, color='#666', va='center')

ax_b.set_xticks(xb)
ax_b.set_xticklabels(QTYPES, fontsize=10)
ax_b.set_ylabel('Exact Match (%)')
ax_b.set_title('(b)  Per question type vs. majority baseline', fontweight='bold')
ax_b.set_ylim(0, 70)
ax_b.grid(axis='y', color='#eeeeee', linewidth=0.5, zorder=0)

# annotation showing LEO is close to majority
leo_dir = next(d[1]['qtype'].get('Direction', 0) for d in model_data if 'LEO' in d[0])
ax_b.annotate('LEO: 4%\n≈ random',
              xy=(0, leo_dir), xytext=(0.35, 8),
              fontsize=7.5, color='#2E7D32',
              arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=0.8))

# legend: modality colours
handles_b = [mpatches.Patch(color=c, label=m, alpha=0.75)
             for m, c in MODALITY_COLOUR.items()]
handles_b += [Line2D([0],[0], color=MAJORITY_COL, lw=2, linestyle='--',
                     label='Majority baseline')]
ax_b.legend(handles=handles_b, fontsize=8, framealpha=0.9, loc='upper left')

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Model Performance vs. Majority-Class Baseline  ·  Hypo3D Benchmark  "
    "(14,885 questions)\n"
    "Majority baseline: always predict most-frequent answer per question type  "
    "(Overall: 11.8%,  Direction: 12.4%,  Scale: 10.4%,  Semantic: 15.8%)",
    fontsize=9.5, fontweight='bold', y=1.01,
)

for ext in ('pdf', 'png'):
    out = FIGS / f'fig_baseline_comparison.{ext}'
    plt.savefig(out, facecolor='white')
    print(f'Saved: {out}')
plt.close()
