"""
Generate publication-quality dissertation figures for Hypo3D.

Produces:
  figs/fig_overall_em.pdf        – Overall EM by model (grouped by modality)
  figs/fig_changetype_heatmap.pdf – EM heatmap: model × change type
  figs/fig_qtype_bars.pdf        – EM by question type per model
  figs/fig_label_effect.pdf      – Effect of scene labels on 2D VLMs
  figs/fig_geometric_metrics.pdf – Pipeline geometric metrics (relocation)
"""

import json, re, os, pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Output directory ────────────────────────────────────────────────────────
FIGS = Path('/rds/general/user/qp23/home/Hypo3D/figs')
FIGS.mkdir(exist_ok=True)

EXP = Path('/rds/general/user/qp23/home/Hypo3D/exp')
GEO = Path('/rds/general/user/qp23/home/Hypo3D/dataset/move_eval_geometric.json')

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

MODALITY_COLOUR = {
    'LLM':    '#4C72B0',
    '2D VLM': '#DD8452',
    '3D VLM': '#55A868',
}

# ── Text normalisation (mirrors metric_compute.py) ───────────────────────────
_REPLACEMENTS = {
    'back and right':'back right','back and left':'back left',
    'front and right':'front right','front and left':'front left',
    'behind and to the right':'back right','behind and to the left':'back left',
    'in front and to the right':'front right',
    'to the':'','by the':'','on the':'','near':'','next':'','corner':'',
    'behind':'back','bottom':'back','top':'front',
    'right side':'right','left side':'left','front side':'front','back side':'back',
    'in front of':'front','on the left of':'left','on the right of':'right',
    'on the left':'left','on the right':'right',
    'north':'front','south':'back','east':'right','west':'left',
    'northwest':'front left','northeast':'front right',
    'southwest':'back left','southeast':'back right',
    'forward':'front','backward':'back','bottom of':'back',
    'left of':'left','right of':'right','front of':'front','back of':'back'
}
_SORTED_KEYS = sorted(_REPLACEMENTS.keys(), key=len, reverse=True)
_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, _SORTED_KEYS)) + r')\b')

def _normalize(text):
    text = text.lower()
    text = _PATTERN.sub(lambda m: _REPLACEMENTS[m.group(0)], text)
    text = re.sub(r'\b(?:a|an|the)\b', '', text).strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# ── Metric computation ────────────────────────────────────────────────────────
CHANGE_TYPES = ['Movement', 'Removal', 'Attribute', 'Addition', 'Replacement']
QUESTION_TYPES = ['Direction', 'Scale', 'Semantic']

def compute_metrics(path):
    data = json.load(open(path))
    total = defaultdict(int)
    exact = defaultdict(int)
    qtotal = defaultdict(int)
    qexact = defaultdict(int)
    overall_t = overall_e = 0

    for scene_id, changes in data.items():
        for change in changes:
            raw_ct = change.get('change_type', '')
            ct = raw_ct.split()[1] if raw_ct else 'Unknown'
            for qa in change['questions_answers']:
                ref  = _normalize(qa['answer'])
                pred = _normalize(qa['predicted_answer'])
                total[ct]  += 1
                overall_t  += 1
                hit = int(ref == pred)
                exact[ct]  += hit
                overall_e  += hit
                for qt in qa['question_type'].split():
                    qtotal[qt] += 1
                    qexact[qt] += hit

    return {
        'overall':  overall_e / overall_t * 100 if overall_t else 0,
        'change':   {ct: exact[ct] / total[ct] * 100 if total[ct] else float('nan')
                     for ct in CHANGE_TYPES},
        'qtype':    {qt: qexact[qt] / qtotal[qt] * 100 if qtotal[qt] else float('nan')
                     for qt in QUESTION_TYPES},
        'n':        overall_t,
    }

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = [
    # (display name,  file,                                        modality)
    ('GPT-4o (Text)',     'contextvqa_GPT4o_text.json',                             'LLM'),
    ('LLaMA 3.2-3B',     'contextvqa_llama-3.2-3B.json',                           'LLM'),
    ('GPT-4o (Vision)',   'contextvqa_GPT4o_with_label_rotated.json',               '2D VLM'),
    ('Claude 3.5',        'contextvqa_Claude-3.5-Sonnet_with_label_rotated.json',   '2D VLM'),
    ('Qwen2-VL-7B',       'contextvqa_Qwen2-VL-7B-Instruct_with_label_rotated.json','2D VLM'),
    ('Qwen2-VL-72B',      'contextvqa_Qwen2-VL-72B-Instruct-AWQ_with_label_rotated.json','2D VLM'),
    ('LLaVA-OV-7B',       'contextvqa_llava-onevision-qwen2-7b-ov-hf_with_label_rotated.json','2D VLM'),
    ('LLaVA-OV-72B',      'contextvqa_llava-onevision-qwen2-72b-ov-hf_with_label_rotated.json','2D VLM'),
    ('LLaVA-3D-7B',       'contextvqa_llava3d-7B.json',                             '3D VLM'),
    ('LEO',               'contextvqa_leo.json',                                    '3D VLM'),
]

LABEL_PAIRS = [
    # (display,  no_label_file,   with_label_file)
    ('GPT-4o',      'contextvqa_GPT4o_no_label_rotated.json',
                    'contextvqa_GPT4o_with_label_rotated.json'),
    ('Claude 3.5',  'contextvqa_Claude-3.5-Sonnet_no_label_rotated.json',
                    'contextvqa_Claude-3.5-Sonnet_with_label_rotated.json'),
    ('Qwen2-VL-7B', 'contextvqa_Qwen2-VL-7B-Instruct_no_label_rotated.json',
                    'contextvqa_Qwen2-VL-7B-Instruct_with_label_rotated.json'),
    ('Qwen2-VL-72B','contextvqa_Qwen2-VL-72B-Instruct-AWQ_no_label_rotated.json',
                    'contextvqa_Qwen2-VL-72B-Instruct-AWQ_with_label_rotated.json'),
    ('LLaVA-OV-7B', 'contextvqa_llava-onevision-qwen2-7b-ov-hf_no_label_rotated.json',
                    'contextvqa_llava-onevision-qwen2-7b-ov-hf_with_label_rotated.json'),
    ('LLaVA-OV-72B','contextvqa_llava-onevision-qwen2-72b-ov-hf_no_label_rotated.json',
                    'contextvqa_llava-onevision-qwen2-72b-ov-hf_with_label_rotated.json'),
]

# ── Collect all results ───────────────────────────────────────────────────────
results = {}
for name, fname, modality in MODELS:
    p = EXP / fname
    if not p.exists():
        print(f'[WARN] missing {fname}')
        continue
    m = compute_metrics(p)
    m['modality'] = modality
    results[name] = m
    print(f'{name:20s}  EM={m["overall"]:5.1f}%')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Overall EM by model, grouped by modality
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

names      = list(results.keys())
ems        = [results[n]['overall'] for n in names]
modalities = [results[n]['modality'] for n in names]
colours    = [MODALITY_COLOUR[m] for m in modalities]

# sort by modality order then EM
order_key = {'LLM': 0, '2D VLM': 1, '3D VLM': 2}
sorted_idx = sorted(range(len(names)),
                    key=lambda i: (order_key[modalities[i]], -ems[i]))
names_s  = [names[i]      for i in sorted_idx]
ems_s    = [ems[i]        for i in sorted_idx]
cols_s   = [colours[i]    for i in sorted_idx]
mods_s   = [modalities[i] for i in sorted_idx]

bars = ax.barh(names_s, ems_s, color=cols_s, height=0.65, edgecolor='white', linewidth=0.5)

# Value labels
for bar, v in zip(bars, ems_s):
    ax.text(v + 0.4, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', ha='left', fontsize=8.5)

# Divider lines between modalities
prev_mod = None
for i, mod in enumerate(mods_s):
    if prev_mod and mod != prev_mod:
        ax.axhline(i - 0.5, color='#cccccc', linewidth=0.8, linestyle='--')
    prev_mod = mod

ax.set_xlim(0, 55)
ax.set_xlabel('Exact Match Accuracy (%)')
ax.set_title('Foundation Model Performance on Hypo3D (with Semantic Labels)')

# Legend
patches = [mpatches.Patch(color=c, label=m)
           for m, c in MODALITY_COLOUR.items()]
ax.legend(handles=patches, loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig(FIGS / 'fig_overall_em.pdf')
plt.savefig(FIGS / 'fig_overall_em.png')
plt.close()
print('Saved fig_overall_em')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Heatmap: model × change type
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.8))

matrix = np.full((len(names_s), len(CHANGE_TYPES)), np.nan)
for i, name in enumerate(names_s):
    for j, ct in enumerate(CHANGE_TYPES):
        v = results[name]['change'].get(ct, np.nan)
        matrix[i, j] = v

cmap = LinearSegmentedColormap.from_list('rg', ['#d73027','#fee08b','#1a9850'], N=256)
im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=65)

ax.set_xticks(range(len(CHANGE_TYPES)))
ax.set_xticklabels(CHANGE_TYPES, rotation=0)
ax.set_yticks(range(len(names_s)))
ax.set_yticklabels(names_s)

# Annotate cells
for i in range(len(names_s)):
    for j in range(len(CHANGE_TYPES)):
        val = matrix[i, j]
        if not np.isnan(val):
            txt_col = 'white' if val < 20 or val > 52 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=8, color=txt_col, fontweight='bold')

cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label('EM Accuracy (%)')

# Horizontal dividers between modalities
mod_boundaries = []
prev = mods_s[0]
for i, m in enumerate(mods_s[1:], 1):
    if m != prev:
        mod_boundaries.append(i - 0.5)
        prev = m
for b in mod_boundaries:
    ax.axhline(b, color='white', linewidth=2)

ax.set_title('Exact Match Accuracy by Model and Change Type (%)')
plt.tight_layout()
plt.savefig(FIGS / 'fig_changetype_heatmap.pdf')
plt.savefig(FIGS / 'fig_changetype_heatmap.png')
plt.close()
print('Saved fig_changetype_heatmap')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – EM by question type per model
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4.5))

qt_colours = {'Direction': '#4C72B0', 'Scale': '#DD8452', 'Semantic': '#55A868'}
n_models = len(names_s)
n_qt     = len(QUESTION_TYPES)
x        = np.arange(n_models)
width    = 0.25

for k, qt in enumerate(QUESTION_TYPES):
    vals = [results[n]['qtype'].get(qt, 0) for n in names_s]
    offset = (k - 1) * width
    bars = ax.bar(x + offset, vals, width, label=qt,
                  color=qt_colours[qt], edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(names_s, rotation=30, ha='right')
ax.set_ylabel('Exact Match Accuracy (%)')
ax.set_title('Performance by Question Type Across Models')
ax.legend(title='Question Type', framealpha=0.9)
ax.set_ylim(0, 75)

# Modality dividers
for b in mod_boundaries:
    ax.axvline(b, color='#cccccc', linewidth=0.9, linestyle='--')

# Modality labels above plot
mod_ranges = {}
prev_mod = mods_s[0]
start = 0
for i, mod in enumerate(mods_s):
    if mod != prev_mod or i == len(mods_s) - 1:
        end = i if mod != prev_mod else i + 1
        mid = (start + end - 1) / 2
        mod_ranges[prev_mod] = mid
        start = i
        prev_mod = mod
for mod, mid in mod_ranges.items():
    ax.text(mid, 72, mod, ha='center', va='bottom', fontsize=8.5,
            color=MODALITY_COLOUR[mod], fontweight='bold')

plt.tight_layout()
plt.savefig(FIGS / 'fig_qtype_bars.pdf')
plt.savefig(FIGS / 'fig_qtype_bars.png')
plt.close()
print('Saved fig_qtype_bars')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Effect of semantic labels (no_label vs with_label)
# ═══════════════════════════════════════════════════════════════════════════════
label_no, label_with = [], []
label_names = []
for name, no_f, with_f in LABEL_PAIRS:
    p_no   = EXP / no_f
    p_with = EXP / with_f
    if not p_no.exists() or not p_with.exists():
        continue
    label_names.append(name)
    label_no.append(compute_metrics(p_no)['overall'])
    label_with.append(compute_metrics(p_with)['overall'])

fig, ax = plt.subplots(figsize=(7, 4))

x = np.arange(len(label_names))
w = 0.35
b1 = ax.bar(x - w/2, label_no,   w, label='No Labels',       color='#4393c3', edgecolor='white')
b2 = ax.bar(x + w/2, label_with, w, label='Semantic Labels',  color='#d6604d', edgecolor='white')

# Delta annotations
for i, (a, b) in enumerate(zip(label_no, label_with)):
    delta = b - a
    sign  = '+' if delta >= 0 else ''
    ax.text(i, max(a, b) + 1, f'{sign}{delta:.1f}', ha='center',
            fontsize=8, color='#444444')

ax.set_xticks(x)
ax.set_xticklabels(label_names, rotation=20, ha='right')
ax.set_ylabel('Exact Match Accuracy (%)')
ax.set_title('Effect of Semantic Object Labels on 2D VLM Performance')
ax.legend(framealpha=0.9)
ax.set_ylim(0, 58)
plt.tight_layout()
plt.savefig(FIGS / 'fig_label_effect.pdf')
plt.savefig(FIGS / 'fig_label_effect.png')
plt.close()
print('Saved fig_label_effect')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Geometric pipeline metrics (relocation)
# ═══════════════════════════════════════════════════════════════════════════════
geo_data = [d for d in json.load(open(GEO))
            if d.get('relation_correct') is not None
            and d.get('placement_error_m') is not None]

relation_correct   = [bool(d['relation_correct'])  for d in geo_data]
placement_error    = [d['placement_error_m']        for d in geo_data]
intersection_frac  = [d['intersection_frac']        for d in geo_data]
labels_geo         = [f"{d['target_label']}\n→{d['relation'].replace('_OF','').replace('_',' ').lower()}"
                      for d in geo_data]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

# ── Panel A: relation correctness rate ──────────────────────────────────────
n_correct = sum(relation_correct)
n_wrong   = len(relation_correct) - n_correct
axes[0].pie([n_correct, n_wrong],
            labels=[f'Correct\n({n_correct}/{len(relation_correct)})',
                    f'Incorrect\n({n_wrong}/{len(relation_correct)})'],
            colors=['#55A868','#e07070'],
            autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 9})
axes[0].set_title('(a) Spatial Relation\nAccuracy')

# ── Panel B: placement error per job ────────────────────────────────────────
colours_b = ['#55A868' if c else '#e07070' for c in relation_correct]
x_b = range(len(placement_error))
axes[1].bar(x_b, placement_error, color=colours_b, edgecolor='white', linewidth=0.4)
axes[1].axhline(np.mean(placement_error), color='#444', linestyle='--', linewidth=1,
                label=f'Mean = {np.mean(placement_error):.2f} m')
axes[1].set_xticks(list(x_b))
axes[1].set_xticklabels(labels_geo, rotation=45, ha='right', fontsize=7)
axes[1].set_ylabel('Placement Error (m)')
axes[1].set_title('(b) Per-Job Placement Error')
axes[1].legend(fontsize=8)

correct_err = [e for e, c in zip(placement_error, relation_correct) if c]
wrong_err   = [e for e, c in zip(placement_error, relation_correct) if not c]
if correct_err:
    axes[1].annotate(f'Correct mean: {np.mean(correct_err):.2f} m',
                     xy=(0.02, 0.96), xycoords='axes fraction',
                     fontsize=7.5, color='#55A868', va='top')
if wrong_err:
    axes[1].annotate(f'Incorrect mean: {np.mean(wrong_err):.2f} m',
                     xy=(0.02, 0.88), xycoords='axes fraction',
                     fontsize=7.5, color='#e07070', va='top')

# ── Panel C: intersection fraction (object collision proxy) ─────────────────
axes[2].bar(x_b, [f*100 for f in intersection_frac],
            color='#DD8452', edgecolor='white', linewidth=0.4)
axes[2].axhline(np.mean(intersection_frac)*100, color='#444', linestyle='--',
                linewidth=1, label=f'Mean = {np.mean(intersection_frac)*100:.1f}%')
axes[2].set_xticks(list(x_b))
axes[2].set_xticklabels(labels_geo, rotation=45, ha='right', fontsize=7)
axes[2].set_ylabel('Intersection Fraction (%)')
axes[2].set_title('(c) Object Intersection\n(Collision Proxy)')
axes[2].legend(fontsize=8)

fig.suptitle('Geometric Evaluation of the Relocation Pipeline', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(FIGS / 'fig_geometric_metrics.pdf')
plt.savefig(FIGS / 'fig_geometric_metrics.png')
plt.close()
print('Saved fig_geometric_metrics')

print(f'\nAll figures written to {FIGS}/')
