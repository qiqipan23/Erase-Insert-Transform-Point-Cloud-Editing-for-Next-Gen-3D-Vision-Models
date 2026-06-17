"""
Generate pipeline_comparison_em.png

Exact Match: Hypo3D baseline vs our MOVE pipeline (no labels vs with labels)
using metric_compute.normalize_text, matched 1182-question Movement subset.

Reads:
  exp/contextvqa_GPT4o_with_label_rotated.json        (Hypo3D baseline)
  dataset/contextvqa_move_eval_gpt4o_labelled_v2.json (our pipeline, with labels)
  dataset/contextvqa_move_eval_gpt4o.json             (our pipeline, no labels)

Output: figs/pipeline_comparison_em.png / .pdf
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metric_compute import normalize_text

HYPO = Path(__file__).parent
FIGS = HYPO / "figs"; FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── Load data ──────────────────────────────────────────────────────────────────
with open(HYPO / "exp/contextvqa_GPT4o_with_label_rotated.json") as f:
    exp = json.load(f)
with open(HYPO / "dataset/contextvqa_move_eval_gpt4o_labelled_v2.json") as f:
    lab_data = json.load(f)
with open(HYPO / "dataset/contextvqa_move_eval_gpt4o.json") as f:
    unlab_data = json.load(f)

# Build lookups: (scene, question) -> qa dict
def build_lookup(data, movement_only=False):
    lookup = {}
    for scene, items in data.items():
        changes = items if isinstance(items, list) else []
        for ch in changes:
            if movement_only and "Movement" not in ch.get("change_type", ""):
                continue
            for qa in ch.get("questions_answers", []):
                lookup[(scene, qa["question"].strip())] = qa
    return lookup

exp_lookup   = build_lookup(exp, movement_only=True)
unlab_lookup = build_lookup(unlab_data)

# ── Compute EM on matched questions ───────────────────────────────────────────
QTYPES = ["Direction", "Scale", "Semantic"]

def compute(dataset, pred_lookup=None):
    total = 0; hits = 0
    qt_t: dict = defaultdict(int); qt_h: dict = defaultdict(int)
    for scene, entries in dataset.items():
        for entry in entries:
            for qa in entry.get("questions_answers", []):
                key = (scene, qa["question"].strip())
                if pred_lookup is not None:
                    if key not in pred_lookup:
                        continue
                    pred = normalize_text(str(pred_lookup[key].get("predicted_answer", "")))
                else:
                    pred = normalize_text(str(qa.get("predicted_answer", "")))
                ref = normalize_text(str(qa.get("answer", "")))
                qt  = str(qa.get("question_type", "")).split()[0]
                h   = int(ref == pred)
                total += 1; hits += h
                qt_t[qt] += 1; qt_h[qt] += h
    overall = hits / total * 100 if total else 0
    return overall, {qt: qt_h[qt] / qt_t[qt] * 100 for qt in qt_t if qt_t[qt] > 0}, total

orig_overall,  orig_qt,  N     = compute(lab_data, exp_lookup)
unlab_overall, unlab_qt, N_u   = compute(lab_data, unlab_lookup)
lab_overall,   lab_qt,   _     = compute(lab_data)

print(f"Baseline : {orig_overall:.1f}%  (N={N})")
print(f"No labels: {unlab_overall:.1f}%  (N={N_u})")
print(f"Labels   : {lab_overall:.1f}%")

# ── Figure ─────────────────────────────────────────────────────────────────────
GROUPS = QTYPES + ["Overall"]

orig_vals  = [orig_qt.get(q, 0)  for q in QTYPES] + [orig_overall]
unlab_vals = [unlab_qt.get(q, 0) for q in QTYPES] + [unlab_overall]
lab_vals   = [lab_qt.get(q, 0)   for q in QTYPES] + [lab_overall]

C_ORIG  = "#2E7D32"   # green
C_UNLAB = "#FF8F00"   # amber
C_LAB   = "#1565C0"   # blue

W = 0.25
x = np.arange(len(GROUPS))

fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

bars_o = ax.bar(x - W,   orig_vals,  W, color=C_ORIG,  label="Hypo3D (text + original scene)",
                edgecolor="white", linewidth=0.5, zorder=3)
bars_u = ax.bar(x,       unlab_vals, W, color=C_UNLAB, label="Ours (rendered 3D edit, no labels)",
                edgecolor="white", linewidth=0.5, zorder=3)
bars_l = ax.bar(x + W,   lab_vals,   W, color=C_LAB,   label="Ours (rendered 3D edit, with labels)",
                edgecolor="white", linewidth=0.5, zorder=3)

for bar, v in (list(zip(bars_o, orig_vals))
             + list(zip(bars_u, unlab_vals))
             + list(zip(bars_l, lab_vals))):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
            f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=bar.get_facecolor())

ax.axvline(len(QTYPES) - 0.5, color="#cccccc", linewidth=1.2, linestyle="--", zorder=2)

ax.set_xticks(x)
ax.set_xticklabels(GROUPS)
ax.set_ylabel("Exact Match (%)")
ax.set_ylim(0, 62)
ax.set_title("Exact Match: Hypo3D baseline vs our MOVE pipeline", fontweight="bold")
ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
ax.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

fig.text(0.5, -0.02,
         f"GPT-4o evaluation · matched {N}-question Movement subset · "
         "normalisation: metric_compute.normalize_text",
         ha="center", fontsize=8, color="#555")

for ext in ("pdf", "png"):
    out = FIGS / f"pipeline_comparison_em.{ext}"
    plt.savefig(out, facecolor="white")
    print(f"Saved: {out}")
plt.close()
