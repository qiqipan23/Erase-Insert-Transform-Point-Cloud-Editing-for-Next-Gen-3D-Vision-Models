"""
Generate two figures for the MOVE pipeline evaluation section.

  figs/fig_pipeline_geometric.pdf/png  — geometric metrics (relation acc,
                                          placement error) per relation
  figs/fig_pipeline_vqa.pdf/png        — GPT-4o VQA accuracy on pipeline
                                          outputs: overall + by question type
                                          + by relation (v1 vs v2)
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

HYPO    = Path("/rds/general/user/qp23/home/Hypo3D")
DS      = HYPO / "dataset"
FIGS    = HYPO / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── Text normalisation ────────────────────────────────────────────────────────
_REP = {
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
    'left of':'left','right of':'right','front of':'front','back of':'back',
}
_SK  = sorted(_REP.keys(), key=len, reverse=True)
_PAT = re.compile(r'\b(' + '|'.join(map(re.escape, _SK)) + r')\b')

def norm(t):
    t = t.lower()
    t = _PAT.sub(lambda m: _REP[m.group(0)], t)
    t = re.sub(r'\b(?:a|an|the)\b', '', t).strip()
    t = re.sub(r'[^a-zA-Z0-9\s]', '', t)
    return t.strip()

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Geometric evaluation
# ════════════════════════════════════════════════════════════════════════════
geo = json.loads((DS / "move_eval_geometric.json").read_text())
valid = [r for r in geo if "error" not in r]

RELS_ORDERED = ["RIGHT_OF","FRONT_OF","UNDER","BACK_OF","ON_TOP_OF","NEXT_TO"]
REL_LABELS   = ["RIGHT_OF","FRONT_OF","UNDER","BACK_OF","ON_TOP_OF","NEXT_TO\n(dir-agnostic)"]

by_rel: dict[str, list] = defaultdict(list)
for r in valid:
    by_rel[r["relation"]].append(r)

rel_acc  = []
rel_pe   = []
rel_n    = []
for rel in RELS_ORDERED:
    items = by_rel.get(rel, [])
    rel_n.append(len(items))
    ra = [i["relation_correct"] for i in items if i["relation_correct"] is not None]
    rel_acc.append(np.mean(ra) * 100 if ra else np.nan)
    pe = [i["placement_error_m"] * 100 for i in items]   # → cm
    rel_pe.append(np.mean(pe) if pe else np.nan)

overall_ra = [r["relation_correct"] for r in valid if r["relation_correct"] is not None]
overall_pe = [r["placement_error_m"] * 100 for r in valid]
intf       = [r["intersection_frac"] * 100 for r in valid
              if not (isinstance(r["intersection_frac"], float) and np.isnan(r["intersection_frac"]))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")

x = np.arange(len(RELS_ORDERED))
W = 0.55

# ── Left: relation accuracy ───────────────────────────────────────────────
C_ACC = ["#2196F3" if not np.isnan(v) else "#BBDEFB" for v in rel_acc]
bars1 = ax1.bar(x, [v if not np.isnan(v) else 0 for v in rel_acc],
                width=W, color=C_ACC, edgecolor="white", linewidth=0.5)
# hatching for direction-agnostic
bars1[-1].set_hatch("//")
bars1[-1].set_facecolor("#BBDEFB")

for bar, v, n in zip(bars1, rel_acc, rel_n):
    if not np.isnan(v):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                 f"{v:.0f}%\n(N={n})", ha="center", va="bottom",
                 fontsize=8.5, color="#111")
    else:
        ax1.text(bar.get_x() + bar.get_width()/2, 2,
                 f"N={n}", ha="center", va="bottom",
                 fontsize=8.5, color="#888")

ax1.axhline(np.mean([v for v in rel_acc if not np.isnan(v)]),
            color="#E53935", linewidth=1.4, linestyle="--",
            label=f"Mean (checkable): {np.nanmean(rel_acc):.0f}%")
ax1.set_xticks(x); ax1.set_xticklabels(REL_LABELS, fontsize=8.5)
ax1.set_ylim(0, 115)
ax1.set_ylabel("Relation Accuracy (%)")
ax1.set_title("(a)  Relation Accuracy by Spatial Relation", fontweight="bold")
ax1.legend(fontsize=8.5, framealpha=0.9)
ax1.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

# ── Right: placement error ────────────────────────────────────────────────
C_PE = ["#FF8F00"] * len(RELS_ORDERED)
bars2 = ax2.bar(x, [v if not np.isnan(v) else 0 for v in rel_pe],
                width=W, color=C_PE, edgecolor="white", linewidth=0.5)
for bar, v, n in zip(bars2, rel_pe, rel_n):
    if not np.isnan(v):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)

ax2.axhline(np.mean(overall_pe), color="#E53935", linewidth=1.4,
            linestyle="--", label=f"Mean: {np.mean(overall_pe):.1f} cm")
ax2.set_xticks(x); ax2.set_xticklabels(REL_LABELS, fontsize=8.5)
ax2.set_ylabel("Placement Error (cm)")
ax2.set_title("(b)  Placement Error by Spatial Relation", fontweight="bold")
ax2.legend(fontsize=8.5, framealpha=0.9)
ax2.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

fig.suptitle(
    f"MOVE Pipeline — Geometric Evaluation  "
    f"({len(valid)} jobs · Rel. Acc. {np.nanmean([v for v in rel_acc if not np.isnan(v)]):.0f}% · "
    f"Placement Err. {np.mean(overall_pe):.1f} cm · "
    f"Intersection {np.mean(intf):.1f}%)",
    fontsize=10.5, fontweight="bold", y=1.01,
)

for ext in ("pdf", "png"):
    plt.savefig(FIGS / f"fig_pipeline_geometric.{ext}", facecolor="white")
print("Saved fig_pipeline_geometric")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — VQA accuracy on pipeline outputs
# ════════════════════════════════════════════════════════════════════════════

def compute_move_vqa(path):
    d = json.loads(Path(path).read_text())
    total = 0; hits = 0
    qt_t = defaultdict(int); qt_h = defaultdict(int)
    rl_t = defaultdict(int); rl_h = defaultdict(int)
    for scene, changes in d.items():
        for ch in changes:
            rel = ch.get("relation", "")
            for qa in ch.get("questions_answers", []):
                ref  = norm(str(qa.get("answer", "")))
                pred = norm(str(qa.get("predicted_answer", "")))
                qt   = str(qa.get("question_type", ""))
                h    = int(ref == pred and bool(pred))
                total += 1; hits += h
                qt_t[qt] += 1; qt_h[qt] += h
                rl_t[rel] += 1; rl_h[rel] += h
    return {
        "overall":  hits / total * 100 if total else 0,
        "n":        total,
        "qtype":    {k: qt_h[k] / qt_t[k] * 100 for k in qt_t},
        "relation": {k: rl_h[k] / rl_t[k] * 100 for k in rl_t},
        "rel_n":    dict(rl_t),
    }

v1 = compute_move_vqa(DS / "contextvqa_move_eval_gpt4o.json")
v2 = compute_move_vqa(DS / "contextvqa_move_eval_gpt4o_v2.json")

QTYPES   = ["Direction", "Scale", "Scale Direction", "Semantic"]
QT_SHORT = ["Direction", "Scale", "Scale+Dir.", "Semantic"]
RELS_VQA = ["BACK_OF","FRONT_OF","LEFT_OF","NEXT_TO","ON_TOP_OF","RIGHT_OF","UNDER"]

fig = plt.figure(figsize=(14, 5), facecolor="white")
gs  = GridSpec(1, 3, figure=fig, wspace=0.35,
               left=0.07, right=0.97, top=0.88, bottom=0.14)

C_V1 = "#1565C0"
C_V2 = "#E53935"

# ── (a) Overall ───────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
bars = ax_a.bar([0, 1], [v1["overall"], v2["overall"]],
                color=[C_V1, C_V2], width=0.45,
                edgecolor="white", linewidth=0.5)
for bar, v in zip(bars, [v1["overall"], v2["overall"]]):
    ax_a.text(bar.get_x() + bar.get_width()/2, v + 0.4,
              f"{v:.1f}%", ha="center", va="bottom", fontsize=10,
              fontweight="bold")
ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(["GPT-4o\nv1", "GPT-4o\nv2"], fontsize=9.5)
ax_a.set_ylim(0, 35)
ax_a.set_ylabel("Exact Match (%)")
ax_a.set_title("(a)  Overall EM", fontweight="bold")
ax_a.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

# ── (b) By question type ─────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
xb = np.arange(len(QTYPES))
W2 = 0.35
v1_qt = [v1["qtype"].get(qt, 0) for qt in QTYPES]
v2_qt = [v2["qtype"].get(qt, 0) for qt in QTYPES]
bars_v1 = ax_b.bar(xb - W2/2, v1_qt, width=W2, color=C_V1,
                   label="v1", edgecolor="white", linewidth=0.4)
bars_v2 = ax_b.bar(xb + W2/2, v2_qt, width=W2, color=C_V2,
                   label="v2", edgecolor="white", linewidth=0.4)
for bar, v in list(zip(bars_v1, v1_qt)) + list(zip(bars_v2, v2_qt)):
    ax_b.text(bar.get_x() + bar.get_width()/2, v + 0.5,
              f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
ax_b.set_xticks(xb)
ax_b.set_xticklabels(QT_SHORT, fontsize=8.5)
ax_b.set_ylim(0, 58)
ax_b.set_ylabel("Exact Match (%)")
ax_b.set_title("(b)  EM by Question Type", fontweight="bold")
ax_b.legend(fontsize=9, framealpha=0.9)
ax_b.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

# ── (c) By relation ───────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
xc = np.arange(len(RELS_VQA))
v1_rl = [v1["relation"].get(r, 0) for r in RELS_VQA]
v2_rl = [v2["relation"].get(r, 0) for r in RELS_VQA]
bars_v1c = ax_c.bar(xc - W2/2, v1_rl, width=W2, color=C_V1,
                    label="v1", edgecolor="white", linewidth=0.4)
bars_v2c = ax_c.bar(xc + W2/2, v2_rl, width=W2, color=C_V2,
                    label="v2", edgecolor="white", linewidth=0.4)
for bar, v in list(zip(bars_v1c, v1_rl)) + list(zip(bars_v2c, v2_rl)):
    ax_c.text(bar.get_x() + bar.get_width()/2, v + 0.4,
              f"{v:.0f}", ha="center", va="bottom", fontsize=7)
# N labels on x axis
rl_labels = [f"{r}\n(N={v1['rel_n'].get(r,0)})" for r in RELS_VQA]
ax_c.set_xticks(xc)
ax_c.set_xticklabels(rl_labels, fontsize=7.5)
ax_c.set_ylim(0, 42)
ax_c.set_ylabel("Exact Match (%)")
ax_c.set_title("(c)  EM by Spatial Relation", fontweight="bold")
ax_c.legend(fontsize=9, framealpha=0.9)
ax_c.grid(axis="y", color="#eeeeee", linewidth=0.5, zorder=0)

fig.suptitle(
    f"GPT-4o VQA on MOVE Pipeline Outputs  "
    f"(408 scenes · 1,122 jobs · 2,152 questions)\n"
    f"v1 = standard top-down prompt  ·  v2 = highlighted scene context",
    fontsize=10, fontweight="bold",
)

for ext in ("pdf", "png"):
    plt.savefig(FIGS / f"fig_pipeline_vqa.{ext}", facecolor="white")
print("Saved fig_pipeline_vqa")
plt.close()
