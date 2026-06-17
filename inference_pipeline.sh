#!/usr/bin/env bash
# ============================================================
# Hypo3D full inference pipeline
# ============================================================
# Prerequisites:
#   - dataset/contextvqa.json          (original benchmark)
#   - dataset/contextvqa_move_eval_*.json  (MOVE pipeline eval sets)
#   - conda envs: placeit3d, pbic, conv_onet, hypo3d, qwen2
#   - export OPENAI_API_KEY=<key>
#   - export ANTHROPIC_API_KEY=<key>
# ============================================================
set -euo pipefail
HYPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HYPO"

# ─────────────────────────────────────────────────────────────
# PART A  Original Hypo3D benchmark (static scene, text change)
# ─────────────────────────────────────────────────────────────
data_path="dataset/contextvqa.json"
if [[ ! -f "$data_path" ]]; then
    echo "ERROR: $data_path not found"; exit 1
fi
echo "=== Part A: Original Hypo3D benchmark ==="

echo "[A1] GPT-4o (vision)"
python 2D-VLM/GPT4o/evaluate.py -f "$data_path"

echo "[A2] Claude-3.5-Sonnet (vision)"
python 2D-VLM/Claude/evaluate.py -f "$data_path"

echo "[A3] GPT-4o (text-only)"
python LLM/GPT4o-text/evaluate.py -f "$data_path"

echo "[A4] Llama-3.2-3B"
python LLM/llama/evaluate.py -f "$data_path"

echo "[A5] Qwen2-VL-7B"
source /rds/general/user/qp23/home/miniconda3/etc/profile.d/conda.sh
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1
conda activate qwen2
python 2D-VLM/Qwen2-VL/evaluate.py -f "$data_path" -m "Qwen/Qwen2-VL-7B-Instruct"

echo "[A6] Qwen2-VL-72B"
python 2D-VLM/Qwen2-VL/evaluate.py -f "$data_path" -m "Qwen/Qwen2-VL-72B-Instruct-AWQ"

echo "[A7] LLaVA-OV-7B"
python 2D-VLM/llava-ov/evaluate.py -f "$data_path" -m "llava-hf/llava-onevision-qwen2-7b-ov-hf"

echo "[A8] LLaVA-OV-72B"
python 2D-VLM/llava-ov/evaluate.py -f "$data_path" -m "llava-hf/llava-onevision-qwen2-72b-ov-hf"

echo "[A9] Compute metrics (Part A)"
conda activate placeit3d
python metric_compute.py

# ─────────────────────────────────────────────────────────────
# PART B  MOVE pipeline (rendered edited scene)
# ─────────────────────────────────────────────────────────────
# Step B1–B4 require the HPC PBS scheduler.
# Submit each step, wait for it to finish, then run the next.
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== Part B: MOVE pipeline ==="
echo ""

echo "─── B1: Run edit jobs (PBS array, ~3h on GPU) ───"
echo "    Submit:  qsub run_move_pipeline_v2.pbs"
echo "    Monitor: qstat -u \$USER"
echo "    Produces: repo/scannet/processed/<scene>/edits/*_v2.ply"
echo ""

echo "─── B2: Background scene completion (PBS) ───"
echo "    Submit:  qsub run_convonet_bkgd.pbs"
echo "    Prereq:  patch_bkgd_pointclouds.py must run first (once per scene)"
echo "    Produces: completion/<scene_job>_bkgd/completed_scene_pointcloud.ply"
echo ""

echo "─── B3: Object instance completion (PBS) ───"
echo "    Submit:  qsub run_pbic_inst.pbs   (SCCNet / PBIC)"
echo "    Produces: edits/<job>_placed.ply"
echo ""

echo "─── B4: Combine bkgd + placed object ───"
echo "    python combine_move_result.py"
echo ""

echo "─── B5: Re-render top-view images ───"
source /rds/general/user/qp23/home/miniconda3/etc/profile.d/conda.sh
conda activate placeit3d
python rerender_and_build_eval_v2.py
echo "    Produces: dataset/contextvqa_move_eval_dense_v2.json"
echo "              dataset/2D_VLM_data/move_edits_top_view_dense_v2/"
echo ""

echo "─── B6: VQA evaluation on MOVE renders ───"
echo "[B6a] GPT-4o no labels"
python 2D-VLM/GPT4o/evaluate_move_dense_v2.py --run

echo "[B6b] GPT-4o with labels"
python 2D-VLM/GPT4o/evaluate_move_labelled.py --run

echo "[B6c] GPT-4o hi-res labels (1600px)"
python 2D-VLM/GPT4o/evaluate_move_labelled_hires.py --run

echo "[B6d] Claude on MOVE"
python 2D-VLM/Claude/evaluate_move.py --run

echo "─── B7: Compute geometric metrics ───"
python eval_move_metrics.py
python eval_round_trip.py

echo "─── B8: Regenerate comparison figure ───"
python generate_orig_vs_pipeline.py

echo ""
echo "=== Done ==="
