# Erase, Insert, Transform: Point Cloud Editing for Next-Gen 3D Vision Models

**Imperial College London** 

---

## Overview

This project investigates whether 3D vision-language models reason better when shown a *physically edited scene* rather than being asked to imagine a hypothetical change from text alone.

We build an automatic point cloud editing pipeline with three core operations:

| Operation | Description |
|---|---|
| **Erase** | Remove an object from a ScanNet scene and reconstruct the exposed background using RANSAC plane fill + structural surface completion + ConvONet |
| **Insert** | Complete a partial object point cloud using SCCNet (PBIC) and place it at a target location |
| **Transform** | Move an object to a new position by composing Erase + Insert, then re-render a top-view image of the edited scene |

Edited scenes are rendered as top-view 2D maps and fed to GPT-4o for VQA evaluation, using the [Hypo3D](https://arxiv.org/abs/2502.00954) benchmark questions as ground truth.

---

## Pipeline Architecture

```
ScanNet scene
     │
     ▼
┌─────────────┐
│   ERASE     │  Remove object → RANSAC plane fill
│             │  → Structural surface completion
│             │  → ConvONet background reconstruction
└──────┬──────┘
       │  completed background
       ▼
┌─────────────┐
│   INSERT    │  SCCNet instance completion
│             │  → Place completed object at target location
└──────┬──────┘
       │  background + placed object
       ▼
┌─────────────┐
│  TRANSFORM  │  Combine → re-render top-view image
│             │  → GPT-4o VQA evaluation
└─────────────┘
```

---

## Repository Structure

```
Erase-Insert-Transform/
├── inference_pipeline.sh         # End-to-end run script (Parts A + B)
├── dataset.py                    # Dataset loading utilities
├── metric_compute.py             # EM / PM metric computation
│
├── 2D-VLM/                       # Vision-language model evaluation
│   ├── GPT4o/
│   │   ├── evaluate.py                    # Original Hypo3D benchmark
│   │   ├── evaluate_move.py               # MOVE renders (unlabelled)
│   │   ├── evaluate_move_dense_v2.py      # MOVE renders (dense)
│   │   ├── evaluate_move_labelled.py      # MOVE renders (with labels)
│   │   └── evaluate_move_labelled_hires.py  # MOVE renders (hi-res 1600px)
│   ├── Claude/
│   │   ├── evaluate.py
│   │   └── evaluate_move.py
│   ├── Qwen2-VL/
│   └── llava-ov/
│
├── LLM/                          # Text-only LLM baselines
│   ├── GPT4o-text/
│   └── llama/
│
├── pipeline/                     # MOVE pipeline runners
│   ├── run_pbic_inst.py          # SCCNet runner
│   ├── patch_bkgd_pointclouds.py # Patch fill points into ConvONet input
│   ├── combine_move_result.py    # Merge completed bkgd + placed object
│   ├── rerender_and_build_eval_v2.py  # Re-render top-view images
│   ├── rerender_denser.py        # Dense top-view renderer
│   └── ...
│
├── eval/                         # Metric computation
│   ├── eval_move_metrics.py      # Geometric: relation accuracy, placement error
│   └── eval_round_trip.py        # Round-trip consistency
│
├── preprocess/                   # Data preprocessing
│   ├── filter_scannet.py         # Filter to available ScanNet scenes
│   ├── extract_inst_ply.py       # Extract per-instance PLY files
│   └── ...
│
├── train/                        # Model training scripts (PBS)
│   ├── train_convonet_finetune.pbs
│   ├── train_hidden_surface_seeded.pbs
│   └── ...
│
└── exp/                          # Evaluation results (JSON)
```

---

## Setup

### 1. Clone

```bash
git clone --recursive https://github.com/qiqipan23/Erase-Insert-Transform-Point-Cloud-Editing-for-Next-Gen-3D-Vision-Models.git
cd Erase-Insert-Transform-Point-Cloud-Editing-for-Next-Gen-3D-Vision-Models
```

### 2. Conda environments

| Environment | Used for |
|---|---|
| `placeit3d` | MOVE pipeline, rendering, metrics |
| `conv_onet` | ConvONet background completion |
| `pbic` | SCCNet object completion |
| `hypo3d` | Dataset utilities |

### 3. API keys

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Data

Download the [Hypo3D benchmark](https://huggingface.co/datasets/MatchLab/Hypo3D) and place it under `dataset/`. ScanNet processed scenes go under `repo/scannet/processed/`.

---

## Running the Pipeline

### Part A — Baseline: text-only hypothetical reasoning

Evaluate models on the original Hypo3D benchmark (static scene image + text change description):

```bash
python 2D-VLM/GPT4o/evaluate.py -f dataset/contextvqa.json
python metric_compute.py
```

### Part B — MOVE rendered-edit pipeline

Physically execute object movements, render the result, re-evaluate:

```bash
# 1. Run edit jobs across all scenes (PBS array job, ~3h GPU)
qsub pipeline/run_move_pipeline_v2.pbs

# 2. Background scene completion
python pipeline/patch_bkgd_pointclouds.py
qsub pipeline/run_convonet_bkgd.pbs

# 3. Object instance completion (SCCNet)
qsub pipeline/run_pbic_inst.pbs

# 4. Combine background + placed object
python pipeline/combine_move_result.py

# 5. Re-render top-view images
python pipeline/rerender_and_build_eval_v2.py

# 6. VQA evaluation on rendered scenes
python 2D-VLM/GPT4o/evaluate_move_labelled_hires.py --run

# 7. Compute metrics
python eval/eval_move_metrics.py
python eval/eval_round_trip.py

# 8. Generate comparison figure
python figures/generate_orig_vs_pipeline.py
```

Or run everything:

```bash
bash inference_pipeline.sh
```

---

## Results

GPT-4o evaluated on ~1,000 matched Movement questions from Hypo3D (partial match EM):

| Condition | Overall | Direction | Scale | Semantic |
|---|---|---|---|---|
| Text baseline (original scene + text description) | 35.2% | 18.1% | 47.7% | 5.6% |
| Pipeline render — no labels | 33.9% | 20.9% | 44.0% | 6.1% |
| Pipeline render — with labels | 32.8% | 18.9% | 43.0% | 7.1% |
| Pipeline render — hi-res labels (1600px) | **35.0%** | **24.9%** | 43.4% | **9.2%** |

Rendering the physically-edited scene at high resolution with instance labels matches the text-only baseline overall, with a +6.8 pp improvement in directional reasoning — suggesting that grounding spatial queries in an actual rendered scene benefits direction-sensitive questions.

---

## Contact

Qiqi Pan — qiqi.pan23@imperial.ac.uk

## License

MIT
