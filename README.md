<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                <img src="docs/static/hypo_icon.png" alt="Icon" style="width:40px; vertical-align:middle; margin-right:10px;">  Hypo3D: Exploring Hypothetical Reasoning in 3D</h1>      
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://yebulabula.github.io/" target="_blank" style="text-decoration: none;">Ye Mao</a>,&nbsp;
    <a href="https://scholar.google.com/citations?user=2Y0-0C8AAAAJ&hl=en" target="_blank" style="text-decoration: none;">Weixun Luo</a>,&nbsp;
    <a href="https://tomtomtommi.github.io/" target="_blank" style="text-decoration: none;">Junpeng Jing</a>,&nbsp;
    <a href="https://anlanqiu.github.io/" target="_blank" style="text-decoration: none;">Anlan Qiu</a>,&nbsp;
    <a href="https://www.imperial.ac.uk/people/k.mikolajczyk"  target="_blank" style="text-decoration: none;">Krystian Mikolajczyk</a>&nbsp;<br/>
&nbsp;<strong>Imperial College London</strong><br/>

<div align="center">
  <a href="https://arxiv.org/abs/2502.00954" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
  <a href="https://matchlab-imperial.github.io/Hypo3D/" target="_blank" rel="external nofollow noopener">
  <img src="https://img.shields.io/badge/Page-Hypo3D-9cf" alt="Project Page"></a>
  <a href="https://docs.google.com/forms/d/e/1FAIpQLSe--CkKIw_aXZpHHIv3OEt2psPsMdqKNkl1NRQN3vd92wHjvA/viewform" rel="external nofollow noopener" target="_blank">
  <img src="https://img.shields.io/badge/Data-Hypo3D-blue" alt="Data"></a>
</div>
</p>

## 📣 Latest Updates

- **[2025-05-01]** 🎉 *Hypo3D has been accepted to ICML 2025!*
- **[2025-02-04]** 📝 *Hypo3D paper preprint is now available on [arXiv](https://arxiv.org/abs/2502.00954).*
- **[2025-02-09]** 📊 *Hypo3D benchmark has been released.*
- **[2025-02-09]** 🧪 *Evaluation scripts for multiple vision-language models are now publicly available.*

## 🔑 Key Takeaways

- **Hypo3D** introduces a novel 3D reasoning benchmark.  
  🧠 **Task Definition**: Given a *past* 3D scene (e.g., point cloud, top-view image, scene captions) and a **context change description**, the goal is to *imagine* the updated scene after the change and answer questions based on that **hypothetical** scene state.

- The benchmark includes **7,727 context changes** and **14,885 QA pairs** spanning **700 indoor scenes**.  
  These changes are categorized into five types:  
  1. **Movement** — Geometric transformations (e.g., translation, rotation)  
  2. **Removal** — Objects taken away from the scene  
  3. **Attribute** — Changes in object properties (e.g., color, open/closed state)  
  4. **Addition** — New objects introduced into the scene  
  5. **Replacement** — Existing objects substituted with different ones

![sicl](docs/static/fig1.png)

---

## About this code

The Hypo3D codebase provides evaluation scripts for the original benchmark (Part A) and a rendered-edit MOVE pipeline (Part B) that physically executes object movements and re-evaluates models on the resulting scenes.

```
Hypo3D/
├── LLM/                              # Text-only LLM evaluation
│   ├── GPT4o-text/                   # GPT-4o text-only mode
│   └── llama/                        # LLaMA-3.2 3B
├── 2D-VLM/                           # 2D vision-language model evaluation
│   ├── Claude/
│   │   ├── evaluate.py               # Original benchmark
│   │   └── evaluate_move.py          # MOVE pipeline renders
│   ├── GPT4o/
│   │   ├── evaluate.py               # Original benchmark
│   │   ├── evaluate_move.py          # MOVE renders (unlabelled)
│   │   ├── evaluate_move_dense_v2.py # MOVE renders (dense, v2)
│   │   ├── evaluate_move_labelled.py # MOVE renders (with labels)
│   │   └── evaluate_move_labelled_hires.py  # MOVE renders (hi-res 1600px)
│   ├── Qwen2-VL/
│   └── llava-ov/
├── exp/                              # Evaluation results (JSON)
├── dataset.py                        # Dataset loading utilities
├── metric_compute.py                 # EM / PM metric computation
├── inference_pipeline.sh             # End-to-end run script (Parts A + B)
│
├── # ── MOVE pipeline ──────────────────────────────────────────
├── run_move_pipeline_v2.pbs          # PBS: run edit jobs (array job)
├── rerender_and_build_eval_v2.py     # Re-render top-view images post-edit
├── rerender_denser.py                # Dense top-view renderer
├── patch_bkgd_pointclouds.py        # Patch ConvONet input with fill points
├── combine_move_result.py            # Merge completed bkgd + placed object
├── eval_move_metrics.py              # Geometric metrics (relation, placement)
├── eval_round_trip.py                # Round-trip consistency evaluation
│
├── # ── Scene completion ────────────────────────────────────────
├── run_convonet_bkgd.pbs             # PBS: ConvONet background completion
├── run_completion_all.pbs            # PBS: all completion jobs
├── run_completion_all_gpu.pbs        # PBS: GPU completion variant
│
├── # ── Object placement ────────────────────────────────────────
├── run_pbic_inst.pbs                 # PBS: SCCNet instance completion
├── run_pbic_inst.py                  # SCCNet runner script
├── run_reconstruct_object.pbs        # PBS: object reconstruction
│
├── # ── Preprocessing ───────────────────────────────────────────
├── filter_scannet.py                 # Filter to ScanNet scenes only
├── extract_inst_ply.py               # Extract per-instance PLY files
├── merge_inst_with_color.py          # Merge instance colours
├── compute_scene_orientation_contexts.py
│
├── # ── Training ────────────────────────────────────────────────
├── train_convonet_finetune.pbs       # PBS: ConvONet fine-tuning
├── train_hidden_surface_seeded.pbs   # PBS: hidden surface model training
├── finetune_hidden_surface.pbs       # PBS: hidden surface fine-tuning
├── merge_hidden_surface_datasets.py  # Merge HS training data
│
└── # ── Figure generation ───────────────────────────────────────
    └── generate_*.py                 # Dissertation / paper figure scripts
```

---

## Setup

### 1. Clone

```bash
git clone --recursive https://github.com/MatchLab-Imperial/Hypo3D.git
cd Hypo3D
```

### 2. Download the benchmark data

```bash
git clone https://huggingface.co/datasets/MatchLab/Hypo3D
mv Hypo3D dataset
```

Expected layout:
```
dataset/
├── contextvqa.json                   # Full benchmark QA pairs
├── LLM_data/                         # Scene captions (for LLMs)
├── 2D_VLM_data/
│   ├── top_view_no_label_rotated/    # Unlabelled top-view maps
│   └── top_view_with_label_rotated/  # Labelled top-view maps
└── 3D_VLM_data/                      # Point clouds / RGB-D (for 3D VLMs)
```

Complete the [data access form](https://forms.gle/w6NCaDjY9FzdSZFEA) to download.

### 3. API keys (for GPT-4o / Claude evaluation)

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the evaluation

### Part A — Original Hypo3D benchmark

Evaluate models on the static scene + text change description setting:

```bash
bash inference_pipeline.sh   # runs all models end-to-end
```

Or run individual models:

```bash
# GPT-4o (vision)
python 2D-VLM/GPT4o/evaluate.py -f dataset/contextvqa.json

# Claude 3.5 Sonnet
python 2D-VLM/Claude/evaluate.py -f dataset/contextvqa.json

# GPT-4o text-only
python LLM/GPT4o-text/evaluate.py -f dataset/contextvqa.json

# Compute EM/PM metrics
python metric_compute.py
```

### Part B — MOVE rendered-edit pipeline

The MOVE pipeline physically executes object movements in ScanNet scenes, renders top-view images of the edited scene, and re-evaluates models on those renders. Requires HPC access (PBS scheduler) and the third-party repos under `repo/`.

```bash
# B1: Run edit jobs across all scenes (PBS array, ~3h on GPU)
qsub run_move_pipeline_v2.pbs

# B2: Background scene completion (ConvONet)
python patch_bkgd_pointclouds.py   # patch fill points into ConvONet input
qsub run_convonet_bkgd.pbs

# B3: Object instance completion (SCCNet/PBIC)
qsub run_pbic_inst.pbs

# B4: Combine completed background + placed object
python combine_move_result.py

# B5: Re-render top-view images
python rerender_and_build_eval_v2.py

# B6: VQA evaluation on rendered scenes
python 2D-VLM/GPT4o/evaluate_move_dense_v2.py --run       # no labels
python 2D-VLM/GPT4o/evaluate_move_labelled.py --run       # with labels
python 2D-VLM/GPT4o/evaluate_move_labelled_hires.py --run # hi-res (1600px)

# B7: Compute metrics
python eval_move_metrics.py    # geometric: relation accuracy, placement error
python eval_round_trip.py      # round-trip consistency

# B8: Generate comparison figure
python generate_orig_vs_pipeline.py
```

---

## 📊 Results

### Part A — Original Hypo3D benchmark

| Model Family | Model | EM (%) | PM (%) |
|---|---|---|---|
| **LLM (Scene Caption)** | Llama-3.2 3B | 26.08 | 29.91 |
| | GPT-4o (text) | **35.54** | **39.65** |
| **2D VLM (No labels)** | Qwen2-VL 7B | 29.68 | 34.47 |
| | Qwen2-VL 72B | 33.39 | 37.51 |
| | LLaVA-OV 7B | 30.62 | 34.34 |
| | LLaVA-OV 72B | **36.38** | **40.13** |
| | Claude 3.5 Sonnet | 20.70 | 30.12 |
| | GPT-4o | 33.58 | 36.75 |
| **2D VLM (Semantic labels)** | Qwen2-VL 7B | 34.40 | 38.91 |
| | Qwen2-VL 72B | 42.45 | 48.25 |
| | LLaVA-OV 7B | 38.93 | 43.51 |
| | LLaVA-OV 72B | 43.81 | 46.83 |
| | Claude 3.5 Sonnet | 41.36 | 51.59 |
| | GPT-4o | **45.50** | **48.82** |
| **3D VLM** | LEO 7B | 14.83 | 22.40 |
| | LLaVA-3D 7B | **31.56** | **35.23** |
| **Human** | | 91.00 | 92.50 |

### Part B — MOVE rendered-edit pipeline (GPT-4o, ~1k matched Movement questions)

| Condition | Overall PM (%) | Direction | Scale | Semantic |
|---|---|---|---|---|
| Original Hypo3D (text + original scene) | 35.2 | 18.1 | 47.7 | 5.6 |
| Pipeline render — no labels | 33.9 | 20.9 | 44.0 | 6.1 |
| Pipeline render — with labels | 32.8 | 18.9 | 43.0 | 7.1 |
| Pipeline render — hi-res labels (1600px) | **35.0** | **24.9** | 43.4 | **9.2** |

Rendering the physically-edited scene at high resolution with instance labels matches text-based hypothetical reasoning on the original scene, with a notable improvement in directional reasoning (+6.8 pp over the text baseline).

---

## Contact

- Ye Mao: ye.mao21@imperial.ac.uk

Please open an issue or submit a pull request for bugs or contributions.

## 💼 License

<a href="https://opensource.org/licenses/MIT" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" />
</a>

## Citation

If you find our benchmark helpful, please cite our paper:

```bibtex
@article{mao2025hypo3d,
  title={Hypo3D: Exploring Hypothetical Reasoning in 3D},
  author={Mao, Ye and Luo, Weixun and Jing, Junpeng and Qiu, Anlan and Mikolajczyk, Krystian},
  journal={arXiv preprint arXiv:2502.00954},
  year={2025}
}
```
