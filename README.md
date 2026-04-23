# RoadBuddy — Dashcam Video QA Baseline

> Baseline system for **Zalo AI Challenge 2025 – RoadBuddy Track**:
> *Understanding Vietnamese traffic from dashcam video via Vision-Language Models.*

---

## Overview

RoadBuddy is a video question-answering (VideoQA) system designed for dashcam footage recorded in Vietnamese traffic conditions. Given a short video clip (5–15 seconds) and a multiple-choice question about traffic signs, signals, or driving instructions, the system returns the correct answer (A/B/C/D).

**Task:**  Input video + Vietnamese MCQ question → Answer (A/B/C/D)
**Constraint:** ≤ 30 seconds inference per sample
**Metric:** Accuracy = correct / total

---

## Pipeline

```
Video (.mp4) + Question
        │
        ▼
[1] Analyze question          → intent, target objects, temporal hints
        │
        ▼
[2] Extract candidate frames  → uniform 1 fps sampling
        │
        ▼
[3] Query-aware frame select  → sharpness (Laplacian) + temporal prior
        │
        ▼
[4] Build Vietnamese prompt   → MCQ format with context hints
        │
        ▼
[5] VLM inference             → Qwen-VL generate + regex parse
        │
        ▼
[6] Save results              → submission.csv + audit.json
```

---

## Repository Structure

```
road-buddy/
├── road_buddy/                  # Core package
│   ├── config.py                # YAML config → typed dataclasses
│   ├── dataio.py                # Dataset I/O, submission CSV
│   ├── video.py                 # Frame extraction (OpenCV)
│   ├── prompting.py             # Prompt builder + answer parser
│   ├── query_aware.py           # Query-aware frame selection
│   ├── pipeline.py              # End-to-end inference orchestration
│   ├── cli.py                   # CLI entry point
│   └── model/
│       └── qwen_vl.py           # Qwen-VL wrapper (generate + regex)
├── configs/
│   └── baseline_qwen.yaml       # Default configuration
├── datasets/
│   ├── train/                   # Training split
│   └── public_test/             # Public test split
├── scripts/
│   ├── run_baseline.slurm       # SLURM job script (HPC cluster)
│   └── analyze_audit.py         # Post-run analysis & report generation
└── outputs/                     # submission.csv, audit.json, analysis/
```

---

## Installation

```bash
git clone <repo-url> && cd road-buddy

conda create -n road_buddy python=3.12 -y
conda activate road_buddy

pip install -e .
# or
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, CUDA ≥ 12.0, VRAM ≥ 24 GB (for 7–8B models in bfloat16)

---

## Quick Start

### 1. Configure

Edit `configs/baseline_qwen.yaml`:

```yaml
model:
  model_name_or_path: Qwen/Qwen3-VL-8B-Instruct   # or Qwen/Qwen2.5-VL-7B-Instruct
  torch_dtype: bfloat16
  enable_thinking: false   # set true only with max_new_tokens >= 2048

runtime:
  max_samples: 50   # null = run full dataset; integer = quick test
```

### 2. Run inference

```bash
python -m road_buddy.cli infer --config configs/baseline_qwen.yaml
```

### 3. Evaluate

```bash
python -m road_buddy.cli eval \
  --submission outputs/submission_baseline_qwen.csv \
  --answer-json datasets/public_test/public_test_with_answers.json
```

### 4. Generate analysis report

```bash
python scripts/analyze_audit.py \
  --audit  outputs/<job_id>/audit.json \
  --output outputs/<job_id>/analysis
```

Output: `report.md`, `summary.json`, and 5 figures (accuracy by intent, confusion matrix, latency distribution, prediction source, frame score).

### 5. Run on HPC cluster (SLURM)

```bash
sbatch scripts/run_baseline.slurm configs/baseline_qwen.yaml
# Logs  → logs/<SLURM_JOB_ID>/job.{out,err}
# Output → outputs/<SLURM_JOB_ID>/{submission.csv,audit.json}
```

---

## Configuration Reference

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model` | `model_name_or_path` | — | HuggingFace model ID |
| `model` | `torch_dtype` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `model` | `enable_thinking` | `null` | `false` = disable thinking mode (Qwen3.5); `null` = model default |
| `sampling` | `num_frames` | `10` | Frames passed to VLM |
| `sampling` | `candidate_frame_multiplier` | `3` | Candidate pool = `num_frames × multiplier` |
| `sampling` | `query_aware_enabled` | `true` | Enable query-aware frame selection |
| `runtime` | `max_samples` | `null` | Limit samples (null = full dataset) |
| `runtime` | `time_limit_sec` | `30` | Per-sample time limit (contest rule) |
| `runtime` | `checkpoint_every_n` | `5` | Save intermediate results every N samples |

---

## Key Design Decisions

**Query-aware frame selection** — Instead of uniform sampling, the system analyzes the question to detect *intent* (temporal, value, direction, identification, existence) and *target objects*, then scores candidate frames by:
- **Sharpness** (Laplacian variance) — removes blurry frames
- **Temporal prior** — weights earlier/later frames based on question intent

**Generate + regex answer extraction** — Instruction-tuned VLMs are optimized for natural text generation. Using `model.generate()` followed by regex parsing is more reliable than extracting logits over A/B/C/D tokens, especially for models with thinking mode. The parser handles thinking blocks (`<think>…</think>`) transparently.

**Graceful degradation** — Each failure mode has a distinct `pred_source` tag in the audit log:

| `pred_source` | Meaning |
|---------------|---------|
| `generate_regex` | Normal path: regex found answer |
| `generate_default` | Regex failed → used first choice |
| `default_on_oom` | CUDA OOM → used first choice |
| `default_on_exception` | Unexpected error → used first choice |
| `default_on_frame_extract_error` | Video unreadable → used first choice |

---

## Supported Models

| Model | VRAM (bfloat16) | Notes |
|-------|----------------|-------|
| `Qwen/Qwen2.5-VL-7B-Instruct` | ~16 GB | Stable, recommended |
| `Qwen/Qwen3-VL-8B-Instruct` | ~18 GB | Latest Qwen VL series |
| `Qwen/Qwen3.5-9B` | ~20 GB | Set `enable_thinking: false` |
---

## Results

| Model | Accuracy | Avg Latency | p95 Latency |
|-------|----------|-------------|-------------|
| Qwen/Qwen3-VL-8B-Instruct | 53.58% | 3.38s | 4.70s |
| *(add your runs here)* | | | |

> Evaluated on `public_test` split (405 samples). Random baseline = 25%.

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{roadbuddy2025,
  title   = {RoadBuddy: A VideoQA Baseline for Vietnamese Dashcam Traffic Understanding},
  author  = { luonvuituoi },
  year    = {2025},
  note    = {Zalo AI Challenge 2025 – RoadBuddy Track},
  url     = {https://github.com/RoadBuddy-lvt}
}
```

---

## License

This repository is released for academic and research purposes.
Dataset usage is subject to the [Zalo AI Challenge terms](https://challenge.zalo.ai).
