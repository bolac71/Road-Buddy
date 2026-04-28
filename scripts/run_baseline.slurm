#!/bin/bash
#SBATCH --job-name=roadbuddy_qwen
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=mps:a100:1
#SBATCH --time=06:00:00

set -euo pipefail

# 1. LOAD MODULES
module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1

# 2. KHAI BÁO BIẾN
REQUIRED_VRAM=24576
REPO="/home/cndt_khanhnd/data/road-buddy"
PYTHON="/datastore/cndt_khanhnd/miniconda3/envs/road_buddy/bin/python"
CONFIG_PATH="${1:-$REPO/configs/qwen3vl_heuristic.yaml}"
RUN_LOG_DIR="$REPO/logs/$SLURM_JOB_ID"
RUN_OUTPUT_DIR="$REPO/outputs/$SLURM_JOB_ID"
RUN_CONFIG_PATH="$RUN_OUTPUT_DIR/config_job_${SLURM_JOB_ID}.yaml"

# 3. KIỂM TRA GPU 
unset CUDA_VISIBLE_DEVICES

CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then echo "$CHECK_OUT"; exit 0; fi
if [ $EXIT_CODE -eq 11 ]; then echo "$CHECK_OUT"; exit 1; fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$CHECK_OUT
fi

# 4. KHỞI TẠO MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | Pipe: gpu$CHECK_OUT"

# 5. CACHE DIRS CHO HUGGING FACE
export TORCH_HOME=/datastore/cndt_khanhnd/.torch_cache
export HF_HOME=/datastore/cndt_khanhnd/.hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE"
mkdir -p "$REPO/logs" "$REPO/outputs" "$RUN_LOG_DIR" "$RUN_OUTPUT_DIR"

# Ghi log theo từng job vào logs/<job_id>/job.out và job.err
exec > >(tee -a "$RUN_LOG_DIR/job.out") 2> >(tee -a "$RUN_LOG_DIR/job.err" >&2)

echo "Run log dir: $RUN_LOG_DIR"
echo "Run output dir: $RUN_OUTPUT_DIR"

HF_TOKEN_FILE="/datastore/cndt_khanhnd/.hf_token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
  export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
fi

# Load .env nếu tồn tại
if [ -f "$REPO/.env" ]; then
  set -a; source "$REPO/.env"; set +a
fi

# 6. KIỂM TRA FILE VÀ VÀO THƯ MỤC
if [ ! -x "$PYTHON" ]; then echo "Python not found: $PYTHON"; exit 1; fi
if [ ! -d "$REPO" ]; then echo "Repo not found: $REPO"; exit 1; fi

cd "$REPO"

# Tạo config tạm theo job để không ghi đè output giữa các lần chạy.
"$PYTHON" - "$CONFIG_PATH" "$RUN_CONFIG_PATH" "$RUN_OUTPUT_DIR" <<'PY'
import pathlib
import sys

import yaml

src_cfg, dst_cfg, run_output_dir = sys.argv[1:4]
src_cfg_path = pathlib.Path(src_cfg).resolve()
src_cfg_dir = src_cfg_path.parent

with src_cfg_path.open("r", encoding="utf-8") as f:
  cfg = yaml.safe_load(f)

def resolve_path(path_value: str | None) -> str | None:
  if not path_value:
    return path_value
  p = pathlib.Path(path_value)
  if p.is_absolute():
    return str(p)
  return str((src_cfg_dir / p).resolve())

paths = cfg.setdefault("paths", {})

# Resolve all input paths first to avoid wrong relative paths when config is moved.
paths["dataset_json"] = resolve_path(paths.get("dataset_json"))
paths["video_root"] = resolve_path(paths.get("video_root"))
paths["answer_json"] = resolve_path(paths.get("answer_json"))

# Per-job outputs
paths["output_csv"] = str(pathlib.Path(run_output_dir) / "submission.csv")
paths["audit_json"] = str(pathlib.Path(run_output_dir) / "audit.json")

with open(dst_cfg, "w", encoding="utf-8") as f:
  yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print(f"Generated run config: {dst_cfg}")
print(f"Run submission path: {paths['output_csv']}")
print(f"Run audit path: {paths['audit_json']}")
PY

"$PYTHON" -u -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")'

# 7. CHẠY CODE INFERENCE
echo "Starting Inference Qwen2.5-VL..."
"$PYTHON" -u -m road_buddy.cli infer --config "$RUN_CONFIG_PATH"

echo "Job completed!"