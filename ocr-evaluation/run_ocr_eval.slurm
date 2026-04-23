#!/bin/bash
#SBATCH --job-name=ocr_eval_qwen
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=mps:l40:1
#SBATCH --time=04:00:00

set -euo pipefail

# 1. LOAD MODULES
module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1

# 2. KHAI BÁO BIẾN
REQUIRED_VRAM=24576
REPO="/home/cndt_khanhnd/data/road-buddy"
PYTHON="/datastore/cndt_khanhnd/miniconda3/envs/road_buddy/bin/python"
RUN_LOG_DIR="$REPO/logs/$SLURM_JOB_ID"
RUN_OUTPUT_DIR="$REPO/outputs/$SLURM_JOB_ID"

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

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"

# 5. CACHE DIRS CHO HUGGING FACE
export TORCH_HOME=/datastore/cndt_khanhnd/.torch_cache
export HF_HOME=/datastore/cndt_khanhnd/.hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE"
mkdir -p "$REPO/logs" "$REPO/outputs" "$RUN_LOG_DIR" "$RUN_OUTPUT_DIR"

# Ghi log theo từng job
exec > >(tee -a "$RUN_LOG_DIR/ocr_eval.out") 2> >(tee -a "$RUN_LOG_DIR/ocr_eval.err" >&2)

echo "Run log dir   : $RUN_LOG_DIR"
echo "Run output dir: $RUN_OUTPUT_DIR"

HF_TOKEN_FILE="/datastore/cndt_khanhnd/.hf_token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
  export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
fi

# Load .env nếu tồn tại
if [ -f "$REPO/.env" ]; then
  set -a; source "$REPO/.env"; set +a
fi

# 6. KIỂM TRA FILE
if [ ! -x "$PYTHON" ]; then echo "Python not found: $PYTHON"; exit 1; fi
if [ ! -d "$REPO" ]; then echo "Repo not found: $REPO"; exit 1; fi

cd "$REPO"

"$PYTHON" -u -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")'

# 7. CHẠY ĐÁNH GIÁ OCR
echo "Starting OCR Evaluation (Qwen3-VL-8B-Instruct)..."

"$PYTHON" -u "$REPO/19-04-2026/ocr_eval.py" \
    --data-dir "$REPO/19-04-2026/vietnamese" \
    --output-dir "$RUN_OUTPUT_DIR" \
    --model "Qwen/Qwen3-VL-8B-Instruct" \
    --dtype bfloat16 \
    --num-samples 100 \
    --max-new-tokens 512 \
    --save-every 10

echo "Job completed! Kết quả tại: $RUN_OUTPUT_DIR"
