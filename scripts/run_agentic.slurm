#!/bin/bash
#SBATCH --job-name=roadbuddy_agentic
#SBATCH --output=/datastore/cndt_khanhnd/road-buddy/logs/qwen_agentic_%j.out
#SBATCH --error=/datastore/cndt_khanhnd/road-buddy/logs/qwen_agentic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=mps:l40:1
#SBATCH --time=06:00:00

set -euo pipefail

# 1. LOAD MODULES
module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1

# 2. KHAI BÁO BIẾN
REQUIRED_VRAM=24576
REPO="/home/cndt_khanhnd/data/road-buddy"
AGENTIC_V2="$REPO/road_buddy_agentic_v2"
PYTHON="/datastore/cndt_khanhnd/miniconda3/envs/road_buddy/bin/python"
CONFIG_PATH="${1:-$AGENTIC_V2/configs/agentic_qwen_base.yaml}"
SUPPORT_PROVIDER="${2:-none}"
SUPPORT_MODEL="${3:-none}"
RUN_TAG="qwen_agentic_${SUPPORT_PROVIDER}_${SLURM_JOB_ID}"
RUN_OUTPUT_DIR="/datastore/cndt_khanhnd/road-buddy/outputs/${RUN_TAG}"

# 3. KIỂM TRA GPU
unset CUDA_VISIBLE_DEVICES

CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then echo "$CHECK_OUT"; exit 0; fi
if [ $EXIT_CODE -eq 11 ]; then echo "$CHECK_OUT"; exit 1; fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$CHECK_OUT
fi

echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# 4. KHỞI TẠO MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# 5. CACHE DIRS
export TORCH_HOME=/datastore/cndt_khanhnd/.torch_cache
export HF_HOME=/datastore/cndt_khanhnd/.hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE"
mkdir -p "$REPO/logs" "$RUN_OUTPUT_DIR"

HF_TOKEN_FILE="/datastore/cndt_khanhnd/.hf_token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
  export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
fi

if [ -f "$REPO/.env" ]; then
  set -a; source "$REPO/.env"; set +a
fi

# 6. KIỂM TRA FILE
if [ ! -x "$PYTHON" ]; then echo "Python not found: $PYTHON"; exit 1; fi
if [ ! -d "$REPO" ]; then echo "Repo not found: $REPO"; exit 1; fi
if [ ! -f "$CONFIG_PATH" ]; then echo "Config not found: $CONFIG_PATH"; exit 1; fi

cd "$REPO"
export PYTHONPATH="$AGENTIC_V2:$REPO:${PYTHONPATH:-}"

nvidia-smi
"$PYTHON" -u -c 'import torch; print("torch:", torch.__version__); print("cuda available:", torch.cuda.is_available()); print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")'

# 7. CHẠY AGENTIC INFERENCE
echo "Starting Agentic Inference..."
"$PYTHON" -u -m road_buddy_agentic.cli.main infer-agentic \
  --config "$CONFIG_PATH" \
  --dataset-json "$REPO/datasets/public_test_with_type.json" \
  --video-root "$REPO/datasets" \
  --answer-json "$REPO/datasets/public_test/public_test_with_answers.json" \
  --max-samples 100 \
  --support-provider "$SUPPORT_PROVIDER" \
  --support-model "$SUPPORT_MODEL" \
  --output-root "$RUN_OUTPUT_DIR"

echo "Job completed!"
