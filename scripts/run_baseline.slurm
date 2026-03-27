#!/bin/bash
#SBATCH --job-name=roadbuddy_qwen
#SBATCH --output=logs/mps_%j.out
#SBATCH --error=logs/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=mps:1
#SBATCH --time=06:00:00

set -euo pipefail

# 1. LOAD MODULES
module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1

# 2. KHAI BÁO BIẾN
REQUIRED_VRAM=16384
REPO="/home/cndt_khanhnd/data/road-buddy"
PYTHON="/datastore/cndt_khanhnd/miniconda3/envs/road_buddy/bin/python"
CONFIG_PATH="$REPO/configs/baseline_qwen.yaml"

# 3. KIỂM TRA GPU 
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
mkdir -p "$REPO/logs"

HF_TOKEN_FILE="/datastore/cndt_khanhnd/.hf_token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
  export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
fi

# 6. KIỂM TRA FILE VÀ VÀO THƯ MỤC
if [ ! -x "$PYTHON" ]; then echo "Python not found: $PYTHON"; exit 1; fi
if [ ! -d "$REPO" ]; then echo "Repo not found: $REPO"; exit 1; fi

cd "$REPO"

"$PYTHON" -u -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")'

# 7. CHẠY CODE INFERENCE
echo "Starting Inference Qwen2.5-VL..."
"$PYTHON" -u -m road_buddy.cli infer --config "$CONFIG_PATH"

echo "Job completed!"