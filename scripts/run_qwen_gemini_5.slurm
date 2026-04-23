#!/bin/bash
#SBATCH --job-name=qwen_gem_5
#SBATCH --output=/datastore/cndt_khanhnd/road-buddy/logs/qwen_gem_5_%j.out
#SBATCH --error=/datastore/cndt_khanhnd/road-buddy/logs/qwen_gem_5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=mps:a100:2
#SBATCH --time=02:00:00

module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1

# Chọn GPU phù hợp bằng helper của server
REQUIRED_VRAM=50000
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
  echo "$CHECK_OUT"
  exit 0
fi
if [ $EXIT_CODE -eq 11 ]; then
  echo "$CHECK_OUT"
  exit 1
fi

# Nếu Slurm chưa set sẵn thì tự set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=$CHECK_OUT
fi

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-gpu$CHECK_OUT
export HF_HOME=/datastore/cndt_khanhnd/.hf_cache
export TORCH_HOME=/datastore/cndt_khanhnd/.torch_cache
export PYTHONPATH=/datastore/cndt_khanhnd/road-buddy:/datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v2

source /datastore/cndt_khanhnd/miniconda3/bin/activate road_buddy

cd /datastore/cndt_khanhnd/road-buddy

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES | Pipe: gpu$CHECK_OUT"

python - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name =", torch.cuda.get_device_name(0))
PY

python -u -m road_buddy_agentic.cli.main infer-agentic \
  --config /datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v2/configs/agentic_qwen_gemini_5.yaml \
  --dataset-json /home/cndt_khanhnd/data/road-buddy/datasets/public_test_with_type.json \
  --video-root /home/cndt_khanhnd/data/road-buddy \
  --answer-json /home/cndt_khanhnd/data/road-buddy/public_test_answers.json \
  --max-samples 5 \
  --support-provider gemini \
  --support-model gemini-2.5-flash \
  --output-root /datastore/cndt_khanhnd/road-buddy/outputs/full_qwen_gemini_5
