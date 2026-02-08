#!/bin/bash
#SBATCH --job-name=llm
#SBATCH --output=/ukp-storage-1/appak/logs/llm%j.log
#SBATCH --error=/ukp-storage-1/appak/logs/llm%j.err

#SBATCH --partition=gpu
#SBATCH --qos=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# ---- Writable dirs on ukp-storage (avoid /home entirely) ----
export HOME=/ukp-storage-1/appak
export TMPDIR=/ukp-storage-1/appak/tmp
export XDG_CACHE_HOME=/ukp-storage-1/appak/.cache
export HF_HOME=/ukp-storage-1/appak/hf_cache
export TORCH_HOME=/ukp-storage-1/appak/torch_cache
export FLASHINFER_WORKSPACE_DIR=/ukp-storage-1/appak/flashinfer_workspace

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$FLASHINFER_WORKSPACE_DIR" \
         /ukp-storage-1/appak/logs

# ---- Initialize conda for non-interactive shells (NO conda init) ----
source /ukp-storage-1/appak/miniconda3/etc/profile.d/conda.sh

conda activate /ukp-storage-1/appak/.conda_envs/llmjudge

# (Optional) avoid flashinfer backend if it causes trouble on sm_75 GPUs
# export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL=/ukp-storage-1/appak/hf_models/Qwen2.5-14B-Instruct-AWQ

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192

