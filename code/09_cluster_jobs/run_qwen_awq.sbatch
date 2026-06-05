#!/bin/bash
#SBATCH --job-name=native_like_json
#SBATCH --output=/mnt/beegfs/work/appak/logs/native_like_json/%x_%A_%a.out
#SBATCH --error=/mnt/beegfs/work/appak/logs/native_like_json/%x_%A_%a.err
#SBATCH --chdir=/mnt/beegfs/work/appak

#SBATCH --partition=gpu
#SBATCH --qos=gpu-small
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00

#SBATCH --array=161-200%2

set -eo pipefail

mkdir -p /mnt/beegfs/work/appak/logs/native_like_json

echo "[debug] job started $(date) on $(hostname)" >&2
echo "[debug] SLURM_JOB_ID=$SLURM_JOB_ID SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}" >&2

export CUDA_HOME=/nfs/cuda/cuda-12.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# --- Make everything write to BeeGFS (writable) ---
export HOME=/mnt/beegfs/work/appak
export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"
mkdir -p "$HOME/.conda" "$CONDA_PKGS_DIRS"

export XDG_CACHE_HOME=$HOME/.cache
export HF_HOME=$XDG_CACHE_HOME/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$XDG_CACHE_HOME/torch

# --- torch.compile / inductor / triton caches ---
export TMPDIR=$HOME/.tmp
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton

# (optional but nice)
export CUDA_CACHE_PATH=$XDG_CACHE_HOME/nvcc
export VLLM_NO_USAGE_STATS=1

# --- vLLM knobs ---
export VLLM_USE_V1=0
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export VLLM_USE_CUDA_GRAPH=0
export VLLM_DISABLE_COMPILE=1
export VLLM_ENFORCE_EAGER=1
export VLLM_LOG_LEVEL=DEBUG

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" \
         "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

# ---- ACTIVATE ENV EARLY ----
source /mnt/beegfs/work/appak/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/beegfs/work/appak/.conda_envs/llmjudge

echo "[debug] CONDA_PREFIX=$CONDA_PREFIX" >&2
which python >&2 || true
python -V >&2 || true

# ---- compiler selection (IMPORTANT for FlashInfer/Ninja) ----
# Prefer conda GCC12 toolchain if present, and force nvcc to use it via CUDAHOSTCXX.
# ---- compiler selection (IMPORTANT for FlashInfer/Ninja) ----
if [[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="$CXX"

  # Force nvcc to use GCC12 explicitly
  export NVCC_PREPEND_FLAGS="-ccbin $CC"
  export CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=$CC -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX"

else
  echo "[FATAL] conda gcc12 toolchain not found in env. Install gcc_linux-64=12 and gxx_linux-64=12 into llmjudge." >&2
  exit 1
fi

echo "[debug] CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX" >&2
$CC --version >&2 || true
$CXX --version >&2 || true

# ---- sanity checks ----
echo "CUDA_HOME=$CUDA_HOME" >&2
which nvcc >&2 || true
nvcc --version >&2 || true

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cc:', torch.cuda.get_device_capability() if torch.cuda.is_available() else None)" >&2

export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=1

# ---- clear flashinfer cache so it rebuilds cleanly (use the real path) ----
rm -rf "$HOME/.cache/flashinfer" || true
rm -rf "$XDG_CACHE_HOME/flashinfer" || true
rm -rf "$HOME/.cache/torch_extensions" || true

# -------- paths --------
LIST="/mnt/beegfs/work/appak/arxiv_100k_txt_file_list.txt"
SCRIPT="/mnt/beegfs/work/appak/qwen_native_like/qwen_native_like_prediction.py"

OUT_ROOT="/mnt/beegfs/work/appak/arxiv_qwen_outputs"
OUT_DIR="$OUT_ROOT/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUT_DIR"

# -------- batching --------
BATCH_SIZE=500
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( START + BATCH_SIZE ))

echo "[job] task_id=$SLURM_ARRAY_TASK_ID start=$START end=$END out_dir=$OUT_DIR host=$(hostname)" >&2

nvidia-smi -L >&2 || true
nvidia-smi >&2 || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2

python3 "$SCRIPT" \
  --list_file "$LIST" \
  --start "$START" \
  --end "$END" \
  --out_dir "$OUT_DIR" \
  --model_path "/mnt/beegfs/work/appak/hf_models/Qwen2.5-14B-Instruct-AWQ" \
  --backend vllm \
  --max_chars 40000 \
  --max_new_tokens 512 \
  --max_model_len 8192 \
  --gpu_mem_util 0.90 \
  --temperature 0.0 \
  --top_p 0.9 \
  --seed 0 \
  --retries 2