#!/bin/bash
#SBATCH --job-name=pdf2json
#SBATCH --output=logs/acl_pdf2json_%A_%a.out
#SBATCH --error=logs/acl_pdf2json_%A_%a.err

# Choose one of these partitions:
#SBATCH --partition=yolo
#SBATCH --qos=yolo             # or: yolo

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00

# Later change this to array=100-114
#SBATCH --array=50-99

set -euo pipefail

# Always use HOME dynamically (never hard-code /home/appak)
echo "HOME=$HOME"
pwd
hostname

# 1) Initialize conda in THIS shell (works in batch)
source /ukp-storage-1/appak/miniconda3/etc/profile.d/conda.sh

# 2) Activate env (use the env NAME if possible)
conda activate /ukp-storage-1/appak/.conda_envs/ocr_py311

# Debug
which python
python --version

# Keep libraries from oversubscribing threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ORT_LOG_SEVERITY_LEVEL=3

python /ukp-storage-1/appak/batch_pdf2txt.py \
  --manifest /ukp-storage-1/appak/acl_pdf_manifest.txt \
  --out_dir /ukp-storage-1/appak/acl_jsons \
  --converter /ukp-storage-1/appak/pdf2txt_with_fitz.py \
  --chunk_size 1000 \
  --skip_existing
