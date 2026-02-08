#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --output=metadata/logs/extract_%A_%a.out
#SBATCH --error=metadata/logs/extract_%A_%a.err

# Choose one of these partitions:
#SBATCH --partition=cpu
#SBATCH --qos=cpu             # or: yolo

#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --nodes=1

#SBATCH --array=50-60

set -euo pipefail

IN_DIR="/mnt/beegfs/work/church/semantic_scholar/releases/2025-09-04/database/papers"
OUT_DIR="/ukp-storage-1/appak/metadata/papers_piece_out"
SCRIPT="/mnt/beegfs/work/appak/extract_paper_metadata.py"


python3 "${SCRIPT}" \
  --in_dir "${IN_DIR}" \
  --out_dir "${OUT_DIR}" \
  --piece "${SLURM_ARRAY_TASK_ID}" \
  --flush_every 200000 \
  --retries 3 \
  --retry_sleep 5 \
  --salvage_on_eof \
  --max_categories 6