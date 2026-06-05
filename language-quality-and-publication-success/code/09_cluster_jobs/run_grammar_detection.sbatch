#!/bin/bash
#SBATCH --job-name=grammar_detect
#SBATCH --output=/mnt/beegfs/work/appak/logs/grammar_detect/%x_%A_%a.out
#SBATCH --error=/mnt/beegfs/work/appak/logs/grammar_detect/%x_%A_%a.err

#SBATCH --partition=gpu
#SBATCH --qos=gpu-small
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00

#SBATCH --array=0-49%2

set -euo pipefail
set -x

INPUT_ROOT="/mnt/beegfs/work/appak/arxiv_txts_100k"
OUTPUT_ROOT="/mnt/beegfs/work/appak/t5-base_grammar_detections_100k"
SCRIPT="/mnt/beegfs/work/appak/final_pipeline/computation_scripts/detect_grammar.py"

TRAIN_INPUT="$INPUT_ROOT/train"
TEST_INPUT="$INPUT_ROOT/test"

TRAIN_OUTPUT="$OUTPUT_ROOT/train"
TEST_OUTPUT="$OUTPUT_ROOT/test"

LOG_DIR="/mnt/beegfs/work/appak/logs/grammar_detect"

MAX_SENTENCES=500
BATCH_SIZE=500

source /mnt/beegfs/work/appak/venvs/gec_pipeline/bin/activate

mkdir -p "$TRAIN_OUTPUT" "$TEST_OUTPUT" "$LOG_DIR"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-50}"

echo "[debug] job started at $(date)"
echo "[debug] running on host $(hostname)"
echo "[debug] array id: $SHARD_ID"
echo "[debug] max sentences per paper: $MAX_SENTENCES"
echo "[debug] batch size: $BATCH_SIZE"

mapfile -t TRAIN_FILES < <(find "$TRAIN_INPUT" -type f -name "*.txt" | sort)
mapfile -t TEST_FILES  < <(find "$TEST_INPUT"  -type f -name "*.txt" | sort)

ALL_FILES=("${TRAIN_FILES[@]}" "${TEST_FILES[@]}")
TOTAL=${#ALL_FILES[@]}

echo "[debug] total files: $TOTAL"
echo "[debug] shard: $SHARD_ID / $N_SHARDS"

processed=0
skipped=0
failed=0

python "$SCRIPT" \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --shard_id "$SHARD_ID" \
    --n_shards "$N_SHARDS" \
    --max_sentences "$MAX_SENTENCES" \
    --batch_size "$BATCH_SIZE"

echo "[debug] finished at $(date)"
echo "[debug] processed=$processed skipped=$skipped failed=$failed