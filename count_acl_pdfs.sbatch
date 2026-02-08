#!/bin/bash
#SBATCH --job-name=acl_pdf_check
#SBATCH --output=/ukp-storage-1/appak/acl_paper_pdfs/acl_pdf_check_%j.out
#SBATCH --error=/ukp-storage-1/appak/acl_paper_pdfs/acl_pdf_check_%j.err
# Choose one of these partitions:
#SBATCH --partition=cpu
#SBATCH --qos=cpu             # or: yolo

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail

OUTDIR="/ukp-storage-1/appak/acl_paper_pdfs"
ALL_IDS="${OUTDIR}/paper_ids_all.txt"

DL_OK="${OUTDIR}/paper_ids_downloadable.txt"
DL_404="${OUTDIR}/paper_ids_nondownloadable.txt"
TMP_OK="${OUTDIR}/.tmp_ok_${SLURM_JOB_ID}.txt"
TMP_404="${OUTDIR}/.tmp_404_${SLURM_JOB_ID}.txt"

mkdir -p "$OUTDIR"
: > "$TMP_OK"
: > "$TMP_404"

# Tune this:
# - Start with 16 or 32 to be polite to ACL + avoid network throttling on cluster
# - If it’s still slow and allowed, increase to 48/64
PARALLEL=32

echo "Starting ACL PDF availability check (parallel=$PARALLEL)"
echo "Input: $ALL_IDS"
echo "Output OK:  $DL_OK"
echo "Output 404: $DL_404"
echo "-----------------------------------"

# Make sure we only have non-empty unique IDs (also avoids re-checking duplicates)
sort -u "$ALL_IDS" | sed '/^\s*$/d' \
| xargs -n 1 -P "$PARALLEL" bash -lc '
  id="$1"
  url="https://aclanthology.org/${id}.pdf"

  # HEAD check with strict timeouts so jobs never hang:
  # --connect-timeout: max seconds to establish TCP/TLS connection
  # --max-time: max total seconds for the request
  # --fail: non-2xx => exit non-zero
  if curl -sI -L --fail --connect-timeout 5 --max-time 15 "$url" >/dev/null; then
    echo "$id"
  else
    echo "$id" >&2
  fi
' _ \
1>> "$TMP_OK" \
2>> "$TMP_404"

# Write final outputs (sorted, unique)
sort -u "$TMP_OK"  > "$DL_OK"
sort -u "$TMP_404" > "$DL_404"

ok=$(wc -l < "$DL_OK" | tr -d " ")
nf=$(wc -l < "$DL_404" | tr -d " ")

echo "-----------------------------------"
echo "ACL PDF availability summary"
echo "Downloadable PDFs     : $ok"
echo "Non-downloadable PDFs : $nf"
echo "Total IDs checked     : $((ok + nf))"
echo "Finished."

# Cleanup temp files
rm -f "$TMP_OK" "$TMP_404"
