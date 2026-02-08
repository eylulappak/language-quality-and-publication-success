#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output=logs/plot%j.log
#SBATCH --error=logs/plot%j.err


# Choose one of these partitions:
#SBATCH --partition=gpu
#SBATCH --qos=gpu-small               # or: yolo

# Request one GPU and reasonable memory
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

python extract_authors_gliner.py \
    --in_txt /ukp-storage-1/appak/sample_papers/2024.propor-1.17_page1.txt \
    --paper_filename 2024.propor-1.17.pdf \
    --out_csv /ukp-storage-1/appak/sample_papers/2024.propor-1.17_authors_gliner.csv \
    --debug






