#!/bin/bash
#SBATCH --job-name=gliner
#SBATCH --output=logs/gliner_%j.log
#SBATCH --error=logs/gliner_%j.err


# Choose one of these partitions:
#SBATCH --partition=gpu
#SBATCH --qos=gpu-small              

# Request one GPU and reasonable memory
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

python /ukp-storage-1/appak/extract_authors_gliner.py --in_json /ukp-storage-1/appak/sample_papers/2024.propor-1.17.json --out_csv /ukp-storage-1/appak/sample_papers/2024.propor-1.17_authors_gliner.csv
