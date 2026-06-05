#!/usr/bin/env python3
"""
Computes seven readability metrics (Flesch-Kincaid, Flesch, Gunning Fog, Coleman-Liau,
Dale-Chall, ARI, Linsear Write) for arXiv abstracts.
Designed as a 50-shard SLURM array job; defaults to shard 0 when run locally.
Input:  data/90k_arxiv_metadata_from_semantic_scholar.csv
Output: data/abstract_readability_50_shards/abstract_readability_shard_NN.csv
"""

import os
import pandas as pd
import numpy as np
from readability import Readability

INPUT_PATH = "data/90k_arxiv_metadata_from_semantic_scholar.csv"
OUT_DIR = "data/abstract_readability_50_shards"

TEXT_COL = "abstract"
N_SHARDS = 50

READABILITY_METRICS = [
    "flesch_kincaid",
    "flesch",
    "gunning_fog",
    "coleman_liau",
    "dale_chall",
    "ari",
    "linsear_write",
]

def safe_score(text):
    out = {}

    for metric in READABILITY_METRICS:
        out[f"abstract_{metric}_mean"] = np.nan

    if pd.isna(text):
        out["abstract_readability_status"] = "missing_abstract"
        return out

    text = str(text).strip()

    # The readability library raises an exception on very short strings; 20 chars is a safe lower bound
    if len(text) < 20:
        out["abstract_readability_status"] = "too_short"
        return out

    try:
        r = Readability(text)

        for metric in READABILITY_METRICS:
            try:
                result = getattr(r, metric)()
                out[f"abstract_{metric}_mean"] = result.score
            except Exception as e:
                print("ERROR:", e)
                out[f"abstract_{metric}_mean"] = np.nan

        out["abstract_readability_status"] = "ok"

    except Exception as e:
        print("ERROR:", e)
        out["abstract_readability_status"] = "readability_error"

    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Falls back to shard 0 when run interactively outside SLURM
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    df = pd.read_csv(
        INPUT_PATH,
        dtype={"arxiv_id": str},  # preserves leading/trailing zeros
        low_memory=False
    )

    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found.")

    n = len(df)
    # Ceiling division ensures the last shard is not empty even if n is not divisible by N_SHARDS
    shard_size = int(np.ceil(n / N_SHARDS))

    start = task_id * shard_size
    end = min(start + shard_size, n)

    shard = df.iloc[start:end].copy()

    print(f"Task {task_id}: processing rows {start} to {end} of {n}")

    results = []

    for i, text in enumerate(shard[TEXT_COL], start=1):
        results.append(safe_score(text))

        if i % 1000 == 0:
            print(f"Task {task_id}: processed {i}/{len(shard)} abstracts")

    # Preserve shard.index so pd.concat aligns rows correctly even after iloc slicing
    readability_df = pd.DataFrame(results, index=shard.index)

    out_df = pd.concat([shard, readability_df], axis=1)

    out_path = os.path.join(
        OUT_DIR,
        f"abstract_readability_shard_{task_id:02d}.csv"
    )

    out_df.to_csv(out_path, index=False)

    print(f"Task {task_id}: saved {len(out_df)} rows to {out_path}")
    print(out_df["abstract_readability_status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()