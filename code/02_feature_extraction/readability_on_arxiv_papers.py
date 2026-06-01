#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from readability import Readability


# -----------------------------
# Config
# -----------------------------

DEFAULT_INPUT_DIR = "data\\arxiv_papers"
DEFAULT_OUTPUT_CSV = "data\\arxiv_papers_readability.csv"
DEFAULT_CHUNK_SIZE = 20

PERCENTILES = [10, 25, 50, 75, 90]

METRICS = [
    "flesch_kincaid",
    "flesch",
    "gunning_fog",
    "coleman_liau",
    "dale_chall",
    "ari",
    "linsear_write",
]


# -----------------------------
# Text processing
# -----------------------------

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def make_chunks(sentences: List[str], chunk_size: int) -> List[str]:
    return [
        " ".join(sentences[i:i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]

# -----------------------------
# Metric computation
# -----------------------------

def safe(func):
    try:
        val = func()
        if val is None or math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    except Exception:
        return None


def compute_metrics(chunk: str) -> Dict[str, Optional[float]]:
    try:
        r = Readability(chunk)

        return {
            "flesch_kincaid": safe(lambda: r.flesch_kincaid().score),
            "flesch": safe(lambda: r.flesch().score),
            "gunning_fog": safe(lambda: r.gunning_fog().score),
            "coleman_liau": safe(lambda: r.coleman_liau().score),
            "dale_chall": safe(lambda: r.dale_chall().score),
            "ari": safe(lambda: r.ari().score),
            "linsear_write": safe(lambda: r.linsear_write().score),
            "smog": safe(lambda: r.smog().score),
        }

    except Exception as e:
        print(f"Metric computation error: {type(e).__name__}: {e}")
        return {k: None for k in METRICS}


# -----------------------------
# Aggregation
# -----------------------------

def aggregate(values: List[Optional[float]], name: str) -> Dict[str, float]:
    arr = np.array([v for v in values if v is not None], dtype=float)

    result = {
        f"{name}_mean": np.nan,
        f"{name}_std": np.nan,
    }

    for p in PERCENTILES:
        result[f"{name}_p{p}"] = np.nan

    if len(arr) == 0:
        return result

    result[f"{name}_mean"] = float(np.mean(arr))
    result[f"{name}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    for p in PERCENTILES:
        result[f"{name}_p{p}"] = float(np.percentile(arr, p))

    return result


# -----------------------------
# Helpers
# -----------------------------

def get_split_from_path(path: Path, input_dir: Path) -> str:
    """
    Infer whether the file comes from the train or test subfolder under input_dir.
    """
    try:
        rel_parts = path.relative_to(input_dir).parts
    except ValueError:
        rel_parts = path.parts

    if len(rel_parts) > 0 and rel_parts[0] in {"train", "test"}:
        return rel_parts[0]

    return "unknown"


def find_txt_files(input_dir: Path, split: str) -> List[Path]:
    split_dir = input_dir / split
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.rglob("*.txt") if p.is_file())

# -----------------------------
# Per paper
# -----------------------------

def process_paper(path: Path, chunk_size: int, input_dir: Path) -> Dict:
    arxiv_id = path.stem  # KEEP EXACT
    split = get_split_from_path(path, input_dir)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {
            "arxiv_id": arxiv_id,
            "split": split,
            "file_path": str(path),
            "status": f"read_error_{e}",
        }

    text = normalize_text(text)
    sentences = split_into_sentences(text)
    chunks = make_chunks(sentences, chunk_size)

    metrics_all = {m: [] for m in METRICS}

    for chunk in chunks:
        if len(chunk.split()) < 100:
            continue

        m = compute_metrics(chunk)
        for k in METRICS:
            metrics_all[k].append(m[k])

    row = {
        "arxiv_id": arxiv_id,
        "split": split,
        "file_path": str(path),
        "n_sentences": len(sentences),
        "n_chunks": len(chunks),
        "status": "ok",
    }

    for metric_name, values in metrics_all.items():
        row.update(aggregate(values, metric_name))

    return row


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    files = find_txt_files(input_dir, args.split)
    files = files[args.shard_idx::args.n_shards]

    if args.limit:
        files = files[:args.limit]

    print(f"Split: {args.split}")
    print(f"Shard {args.shard_idx}/{args.n_shards}: {len(files)} files assigned")

    rows = []
    skipped = 0

    for i, f in enumerate(files, 1):
        arxiv_id = f.stem

        rows.append(process_paper(f, args.chunk_size, input_dir))

        if i % 100 == 0 or i == len(files):
            print(f"{i}/{len(files)} done (skipped={skipped})")

    print(f"Skipped {skipped} already processed papers")

    df = pd.DataFrame(rows)

    if not df.empty:
        if "arxiv_id" in df.columns:
            df["arxiv_id"] = df["arxiv_id"].astype(str)
        if "split" in df.columns:
            df["split"] = df["split"].astype(str)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("Saved to:", args.output_csv)


if __name__ == "__main__":
    main()