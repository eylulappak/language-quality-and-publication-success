#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Compute GPT-2 perplexity for abstracts using Slurm array shards.")
    p.add_argument("--input_csv", default="data\\90k_arxiv_metadata_from_semantic_scholar.csv")
    p.add_argument("--output_dir", default="data\\abstract_gpt2_ppl_outputs_50shards")
    p.add_argument("--model_name", default="openai-community/gpt2")
    p.add_argument("--chunk_size", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_chars", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--array_index", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=50)
    p.add_argument("--output_prefix", default="gpt2_ppl_abstracts")
    p.add_argument("--skip_existing_output", action="store_true")
    return p.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    return device_arg


def resolve_dtype(dtype_arg, device):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float16 if device == "cuda" else torch.float32


def chunked(seq: List[torch.Tensor], size: int) -> Iterable[List[torch.Tensor]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@torch.inference_mode()
def compute_perplexity_for_text(text, tokenizer, model, device, chunk_size, batch_size):
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]

    seq_len = len(token_ids)

    if seq_len < 2:
        return {
            "status": "too_short",
            "n_tokens": seq_len,
            "n_pred_tokens": 0,
            "n_windows": 0,
            "neg_log_likelihood": "",
            "perplexity": "",
        }

    windows_input = []
    windows_labels = []

    for begin in range(0, seq_len, chunk_size):
        end = min(begin + chunk_size, seq_len)
        ids_list = token_ids[begin:end]

        if len(ids_list) < 2:
            continue

        ids = torch.tensor(ids_list, dtype=torch.long)
        labels = ids.clone()
        labels[0] = -100

        windows_input.append(ids)
        windows_labels.append(labels)

    total_nll = 0.0
    total_pred_tokens = 0

    for batch_inputs, batch_labels in zip(
        chunked(windows_input, batch_size),
        chunked(windows_labels, batch_size),
    ):
        input_batch = torch.nn.utils.rnn.pad_sequence(
            batch_inputs,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(device)

        label_batch = torch.nn.utils.rnn.pad_sequence(
            batch_labels,
            batch_first=True,
            padding_value=-100,
        ).to(device)

        outputs = model(input_ids=input_batch, labels=label_batch)

        valid_labels = int((label_batch != -100).sum().item())
        batch_nll = float(outputs.loss.item()) * valid_labels

        total_nll += batch_nll
        total_pred_tokens += valid_labels

    ppl = math.exp(total_nll / total_pred_tokens) if total_pred_tokens > 0 else ""

    return {
        "status": "ok",
        "n_tokens": seq_len,
        "n_pred_tokens": total_pred_tokens,
        "n_windows": len(windows_input),
        "neg_log_likelihood": total_nll,
        "perplexity": ppl,
    }


def main():
    args = parse_args()

    array_index = args.array_index
    if array_index is None:
        array_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    num_shards = args.num_shards

    if array_index < 0 or array_index >= num_shards:
        raise ValueError(f"Invalid array_index={array_index}; expected 0 to {num_shards - 1}")

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_csv = output_dir / f"{args.output_prefix}.shard_{array_index:03d}_of_{num_shards:03d}.csv"

    if args.skip_existing_output and shard_csv.exists() and shard_csv.stat().st_size > 0:
        print(f"Shard output already exists, skipping: {shard_csv}")
        return

    print(f"Reading input CSV: {input_csv}")

    needed_cols = ["corpusid", "arxiv_id", "abstract"]
    df = pd.read_csv(
        input_csv,
        usecols=needed_cols,
        dtype={"corpusid": str, "arxiv_id": str, "abstract": str},
        low_memory=False,
    )

    df = df.dropna(subset=["corpusid", "abstract"]).reset_index(drop=True)

    if args.max_chars is not None:
        df["abstract"] = df["abstract"].astype(str).str.slice(0, args.max_chars)

    shard_df = df.iloc[array_index::num_shards].copy()

    print(f"SLURM_ARRAY_TASK_ID: {array_index}")
    print(f"Number of shards: {num_shards}")
    print(f"Total rows loaded: {len(df):,}")
    print(f"Rows in this shard: {len(shard_df):,}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    max_length = args.max_length or int(getattr(model.config, "n_positions", 1024))
    chunk_size = args.chunk_size or max_length

    if chunk_size > max_length:
        raise ValueError(f"chunk_size ({chunk_size}) must be <= max_length ({max_length})")

    fieldnames = [
        "corpusid",
        "arxiv_id",
        "abstract",
        "n_tokens",
        "n_pred_tokens",
        "n_windows",
        "chunk_size",
        "max_length",
        "model_name",
        "status",
        "neg_log_likelihood",
        "perplexity",
    ]

    with shard_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(shard_df.itertuples(index=False), start=1):
            corpusid = row.corpusid
            arxiv_id = row.arxiv_id
            abstract = str(row.abstract).strip()

            try:
                if not abstract:
                    result = {
                        "status": "empty_abstract",
                        "n_tokens": "",
                        "n_pred_tokens": "",
                        "n_windows": "",
                        "neg_log_likelihood": "",
                        "perplexity": "",
                    }
                else:
                    result = compute_perplexity_for_text(
                        text=abstract,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        chunk_size=chunk_size,
                        batch_size=args.batch_size,
                    )

                out_row = {
                    "corpusid": corpusid,
                    "arxiv_id": arxiv_id,
                    "abstract": abstract,
                    "n_tokens": result["n_tokens"],
                    "n_pred_tokens": result["n_pred_tokens"],
                    "n_windows": result["n_windows"],
                    "chunk_size": chunk_size,
                    "max_length": max_length,
                    "model_name": args.model_name,
                    "status": result["status"],
                    "neg_log_likelihood": result["neg_log_likelihood"],
                    "perplexity": result["perplexity"],
                }

            except Exception as e:
                out_row = {
                    "corpusid": corpusid,
                    "arxiv_id": arxiv_id,
                    "abstract": abstract,
                    "n_tokens": "",
                    "n_pred_tokens": "",
                    "n_windows": "",
                    "chunk_size": chunk_size,
                    "max_length": max_length,
                    "model_name": args.model_name,
                    "status": f"error: {repr(e)}",
                    "neg_log_likelihood": "",
                    "perplexity": "",
                }

            writer.writerow(out_row)

            if i % 100 == 0 or i == len(shard_df):
                print(f"Processed {i:,}/{len(shard_df):,} abstracts", flush=True)

    print(f"Wrote: {shard_csv}")


if __name__ == "__main__":
    main()