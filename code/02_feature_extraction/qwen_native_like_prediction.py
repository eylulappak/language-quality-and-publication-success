#!/usr/bin/env python3
"""
Run Qwen2.5-14B-Instruct-AWQ on merged paper text (.txt) files and write ONE JSON per paper into an output folder.

Input modes:
  1) --in_txt FILE.txt
  2) --in_dir DIR --glob "*.txt"
  3) --list_file FILELIST.txt --start N --end M   (recommended for Slurm arrays)

Output:
  For each input txt, write:
    OUT_DIR/<paper_id>.native_like.json

paper_id is derived from filename:
  1952.earlymt-1.1.json.sections.txt -> 1952.earlymt-1.1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from transformers import AutoTokenizer

HOME = "/mnt/beegfs/work/appak"
os.environ.setdefault("HOME", HOME)
os.environ.setdefault("XDG_CACHE_HOME", f"{HOME}/.cache")
os.environ.setdefault("HF_HOME", f"{HOME}/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
os.environ.setdefault("TORCH_HOME", f"{HOME}/.cache/torch")
os.environ.setdefault("TMPDIR", f"{HOME}/.tmp")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{HOME}/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", f"{HOME}/.cache/triton")
os.environ.setdefault("CUDA_CACHE_PATH", f"{HOME}/.cache/nvcc")
os.environ["VLLM_LOG_LEVEL"] = "DEBUG"

for k in [
    "XDG_CACHE_HOME",
    "HF_HOME",
    "TORCH_HOME",
    "TMPDIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
]:
    Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

# ----------------------------
# Prompt (UPDATED)
# ----------------------------

SYSTEM_PROMPT= '''You are a linguistics expert assessing whether a text is written in native-like academic English.
You will receive excerpts from a scientific paper merged into a single text. Evaluate the overall linguistic quality of the writing. Assess language only; do not speculate about the authors' nationality, affiliation, or background.
Base your judgment on systematic patterns across the whole text, not on isolated typos. Focus on observable linguistic features such as:

• Grammatical control: correct use of articles and determiners (a/an/the), correct subject-verb agreement, proper verb forms and tense, and correct prepositions.  
• Lexical usage: natural academic collocations, appropriate word choice, and the absence of literal-translation-like clauses or sentences.  
• Sentence structure: well-formed, fluent sentences with natural clause structure and word order.

Ignore minor spelling mistakes andOCR/PDF artifacts (e.g., line breaks, “\n”, hyphenation splits, citation numbers, and section numbering).
Return strictly valid JSON with:
• paper_id (string)  
• native_like_score (integer 0-10, where 0 = clearly non-native-like, 5 = mixed/uncertain, 10 = indistinguishable from strong native academic writing)  
• confidence (float 0-1)  
• verdict (1 = native-like, 0 = non-native-like)  
• reasons (3-5 bullet points referencing specific observable linguistic patterns and giving examples from the paper)
'''

def read_list_file(list_path: Path) -> List[Path]:
    files: List[Path] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                files.append(Path(p))
    return files


def derive_paper_id_from_path(p: Path) -> str:
    name = p.name
    for suf in [".json.sections.txt", ".sections.txt", ".txt"]:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    name = re.sub(r"\.json\.sections$", "", name)
    name = re.sub(r"\.json$", "", name)
    return name or p.stem


def load_txt(path: Path, max_chars: int) -> str:
    txt = path.read_text(encoding="utf-8", errors="replace").strip()
    if max_chars > 0 and len(txt) > max_chars:
        txt = txt[:max_chars] + "\n[TRUNCATED]"
    return txt


# ----------------------------
# Model backend
# ----------------------------

def load_backend(
    model_path: str,
    dtype: str,
    gpu_mem_util: float,
    max_model_len: int,
    prefer: str = "vllm",
):
    if prefer not in ("vllm", "transformers"):
        raise ValueError(f"--backend must be 'vllm' or 'transformers' (got {prefer})")

    if prefer == "vllm":
        try:
            from vllm import LLM, SamplingParams

            llm = LLM(
                model=model_path,
                dtype=dtype,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_mem_util,
                max_model_len=max_model_len,
                enforce_eager=True,
                disable_log_stats=True,
                quantization="awq",
            )

            def _generate(messages, max_new_tokens, temperature, top_p, seed):
                tok = llm.get_tokenizer()
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                sp = SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
                out = llm.generate([prompt], sp)[0]
                return out.outputs[0].text

            return "vllm", _generate

        except Exception as e:
            print("[ERROR] vLLM backend failed:", repr(e), file=sys.stderr)
            raise RuntimeError("vLLM backend failed (see error above).") from e

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16,
        )
        model.eval()

        def _generate(messages, max_new_tokens, temperature, top_p, seed):
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tok.eos_token_id,
            )
            if seed is not None:
                torch.manual_seed(seed)
            with torch.no_grad():
                out_ids = model.generate(**inputs, **gen_kwargs)

            gen_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
            return tok.decode(gen_ids, skip_special_tokens=True)

        return "transformers", _generate

    except Exception as e:
        print("[ERROR] Transformers backend failed:", repr(e), file=sys.stderr)
        raise RuntimeError("Transformers backend failed (see error above).") from e


# ----------------------------
# Output parsing
# ----------------------------

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

# Matches a comma that is followed only by whitespace and then a closing } or ]
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

def repair_json_text(s: str) -> str:
    """
    Minimal JSON repair:
      - remove trailing commas before } or ]
    """
    # repeatedly apply in case there are nested trailing commas
    prev = None
    while prev != s:
        prev = s
        s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s

def extract_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()

    # 1) try direct parse (with repair)
    try:
        repaired = repair_json_text(raw)
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) try extracting first {...} blob, then parse (with repair)
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        raise ValueError("Model output did not contain a JSON object.")

    obj_txt = repair_json_text(m.group(0))
    obj = json.loads(obj_txt)
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON was not an object.")
    return obj


def clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(float(x))
    except Exception:
        return default
    return max(lo, min(hi, v))


def clamp_float(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


def normalize_reasons(reasons: Any) -> List[str]:
    if isinstance(reasons, list):
        return [str(r).strip() for r in reasons if str(r).strip()]
    if isinstance(reasons, str):
        lines = [ln.strip(" \t-•*") for ln in reasons.splitlines()]
        return [ln for ln in lines if ln]
    return []


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def is_error_output(obj: dict) -> bool:
    # Treat as "error" if reasons contains an ERROR marker or confidence is 0 with an error-ish reasons.
    reasons = obj.get("reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    if isinstance(reasons, list):
        joined = "\n".join(str(x) for x in reasons)
        if "ERROR:" in joined or "failed to parse" in joined or "maximum model length" in joined:
            return True
    # Optional: many of your error files have confidence=0.0
    # (don’t rely ONLY on this; some real predictions could be 0.0)
    return False
# ----------------------------
# Main processing
# ----------------------------

def out_path_for(paper_id: str, out_dir: Path) -> Path:
    return out_dir / f"{paper_id}.native_like.json"


def process_one_txt(
    txt_path: str,
    out_dir: str,
    generate,
    system_prompt: str,
    tok,
    max_model_len: int,
    max_chars: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    retries: int,
    overwrite: bool,
) -> Tuple[bool, Optional[str]]:

    paper_id = derive_paper_id_from_path(Path(txt_path))
    out_path = out_path_for(paper_id, Path(out_dir))

    # Skip logic:
    # - If output exists and is NOT an error -> skip (save time)
    # - If output exists and IS an error -> re-run and overwrite
    # - If output doesn't exist -> run
    if out_path.exists() and not overwrite:
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and not is_error_output(existing):
                return False, None  # skipped (already good)
            # else: it's an error output -> fall through and re-run
        except Exception:
            # If existing JSON is corrupted/unreadable, re-run and overwrite it
            pass

    text = load_txt(Path(txt_path), max_chars=max_chars)

    # ---- truncate by TOKENS to fit model limit ----
    SAFETY = 64  # leave some buffer
    header = (
        f"paper_id: {paper_id}\n"
        "Paper text (merged excerpts):\n\n"
    )

    # overhead = tokens for system+header (without the paper body)
    base_messages = [
        {"role": "system", "content": system_prompt},  # <-- use system_prompt
        {"role": "user", "content": header},
    ]
    base_ids = tok.apply_chat_template(
        base_messages, tokenize=True, add_generation_prompt=True
    )
    overhead = len(base_ids)

    # available tokens for the paper text
    available = max_model_len - overhead - max_new_tokens - SAFETY
    if available < 64:
        available = 64  # at least keep something

    text_ids = tok.encode(text, add_special_tokens=False)
    if len(text_ids) > available:
        text_ids = text_ids[:available]
        text = tok.decode(text_ids, skip_special_tokens=True) + "\n[TRUNCATED_TO_FIT_MODEL_LEN]"
    # ---------------------------------------------

    user_content = header + text

    messages = [
        {"role": "system", "content": system_prompt},  # <-- use system_prompt
        {"role": "user", "content": user_content},
    ]

    last_err: Optional[str] = None
    raw: str = ""

    for attempt in range(retries + 1):
        try:
            raw = generate(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )

            obj = extract_json_object(raw)

            out_obj: Dict[str, Any] = {}
            out_obj["paper_id"] = paper_id
            out_obj["native_like_score"] = clamp_int(obj.get("native_like_score"), 0, 10, 5)
            out_obj["confidence"] = clamp_float(obj.get("confidence"), 0.0, 1.0, 0.5)
            out_obj["verdict"] = clamp_int(obj.get("verdict"), 0, 1, 0)
            out_obj["reasons"] = normalize_reasons(obj.get("reasons"))

            if len(out_obj["reasons"]) < 3:
                while len(out_obj["reasons"]) < 3:
                    out_obj["reasons"].append("Insufficiently structured reasons returned.")
            if len(out_obj["reasons"]) > 8:
                out_obj["reasons"] = out_obj["reasons"][:8]

            write_json_atomic(out_path, out_obj)
            return True, None

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    fail = {
        "paper_id": paper_id,
        "native_like_score": 5,
        "confidence": 0.0,
        "verdict": 0,
        "reasons": [f"ERROR: failed to parse model output after retries ({last_err})"],
        "_source_txt": str(txt_path),
    }

    write_json_atomic(out_path, fail)
    return False, last_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--model_path", default="/mnt/beegfs/work/appak/hf_models/Qwen2.5-14B-Instruct-AWQ")
    ap.add_argument("--max_chars", type=int, default=40000)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backend", type=str, default="vllm", choices=["vllm", "transformers"])
    ap.add_argument("--in_txt", type=str)
    ap.add_argument("--in_dir", type=str)
    ap.add_argument("--glob", type=str, default="*.txt")
    ap.add_argument("--list_file", type=str)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)

    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    use_fast=True,
    )

    backend_name, generate = load_backend(
        model_path=args.model_path,
        dtype=args.dtype,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        prefer=args.backend,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []

    if args.in_txt:
        files = [Path(args.in_txt)]
    elif args.list_file:
        all_files = read_list_file(Path(args.list_file))
        n = len(all_files)
        start = max(0, args.start)
        end = n if args.end < 0 else min(n, args.end)
        files = all_files[start:end]
    elif args.in_dir:
        files = sorted(Path(args.in_dir).glob(args.glob))
    else:
        raise SystemExit("Provide --in_txt, or --list_file, or --in_dir.")

    total = 0
    wrote = 0
    skipped = 0
    failed = 0

    for i, p in enumerate(files, 1):
        total += 1
        did_write, err = process_one_txt(
            txt_path=str(p),
            out_dir=str(out_dir),
            generate=generate,
            system_prompt=SYSTEM_PROMPT,
            tok=tok,
            max_model_len=args.max_model_len,
            max_chars=args.max_chars,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            retries=args.retries,
            overwrite=args.overwrite,
        )

        if err is not None:
            failed += 1
        elif did_write:
            wrote += 1
        else:
            skipped += 1

    print(f"[DONE] total={total} wrote={wrote} skipped={skipped} failed={failed} out_dir={out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
