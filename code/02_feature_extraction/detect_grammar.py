#!/usr/bin/env python3
"""
Detects grammar errors in paper TXT files using a T5-based GEC model (unbabel/gec-t5-small).
A correction is counted as a grammar edit only when it changes grammatically meaningful
content — not formatting, hyphenation, or citation artifacts. Supports single-file mode
and sharded batch mode.
Input:  paper TXT files (not included in submission)
Output: JSON per paper with grammar_edits count and errors_per_sentence rate
"""

import argparse
import json
from pathlib import Path
import re
import unicodedata
from difflib import SequenceMatcher

import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "unbabel/gec-t5-small"

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("[debug] torch cuda available:", torch.cuda.is_available(), flush=True)
print("[debug] cuda device count:", torch.cuda.device_count(), flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[debug] using device:", device, flush=True)


def correct_sentences(sentences, batch_size):
    prompts = ["gec: " + s for s in sentences]

    corrected_all = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        # 256 tokens matches the model's training context; longer sentences are truncated rather than rejected
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=1,  # greedy decoding for speed; beam search is not needed for an edit-detection proxy
                do_sample=False
            )

        corrected_all.extend(
            tokenizer.batch_decode(outputs, skip_special_tokens=True)
        )

    return [x.strip() for x in corrected_all]


def strip_year_citations(text: str) -> str:
    return re.sub(
        r"\([^()]*\b\d{4}\s*[a-z]?(?:\s*,\s*[a-z])?[^()]*[)\];.,:]?",
        "",
        text,
        flags=re.IGNORECASE
    )


def only_linebreak_hyphenation_change(original: str, corrected: str) -> bool:
    # The `\n not in original` guard ensures this rule only suppresses changes caused by
    # PDF line-break artifacts, not genuine hyphen edits in clean text.
    if "\n" not in original:
        return False

    o_norm = normalize_for_compare(original)
    c_norm = normalize_for_compare(corrected)
    return o_norm == c_norm


def fix_linebreak_hyphenation(text: str) -> str:
    # sig-\nnificant -> significant
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # implicit\nhypothesis -> implicit hypothesis
    text = re.sub(r"(?<=\w)\s*\n\s*(?=\w)", " ", text)

    # mi- graine -> migraine, pro- cess -> process
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", text)

    return text


def fix_suffix_duplication(text: str) -> str:
    # understandinging → understanding
    text = re.sub(r"\b(\w+?)(ing)\2\b", r"\1\2", text)

    # learneded → learned
    text = re.sub(r"\b(\w+?)(ed)\2\b", r"\1\2", text)

    # modelses → models
    text = re.sub(r"\b(\w+?)(s)\2\b", r"\1\2", text)

    return text   


def single_token_semantic_change(original: str, corrected: str) -> bool:
    o_tokens = tokens_for_lexical_check(original)
    c_tokens = tokens_for_lexical_check(corrected)

    if len(o_tokens) != len(c_tokens):
        return False

    changes = [(a, b) for a, b in zip(o_tokens, c_tokens) if a != b]

    if len(changes) != 1:
        return False

    a, b = changes[0]

    # ignore semantic substitutions (answering → solving, model → system, etc.)
    if len(a) >= 4 and len(b) >= 4:
        return True

    return False


def normalize_for_compare(text: str) -> str:
    text = fix_linebreak_hyphenation(text)
    text = unicodedata.normalize("NFKC", text)

    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
            .replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
            .replace("×", " x ")
    )

    # remove common non-grammatical symbols/markers in papers
    text = re.sub(r"[∗*†‡§¶]", " ", text)

    # remove non-ascii comparison-noise symbols like arrows, math markers, etc.
    text = text.encode("ascii", "ignore").decode("ascii")

    text = strip_year_citations(text)

    # normalize alnum splits like 2017 b -> 2017b, 6 B -> 6B, CWIG 3 G 2 -> CWIG3G2
    text = re.sub(r"(?<=\d)\s+(?=[a-zA-Z])", "", text)
    text = re.sub(r"(?<=[a-zA-Z])\s+(?=\d)", "", text)
    text = re.sub(r"(?<=[A-Z])\s+(?=[A-Z0-9])", "", text)
    text = re.sub(r"(?<=[0-9])\s+(?=[A-Z])", "", text)

    # normalize slash spacing: Y -1/ DL -1 -> Y-1/DL-1, Pubmed / Medline -> Pubmed/Medline
    text = re.sub(r"\s*/\s*", "/", text)

    # normalize hyphen spacing around alnum notation: Y -1 -> Y-1, DL -1 -> DL-1
    text = re.sub(r"(?<=[A-Za-z0-9])\s*-\s*(?=[A-Za-z0-9])", "-", text)

    # normalize punctuation / bracket / quote spacing
    text = re.sub(r"\s*([,.;:!?])\s*", r"\1", text)
    text = re.sub(r"\s*([()\[\]{}\"'])\s*", r"\1", text)

    # normalize operator spacing for math-like text
    text = re.sub(r"\s*([=+\-x])\s*", r"\1", text)

    # normalize section-number spacing: ". 3.3" -> ".3.3"
    text = re.sub(r"\.\s+(?=\d+\.)", ".", text)

    # normalize variants like e. g ., e.g ., and e. g . to e.g.,
    text = re.sub(r"\be\s*\.\s*g\s*\.\s*,?", "e.g.,", text, flags=re.IGNORECASE)

    # normalize letter-hyphen variants for comparison only:
    # co-occurrence -> cooccurrence, domain-independent -> domainindependent
    text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", text)

    text = fix_suffix_duplication(text)

    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def only_non_grammatical_change(original: str, corrected: str) -> bool:
    # Each branch catches a different class of false positives: formatting noise, citation
    # removal, math symbol variants, single-token semantic swaps, and notation differences.
    o_norm = normalize_for_compare(original)
    c_norm = normalize_for_compare(corrected)

    # exact match after normalization -> non-grammatical / formatting-only
    if o_norm == c_norm:
        return True

    if only_linebreak_hyphenation_change(original, corrected):
        return True

    if is_name_or_term_hallucination(original, corrected):
        return True

    o_no_cit = strip_year_citations(
        unicodedata.normalize("NFKC", fix_linebreak_hyphenation(original))
    )
    c_no_cit = strip_year_citations(
        unicodedata.normalize("NFKC", fix_linebreak_hyphenation(corrected))
    )
    if normalize_for_compare(o_no_cit) == normalize_for_compare(c_no_cit):
        return True

    o_math = (
        unicodedata.normalize("NFKC", fix_linebreak_hyphenation(original))
        .replace("×", " ")
        .replace("−", " ")
        .replace("–", " ")
        .replace("—", " ")
    )
    c_math = (
        unicodedata.normalize("NFKC", fix_linebreak_hyphenation(corrected))
        .replace("×", " ")
        .replace("−", " ")
        .replace("–", " ")
        .replace("—", " ")
    )
    if normalize_for_compare(o_math) == normalize_for_compare(c_math):
        return True

    if single_token_semantic_change(original, corrected):
        return True    

    # softer comparison for notation / PDF artifacts only
    # IMPORTANT: keep commas/periods/colons/semicolons because these can be grammatical
    def soft_notation_compare(text: str) -> str:
        text = normalize_for_compare(text)
        text = re.sub(r"[\s\-\[\]\(\)\{\}/\'\"]+", "", text)
        return text

    if soft_notation_compare(original) == soft_notation_compare(corrected):
        return True

    return False
    

def tokens_for_lexical_check(text: str) -> list[str]:
    text = fix_linebreak_hyphenation(text)
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("×", " ")
            .replace("−", " ")
            .replace("–", " ")
            .replace("—", " ")
    )
    return re.findall(r"[A-Za-z0-9]+", text)


def is_name_or_term_hallucination(original: str, corrected: str) -> bool:
    o_tokens = tokens_for_lexical_check(original)
    c_tokens = tokens_for_lexical_check(corrected)

    if len(o_tokens) != len(c_tokens):
        return False

    changed = [(a, b) for a, b in zip(o_tokens, c_tokens) if a != b]
    if len(changed) != 1:
        return False

    a, b = changed[0]
    sim = SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # Threshold (0.72 similarity, min length 6) catches model hallucinations of proper nouns /
    # technical terms while still allowing genuine short-word grammar corrections to pass through.
    if len(a) >= 6 and len(b) >= 6 and sim >= 0.72:
        return True

    return False


def bad_generation(original: str, corrected: str) -> bool:
    o = original.strip()
    c = corrected.strip()

    if not c:
        return True

    # 0.85 lower bound allows minor deletions; anything shorter is almost certainly a truncated generation
    if len(c) < 0.85 * len(o):
        return True

    # 1.8 upper bound catches repetition loops; the T5-small GEC model is prone to them on unusual inputs
    if len(c) > 1.8 * len(o):
        return True

    # repeated junk
    if re.search(r"\b(\w+)( \1){2,}\b", c.lower()):
        return True

    # original ended with sentence punctuation but corrected does not
    if o.endswith((".", "!", "?")) and not c.endswith((".", "!", "?")):
        return True

    # corrected ends with a suspicious unfinished connector
    if re.search(
        r"\b(and|or|but|if|that|which|who|whom|whose|when|where|while|because|although|though|since|unless|until|than|to|of|in|on|at|for|with|by|from|as|is|are|was|were|be|been|being|a|an|the|only|even|very|more|most)\s*$",
        c.lower()
    ):
        return True

    return False


def count_edits(original: str, corrected: str) -> int:
    if bad_generation(original, corrected):
        return 0
    if only_non_grammatical_change(original, corrected):
        return 0
    return 1


def limit_sentences_evenly(sentences, max_sentences):
    if max_sentences is None or len(sentences) <= max_sentences:
        return sentences
    # Uniform stride sampling rather than head-truncation keeps sentences from the full paper body
    step = len(sentences) / max_sentences
    return [sentences[int(i * step)] for i in range(max_sentences)]


def process_file(input_path: Path, output_path: Path, max_sentences=None, batch_size=64, save_details=False):
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    doc = nlp(text)

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    sentences = limit_sentences_evenly(sentences, max_sentences)

    corrected_sentences = correct_sentences(sentences, batch_size=batch_size)

    results = []
    total_edits = 0

    for original, corrected in zip(sentences, corrected_sentences):
        edits = count_edits(original, corrected)
        total_edits += edits

        if save_details:
            results.append({
                "sentence": original,
                "corrected": corrected,
                "edit": edits,
                "original_normalized": normalize_for_compare(original),
                "corrected_normalized": normalize_for_compare(corrected),
                "invalid_generation": bad_generation(original, corrected),
                "non_grammatical_change": only_non_grammatical_change(original, corrected),
                "linebreak_hyphenation_change": only_linebreak_hyphenation_change(original, corrected),
            })

    output = {
        "file": str(input_path),
        "sentences": len(sentences),
        "grammar_edits": total_edits,
        "errors_per_sentence": total_edits / max(len(sentences), 1),
    }

    if save_details:
        output["details"] = results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    # old single-file mode
    parser.add_argument("--input")
    parser.add_argument("--output")

    # new shard mode
    parser.add_argument("--input_root")
    parser.add_argument("--output_root")
    parser.add_argument("--shard_id", type=int)
    parser.add_argument("--n_shards", type=int)

    parser.add_argument("--max_sentences", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_details", action="store_true")

    args = parser.parse_args()

    # -------------------------------
    # Single-file mode
    # -------------------------------
    if args.input and args.output:
        process_file(
            Path(args.input),
            Path(args.output),
            max_sentences=args.max_sentences,
            batch_size=args.batch_size,
            save_details=args.save_details,
        )
        return

    # -------------------------------
    # Shard mode
    # -------------------------------
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    train_input = input_root / "train"
    test_input = input_root / "test"

    train_files = sorted(train_input.glob("*.txt"))
    test_files = sorted(test_input.glob("*.txt"))

    all_files = train_files + test_files
    total = len(all_files)

    print(f"[debug] total files: {total}", flush=True)
    print(f"[debug] shard: {args.shard_id} / {args.n_shards}", flush=True)
    print(f"[debug] batch size: {args.batch_size}")
    print(f"[debug] max sentences: {args.max_sentences}")

    processed = 0
    skipped = 0
    failed = 0

    for i in range(args.shard_id, total, args.n_shards):
        infile = all_files[i]

        print(f"[debug] processing {infile}", flush=True)

        if infile.parent.name == "train":
            outfile = output_root / "train" / f"{infile.stem}.json"
        else:
            outfile = output_root / "test" / f"{infile.stem}.json"

        if outfile.exists() and outfile.stat().st_size > 0:
            skipped += 1
            continue

        print(f"[debug] processing {infile}")

        try:
            process_file(
                infile,
                outfile,
                max_sentences=args.max_sentences,
                batch_size=args.batch_size,
                save_details=args.save_details,
            )
            processed += 1
        except Exception as e:
            failed += 1
            print(f"[error] failed {infile}: {e}", flush=True)

    print(f"[debug] processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()