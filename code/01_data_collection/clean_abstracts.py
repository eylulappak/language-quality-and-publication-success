#!/usr/bin/env python3
"""
Cleans and deduplicates arXiv abstracts from the Semantic Scholar metadata dump.
Applies text normalization (HTML entities, LaTeX, encoding artifacts), filters
abstracts that are too short/long, placeholder text, boilerplate, or symbol-heavy,
then removes exact and near-duplicate abstracts.
Input:  data/90k_arxiv_metadata_from_semantic_scholar.csv
Output: cleaned CSV + audit CSV of removed rows.
"""
import re
import html
import hashlib
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

IN_PATH = Path("data/90k_arxiv_metadata_from_semantic_scholar.csv")
OUT_PATH = Path("data/90k_arxiv_metadata_from_semantic_scholar_cleaned.csv")
REMOVED_PATH = Path("data/90k_arxiv_metadata_from_semantic_scholar_removed.csv")

MIN_CHARS = 100
MIN_WORDS = 20
MIN_ALPHA_CHARS = 30
MIN_ALPHA_RATIO = 0.50
MAX_PUNCT_RATIO = 0.30
# MAX_CHARS / MAX_WORDS are set high enough to catch full-text papers accidentally stored as abstracts
MAX_CHARS = 5000
MAX_WORDS = 3000
# 0.20 is intentionally lenient: highly technical abstracts with many repeated symbols can still pass
MIN_LEXICAL_DIVERSITY = 0.20
# 0.95 rather than 1.0 so near-identical abstracts with minor whitespace edits are still caught
NEAR_DUP_THRESHOLD = 0.95

PLACEHOLDERS = {
    "no abstract available",
    "n/a",
    "na",
    "none",
    "not provided",
    "abstract unavailable",
    "no abstract",
    "null",
}

BOILERPLATE_PATTERNS = [
    r"copyright\s*©?",
    r"all rights reserved",
    r"published by",
    r"publisher['’]s note",
    r"springer nature remains neutral",
    r"elsevier",
]

REFERENCE_PATTERNS = [
    r"^\s*\[\d+\]",
    r"\bdoi\s*:",
    r"\breferences\b",
    r"\bbibliography\b",
]

def clean_text(x):
    if pd.isna(x):
        return ""

    x = str(x)

    # Decode HTML entities: &amp; -> &, &lt; -> <
    x = html.unescape(x)

    # Remove common broken encoding artifacts
    x = x.replace("ï»¿", " ").replace("Ã©", "é").replace("Ã¨", "è")
    x = x.replace("Ã¶", "ö").replace("Ã¼", "ü").replace("Ã¤", "ä")
    # The first two replace targets look identical but are distinct mojibake: one decodes to en-dash (U+2013), the other to em-dash (U+2014).
    x = x.replace("â€“", "-").replace("â€”", "-").replace("â€˜", "'").replace("â€™", "'")
    x = x.replace("â€œ", '"').replace("â€", '"')

    # Remove XML/HTML/JATS/MathML tags
    x = re.sub(r"<[^>]+>", " ", x)

    # Remove LaTeX commands but keep possible text around them
    x = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^{}]*\})?", " ", x)

    # Remove line breaks and unusual separators
    x = x.replace("\u2028", " ").replace("\u2029", " ")
    x = x.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Minimum 10 chars inside $…$ to avoid stripping single-letter math variables like $x$
    x = re.sub(r"\$[^$]{10,}\$", " ", x)
    x = re.sub(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", " ", x, flags=re.DOTALL)

    # Collapse whitespace
    x = " ".join(x.split())

    return x.strip()

def word_list(text):
    return re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text.lower())

def get_stats(text):
    total_chars = len(text)
    alpha_chars = sum(c.isalpha() for c in text)
    punct_chars = sum((not c.isalnum() and not c.isspace()) for c in text)
    words = word_list(text)
    total_words = len(words)
    unique_words = len(set(words))

    return {
        "clean_abstract_length": total_chars,
        "word_count": total_words,
        "alpha_chars": alpha_chars,
        "alpha_ratio": alpha_chars / total_chars if total_chars else 0,
        "punct_ratio": punct_chars / total_chars if total_chars else 0,
        "lexical_diversity": unique_words / total_words if total_words else 0,
        "period_count": text.count("."),
        "symbol_ratio": sum((not c.isalnum() and not c.isspace() and c not in ".,;:!?-'\"()") for c in text) / total_chars if total_chars else 0,
    }

def has_verb_like_word(text):
    words = word_list(text)
    common_verbs = {
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did",
        "show", "shows", "shown", "study", "studies", "studied",
        "present", "presents", "presented", "propose", "proposes", "proposed",
        "find", "finds", "found", "demonstrate", "demonstrates", "demonstrated",
        "analyze", "analyzes", "analysed", "analyzed", "investigate", "investigates",
        "use", "uses", "used", "provide", "provides", "provided",
    }

    if any(w in common_verbs for w in words):
        return True

    # crude English verb suffix heuristic
    if any(w.endswith(("ed", "ing", "izes", "ises")) for w in words if len(w) > 5):
        return True

    return False

def removal_reason(text, stats):
    low = text.lower().strip()

    if low in PLACEHOLDERS:
        return "placeholder_abstract"

    if stats["clean_abstract_length"] < MIN_CHARS:
        return "too_short_chars"

    if stats["word_count"] < MIN_WORDS:
        return "too_few_words"

    if stats["alpha_chars"] < MIN_ALPHA_CHARS:
        return "too_few_alphabetic_chars"

    if stats["alpha_ratio"] <= MIN_ALPHA_RATIO:
        return "low_alphabetic_ratio"

    if stats["punct_ratio"] >= MAX_PUNCT_RATIO:
        return "high_punctuation_ratio"

    if stats["clean_abstract_length"] > MAX_CHARS:
        return "too_long_chars_possible_full_text"

    if stats["word_count"] > MAX_WORDS:
        return "too_long_words_possible_full_text"

    if stats["lexical_diversity"] < MIN_LEXICAL_DIVERSITY:
        return "low_lexical_diversity"

    # 0.15 symbol ratio is stricter than the punct_ratio check because it targets non-standard chars specifically
    if stats["symbol_ratio"] > 0.15:
        return "formula_or_symbol_heavy"

    if stats["period_count"] == 0:
        return "no_period"

    if not has_verb_like_word(text):
        return "no_verb_like_word"

    if any(re.search(p, low) for p in BOILERPLATE_PATTERNS):
        return "publisher_boilerplate"

    if any(re.search(p, low) for p in REFERENCE_PATTERNS):
        return "reference_like_content"

    # too much remaining markup after cleaning
    if low.count("<") + low.count(">") >= 3:
        return "remaining_markup_corruption"

    # single-letter token count > 30 indicates the abstract collapsed into a symbol soup from math-heavy papers
    if len(re.findall(r"\b[a-zA-Z]\b", text)) > 30:
        return "many_single_letter_formula_fragments"

    return ""

def normalize_for_duplicate(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text

def md5_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

print("Reading CSV...")
df = pd.read_csv(IN_PATH, dtype={"arxiv_id": str, "file_path": str}, low_memory=False)
print("Input rows:", len(df))

if "abstract" not in df.columns:
    raise ValueError("No 'abstract' column found.")

print("Cleaning abstract text...")
df["abstract_original"] = df["abstract"]
df["abstract"] = df["abstract"].apply(clean_text)

stats_df = df["abstract"].apply(get_stats).apply(pd.Series)
df = pd.concat([df, stats_df], axis=1)

print("Applying quality filters...")
df["removal_reason"] = df.apply(lambda r: removal_reason(r["abstract"], {
    "clean_abstract_length": r["clean_abstract_length"],
    "word_count": r["word_count"],
    "alpha_chars": r["alpha_chars"],
    "alpha_ratio": r["alpha_ratio"],
    "punct_ratio": r["punct_ratio"],
    "lexical_diversity": r["lexical_diversity"],
    "period_count": r["period_count"],
    "symbol_ratio": r["symbol_ratio"],
}), axis=1)

removed_basic = df[df["removal_reason"] != ""].copy()
kept = df[df["removal_reason"] == ""].copy()

print("After basic filters:", len(kept))
print("Removed by basic filters:", len(removed_basic))

print("Removing exact duplicate abstracts...")
kept["abstract_norm"] = kept["abstract"].apply(normalize_for_duplicate)

exact_dup_mask = kept.duplicated(subset=["abstract_norm"], keep="first")
removed_exact = kept[exact_dup_mask].copy()
removed_exact["removal_reason"] = "exact_duplicate_abstract"
kept = kept[~exact_dup_mask].copy()

print("After exact duplicate removal:", len(kept))
print("Removed exact duplicates:", len(removed_exact))

print("Removing near-duplicate abstracts...")
kept = kept.reset_index(drop=True)
# 120-char prefix buckets candidates so SequenceMatcher only runs within groups that share the same prefix, avoiding O(n^2) all-pairs comparisons.
kept["near_dup_key"] = kept["abstract_norm"].str[:120]

seen_by_key = {}
keep_mask = []
near_dup_reasons = []

for idx, row in kept.iterrows():
    text = row["abstract_norm"]
    key = row["near_dup_key"]

    is_dup = False

    candidates = seen_by_key.get(key, [])
    for prev_text in candidates:
        if SequenceMatcher(None, text, prev_text).ratio() >= NEAR_DUP_THRESHOLD:
            is_dup = True
            break

    if is_dup:
        keep_mask.append(False)
        near_dup_reasons.append("near_duplicate_abstract")
    else:
        keep_mask.append(True)
        near_dup_reasons.append("")
        seen_by_key.setdefault(key, []).append(text)

kept["near_dup_removal_reason"] = near_dup_reasons

removed_near = kept[~pd.Series(keep_mask)].copy()
removed_near["removal_reason"] = "near_duplicate_abstract"

kept = kept[pd.Series(keep_mask)].copy()

print("After near duplicate removal:", len(kept))
print("Removed near duplicates:", len(removed_near))

# Recalculate final abstract fields
kept["abstract_length"] = kept["abstract"].str.len()
kept["abstract_md5_hash"] = kept["abstract"].apply(md5_text)

# Drop helper columns from final clean dataset
helper_cols = [
    "abstract_original",
    "clean_abstract_length",
    "word_count",
    "alpha_chars",
    "alpha_ratio",
    "punct_ratio",
    "lexical_diversity",
    "period_count",
    "symbol_ratio",
    "removal_reason",
    "abstract_norm",
    "near_dup_key",
    "near_dup_removal_reason",
]

final = kept.drop(columns=[c for c in helper_cols if c in kept.columns])

removed = pd.concat([removed_basic, removed_exact, removed_near], ignore_index=True)

print("Writing outputs...")
final.to_csv(OUT_PATH, index=False)
removed.to_csv(REMOVED_PATH, index=False)

print("=" * 80)
print("Input rows:", len(df))
print("Final cleaned rows:", len(final))
print("Total removed:", len(removed))
print("Cleaned output:", OUT_PATH)
print("Removed rows output:", REMOVED_PATH)
print()
print("Removal counts:")
print(removed["removal_reason"].value_counts(dropna=False).to_string())