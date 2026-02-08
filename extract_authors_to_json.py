#!/usr/bin/env python3
"""
Frontiers-friendly page-1 extraction producing TWO separate outputs:

- authors:      paper_id, author_name, email
- affiliations: paper_id, affiliation_id, affiliation_text

Key fix:
- Affiliations are split using bracket markers like [2], [3] BEFORE brackets are removed,
  so they don't collapse into a single "1..." affiliation.

Usage:
  python extract_frontiers_page1_split.py \
    --in_txt /path/page1.txt \
    --paper_filename 280974777.pdf \
    --out_json /path/out.json \
    --debug

Or CSV:
  --out_authors_csv authors.csv --out_affils_csv affils.csv
"""

import argparse
import json
import os
import ntpath
import re
from typing import List, Tuple

import pandas as pd

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

META_LINE_RE = re.compile(
    r"^\s*(type\b|published\b|doi\b|open\s+access\b|front\.|frontiersin\.org|01\s*$)\b",
    flags=re.IGNORECASE,
)

HARD_STOP_RE = re.compile(
    r"^\s*(abstract|introduction|keywords?|received|accepted|citation|copyright)\b",
    flags=re.IGNORECASE,
)

EDITED_BY_RE = re.compile(r"^\s*edited\s+by\b", flags=re.IGNORECASE)
REVIEWED_BY_RE = re.compile(r"^\s*reviewed\s+by\b", flags=re.IGNORECASE)
CORRESP_RE = re.compile(r"^\s*\*?\s*correspondence\b", flags=re.IGNORECASE)

# "1Department ..." at the beginning of a line
AFFIL_START_RE = re.compile(r"^\s*(\d+)\s*[A-Za-z]", flags=re.IGNORECASE)

# bracket markers in affiliations like [2], [3]
BRACKET_MARKER_RE = re.compile(r"\[\s*(\d+)\s*\]")

# remove bracket chunks for AUTHOR parsing only (not affiliation parsing)
BRACKET_CHUNK_RE = re.compile(r"\[[^\]]*\]")


def safe_paper_id(filename: str) -> str:
    base = ntpath.basename(filename)
    if base.lower().endswith(".pdf"):
        return base[:-4]
    return os.path.splitext(base)[0]


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def normalize_glued(s: str) -> str:
    s = re.sub(r"\band(?=[A-Z])", "and ", s)            # "andCarlo" -> "and Carlo"
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)         # ElisaGatta -> Elisa Gatta
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_common(text: str) -> str:
    # DO NOT remove bracket chunks here; we need them for affiliation splitting
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def preprocess_for_authors(text: str) -> str:
    # For author detection, remove bracket chunks like [1,2] / [*]
    text = BRACKET_CHUNK_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_emails(text: str) -> List[str]:
    found = EMAIL_RE.findall(text)
    seen = set()
    out = []
    for e in found:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def looks_like_person_name(token: str) -> bool:
    token = token.strip().strip(",;")
    if len(token.split()) < 2:
        return False
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", token):
        return False
    if re.search(r"\b(meta-analysis|systematic review|association|phenotypes)\b",
                 token, flags=re.IGNORECASE):
        return False
    return True


def looks_like_author_list_line(line: str) -> bool:
    l = normalize_glued(line)
    if HARD_STOP_RE.match(l) or META_LINE_RE.match(l):
        return False
    if AFFIL_START_RE.match(l):
        return False
    if ("," not in l) and not re.search(r"\band\b", l, flags=re.IGNORECASE):
        return False

    blob = re.sub(r"\s+\band\b\s+", ", ", l, flags=re.IGNORECASE)
    parts = [p.strip() for p in blob.split(",") if p.strip()]

    good = 0
    for p in parts:
        p = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'\- ]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if looks_like_person_name(p):
            good += 1
    return good >= 2


def remove_editor_reviewer_blocks(lines: List[str]) -> List[str]:
    out = []
    skip = False
    for ln in lines:
        t = ln.strip()
        if not t:
            continue

        if EDITED_BY_RE.match(t) or REVIEWED_BY_RE.match(t):
            skip = True
            continue

        if skip:
            if CORRESP_RE.match(t) or AFFIL_START_RE.match(t) or looks_like_author_list_line(t):
                skip = False
            else:
                continue

        out.append(t)
    return out


def extract_author_names(lines_for_authors: List[str], debug: bool = False) -> List[str]:
    # stop before affiliations start
    aff_idx = None
    for i, ln in enumerate(lines_for_authors):
        if AFFIL_START_RE.match(ln):
            aff_idx = i
            break
    search_upto = aff_idx if aff_idx is not None else len(lines_for_authors)

    author_lines = []
    for i in range(search_upto):
        ln = lines_for_authors[i]
        if HARD_STOP_RE.match(ln):
            break
        if looks_like_author_list_line(ln):
            author_lines.append(ln)

    blob = normalize_glued(" ".join(author_lines))
    blob = re.sub(r"\s+\band\b\s+", ", ", blob, flags=re.IGNORECASE)

    parts = [p.strip().strip(",;") for p in blob.split(",") if p.strip()]
    names = []
    for p in parts:
        p = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'\- ]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if looks_like_person_name(p):
            names.append(p)

    seen = set()
    out = []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)

    if debug:
        print("author_lines:", author_lines)
        print("authors:", out)

    return out


def extract_affiliations(lines_raw: List[str], debug: bool = False) -> List[Tuple[str, str]]:
    """
    Extract Frontiers affiliations robustly:
    - Start at first line beginning with '1...'
    - Concatenate until HARD_STOP
    - Then split into chunks on bracket markers [2], [3], ... (kept in raw)
    - Affiliation 1 is the text before first [2] marker.
    """
    start = None
    first_id = None
    for i, ln in enumerate(lines_raw):
        m = AFFIL_START_RE.match(ln)
        if m:
            start = i
            first_id = m.group(1)
            break
    if start is None:
        return []

    block_lines = []
    for j in range(start, len(lines_raw)):
        ln = lines_raw[j].strip()
        if not ln:
            continue
        if HARD_STOP_RE.match(ln):
            break
        block_lines.append(ln)

    block = normalize_glued(" ".join(block_lines))

    # Remove the leading "1" token if it's "1Department ..." (keep the text)
    if first_id:
        block = re.sub(rf"^\s*{re.escape(first_id)}\s+", "", block)

    # Split on [n] markers
    # re.split keeps captured group: ["text for 1", "2", "text for 2", "3", "text for 3", ...]
    parts = re.split(r"\[\s*(\d+)\s*\]", block)

    out: List[Tuple[str, str]] = []

    # parts[0] corresponds to affiliation 1 text
    aff1_text = parts[0].strip().strip(",;")
    if aff1_text:
        out.append(("1", re.sub(r"\s+", " ", aff1_text).strip()))

    # subsequent pairs are (id, text)
    idx = 1
    while idx + 1 < len(parts):
        aff_id = parts[idx].strip()
        aff_text = parts[idx + 1].strip().strip(",;")
        aff_text = re.sub(r"\s+", " ", aff_text).strip()
        if aff_id and aff_text:
            out.append((aff_id, aff_text))
        idx += 2

    if debug:
        print("affiliation_block (first 400 chars):", block[:400])
        print("affiliations parsed:", out[:10], f"(total {len(out)})")

    return out


def best_email_for_name(name: str, emails: List[str]) -> str:
    if not emails:
        return ""
    surname = name.split()[-1].lower()
    for e in emails:
        if surname and surname in e.split("@", 1)[0].lower():
            return e
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True)
    ap.add_argument("--paper_filename", required=True)
    ap.add_argument("--out_authors_csv", default=None)
    ap.add_argument("--out_affils_csv", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    paper_id = safe_paper_id(args.paper_filename)

    raw = load_text(args.in_txt)
    raw_common = preprocess_common(raw)

    # Lines for affiliation extraction MUST preserve bracket markers
    lines_raw = [ln.strip() for ln in raw_common.splitlines() if ln.strip()]
    lines_raw = [ln for ln in lines_raw if not META_LINE_RE.match(ln)]
    lines_raw = remove_editor_reviewer_blocks(lines_raw)

    # Lines for author extraction: can strip bracket chunks safely
    authors_text = preprocess_for_authors(raw_common)
    lines_auth = [ln.strip() for ln in authors_text.splitlines() if ln.strip()]
    lines_auth = [ln for ln in lines_auth if not META_LINE_RE.match(ln)]
    lines_auth = remove_editor_reviewer_blocks(lines_auth)

    # Emails from raw (with bracket expansions already done upstream in your txt creation; otherwise add expansion here)
    full_text_for_emails = "\n".join(lines_raw)
    emails = extract_emails(full_text_for_emails)

    authors = extract_author_names(lines_auth, debug=args.debug)
    author_rows = [{"paper_id": paper_id, "author_name": a, "email": best_email_for_name(a, emails)} for a in authors]
    if not author_rows:
        author_rows = [{"paper_id": paper_id, "author_name": "", "email": ""}]

    affils = extract_affiliations(lines_raw, debug=args.debug)
    affil_rows = [{"paper_id": paper_id, "affiliation_id": aff_id, "affiliation_text": aff_text} for aff_id, aff_text in affils]
    if not affil_rows:
        affil_rows = [{"paper_id": paper_id, "affiliation_id": "", "affiliation_text": ""}]

    if args.out_json:
        payload = {"paper_id": paper_id, "authors": author_rows, "affiliations": affil_rows}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON: {args.out_json}")
        return

    if not args.out_authors_csv or not args.out_affils_csv:
        raise SystemExit("Provide --out_authors_csv and --out_affils_csv OR --out_json")

    pd.DataFrame(author_rows, columns=["paper_id", "author_name", "email"]).to_csv(args.out_authors_csv, index=False)
    pd.DataFrame(affil_rows, columns=["paper_id", "affiliation_id", "affiliation_text"]).to_csv(args.out_affils_csv, index=False)

    print(f"Wrote authors CSV: {args.out_authors_csv}")
    print(f"Wrote affiliations CSV: {args.out_affils_csv}")
    print(f"paper_id = {paper_id}")


if __name__ == "__main__":
    main()
