#!/usr/bin/env python3
"""
Extract author info from a PAGE-1 plain-text file (Frontiers-friendly).

Output CSV columns:
  paper_id, author_name, email, affiliation

Key fixes for Frontiers-style pages:
- Explicitly drop EDITED BY / REVIEWED BY blocks (editors/reviewers are not authors)
- Prefer author lines near *CORRESPONDENCE and/or right before first numbered affiliation ("1Department ...")
- Parse glued names (e.g., "ElisaGatta" -> "Elisa Gatta") and "and" joins
- Affiliation taken as FULL numbered affiliation block (lines starting with digits)

Usage:
  python extract_authors_from_txt.py --in_txt page1.txt --paper_filename 280974777.pdf --out_csv out.csv --debug
"""

import argparse
import os
import ntpath
import re
from typing import List, Optional, Tuple

import pandas as pd

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

CUTOFF_RE = re.compile(
    r"^\s*(abstract|introduction|keywords?|received|accepted|published|citation|copyright)\b",
    flags=re.IGNORECASE,
)

# Strip bracket markers like [1,2], [*]
BRACKET_RE = re.compile(r"\[[^\]]*\]")
LEADING_DIGIT_RE = re.compile(r"(?m)^\s*\d+(?=\S)")
DIGIT_WORD_RE = re.compile(r"(?<=\s)\d+(?=[A-Za-z])")

# Frontiers / page metadata junk
META_LINE_RE = re.compile(
    r"^\s*(type\b|published\b|doi\b|open\s+access\b|front\.|frontiersin\.org|01\s*$)\b",
    flags=re.IGNORECASE,
)

# Headings that start non-author blocks we want to skip
EDITED_BY_RE = re.compile(r"^\s*edited\s+by\b", flags=re.IGNORECASE)
REVIEWED_BY_RE = re.compile(r"^\s*reviewed\s+by\b", flags=re.IGNORECASE)
CORRESP_RE = re.compile(r"^\s*\*?\s*correspondence\b", flags=re.IGNORECASE)

# Numbered affiliation line start: "1Department ..." or "2Brazilian ..."
AFFIL_START_RE = re.compile(r"^\s*\d+\s*[A-Za-z]", flags=re.IGNORECASE)

# Affiliation-ish keywords (extra help)
AFFIL_HINT_RE = re.compile(
    r"\b(university|universidade|federal|institute|institut|program|department|dept\.|school|"
    r"faculty|laboratory|lab|centre|center|council|academy|hospital|"
    r"graduate|research|group|college|asst|irccs|istituti|unit)\b",
    flags=re.IGNORECASE,
)


def safe_paper_id(filename: str) -> str:
    base = ntpath.basename(filename)
    if base.lower().endswith(".pdf"):
        return base[:-4]
    return os.path.splitext(base)[0]


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def preprocess(text: str) -> str:
    # remove [..] markers and digit artefacts
    text = BRACKET_RE.sub("", text)
    text = LEADING_DIGIT_RE.sub("", text)
    text = DIGIT_WORD_RE.sub("", text)
    # normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_emails(text: str) -> List[str]:
    seen = set()
    out = []
    for m in EMAIL_RE.finditer(text):
        e = m.group(0)
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def normalize_glued_name(s: str) -> str:
    """
    Turn "ElisaGatta" -> "Elisa Gatta", "VirginiaMaltese" -> "Virginia Maltese".
    Also fixes "...Bertagna andCarloCappelli" spacing.
    """
    s = re.sub(r"\band(?=[A-Z])", "and ", s)                    # "andCarlo" -> "and Carlo"
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)                 # lower->Upper boundary
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_probable_author_line(line: str) -> bool:
    # must contain letters, and at least one comma or 'and' (common in author lists)
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", line):
        return False
    if META_LINE_RE.match(line):
        return False
    # avoid plain affiliation lines
    if AFFIL_START_RE.match(line) or AFFIL_HINT_RE.search(line):
        return False
    # author list usually has commas or 'and'
    if ("," in line) or re.search(r"\band\b", line):
        return True
    return False


def split_names_from_author_blob(blob: str) -> List[str]:
    """
    Parse names from a blob like:
      "ElisaGatta, VirginiaMaltese, FrancescoDondi, ... and CarloCappelli"
    """
    blob = normalize_glued_name(blob)

    # replace " and " with comma to split uniformly
    blob = re.sub(r"\s+\band\b\s+", ", ", blob, flags=re.IGNORECASE)

    parts = [p.strip().strip(",;") for p in blob.split(",") if p.strip()]
    names = []
    for p in parts:
        p = p.strip()
        # strip non-name chars
        p = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'\- ]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        # require at least first+last
        if len(p.split()) >= 2 and not AFFIL_HINT_RE.search(p):
            names.append(p)

    # de-dup preserve order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def remove_editor_reviewer_blocks(lines: List[str]) -> List[str]:
    """
    Remove lines inside EDITED BY / REVIEWED BY sections.
    We stop removing when we hit a known next header or a likely author list.
    """
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
            # stop skipping when we hit correspondence, first author list, or first affiliation
            if CORRESP_RE.match(t) or is_probable_author_line(t) or AFFIL_START_RE.match(t):
                skip = False
            else:
                continue  # keep skipping

        out.append(t)
    return out


def find_affiliation_block(lines: List[str]) -> str:
    """
    Prefer numbered affiliations starting with digits; collect consecutive lines that look like affiliations.
    """
    start = None
    for i, ln in enumerate(lines):
        if AFFIL_START_RE.match(ln):
            start = i
            break
    if start is None:
        # fallback: first line with affiliation keywords
        for i, ln in enumerate(lines):
            if AFFIL_HINT_RE.search(ln):
                start = i
                break
    if start is None:
        return ""

    aff = [lines[start].strip()]
    for j in range(start + 1, len(lines)):
        ln = lines[j].strip()
        if not ln:
            break
        if CUTOFF_RE.match(ln):
            break
        # keep if still looks like affiliation continuation
        if AFFIL_START_RE.match(ln) or AFFIL_HINT_RE.search(ln) or "," in ln:
            aff.append(ln)
        else:
            break

    return " ".join(aff).strip()


def find_author_blob(lines: List[str]) -> str:
    """
    Priority:
      1) window around *CORRESPONDENCE
      2) lines immediately before first affiliation start
      3) first probable author line in top part
    """
    # 1) Around correspondence
    for i, ln in enumerate(lines):
        if CORRESP_RE.match(ln):
            window = lines[max(0, i - 8): min(len(lines), i + 8)]
            # pick likely author lines inside window (excluding correspondence label itself)
            cand = [w for w in window if is_probable_author_line(w)]
            if cand:
                return " ".join(cand)

    # 2) Before first affiliation
    aff_idx = None
    for i, ln in enumerate(lines):
        if AFFIL_START_RE.match(ln):
            aff_idx = i
            break
    if aff_idx is not None:
        # take up to last 3 lines before affiliations that look author-ish
        prev = lines[max(0, aff_idx - 6): aff_idx]
        cand = [w for w in prev if is_probable_author_line(w)]
        if cand:
            return " ".join(cand)

    # 3) Fallback: first probable author line
    for ln in lines:
        if is_probable_author_line(ln):
            return ln

    return ""


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
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--no_clean", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    paper_id = safe_paper_id(args.paper_filename)

    raw = load_text(args.in_txt)
    text = raw if args.no_clean else raw  # keep your file "as is" unless you want cutoff; Frontiers needs more context
    text = preprocess(text)

    # Work line-wise
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Stop at keywords/introduction only after we’ve seen affiliations/correspondence
    # (we keep more context because Frontiers has lots of metadata lines)
    lines = [ln for ln in lines if not META_LINE_RE.match(ln)]

    # Remove editor/reviewer blocks entirely
    lines = remove_editor_reviewer_blocks(lines)

    # Emails from full remaining text
    full_for_emails = "\n".join(lines)
    emails = extract_emails(full_for_emails)

    # Affiliation block (full)
    affiliation = find_affiliation_block(lines)

    # Author blob
    author_blob = find_author_blob(lines)
    author_names = split_names_from_author_blob(author_blob) if author_blob else []

    if args.debug:
        print("paper_id:", paper_id)
        print("author_blob:", author_blob)
        print("authors:", author_names)
        print("affiliation:", affiliation)
        print("emails:", emails[:10])

    rows = []
    for n in author_names:
        rows.append(
            {
                "paper_id": paper_id,
                "author_name": n,
                "email": best_email_for_name(n, emails),
                "affiliation": affiliation,
            }
        )

    if not rows:
        rows.append({"paper_id": paper_id, "author_name": "", "email": "", "affiliation": ""})

    df = pd.DataFrame(rows, columns=["paper_id", "author_name", "email", "affiliation"])
    if os.path.exists(args.out_csv):
        df.to_csv(args.out_csv, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(args.out_csv, index=False, encoding="utf-8")

    print(f"Wrote {len(df)} row(s) to {args.out_csv}")
    print(f"paper_id = {paper_id}")


if __name__ == "__main__":
    main()
