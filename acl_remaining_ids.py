#!/usr/bin/env python3
"""
Extract ACL Anthology paper IDs from a BibTeX file and identify which PDFs are missing.

Input:
  - /ukp-storage-1/appak/acl_papers/anthology.bib

Outputs (in the same directory):
  - paper_ids_all.txt        : all unique extracted IDs
  - paper_ids_missing.txt    : IDs whose <ID>.pdf is not present in OUTDIR
"""

from __future__ import annotations

import re
from pathlib import Path


BIB_PATH = Path("/ukp-storage-1/appak/acl_papers/anthology.bib")
OUTDIR = Path("/ukp-storage-1/appak/acl_papers")

ALL_IDS_TXT = OUTDIR / "paper_ids_all.txt"
MISSING_IDS_TXT = OUTDIR / "paper_ids_missing.txt"

# Matches:
#   https://aclanthology.org/2025.yrrsds-1.0/
#   https://aclanthology.org/2025.yrrsds-1.0
#   https://aclanthology.org/2025.yrrsds-1.0.pdf   (we'll normalize to remove .pdf)
ACL_URL_RE = re.compile(r"https?://aclanthology\.org/([^/\s\"}]+)")


def extract_ids_from_bib(bib_text: str) -> list[str]:
    ids: list[str] = []
    for m in ACL_URL_RE.finditer(bib_text):
        raw = m.group(1).strip()
        # Normalize:
        # - remove trailing punctuation that sometimes sticks to URLs in BibTeX
        raw = raw.rstrip(".,;")
        # - strip a possible .pdf suffix
        if raw.endswith(".pdf"):
            raw = raw[:-4]
        # - strip a possible trailing slash (in case regex captured it—usually it won't)
        raw = raw.rstrip("/")
        if raw:
            ids.append(raw)
    return ids


def main() -> None:
    if not BIB_PATH.exists():
        raise FileNotFoundError(f"BibTeX file not found: {BIB_PATH}")

    if not OUTDIR.exists():
        OUTDIR.mkdir(parents=True, exist_ok=True)

    bib_text = BIB_PATH.read_text(encoding="utf-8", errors="replace")

    ids = extract_ids_from_bib(bib_text)
    unique_ids = sorted(set(ids))

    # Save all IDs
    ALL_IDS_TXT.write_text("\n".join(unique_ids) + ("\n" if unique_ids else ""), encoding="utf-8")
    print(f"[OK] Extracted {len(unique_ids)} unique ACL IDs")
    print(f"     Wrote: {ALL_IDS_TXT}")

    # Determine which PDFs exist (match by <ID>.pdf)
    missing: list[str] = []
    for acl_id in unique_ids:
        pdf_path = OUTDIR / f"{acl_id}.pdf"
        if not pdf_path.exists():
            missing.append(acl_id)

    # Save missing IDs
    MISSING_IDS_TXT.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    print(f"[OK] Missing PDFs: {len(missing)}")
    print(f"     Wrote: {MISSING_IDS_TXT}")

    # Optional: quick hint
    if missing:
        print("\nTip: You can download missing PDFs with:")
        print(f"  cat {MISSING_IDS_TXT} | xargs -n 1 -P 8 bash -lc 'curl -L -C - --fail --retry 5 --retry-delay 2 -o \"{OUTDIR}/$1.pdf\" \"https://aclanthology.org/$1.pdf\"' _")


if __name__ == "__main__":
    main()

