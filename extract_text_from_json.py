#!/usr/bin/env python3
"""
Extract ALL plain text (one line per textline) from PAGE 1
of a parsed-paper JSON and write it to a .txt file.

- No filtering by boxclass (keeps title, headers, authors, etc.)
- Sorts lines by visual position (y, x)
- Writes one reconstructed line per textline

Usage:
  python page1_json_to_txt_all.py \
    --in_json /path/paper.json \
    --out_txt /path/page1.txt
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def find_page1(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the page dict for page_number==1, else fallback to first page."""
    pages = doc.get("pages", [])
    if not pages:
        return None

    for p in pages:
        if p.get("page_number") == 1:
            return p

    return pages[0]


def reconstruct_line(textline: Dict[str, Any]) -> str:
    """Join all spans inside a textline into one string."""
    spans = textline.get("spans", []) or []
    return "".join((s.get("text") or "") for s in spans).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Input parsed-paper JSON")
    ap.add_argument("--out_txt", required=True, help="Output .txt file")
    ap.add_argument(
        "--drop_empty",
        action="store_true",
        help="Skip empty reconstructed lines",
    )
    args = ap.parse_args()

    # Load JSON
    with open(args.in_json, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Get page 1
    page1 = find_page1(doc)
    if page1 is None:
        raise SystemExit("ERROR: No pages found in JSON.")

    collected: List[Tuple[float, float, str]] = []

    # Loop through ALL boxes and ALL textlines
    for box in page1.get("boxes", []) or []:
        for tl in box.get("textlines", []) or []:

            bbox = tl.get("bbox") or [0, 0, 0, 0]
            y0 = float(bbox[1]) if len(bbox) > 1 else 0.0
            x0 = float(bbox[0]) if len(bbox) > 0 else 0.0

            line = reconstruct_line(tl)

            if args.drop_empty and not line:
                continue

            collected.append((y0, x0, line))

    # Sort lines in reading order: top-to-bottom, left-to-right
    collected.sort(key=lambda t: (t[0], t[1]))

    # Ensure output folder exists
    out_dir = os.path.dirname(os.path.abspath(args.out_txt))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write output
    with open(args.out_txt, "w", encoding="utf-8", newline="\n") as f:
        for _, _, line in collected:
            f.write(line + "\n")

    print(f"Wrote {len(collected)} lines to: {args.out_txt}")


if __name__ == "__main__":
    main()
