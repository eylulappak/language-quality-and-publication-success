#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

KEEP_CLASSES = {"section-header", "text"}

REFERENCES_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"(?:\d+(?:\.\d+)*)?\s*"
    r"(?:references|bibliography|works\s+cited|literature\s+cited|reference\s+list)"
    r")\s*$",
    re.IGNORECASE,
)

# "recogni- tion" -> "recognition"
INTRA_LINE_HYPHEN_RE = re.compile(r"([A-Za-z])-\s+([A-Za-z])")

# "attrac-" at end of line
LINE_END_HYPHEN_RE = re.compile(r"^(.*\b[A-Za-z]+)-\s*$")

# "tive example" at start of next line
LINE_START_FRAGMENT_RE = re.compile(r"^([A-Za-z]+)(.*)$")


def is_references_header(line: str) -> bool:
    return bool(REFERENCES_HEADER_RE.match(line.strip()))


def normalize_line_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("­", "")
    text = re.sub(r"\s+", " ", text).strip()

    while True:
        new_text = INTRA_LINE_HYPHEN_RE.sub(r"\1\2", text)
        if new_text == text:
            break
        text = new_text

    return text


def repair_hyphenated_lines(items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    repaired: List[Tuple[str, str]] = []
    i = 0

    while i < len(items):
        current_text, current_class = items[i]
        current_text = normalize_line_text(current_text)

        while i + 1 < len(items):
            next_text, next_class = items[i + 1]
            next_text = normalize_line_text(next_text)

            # only merge split words across lines when both are normal text
            if current_class != "text" or next_class != "text":
                break

            m_end = LINE_END_HYPHEN_RE.match(current_text)
            if not m_end:
                break

            m_start = LINE_START_FRAGMENT_RE.match(next_text)
            if not m_start:
                break

            fragment = m_start.group(1)
            rest = m_start.group(2)

            if not fragment or not fragment[0].islower():
                break

            current_text = m_end.group(1) + fragment + rest
            i += 1

        repaired.append((current_text, current_class))
        i += 1

    return repaired


def join_lines_within_paragraphs(items: List[Tuple[str, str]]) -> List[str]:
    output: List[str] = []
    current_parts: List[str] = []

    def flush_paragraph():
        nonlocal current_parts
        if current_parts:
            output.append(" ".join(current_parts))
            current_parts = []

    for text, boxclass in items:
        text = text.strip()
        if not text:
            flush_paragraph()
            continue

        # Keep section-headers on their own lines
        if boxclass == "section-header":
            flush_paragraph()
            output.append(text)
            continue

        # Join consecutive text lines into one paragraph
        if boxclass == "text":
            current_parts.append(text)
        else:
            flush_paragraph()
            output.append(text)

    flush_paragraph()
    return output


def extract_lines(json_path: Path) -> List[str]:
    items: List[Tuple[str, str]] = []
    ignore_mode = False

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])

    for page in pages:
        boxes = page.get("boxes", [])

        for box in boxes:
            boxclass = box.get("boxclass")

            if boxclass not in KEEP_CLASSES:
                continue

            textlines = box.get("textlines", [])

            for tl in textlines:
                spans = tl.get("spans", [])

                span_texts = []
                for span in spans:
                    txt = span.get("text", "").strip()
                    if txt:
                        span_texts.append(txt)

                if not span_texts:
                    continue

                line = normalize_line_text(" ".join(span_texts))

                if is_references_header(line):
                    ignore_mode = True
                    break

                if not ignore_mode:
                    items.append((line, boxclass))

            if ignore_mode:
                break

        if ignore_mode:
            break

    items = repair_hyphenated_lines(items)
    lines = join_lines_within_paragraphs(items)

    return lines


def process_one(in_path: Path, out_dir: Path, overwrite: bool = False) -> bool:
    out_path = out_dir / (in_path.stem + ".txt")

    if out_path.exists() and not overwrite:
        return False

    lines = extract_lines(in_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = out_path.with_suffix(".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    tmp.replace(out_path)
    return True


def read_list_file(path: Path) -> List[Path]:
    files = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                files.append(Path(p))
    return files


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--in_json")
    ap.add_argument("--in_dir")
    ap.add_argument("--glob", default="*.json")

    ap.add_argument("--list_file")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.in_json:
        wrote = process_one(Path(args.in_json), out_dir, args.overwrite)
        print(f"[single] {'wrote' if wrote else 'skip'} {args.in_json}")
        return

    if args.list_file:
        files = read_list_file(Path(args.list_file))
        n = len(files)

        start = max(0, args.start)
        end = n if args.end < 0 else min(n, args.end)

        wrote = skip = err = 0

        for i in range(start, end):
            p = files[i]

            try:
                did = process_one(p, out_dir, args.overwrite)
                if did:
                    wrote += 1
                else:
                    skip += 1
            except Exception as e:
                err += 1
                print(f"[ERROR] {p}: {e}")

            if (i - start + 1) % 500 == 0:
                print(f"[progress] {i-start+1}/{end-start}")

        print(f"[done] wrote={wrote} skip={skip} err={err}")
        return

    if args.in_dir:
        files = sorted(Path(args.in_dir).glob(args.glob))

        wrote = skip = 0

        for i, p in enumerate(files, 1):
            did = process_one(p, out_dir, args.overwrite)
            if did:
                wrote += 1
            else:
                skip += 1

            if i % 1000 == 0:
                print(f"[progress] {i}/{len(files)}")

        print(f"[done] total={len(files)} wrote={wrote} skip={skip}")
        return

    raise SystemExit("Provide --in_json, --list_file, or --in_dir.")


if __name__ == "__main__":
    main()