#!/usr/bin/env python
"""
Converts a single PDF to a structured JSON file using PyMuPDF (fitz/pymupdf4llm).
The JSON produced is the required input format for json2txt.py.
Usage: --input paper.pdf [--output output_dir/]
Paper PDFs are not included in the submission.
"""

import sys,fitz,argparse,json
# import pymupdf
import pymupdf.layout
import pymupdf4llm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="a pdf file", required=True)
parser.add_argument("--output", help="a directory", default=None)
args = parser.parse_args()

doc = fitz.open(args.input)
# pymupdf4llm.to_json produces structured JSON (pages → boxes → textlines) consumed by json2txt.py
j = pymupdf4llm.to_json(doc)
outf = args.output
if outf is None:
    # default to same location as the input PDF to keep JSON alongside the source file
    outf = args.input

# write as bytes so the JSON encoding from pymupdf4llm is preserved exactly
Path(outf).with_suffix('.json').write_bytes(j.encode())

# for i, page in enumerate(doc):
#     print('<h2>Page %d</h2>' % (i))
#     print(page.get_text("blocks"))
#     doc.close()
