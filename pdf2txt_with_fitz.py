#!/usr/bin/env python

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
j = pymupdf4llm.to_json(doc)
outf = args.output
if outf is None:
    outf = args.input

Path(outf).with_suffix('.json').write_bytes(j.encode())

# for i, page in enumerate(doc):
#     print('<h2>Page %d</h2>' % (i))
#     print(page.get_text("blocks"))
#     doc.close()
