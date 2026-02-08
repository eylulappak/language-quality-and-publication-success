from pathlib import Path

PDF_DIR = Path("/ukp-storage-1/appak/acl_paper_pdfs")
manifest = Path("/ukp-storage-1/appak/acl_pdf_manifest.txt")

pdfs = sorted(PDF_DIR.glob("*.pdf"))
manifest.parent.mkdir(parents=True, exist_ok=True)

with manifest.open("w", encoding="utf-8") as f:
    for p in pdfs:
        f.write(str(p) + "\n")

print(f"Wrote {len(pdfs)} paths to {manifest}")
