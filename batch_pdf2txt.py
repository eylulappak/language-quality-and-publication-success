import argparse
import os
import subprocess
import sys
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--converter", type=Path, required=True)
    ap.add_argument("--task_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    ap.add_argument("--chunk_size", type=int, default=100)
    ap.add_argument("--skip_existing", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()

    if not args.converter.exists():
        print(f"ERROR: converter not found: {args.converter}", file=sys.stderr)
        sys.exit(2)
    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(2)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    total = len(lines)

    start = args.task_id * args.chunk_size
    end = min(start + args.chunk_size, total)

    if start >= total:
        print(f"Task {args.task_id}: start={start} >= total={total}. Nothing to do.")
        return

    print(f"Task {args.task_id}: processing lines [{start}, {end}) out of {total}")

    failures = 0
    for i in range(start, end):
        pdf_path = Path(lines[i])
        paper_id = pdf_path.stem
        out_json = args.out_dir / f"{paper_id}.json"

        if args.skip_existing and out_json.exists() and out_json.stat().st_size > 0:
            continue

        cmd = ["python", str(args.converter), "--input", str(pdf_path), "--output", str(out_json)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f"FAILED: {pdf_path} -> {out_json} ({e})", file=sys.stderr)

    print(f"Task {args.task_id}: done. failures={failures}")

    # If you want the job to be marked failed when any failure occurs:
    # if failures > 0:
    #     sys.exit(1)

if __name__ == "__main__":
    main()
