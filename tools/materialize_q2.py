#!/usr/bin/env python3
"""Materialize the exact published HQ8.606B sparse 2-bit state without a GPU."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/epoch001.manifest.json"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path, help="New output Q2 file; will NOT overwrite")
    args = ap.parse_args()
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = ROOT / spec["snapshot"]
    logical_bytes = int(spec["logical_size_bytes"])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with output.open("xb") as dst:
            dst.truncate(logical_bytes)
            with rows.open("r", encoding="utf-8") as src:
                for line in src:
                    if not line.strip() or line.startswith("#"):
                        continue
                    index_str, value_str = line.rstrip("\n").split("\t")
                    index, value = int(index_str), int(value_str, 16)
                    if index != written * 65536 or index >= int(spec["packed_words"]) or value != 3:
                        raise ValueError(f"unexpected sparse word at row {written}: {line.strip()}")
                    dst.seek(index * 4)
                    dst.write(value.to_bytes(4, "little"))
                    written += 1
            if written != int(spec["nonzero_word_count"]):
                raise ValueError(f"expected {spec['nonzero_word_count']} entries; got {written}")
            dst.flush()
            os.fsync(dst.fileno())
    except Exception:
        # A partial/corrupt output is never accepted as a release snapshot.
        output.unlink(missing_ok=True)
        raise
    st = output.stat()
    print(json.dumps({
        "status": "MATERIALIZED", "file": str(output),
        "logical_bytes": st.st_size, "nonzero_words": written,
        "allocated_bytes_linux": getattr(st, "st_blocks", 0) * 512,
        "expected_sha256": spec["sha256_materialized_q2"],
        "next": "python3 tools/verify_q2.py --input " + str(output)
    }, indent=2))

if __name__ == "__main__":
    main()
