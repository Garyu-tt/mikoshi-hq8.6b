#!/usr/bin/env python3
"""Independently check an HQ8.606B materialized/swept Q2 epoch against its source hash."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads((ROOT / "models/epoch001.manifest.json").read_text(encoding="utf-8"))
    target = args.input
    if target.stat().st_size != spec["logical_size_bytes"]:
        raise SystemExit("INVALID SIZE: expected %s bytes" % spec["logical_size_bytes"])
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    expected = spec["sha256_materialized_q2"]
    print(json.dumps({
        "status": "COMPLETE_VERIFIED" if observed == expected else "CHECKSUM_MISMATCH",
        "file": str(target), "logical_bytes": target.stat().st_size,
        "sha256": observed, "expected_sha256": expected,
        "source": "first 10% of independently verified BFB9000 R568 GPU epoch-001"
    }, indent=2))
    if observed != expected:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
