#!/usr/bin/env python3
"""CPU-only release integrity tests, no CUDA or third-party packages required."""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "models/epoch001.manifest.json").read_text(encoding="utf-8"))

class ReleaseTests(unittest.TestCase):
    def test_exact_ten_percent(self):
        self.assertEqual(SPEC["parent_virtual_addresses"], 86_060_000_000)
        self.assertEqual(SPEC["virtual_addresses"], 8_606_000_000)
        self.assertEqual(10 * SPEC["virtual_addresses"], SPEC["parent_virtual_addresses"])
        self.assertEqual(SPEC["packed_words"] * 16, SPEC["virtual_addresses"])
        self.assertEqual(SPEC["logical_size_bytes"], SPEC["packed_words"] * 4)
        self.assertEqual(SPEC["logical_size_bytes"], 2_151_500_000)

    def test_sparsity_file_is_lossless(self):
        count = 0
        with (ROOT / SPEC["snapshot"]).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                index, value = line.strip().split("\t")
                self.assertEqual(int(index), count * 65536)
                self.assertEqual(int(value, 16), 3)
                self.assertLess(int(index), SPEC["packed_words"])
                count += 1
        self.assertEqual(count, SPEC["nonzero_word_count"])
        self.assertEqual(count, 8208)

    def test_expected_full_byte_stream_hash(self):
        # Independent streaming SHA over the zero-plus-3 repeated word structure.
        # This uses <1 MiB RAM despite logically representing >2 GiB.
        pattern = bytearray(65536 * 4)
        pattern[0] = 3
        n, rest = divmod(SPEC["logical_size_bytes"], len(pattern))
        digest = hashlib.sha256()
        for _ in range(n):
            digest.update(pattern)
        digest.update(pattern[:rest])
        self.assertEqual(digest.hexdigest(), SPEC["sha256_materialized_q2"])
        self.assertEqual(digest.hexdigest(), "b6a6109b2c5a0bb304d940e04e7027f276fd1ad4422b2e0005a29055b5dddb3f")

if __name__ == "__main__":
    unittest.main(verbosity=2)
