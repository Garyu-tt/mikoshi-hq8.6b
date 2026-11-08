# Mikoshi HQ8.6B: GPU neural fabric (10% release)

**A reproducible, 8,606,000,000-address synthetic neural-state experiment.** This is **exactly one tenth** of the 86,060,000,000-address BFB9000 R568 GPU research run, not an unrelated toy with a similar name.

> **Research honesty:** This is a **synthetic 2-bit quantized proxy**, **not** a reconstructed human connectome, **not** 8.6 billion anatomically observed neurons or synapses, and **not** an AI language model or a demonstrated conscious system. Each address receives a GPU update during a sweep, but most states remain zero. The coupling is an **artificial directed nearest-neighbour ring** with one synthetic incoming dependency per address.

## What's in this repository

- `gpu/hq86b_gpu.py`: NVIDIA CUDA Driver API + embedded PTX, streaming tiled update, CPU-vs-GPU self-test, restartable sweep, and sparse on-disk output. No PyTorch or CUDA Toolkit needed.
- `models/epoch001.sparse.tsv`: **the actual non-zero 32-bit words** from the first 10% of the verified full-86.06B R568 epoch-001 GPU result. Every unlisted word is zero. This is a **lossless sparse encoding** of the released epoch, not an illustrative sample.
- `models/epoch001.manifest.json`: precise census, representation, original source provenance and hashes.
- `tools/materialize_q2.py`: reconstruct the exact 2,151,500,000-byte logical Q2 state file from the sparse release.
- `tools/verify_q2.py`: verify a Q2 file against its expected whole-file checksum and sparse encoding.
- `tests/test_release.py`: CPU-only deterministic release tests.
- `validation/rtx3050-laptop-wsl2-2026-11-08.json`: completed 8.606B GPU sweep and independent SHA-256 validation.

## Quick start

Clone the repository, then use Python 3.10+:

```bash
# Examine the included epoch without a GPU
python3 tests/test_release.py

# Reconstruct the EXACT released 8.606B-address state on your local disk
# Logical size 2,151,500,000 bytes; actual disk allocation is much lower on sparse-file systems
python3 tools/materialize_q2.py --output ./runs/epoch001-released.q2
python3 tools/verify_q2.py --input ./runs/epoch001-released.q2

# If you have a compatible NVIDIA GPU and working CUDA driver:
python3 gpu/hq86b_gpu.py --selftest
python3 gpu/hq86b_gpu.py --sweep --root ./runs/new-sweep
python3 tools/verify_q2.py --input ./runs/new-sweep/epoch001-generated.q2
```

**The GPU run generates a fresh epoch from zero state** (it does not fine-tune the included snapshot). It refuses to overwrite a committed result. For a new run, use a new `--root` directory. A different `--chunk-words` (1,024 to 33,554,432) trades VRAM usage against transfer overhead; default is 16,777,216 packed words (64 MiB per tile, with separate input/output buffers).

## Hardware and verified source

The **parent 86.06B-address R568** experiment completed and was independently hash-verified on **NVIDIA RTX 3050 Laptop GPU, 4 GiB VRAM, compute capability 8.6, Ubuntu under WSL2**, on **2026-11-08**. It processed 321 GPU tiles; full logical state was 21,515,000,000 bytes, not resident in 4 GiB VRAM. The 8.606B release is the **first contiguous 10% of its verified epoch-001 output**, independently hashed against the original file. The **published 8.606B GPU code was then run independently on the same RTX 3050 Laptop GPU** on **2026-11-08**: **33 tiles, 43.8 seconds** for the GPU+host streaming sweep, followed by **COMPLETE_VERIFIED** against the released whole-file SHA-256. See [the reproducible validation record](validation/rtx3050-laptop-wsl2-2026-11-08.json). This is an end-to-end sweep time, **not a kernel-only benchmark**; no other GPU architectures are claimed tested.

This exact PTX build targets **sm_86**. An NVIDIA CUDA driver and a compatible GPU architecture are required for the GPU path; non-NVIDIA cards are **not supported** by this backend. CPU scripts can inspect and reconstruct the published snapshot but do not offer practical 8.6B-state recurrent GPU performance.

## Exact size and integrity

| Property | Value |
| --- | ---: |
| Released virtual addresses | **8,606,000,000** |
| Portion of 86.06B parent | **10.000%** |
| State encoding | **2 bits/address, 16 addresses/32-bit word, little endian** |
| Logical packed Q2 bytes | **2,151,500,000** |
| Explicit non-zero words in epoch 001 | **8,208** |
| Other packed words | **all zero** |
| Whole-file reconstructed Q2 SHA-256 | `b6a6109b2c5a0bb304d940e04e7027f276fd1ad4422b2e0005a29055b5dddb3f` |

The sparse TSV plus manifest is lossless: the materialized byte stream hashes exactly to the **prefix of the verified 86.06B GPU output**. The committed sparse representation is intentionally compact: we do not put a 2.15 GB mostly-zero file into Git, and the TSV is not a substitute for unobserved biological wiring.

## Provenance and limitations

Parent: BFB9000 R568 full-address GPU synthetic proxy, 2026-11-08. Original reference run processed 86,060,000,000 state transitions with a deterministic one-neighbour update and sparse stimulation. Its 86.06B Q2 SHA-256 is `dfa422e9fb75a8033b9f3f1db5daa934b97ba277328e687b9c3bd432ed879eb0`. The richer BFB9000 R167/R168 409,600-active-synthetic-neuron recurrent CUDA model is **different** from this large-scale Q2 proxy.

**No training corpus, biological neurite graph, learned weights, measured human connectivity, or full GPU-resident 8.6B tensor is claimed or bundled.** Treat this as a scalable reproducible systems/research artifact.

License: MIT. Contributions and independent GPU reports welcome, especially with hardware, driver version, commands, checksums and observed wall-time.
