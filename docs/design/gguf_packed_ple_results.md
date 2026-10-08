# Packed GGUF PLE results

When a complete PLE table cannot fit the fair pinned-host budget, the CPU
offloader gathers requested GGUF rows and normally dequantizes them before
publishing FP16 embeddings. The GPU then waits for the CPU result inside its
graph. IQ4_NL rows can cross this boundary in their original packed form.

## Data flow and numerical contract

The loader declares the source format, head count, row width and output dtype.
An eligible CPU-owned IQ4_NL table uses byte results; local pinned and cascade
table paths retain their existing behavior. Other formats, dtypes and devices
report an admission reason and use the existing result path. The policy is
`KernelConfig.ple_packed_gguf_results` and is enabled by default.

The CPU computes the same n-gram IDs, copies complete packed rows in request
order, and publishes the existing release flag. The consumer stream waits,
copies the small packet and reconstructs `FP32(scale) * FP32(codebook[code])`
before the final FP16 conversion. No quantization code or coefficient changes.
Finite and FP16-overflow checks remain on the CPU; uncommon large scales use
the official decoder for the value-dependent overflow check.

For 16 heads of width 160, each token transfers 1,440 packed bytes instead of
5,120 FP16 bytes. Table placement and ownership remain unchanged. No complete
table copy, second weight representation, private DSO or environment variable
is introduced. Output buffers retain fixed graph addresses, and consumers
acknowledge them only after decoding and using the result. CUDA and mapped
transports share the same negotiated result shape.

The CPU process is spawned from a configuration snapshot taken before GPU
weight loading. Result geometry therefore travels in each GPU registration,
after loading resolves capabilities. The CPU checks agreement across all
DP/TP consumers and validates the descriptor against its actual loaded row
type and dimensions before validating or allocating transport buffers. No
live model or loader closure crosses the spawn boundary.

## Measurement

The initial four-V100 control uses TP4, device E4M3 target history, FP16 draft
history, MTP4, FULL graphs, CUDA 12.8 and Torch 2.10.0. Its unprofiled C1 is
19.809 ms/round and C4 is 43.113 ms/round. Full-table pinned PLE admission was
rejected by host capacity. A separate graph-node trace has an early target
wait of roughly 2.3 ms before copying the 25,600-byte M5 PLE result. This
profiled wait is not an unprofiled end-to-end speedup estimate.

A same-process ABBA diagnostic on real IQ4_NL rows observed the following warm,
rotating-prefix producer costs. Both arms use the normally installed wheel
from `ec0e22c17d`, Torch 2.10.0 and one CPU thread. All 32 prefix batches per
shape produce identical FP16 result bytes after official GGUF dequantization.
These measurements exclude IPC and GPU consumption.

| Tokens | Existing producer | Packed producer | Possible CPU reduction |
| --- | ---: | ---: | ---: |
| 5 | 0.832 ms | 0.242 ms | 0.590 ms |
| 20 | 2.557 ms | 0.366 ms | 2.192 ms |

The wheel SHA-256 is
`5ed7c3151b4241ba00f3b6a0cee2abd8af5d4eb5d8fd9d60de13c6ce6f05e371`.
All 17 native modules are unchanged from the device-history measurement
artifact. The packed-result decoder is registered Python/Triton source in the
wheel. No separate extension is loaded.

Flattening the official CPU decoder's small row batches reduced M5 by only
0.24 ms and is not implemented. Transferring packed rows removes that CPU
dequantization work rather than changing its batch size. The packed M20 path
also admits the existing exact scalar n-gram algorithm through 32 tokens.

The reproducible producer diagnostic compares both arms in alternating ABBA
order in one process, checks every warmup result byte against official GGUF
dequantization, and reads only requested rows from retained file mappings:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python \
  benchmarks/benchmark_packed_ple_results.py MODEL.gguf acceptance.json \
  --output producer-ab.json
```

CPU producer checks, GPU decoder/transport replay checks and same-wheel model
C1/C4/teacher-forcing/acceptance are separate gates. No model improvement is
admitted until the last gate passes. Both model arms must have identical
actual PLE placement; requested kernel flags alone do not establish that.

The normally installed wheel passes all 19 packed-result tests on V100,
including M1/5/20/512 official decoder comparisons, changing graph inputs,
and M5/20 mapped-buffer consumption followed by acknowledgement. The model
A/B disables direct QSA and complete-table pinned decode in both arms, and
changes only the packed-result capability.
