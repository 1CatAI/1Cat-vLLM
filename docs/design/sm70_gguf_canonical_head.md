# Canonical GGUF vocabulary head on SM70

The Flash-Next IQ3_S output tensor is Q6_K. The dense loading route expands its
TP4 vocabulary shard to 317,849,600 FP16 bytes. The existing TurboMind affine
decoder can read the canonical Q6_K bit planes and coefficients instead,
without retaining a full FP16 matrix.

## Operator comparison

The benchmark uses the real output tensor, TP4 rank 0 shape `[62080, 2560]`,
with synthetic FP16 activations. Correctness compares against official FP32
dequantization and FP32 matmul. Both candidate and dense control use FP32
accumulation. The canonical projection has no FP16 cache and selects the
packaged `TurboMindGgufAffineKernel`.

Hardware: one V100 SXM2 32 GB, driver 580.173.02. Runtime: Torch 2.10 CUDA
12.8, Python 3.12.3, `1.5.2.dev466+g8a649367f.precompiled`. Benchmark source:
`f5a3fb8f9c`. Graph timings are medians of five samples of 100 replays.

| Storage | Bytes per rank |
| --- | ---: |
| Original Q6_K | 130,368,000 |
| Canonical packed weights and coefficients | 158,924,800 |
| Dense FP16 | 317,849,600 |

| M | Dense FP16 (µs) | Canonical (µs) | Saved (µs) | Dense relative L2 | Canonical relative L2 | Canonical max absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 449.782 | 252.303 | 197.478 | 2.76e-4 | 3.17e-4 | 1.93e-4 |
| 5 | 455.393 | 209.091 | 246.303 | 2.98e-4 | 3.52e-4 | 1.78e-4 |
| 8 | 459.018 | 213.340 | 245.678 | 2.88e-4 | 3.36e-4 | 1.91e-4 |
| 20 | 472.668 | 289.485 | 183.183 | 2.91e-4 | 3.41e-4 | 2.30e-4 |

Top-1 agrees with the official FP32 reference for all tested rows. Coefficient
expansion to FP16 has a maximum weight error of 6.10e-5 and relative L2 error
of 1.97e-4. For Q6_K, projection error is slightly larger than the dense FP16
control; this is a measured representation tradeoff, not reduced accumulation
precision. These synthetic rows do not establish model quality.

A verifier at M=5 plus four M=1 draft calls suggests about 1.04 ms less head
service per speculative round. This is an operator estimate; acceptance,
scheduling, and complete-round latency are not measured by this benchmark.

Reproduce with `benchmarks/kernels/benchmark_sm70_gguf_head.py MODEL.gguf
--tp 4 --rank 0 --output RESULT.json` under an exclusive GPU lease.

## Integration status

The existing `GGUFLMHeadMethod` policy now admits the measured Q6_K shard at
M=1..20. The opaque projection forwards the actual canonical bit width and
group size: Q6_K uses 4+2 bit planes with group16. Its packed fallback also receives the actual
source type. The existing Q4_K M=2..16 policy is retained.

The method retains original packed bytes for unmeasured M values. Therefore
resident head payload is about 289,292,800 bytes per rank, comprising raw and
canonical streams, rather than the canonical-only footprint in the table.
Admitted projections read the canonical stream; no full FP16 head is cached.

Qwen4Exp passes GGUF quantization configuration to its vocabulary head.
Its architecture adapter still needs to classify the head as a projection
instead of requesting dense expansion. A shared target/draft head must
preserve that same object and vocabulary layout. Model integration follows
the dense/HC complete-round measurement and is checked with natural greedy
output and the later combined C1/C4 comparison.

## LM-head method measurement

The `--lm-head-method` comparison additionally exercises real LM-head
post-load preparation and dispatch, rather than calling the projection
directly. Source `ae536e80ae`, ordinary precompiled package version
`1.5.2.dev498` built from that commit, uses the same hardware, tensor and graph
timing method. The measured Q6_K route prepares canonical 4+2 bit planes
and admits every tested M. Top-1 matches the official FP32 reference for
all tested rows.

| M | Dense FP16 (µs) | Packed LM-head method (µs) |
| --- | ---: | ---: |
| 1 | 449.608 | 252.201 |
| 5 | 455.137 | 208.558 |
| 8 | 458.619 | 211.671 |
| 20 | 472.330 | 289.751 |

The M=5 verifier plus four M=1 draft calls estimates 1.036 ms less service
per round. Q6_K projection relative L2 is 3.51e-4 at M=5, compared with
2.98e-4 for dense FP16. Original packed fallback plus canonical data remains
289,292,800 bytes per rank; admitted projections read only the canonical
stream. Complete-round measurements must account for acceptance separately.
