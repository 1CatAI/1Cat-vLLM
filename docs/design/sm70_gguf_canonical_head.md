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

This change supplies the operator comparison. Model loading still needs to
retain the quantized head and use the canonical linear lifecycle rather than
embedding preparation. A shared target/draft head must preserve that same
object and vocabulary layout. Model integration follows the dense/HC
complete-round measurement and is checked with natural greedy output and
the later combined C1/C4 comparison.
