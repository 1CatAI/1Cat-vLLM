# GGUF mixed projection output on SM70

Mixed projections can use different canonical GGUF families while retaining
checkpoint projection order. Previously each projection allocated an independent
FP16 output, followed by concatenation. Existing affine, LUT4 and lattice GEMM
operators accept an output row stride, so aligned column views can share one
merged output allocation.

The opaque `prepared_gguf_mixed_projection` operator chooses using actual M.
M5 and M20 use direct column writes when every prepared output is aligned to
32 columns and has no padding, active FP16 cache or BLAS policy. Other M retain
the existing independent projection policy and concatenation. Padding or an
unprepared projection retains the previous path and reports a capability reason.
Bias and architecture input layout transformations keep their original order.
No decoder, weight format, activation precision or accumulation policy changes.

## Validation

An ordinary source-containing wheel built for V100-SXM2-32GB with CUDA12.8,
Torch2.10.0+cu128 and Python3.12 passes 10 focused mixed-output GPU checks: mixed affine/LUT4/
lattice outputs at M5/M20/M32, padded fallback, compiled fullgraph calls,
graph replay and contiguous outputs. Mixed output
comparisons are bitwise against independent projection outputs. The existing
original-block expert operators remain registered in the normal extension.

The Flash-Next IQ3_S TP4 projection descriptors are:

| Projection | Source types | Local N | K |
|---|---|---|---|
| GDN QKV + Z | Q6_K + Q4_K | 2560 + 1536 | 2560 |
| Shared gate + up | Q4_K + IQ4_XS | 160 + 160 | 2560 |

CUDA graph microbenchmarks alternate control/candidate order, use 64 warmup
replays, then 12 alternating epochs of 128 replays each. Medians exclude the first
two epochs. Each replay cycles distinct physical weight banks: six QKV/Z banks
occupy 54,067,200 bytes; 39 shared gate/up banks occupy 18,969,600 bytes. Both
exceed twice the V100 6 MiB L2. Outputs match bitwise.

| Projection | M | Independent + concat (µs) | Direct output (µs) |
|---|---:|---:|---:|
| GDN QKV + Z | 5 | 43.395 | 34.532 |
| GDN QKV + Z | 20 | 42.944 | 41.201 |
| Shared gate + up | 5 | 31.734 | 29.311 |
| Shared gate + up | 20 | 31.538 | 29.007 |

A repeated single-bank QKV/Z M5 test measured 78.140 versus 35.008 µs. The gain
shrinks with distinct weight banks; the single-bank result is not a model speed
claim. Direct writes also change destination stride.

A combined TP4 Flash-Next IQ3_S/FP16-MTP4 run completes eight natural requests of
600 tokens with plausible text. It reports C1 verification-round mean 24.252 ms
(I8192/O256) and C4 mean 49.728 ms (I128/O600). That run also enables IQ3_XXS
original expert banks, so these absolute values do not isolate this change.
Both use FP16 activations/KV, FP32 recurrent state and accumulation, FULL graphs,
max length9216, max batch512, four sequences and memory utilization0.95.
The scoped microbenchmark establishes the projection benefit.
