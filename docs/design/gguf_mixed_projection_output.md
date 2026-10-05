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
Torch2.10.0+cu128 and Python3.12 passes 16 focused GPU checks: mixed affine/LUT4/
lattice outputs at M5/M20/M32, padded fallback, compiled fullgraph calls,
changed-input expert graph replay and IQ2_S M20 canonical fallback. Mixed output
comparisons are bitwise against independent projection outputs. The existing
original-block expert operators remain registered in the normal extension.

The Flash-Next IQ3_S TP4 projection descriptors are:

| Projection | Source types | Local N | K |
|---|---|---|---|
| GDN QKV + Z | Q6_K + Q4_K | 2560 + 1536 | 2560 |
| Shared gate + up | Q4_K + IQ4_XS | 160 + 160 | 2560 |

Repeated-weight CUDA graph microbenchmarks alternate control/candidate order,
use 64 warmup replays, then 12 alternating epochs of 128 replays each. Medians
exclude the first two epochs. FP16 results match bitwise.

| Projection | M | Independent + concat (µs) | Direct output (µs) |
|---|---:|---:|---:|
| GDN QKV + Z | 5 | 78.140 | 35.008 |
| GDN QKV + Z | 20 | 42.988 | 41.568 |
| Shared gate + up | 5 | 30.300 | 28.908 |
| Shared gate + up | 20 | 30.080 | 28.712 |

These repeated-weight results do not establish model step savings. In particular,
the large M5 QKV/Z difference is stable across alternating epochs but is not
attributed solely to eliminating a concatenation launch. The merged output also
changes the GEMM destination stride. Cache-exceeding bank measurements and the
matched model run are required before final promotion.
