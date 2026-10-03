# Canonical GGUF LUT4 on SM70

IQ4_NL/IQ4_XS preserve nonlinear nibble indices and use the official IQ4
table. MXFP4/NVFP4 preserve E2M1 indices and expand source scale bytes into
FP16 coefficients. Both tables share TurboMind's native U4 operand packing,
FP16 activation path, mma884 loops and grouped expert scheduling.

| Source | Table | Group | Scale conversion |
| --- | --- | --- | --- |
| IQ4_NL | IQ4 nonlinear | 32 | Source FP16 scale, exact |
| IQ4_XS | IQ4 nonlinear | 32 | Expand signed nested scale products into FP16 |
| MXFP4 | E2M1 | 32 | Expand E8M0 scale; reject FP16 overflow/underflow |
| NVFP4 | E2M1 | 16 | Expand unsigned E4M3 scale, exact when representable |

Scales use a 16-bit metadata carrier. Distinct table tags keep these kernels
separate from affine U4 and native FP4 descriptors. Dense tuning keys are
also separate. No indices are requantized and activation precision is unchanged.

The IQ table restores four unsigned `value + 128` bytes with register byte
permutations, then forms exact FP16 pairs with the 1024 mantissa trick.
E2M1 uses the existing TurboMind bit decoder. Thus there is one LUT4 decoder
family with two fixed tables rather than one GEMM implementation per GGUF type.

## Correctness and integration

Eleven CPU checks compare official reconstruction, preserve index values,
exercise TP4 slices across IQ4_XS superblocks and reject unrepresentable
MXFP4 scales. IQ4_NL, MXFP4 and NVFP4 fixtures reconstruct exactly. IQ4_XS
expanded coefficients are checked against the official FP32 formula, including
non-power-of-two products and their relative error.

Twelve GPU checks pass for all four source formats, M=1–8192, distinct and
empty experts, CUDA graphs and framework full-graph tracing. Both scalar and
four-index register lookup revisions pass. The initial extension import
exposed missing global binding wrappers; these were added before the GPU
checks. No result from the failed import is counted as numerical validation.

Canonical LUT4 storage has a dedicated mixed-precision configuration and
kernel selector entry. Admission reports codec, activation, group, packing
or missing-operator failures. The path is enabled by default, uses the existing
kernel lifecycle and adds no runtime environment variables. Model wiring is
separate.

## First real-shape measurements

V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128, FP16 activations and FP32 MMA
accumulation. Twenty event-timed iterations follow 100 ms warmup per route.
The source is expert 0 of `blk.0.ffn_down_exps.weight` in Flash-Next IQ3_XXS,
stored as IQ4_NL, N=2560/K=640. These are individual projections, not FFN or
model throughput. AWQ uses the same dimensions with valid group128 U4 storage;
it is a speed comparison, not a quality-equivalent checkpoint.

| Decoder | M | GGUF graph (us) | AWQ graph (us) | DQ + cuBLAS graph (us) |
| --- | --- | --- | --- | --- |
| Scalar lookup | 1 | 11.37 | 10.39 | 20.99 |
| Scalar lookup | 512 | 51.35 | 33.18 | 55.65 |
| Scalar lookup | 8192 | 599.81 | 437.66 | 690.84 |
| Four-index lookup | 512 | 43.32 | 33.33 | 55.65 |
| Four-index lookup | 8192 | 484.25 | 438.48 | 692.94 |

Coefficient reconstruction is exact; operator output relative L2 is about
0.000207. The four-index decoder improves large-M speed by about 19%, leaving
roughly 10% versus AWQ at M=8192 and a larger gap at M=512. No model performance
objective is considered complete.

The second M=1 probe had common graph/host scheduling inflation across routes
under CPU load and is excluded from the table. The harness now captures eight
device invocations per replay when output has at most ten million elements,
and divides elapsed time by that count. Larger outputs retain one invocation
to bound graph-pool memory. Eager timing is unchanged. Full IQ4_NL sweeps and actual grouped expert speed appear below. Real IQ4_XS and TP4 measurements follow below. Normal-wheel validation
remains a separate gate before model integration.

## Full operator measurements

Updated graph timings use the repeated-invocation protocol above. All
routes for each row share that protocol. Times are microseconds.

### Dense expert projection

| M | GGUF | AWQ | MMVQ | MMQ | DQ + cuBLAS |
| --- | --- | --- | --- | --- | --- |
| 1 | 8.93 | 9.73 | 8.31 | unavailable | 20.29 |
| 2 | 8.88 | 9.87 | 9.12 | unavailable | 20.62 |
| 4 | 9.76 | 9.96 | 10.52 | unavailable | 20.76 |
| 8 | 9.52 | 11.04 | 11.72 | 18.80 | 20.98 |
| 16 | 9.59 | 13.68 | unavailable | 21.00 | 21.33 |
| 32 | 10.82 | 19.50 | unavailable | 27.04 | 23.29 |
| 64 | 16.10 | 13.17 | unavailable | unavailable | 33.50 |
| 128 | 19.13 | 14.66 | unavailable | unavailable | 32.94 |
| 512 | 42.02 | 31.65 | unavailable | unavailable | 55.61 |
| 2048 | 118.50 | 105.01 | unavailable | unavailable | 188.31 |
| 8192 | 484.51 | 433.82 | unavailable | unavailable | 697.45 |

### Grouped projection: four distinct experts

| M | GGUF grouped | AWQ grouped | MoE MMVQ | MoE MMQ | DQ + cuBLAS eager |
| --- | --- | --- | --- | --- | --- |
| 1 | 14.59 | 9.63 | 9.17 | unavailable | 98.87 |
| 2 | 14.94 | 10.30 | 10.30 | unavailable | 137.68 |
| 4 | 15.61 | 11.12 | 12.38 | unavailable | 213.86 |
| 8 | 16.94 | 13.50 | 17.02 | 28.08 | 213.86 |
| 16 | 16.95 | 13.70 | 30.59 | 31.34 | 216.17 |
| 32 | 17.14 | 13.76 | 58.04 | 40.59 | 217.40 |
| 64 | 17.95 | 17.47 | 112.82 | 57.38 | 277.09 |
| 128 | 23.33 | 20.22 | 257.61 | 93.62 | 263.83 |
| 512 | 56.69 | 47.46 | 1028.52 | 187.10 | 224.56 |
| 2048 | 118.92 | 109.61 | 4118.55 | 686.00 | 364.80 |
| 8192 | 497.36 | 449.02 | 16443.95 | 2696.35 | 1163.37 |

Grouped rows are already sorted, with one assigned expert per row.
Router, sorting and the full MoE FFN are excluded. The reference
grouped dequantization path requires a host sort and cannot be captured;
its column therefore uses eager timings. Empty experts occur at small M.
At M=8192, GGUF grouped is 497.36 us versus AWQ grouped 449.02 us,
MMQ 2696.35 us and MMVQ 16443.95 us. The remaining AWQ difference must be addressed before model connection.

## Dense IQ4_XS projection

The 27B `blk.0.ffn_gate.weight` projection has N=17408/K=5120.
Nested coefficients have maximum absolute reconstruction error
6.0558319091796875e-05 and relative L2 error 0.00016060 against the
official FP32 dequantization formula. GEMM output relative L2 is about
0.0003163. No indices are changed.

| M | GGUF | AWQ | MMVQ | MMQ | DQ + cuBLAS |
| --- | --- | --- | --- | --- | --- |
| 1 | 93.62 | 69.29 | 72.26 | unavailable | 962.25 |
| 2 | 82.04 | 74.43 | 76.91 | unavailable | 963.64 |
| 4 | 83.11 | 74.65 | 88.78 | unavailable | 964.65 |
| 8 | 83.80 | 72.72 | 127.26 | 168.01 | 968.59 |
| 16 | 95.39 | 95.55 | unavailable | 189.81 | 981.73 |
| 32 | 142.22 | 114.86 | unavailable | 262.53 | 976.99 |
| 64 | 293.20 | 206.87 | unavailable | unavailable | 1092.31 |
| 128 | 476.41 | 461.16 | unavailable | unavailable | 1272.03 |
| 512 | 1581.64 | 1401.85 | unavailable | unavailable | 1837.33 |
| 2048 | 5699.28 | 5022.21 | unavailable | unavailable | 5199.92 |
| 8192 | 23573.15 | 20565.66 | unavailable | unavailable | 18498.20 |

Large-M fused IQ4_XS retains about a 14–15% difference versus AWQ.
Canonical dequantization is a separate candidate to investigate; these
measurements do not justify model integration yet.

## TP4 expert down projection

Flash-Next remains TP4. IQ4_NL down weights slice from K=640 to K=160,
preserving complete group32 blocks. Coefficient reconstruction is exact.
AWQ group128 cannot represent this local K, so the native comparison uses
NVFP4 group16 at the same N=2560/K=160. It compares speed, not checkpoint
quality. All timings below use CUDA graphs and are microseconds.

| M | GGUF | Native NVFP4 | MMVQ | MMQ | DQ + cuBLAS |
| --- | --- | --- | --- | --- | --- |
| 1 | 5.29 | 5.33 | 7.60 | unavailable | 9.29 |
| 2 | 5.15 | 5.36 | 8.52 | unavailable | 9.22 |
| 4 | 5.00 | 5.45 | 9.27 | unavailable | 10.25 |
| 8 | 5.15 | 5.60 | 10.23 | 13.71 | 9.86 |
| 16 | 5.29 | 5.68 | unavailable | 14.72 | 10.07 |
| 32 | 5.28 | 5.42 | unavailable | 17.79 | 10.56 |
| 64 | 5.36 | 5.95 | unavailable | unavailable | 11.89 |
| 128 | 8.35 | 8.45 | unavailable | unavailable | 13.15 |
| 512 | 18.58 | 17.91 | unavailable | unavailable | 29.08 |
| 2048 | 44.55 | 41.37 | unavailable | unavailable | 89.84 |
| 8192 | 175.82 | 170.39 | unavailable | unavailable | 320.61 |

These measurements keep tensor parallelism and do not substitute expert
parallelism. Canonical slicing across larger source superblocks is tested
separately; unsupported raw reference slices report their block boundary
reason rather than benchmarking malformed packed weights.

## Installed artifact validation

A normal wheel built from source revision `f01116d3af` passes 34 CPU
checks and 55 GPU checks (one non-SM70 test skipped) in a fresh environment.
The installed TP4 projection measures M=512 at 18.57 us versus native NVFP4
17.91 us, and M=8192 at 179.15 us versus 169.16 us. No preload or private
shared library is required. The core has no RPATH/RUNPATH and depends only
on standard CUDA, Torch and system libraries. Source, wheel and installed
core hashes match.

- Wheel SHA256: `7ed9e273c45b3a0c398296b7e8a74ef43c8537a5b4f9fd2eb990753272f3ff67`
- Core SHA256: `f28920f322669081343e3bf207401123b44e5c35bbe985667f52b289d10d3948`

After integrating main revision `d582b816e6`, the normal build and wheel
were regenerated. The installed LUT4 checks pass again (12 tests); the
GGUF decoder source is unchanged by that integration.

- Updated wheel SHA256: `b16a437f38ad7a58fb6d465e5e2df4a1c62319f256ac0c4dfc906eb88636cc6b`
- Updated core SHA256: `8c811b451b8f1853abe5846640ca27fdbde7934f0d9c7eab11e19fa2dfbf7d62`
