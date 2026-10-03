# GGUF affine bit planes on SM70

Q3_K, Q5_0/Q5_1, Q5_K and Q6_K retain their integer codes and normalize into
U2/U4 low planes, one/two-bit high planes and FP16 affine coefficients.
Activation precision and FP32 MMA accumulation follow native TurboMind.
Model integration is separate.

| Source | Low/high widths | Group | Coefficients |
| --- | --- | --- | --- |
| Q3_K | 2 + 1 | 16 | Expand signed nested scales; additive min is -4 scale |
| Q5_0 | 4 + 1 | 32 | Source scale; additive min is -16 scale |
| Q5_1 | 4 + 1 | 32 | Source scale and additive min |
| Q5_K | 4 + 1 | 32 | Expand nested scale/min products |
| Q6_K | 4 + 2 | 16 | Expand signed nested scales; additive min is -32 scale |

Q5/Q6 use an aligned 64-bit carrier with FP16 scale/min in its lower word
and the little-endian high plane in its upper word. Q3_K instead uses a
32-bit carrier: FP16 scale and 16 high bits. Its centered form is
`(code - 4) * scale`; preparation checks exact equality of the independently
rounded min and `-4 * scale`. A mismatch requires the reference fallback,
rather than changing the rounded coefficients. All 5,570,560 groups of the
measured 27B Q3_K tensor satisfy this condition. The carrier reduces Q3's
canonical storage from six to four bits per weight, including metadata.

Low codes reuse the native packed U2/U4 weight layout. Register decoding
combines both planes before FP16 affine FMA (Q5/Q6) or centered multiplication
(Q3), then mma884. No code values are requantized.

This metadata changes the decoding contract, so its GEMM descriptor uses a
separate bit-plane and centered-Q3 quantization tags. Dense tuning also uses independent keys
for widths 3/5/6. Existing U2/U4/U8 descriptors retain their original tag.
Tiles start on complete metadata groups. Canonical groups, rather than the
original GGUF superblocks, govern TP slicing.

## Correctness

Twenty-three CPU checks pass across these codecs and the existing affine
codecs. Ten new checks compare source formulas with official reconstruction,
preserve all bit-plane codes and cover TP4 K slices that cut source blocks.
The initial source extension passes 42 GPU checks (one existing raw Q4_K
K=160 fixture is skipped), including M=1–8192, grouped GEMM with distinct
and empty experts, graph replay and framework full-graph tracing.

The first full GPU run lacked installed package metadata in its test
environment and could not identify the CUDA platform after the eight new
direct-operator checks passed. Installing the declared normal package
metadata resolved this test-environment failure; the complete run then passed.

## Initial measurements and profiling

Measurements use V100-SXM2-32GB, CUDA 12.8 and Torch 2.10.0+cu128, FP16
activations, 100 ms warmup per route and 20 event-timed iterations. Graph
capture warms its stream and three replay calls. These are full projection
shapes from the actual Qwen3.8-27B UD-Q4_K_M file, not TP4 model throughput.
The AWQ comparison has the same N/K with valid group128 U4 storage and is
not a quality-equivalent checkpoint. Cached FP16 excludes dequantization.

The initial bit-plane path is slower than AWQ at concurrency and prefill
shapes. Q6_K N=5120/K=6144/M=8192 takes about 9500 us versus 7073 us for
AWQ and 6270 us for dequantization plus cuBLAS. This is a rejected performance
baseline, not evidence that the performance objective has been reached.

The matched-shape dispatch probe selects CTA128x256x16 for both Q6 and AWQ,
with identical shared-memory size and one active CTA per SM. Q6 uses 252
registers versus AWQ's 250. Static disassembly contains 48 integer-to-FP16
conversion instructions for Q6 versus two for AWQ, supporting a register
decoder experiment that combines both planes as integer mantissa bits before
converting pairs to FP16. Static instruction counts are not dynamic counters.
Nsight Compute hardware-counter collection was denied by the driver; no
counter-based bottleneck claim is made.

The paired decoder restores integers below 64 using the FP16 1024 mantissa
trick. All eleven new GPU checks pass after this change. A normal wheel also
passes 23 CPU and 42 GPU checks in a fresh installed runtime. Q6's matched
M=8192 shape improves from 9500 to 8134 us, versus AWQ's 7099 us; a separate
same-shape probe gives 8057 versus 7043 us. The remaining gap is material.

The compact centered-Q3 variant passes twelve new GPU checks, including a
regression that rejects nonredundant min coefficients near FP16 underflow.
Its real 27B M=1/512/8192 probe measures 96.31/1539.53/23087.62 us versus
AWQ's 65.84/1395.97/20494.28 us. The full updated sweep appears below. The initial
tables below retain the original decoder's negative baseline. Model connection
remains pending performance and model-level quality checks.

### blk.0.ffn_up.weight, Q3_K, N=17408, K=5120

Coefficient reconstruction: maximum absolute error 3.05175781e-05; relative L2 0.000198725886.

| M | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS graph | Cached FP16 graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 124.21 | 65.89 | 108.65 | unavailable | 863.18 | 310.37 |
| 2 | 124.42 | 72.86 | 115.71 | unavailable | 873.42 | 310.32 |
| 4 | 123.34 | 72.65 | 167.78 | unavailable | 869.79 | 312.47 |
| 8 | 124.21 | 70.09 | 255.18 | 189.80 | 870.66 | 314.93 |
| 16 | 133.99 | 95.49 | unavailable | 220.62 | 879.26 | 321.74 |
| 32 | 173.57 | 114.33 | unavailable | 317.70 | 878.28 | 315.55 |
| 64 | 369.97 | 208.79 | unavailable | unavailable | 996.71 | 437.76 |
| 128 | 575.49 | 462.18 | unavailable | unavailable | 1160.86 | 594.94 |
| 512 | 1861.73 | 1396.28 | unavailable | unavailable | 1753.75 | 1100.60 |
| 2048 | 6574.39 | 5024.41 | unavailable | unavailable | 5000.24 | 4152.99 |
| 8192 | 26814.93 | 20464.03 | unavailable | unavailable | 18222.59 | 16575.85 |

### blk.0.attn_gate.weight, Q5_K, N=6144, K=5120

Coefficient reconstruction: maximum absolute error 0.000104904175; relative L2 0.000688216189.

| M | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS graph | Cached FP16 graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 53.71 | 37.43 | 35.79 | unavailable | 273.61 | 92.67 |
| 2 | 49.61 | 37.63 | 36.66 | unavailable | 279.50 | 99.28 |
| 4 | 50.07 | 37.73 | 46.54 | unavailable | 291.48 | 100.25 |
| 8 | 50.53 | 38.50 | 78.44 | 77.41 | 289.54 | 104.55 |
| 16 | 58.01 | 44.95 | unavailable | 90.52 | 289.54 | 105.88 |
| 32 | 64.36 | 57.09 | unavailable | 125.54 | 291.28 | 105.63 |
| 64 | 130.92 | 96.36 | unavailable | unavailable | 358.20 | 173.06 |
| 128 | 180.68 | 148.89 | unavailable | unavailable | 407.40 | 212.79 |
| 512 | 696.73 | 564.07 | unavailable | unavailable | 650.14 | 436.38 |
| 2048 | 2325.96 | 1805.41 | unavailable | unavailable | 1807.26 | 1518.75 |
| 8192 | 9788.88 | 7476.17 | unavailable | unavailable | 6601.52 | 6022.20 |

### blk.1.ssm_out.weight, Q6_K, N=5120, K=6144

Coefficient reconstruction: maximum absolute error 0.000465393066; relative L2 0.000201302042.

| M | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS graph | Cached FP16 graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 66.30 | 33.18 | 45.21 | unavailable | 268.13 | 87.14 |
| 2 | 65.74 | 33.13 | 47.21 | unavailable | 293.48 | 121.86 |
| 4 | 66.10 | 33.54 | 56.63 | unavailable | 292.92 | 122.21 |
| 8 | 59.75 | 34.46 | 78.85 | 77.67 | 294.20 | 121.50 |
| 16 | 71.32 | 42.24 | unavailable | 90.01 | 294.66 | 120.27 |
| 32 | 79.10 | 52.79 | unavailable | 126.46 | 297.01 | 127.54 |
| 64 | 123.85 | 87.04 | unavailable | unavailable | 349.39 | 167.12 |
| 128 | 169.37 | 129.54 | unavailable | unavailable | 415.85 | 249.96 |
| 512 | 581.22 | 439.09 | unavailable | unavailable | 554.75 | 367.92 |
| 2048 | 2317.36 | 1780.17 | unavailable | unavailable | 1714.02 | 1445.22 |
| 8192 | 9500.11 | 7073.18 | unavailable | unavailable | 6270.36 | 5772.75 |

## Updated operator measurements

Times are microseconds from the same graph/eager benchmark contract above.
Q3 uses the compact centered carrier; Q5/Q6 use paired register decoding.
Flash-Next tensors below are dense/shared projections. Its downloaded file
has no Q3/Q5/Q6 stacked expert tensors; grouped correctness uses distinct
expert fixtures, while actual affine expert speed evidence is the Q2_0
measurement in `gguf_turbomind_u2.md`. No full FFN or model speed is claimed.

### 27B centered Q3: blk.0.ffn_up.weight, Q3_K, N=17408, K=5120

Coefficient maximum absolute error 3.05175781e-05; relative L2 0.000198725886.

| M | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS graph | Cached FP16 graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 97.74 | 65.89 | 108.54 | unavailable | 858.06 | 309.71 |
| 2 | 96.46 | 73.06 | 115.97 | unavailable | 859.03 | 310.73 |
| 4 | 97.64 | 72.76 | 167.73 | unavailable | 872.45 | 311.96 |
| 8 | 98.87 | 69.84 | 254.72 | 190.26 | 871.12 | 313.96 |
| 16 | 110.80 | 95.28 | unavailable | 221.03 | 877.11 | 321.64 |
| 32 | 150.53 | 115.15 | unavailable | 317.08 | 884.02 | 312.63 |
| 64 | 329.57 | 209.41 | unavailable | unavailable | 994.46 | 436.58 |
| 128 | 497.56 | 462.39 | unavailable | unavailable | 1143.96 | 594.94 |
| 512 | 1554.84 | 1397.15 | unavailable | unavailable | 1765.02 | 1100.13 |
| 2048 | 5655.14 | 5060.81 | unavailable | unavailable | 5030.09 | 4178.64 |
| 8192 | 23051.62 | 20456.50 | unavailable | unavailable | 18333.54 | 16664.32 |

### 27B paired Q5/Q6: blk.0.attn_gate.weight, Q5_K, N=6144, K=5120

Coefficient maximum absolute error 0.000104904175; relative L2 0.000688216189.

| M | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS graph | Cached FP16 graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 55.30 | 37.73 | 35.74 | unavailable | 273.66 | 92.72 |
| 2 | 48.84 | 37.58 | 36.61 | unavailable | 279.04 | 98.76 |
| 4 | 49.10 | 38.09 | 46.69 | unavailable | 291.53 | 99.89 |
| 8 | 49.97 | 38.35 | 78.44 | 77.11 | 289.08 | 104.55 |
| 16 | 56.42 | 45.47 | unavailable | 90.57 | 289.59 | 105.73 |
| 32 | 62.26 | 56.78 | unavailable | 125.90 | 290.25 | 105.63 |
| 64 | 118.94 | 96.56 | unavailable | unavailable | 358.14 | 173.82 |
| 128 | 168.09 | 154.47 | unavailable | unavailable | 407.40 | 214.22 |
| 512 | 639.85 | 560.03 | unavailable | unavailable | 659.87 | 435.66 |
| 2048 | 2146.41 | 1805.36 | unavailable | unavailable | 1791.03 | 1511.42 |
| 8192 | 9147.19 | 7623.37 | unavailable | unavailable | 6622.82 | 6036.89 |

### 27B paired Q5/Q6: blk.1.ssm_out.weight, Q6_K, N=5120, K=6144

Coefficient maximum absolute error 0.000465393066; relative L2 0.000201302042.

| M | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS graph | Cached FP16 graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 66.51 | 33.13 | 45.31 | unavailable | 267.98 | 86.63 |
| 2 | 66.25 | 33.13 | 47.00 | unavailable | 293.53 | 111.36 |
| 4 | 65.84 | 33.43 | 56.58 | unavailable | 294.14 | 111.05 |
| 8 | 61.03 | 34.20 | 79.51 | 77.52 | 294.04 | 111.46 |
| 16 | 69.99 | 42.29 | unavailable | 89.96 | 295.01 | 115.66 |
| 32 | 77.16 | 52.79 | unavailable | 126.67 | 296.91 | 112.74 |
| 64 | 109.67 | 87.09 | unavailable | unavailable | 349.39 | 167.73 |
| 128 | 147.87 | 127.54 | unavailable | unavailable | 415.39 | 243.00 |
| 512 | 490.70 | 437.76 | unavailable | unavailable | 560.95 | 367.05 |
| 2048 | 1959.73 | 1780.17 | unavailable | unavailable | 1723.75 | 1448.76 |
| 8192 | 8134.40 | 7099.29 | unavailable | unavailable | 6248.60 | 5777.56 |

### Flash-Next paired Q5/Q6: blk.2.ffn_up_shexp.weight, Q5_K, N=640, K=2560

Coefficient maximum absolute error 0.000195026398; relative L2 0.000826338383.

| M | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS graph | Cached FP16 graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 23.14 | 34.97 | 28.11 | unavailable | 37.32 | 23.30 |
| 2 | 25.65 | 21.15 | 9.16 | unavailable | 27.65 | 23.30 |
| 4 | 17.00 | 27.75 | 25.86 | unavailable | 36.15 | 28.62 |
| 8 | 23.40 | 25.65 | 29.49 | 23.71 | 25.55 | 26.11 |
| 16 | 26.11 | 21.20 | unavailable | 20.28 | 25.19 | 29.85 |
| 32 | 21.61 | 24.12 | unavailable | 28.11 | 31.39 | 11.78 |
| 64 | 42.09 | 38.96 | unavailable | unavailable | 30.67 | 28.26 |
| 128 | 36.25 | 39.17 | unavailable | unavailable | 36.97 | 28.26 |
| 512 | 77.77 | 54.27 | unavailable | unavailable | 75.62 | 38.96 |
| 2048 | 205.00 | 130.20 | unavailable | unavailable | 143.51 | 95.85 |
| 8192 | 911.10 | 422.66 | unavailable | unavailable | 465.46 | 322.20 |

### Flash-Next paired Q5/Q6: blk.1.attn_gate.weight, Q6_K, N=6144, K=2560

Coefficient maximum absolute error 0.000122070312; relative L2 0.000189462327.

| M | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS graph | Cached FP16 graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 37.94 | 83.15 | 27.70 | unavailable | 140.08 | 45.47 |
| 2 | 267.32 | 23.65 | 31.44 | unavailable | 147.15 | 52.63 |
| 4 | 32.77 | 21.15 | 36.25 | unavailable | 148.28 | 55.50 |
| 8 | 33.95 | 25.50 | 48.38 | 47.46 | 150.12 | 57.24 |
| 16 | 34.15 | 28.21 | unavailable | 56.93 | 147.87 | 54.12 |
| 32 | 41.16 | 32.92 | unavailable | 77.21 | 150.84 | 56.99 |
| 64 | 256.82 | 53.30 | unavailable | unavailable | 171.11 | 81.25 |
| 128 | 85.91 | 87.60 | unavailable | unavailable | 248.37 | 154.11 |
| 512 | 373.76 | 350.57 | unavailable | unavailable | 353.43 | 265.47 |
| 2048 | 1035.01 | 940.29 | unavailable | unavailable | 957.80 | 772.71 |
| 8192 | 4324.51 | 3862.32 | unavailable | unavailable | 3545.45 | 3071.44 |

### Flash-Next paired Q5/Q6: blk.3.attn_output.weight, Q6_K, N=2560, K=6144

Coefficient maximum absolute error 0.000244140625; relative L2 0.000195282298.

| M | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS graph | Cached FP16 graph |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 41.47 | 24.22 | 27.49 | unavailable | 141.00 | 44.95 |
| 2 | 35.89 | 31.23 | 29.34 | unavailable | 160.51 | 60.72 |
| 4 | 240.33 | 30.67 | 36.97 | unavailable | 160.05 | 60.72 |
| 8 | 37.94 | 26.78 | 46.18 | 49.72 | 156.36 | 60.77 |
| 16 | 41.01 | 35.84 | unavailable | 55.04 | 150.02 | 56.73 |
| 32 | 46.49 | 46.80 | unavailable | 76.08 | 160.10 | 64.61 |
| 64 | 67.84 | 74.85 | unavailable | unavailable | 170.24 | 79.82 |
| 128 | 108.75 | 98.20 | unavailable | unavailable | 207.00 | 102.14 |
| 512 | 257.89 | 230.86 | unavailable | unavailable | 290.00 | 197.99 |
| 2048 | 968.91 | 873.22 | unavailable | unavailable | 841.01 | 719.82 |
| 8192 | 4057.60 | 3605.61 | unavailable | unavailable | 3159.71 | 2908.36 |

At M=2048–8192, dequantization plus cuBLAS is faster than the fused
bit-plane kernel for these projections. A canonical dequantization
operator with FP32-accumulating GEMM is the next candidate; reference
timings alone do not implement or select that route. Small-M and
intermediate-M defaults also require the completed candidate comparison.
The remaining AWQ gap is recorded rather than treated as a completed
performance objective. Model integration and model-level quality remain
separate work.
