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

The aligned 64-bit metadata carrier stores FP16 scale/min in its lower word
and the little-endian high plane in its upper word. Q3_K uses 16 of the
upper word's bits; Q5 and Q6 use all 32. Low codes reuse the native packed
U2/U4 weight layout. Register decoding combines both planes before FP16
affine FMA and mma884. No code values are requantized.

This metadata changes the decoding contract, so its GEMM descriptor uses a
separate bit-plane quantization tag. Dense tuning also uses independent keys
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
trick. All eleven new GPU checks pass after this change, including all five
source formats, grouped GEMM, graph replay and framework tracing. The next
same-shape speed measurement is pending shared GPU availability; the initial
tables below describe the previous decoder. Model connection remains pending.

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
