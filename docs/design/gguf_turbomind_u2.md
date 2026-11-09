# GGUF U2 affine operators on SM70

This extends canonical affine storage with two-bit integer codes, group16
coefficients and the existing FP16 mma884 dense/grouped pipeline. It does not
connect a new model loader or lower activation or accumulation precision.

## Storage and source formats

Eight logical codes occupy one 16-bit word. Four even-index codes occupy its
low byte and four odd-index codes its high byte; register decoding restores
FP16 pairs before scale/min FMA. Coefficients retain the existing packed FP16
scale plus additive minimum carrier. Dense and grouped descriptors declare
group16 and group32 independently.

| Source | Canonical width | Group | Conversion |
| --- | --- | --- | --- |
| Q2_K | U2 | 16 | Expand nested scale/min products to FP16 |
| Q2_0 | U2 | 32 | Preserve codes and repeat its block64 scale/min |
| Q1_0 | U2 | 32 | Preserve signs as codes 0/2; retain scale without doubling |
| TQ1_0 | U2 | 32 | Decode wrapped base-three lanes to codes 0/1/2 |
| TQ2_0 | U2 | 32 | Reorder source two-bit lanes, preserving all values |

Q1_0's 0/2 encoding keeps finite coefficients when its FP16 scale is 65504.
Q2_0, Q1_0 and ternary reconstruction is exact against source formulas.
For a decimal-scale Q2_K fixture, maximum absolute coefficient reconstruction
error is 2.288818359375e-5 and relative L2 error is 6.89912267101758e-5.
Overflowing expanded coefficients are rejected rather than silently clipped.
Block layouts follow gguf-py and the pinned llama.cpp reference described in
[the fallback design](gguf_native_sm70.md); no reference source is copied here.

## TP4 and admission

Transcoding precedes TP slicing. Flash-Next down projections with full K=640
become four K=160 shards; each boundary aligns with a canonical group32 even
though it cuts the original Q2_0 block64. Reconstructing the full projection
from the four local outputs is covered by the GPU correctness test. Expert
parallelism is not required for these shards.

The mixed-precision selector admits source type, U2 width, FP16 activation,
canonical group and local output packing together. Wrong group, unavailable
codec, operator or output packing receives an explicit rejection reason.
Operators are enabled by capability, without a new environment switch.
The group-size argument defaults to 32, preserving the U4/U8 operator API.

## Correctness and baseline measurements

The first source-built extension passes 29 GPU checks together with the
U4/U8 suite (one invalid raw Q4_K K=160 fixture is skipped). U2 coverage
includes all five source types, M=1/2/4/8/16/32/64/128/512/2048/8192,
graph replay, group16/group32 grouped GEMM and empty experts. Thirteen CPU
transcode checks pass. Additional framework preparation/tracing and packaged
artifact checks accompany subsequent revisions.

The baseline below uses V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128, FP16
activations and source-built normal `_C`/packaged `_C_gguf` targets. It uses
the first expert of Flash-Next IQ3_XXS `blk.1.ffn_down_exps.weight`, Q2_0,
N=2560/K=640. These are full expert projections, not TP4 model timings.
Canonical reconstruction error for these actual weights is zero. Output
relative L2 error is approximately 2.1e-4 against FP16 reconstructed weights
and FP32 reference accumulation.

Each route has at least 100 ms warmup and 20 event-timed iterations. Graph
capture warms the capture stream and three graph replays before timing.
The AWQ comparator measures the same shape with group128 storage; it is a
speed comparator, not an equivalently quantized checkpoint. Cached FP16 is
a lower bound that excludes per-call dequantization.

The initial narrow N128/K32 U2 prefill tile is slower than native AWQ at
large M. A matched M=8192 trace localizes this to GPU kernel execution:
U2 averages 768.76 us with CTA128x128x32, while AWQ averages 425.70 us with
CTA128x256x16. Completing the native prefill tile repertoire is the next
experiment. These baseline results do not qualify a model performance claim
or a final default-route policy.

The grouped benchmark measures already sorted rows, one expert per row and
four distinct checkpoint experts. It excludes routing, sorting and the FFN
activation. Host-sorted reference dequantization plus cuBLAS is measured only
in eager mode and records its graph rejection reason explicitly.

### Initial dense graph baseline (microseconds)

| M | GGUF U2 | AWQ | MMVQ | MMQ | DQ + cuBLAS | Cached FP16 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.98 | 10.34 | 8.35 | unavailable | 14.90 | 6.04 |
| 2 | 8.76 | 10.44 | 9.16 | unavailable | 15.16 | 6.35 |
| 4 | 8.70 | 10.60 | 12.34 | unavailable | 15.36 | 6.30 |
| 8 | 9.11 | 11.72 | 12.95 | 18.48 | 15.46 | 6.50 |
| 16 | 9.93 | 14.28 | unavailable | 20.68 | 17.41 | 7.27 |
| 32 | 10.80 | 20.22 | unavailable | 26.88 | 17.92 | 8.40 |
| 64 | 16.59 | 13.98 | unavailable | unavailable | 28.57 | 11.26 |
| 128 | 22.32 | 15.36 | unavailable | unavailable | 26.93 | 12.70 |
| 512 | 40.76 | 32.87 | unavailable | unavailable | 50.12 | 29.24 |
| 2048 | 163.17 | 107.72 | unavailable | unavailable | 179.20 | 91.49 |
| 8192 | 761.45 | 436.02 | unavailable | unavailable | 679.73 | 351.64 |

### Dense graph after restoring native prefill tiles (microseconds)

The added tile candidates pass 31 GPU checks (one skipped raw fixture),
including U2 framework preparation and full-graph tracing. M=2048 and
M=8192 now approach AWQ; M=64–512 still need further analysis.

| M | GGUF U2 | AWQ | MMVQ | MMQ | DQ + cuBLAS | Cached FP16 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.96 | 10.29 | 8.29 | unavailable | 14.85 | 6.04 |
| 2 | 8.65 | 10.60 | 9.32 | unavailable | 15.00 | 6.30 |
| 4 | 8.70 | 10.60 | 12.44 | unavailable | 15.26 | 6.35 |
| 8 | 9.11 | 11.83 | 12.90 | 18.64 | 15.36 | 6.50 |
| 16 | 9.93 | 14.49 | unavailable | 20.63 | 16.33 | 7.22 |
| 32 | 10.80 | 20.07 | unavailable | 26.78 | 17.92 | 8.40 |
| 64 | 16.59 | 13.93 | unavailable | unavailable | 28.57 | 11.32 |
| 128 | 22.27 | 15.36 | unavailable | unavailable | 26.93 | 12.65 |
| 512 | 42.04 | 33.64 | unavailable | unavailable | 50.48 | 29.75 |
| 2048 | 113.77 | 107.01 | unavailable | unavailable | 178.69 | 90.21 |
| 8192 | 455.01 | 432.95 | unavailable | unavailable | 681.16 | 350.41 |

### Initial grouped comparison (microseconds)

These three points use the initial tile set, before the prefill change.

| Total rows | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS eager |
| --- | --- | --- | --- | --- | --- |
| 1 | 15.05 | 10.50 | 9.73 | unavailable | 123.24 |
| 8 | 17.36 | 14.39 | 15.16 | 26.83 | 276.48 |
| 512 | 48.64 | 48.44 | unavailable | unavailable | 308.07 |

### Complete grouped graph comparison after the tile change

All eleven points preserve approximately 2.1e-4 output relative L2 error.
The explicit MoE wrappers chunk large batches; reference MMQ rejects M=1.
At M=2048 and M=8192 the grouped U2 operator is near AWQ. Small grouped
workloads retain a gap and need further decode/batch work.

| Total rows | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ + cuBLAS eager |
| --- | --- | --- | --- | --- | --- |
| 1 | 14.95 | 10.44 | 9.22 | unavailable | 136.65 |
| 2 | 14.69 | 11.01 | 10.29 | unavailable | 137.78 |
| 4 | 15.16 | 11.93 | 11.72 | unavailable | 214.84 |
| 8 | 17.31 | 14.23 | 15.05 | 26.62 | 209.87 |
| 16 | 17.46 | 14.34 | 26.62 | 30.21 | 214.73 |
| 32 | 17.51 | 14.54 | 48.84 | 39.37 | 214.32 |
| 64 | 17.31 | 18.07 | 94.05 | 56.32 | 280.99 |
| 128 | 23.40 | 20.94 | 185.45 | 90.62 | 258.71 |
| 512 | 52.07 | 48.18 | 757.15 | 188.21 | 230.50 |
| 2048 | 108.75 | 110.54 | 3517.44 | 684.85 | 338.53 |
| 8192 | 461.06 | 448.31 | 14033.41 | 2682.88 | 1146.78 |

## Packaged artifact validation

The normal wheel installs in a separate Python 3.12 environment and passes
13 CPU and 31 GPU checks (one skipped fixture), including framework tracing
and CUDA graph replay. The run uses installed package imports without source
overrides or private library preloads. The core extension has no RPATH and
its hash matches the CMake-installed core stored in the wheel.

- Wheel SHA256: `21b030bbb0cc544429afd792ae1efdf895f1b0bf1dec67956f80389cb0e8aebc`.
- Core extension SHA256: `1f307a975ab90fad13156762ffd3462969696166013386a95ed984d02434fdcb`.
- Reference extension SHA256: `74ed944b8abb0f8679757a4e1bf0acef453f4a9803002f2c47b35b47f39d163f`.
- Wheel version/source: `1.5.2.dev216+g543bd12b2.precompiled`. Later changes
  record documentation and benchmark reference admission only.
