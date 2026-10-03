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
