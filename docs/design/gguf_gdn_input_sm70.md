# GGUF GDN input projection on SM70

The Qwen3.5-derived TP4 GDN input combines quantized QKV and Z with small
floating b/a projections. The mixed-CTA design follows the existing channel
FP8 verifier operator: quantized CTAs write QKV/Z directly, while additional
CTAs in the same launch compute b/a with FP32 FMA and reductions. There is no
concatenation or output-copy launch.

The initial scope is M8 with K5120, QKV N2560, Z N1536, and b/a N12 each.
The real layer-6 GGUF stores QKV and Z in IQ3_S; alpha and beta are BF16.
Loading checks finite values and the maximum absolute value before FP16
conversion, and reports any underflow or conversion error. It must check that
conversion and restore the grouped GDN value-head
order before TP selection. Q and K each contribute 512 rank-local rows; V and
Z each contribute 1536. The compact quantized payload remains 9,011,200
bytes; the floating tail contributes 245,760 bytes.

## Operator boundary

The quantized calculation uses the shared `LatticeCompactDecoder<21>` from
PR #897. It retains source scales and forms the same final FP16 operands as
official FP32 dequantization followed by FP16 rounding. Every dot product and
split reduction uses FP32. The b/a branch is copied from the qualified mixed
FP8 projection in PR #893, including its 256-thread row reduction. Decoder
changes belong to the shared GGUF implementation.

The research entry uses 128 quantized CTAs and 96 b/a CTAs, each with 512
threads. Its quantized K16 loop compiles to 154 SASS instructions, including
16 HMMA instructions, in both the fused and quantized-only variants. Both
use 40 registers per thread, 18,432 bytes shared memory, and no spills. This
is a wiring baseline, not the requested instruction-count optimization.

## Validation contract

Use actual layer-6 TP4 rank-0 weights after the adapter's head permutation.
Compare all compact weights against the official FP32 reader exactly, compare
quantized projections against FP32 dense products, and compare b/a against
FP64 products of the loaded FP16 weights. The fused and quantized-only QKV/Z
outputs must be bitwise equal. Repeated cold-L2 CUDA Graph calls measure only
the projection between external CUDA events; a 16 MiB flush precedes every
call outside the event interval. Record GPU clocks and run matched controls
in the same process.

The operator is research-only. It is not registered in the production build,
model dispatch, or kernel capability registry, and it is not eligible for a
performance PR until the source-complete artifact and runtime gates pass.
The M8 mixed projection target is 20 microseconds. No model benchmarks or
model traces are run during this operator screen.

The CPU conversion screen found all 245,760 values finite in each full
alpha/beta matrix. Their maximum absolute values are 0.2177734375 and
0.1943359375 respectively. FP16 conversion introduces a maximum absolute
error of 2.9802322387695312e-8 and one underflow-to-zero in each full matrix;
there is no overflow. The rank-0 restored value-head indices are
`0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35`.

The complete 64-layer file contains 48 GDN layers. Only layers
`5, 6, 8, 24, 28, 30, 33, 57` have IQ3_S for both QKV and Z; the other 40
use mixed types, including IQ3_XXS, IQ4_XS, and K-quants. All 48 b/a pairs are
BF16. This initial same-type operator can only claim projected savings over
its eight admitted layers; extending the mixed projection must retain
format-specific decoding for the remaining layers.

## Initial cold-L2 graph results

The layer-6 screen uses a V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128,
TP4 rank-0 weights, and FP16 input. GGUF points record 1290 MHz SM and
877 MHz memory clocks at 38 degrees Celsius. The controls run in the same
process; their clocks were not sampled separately.

| Projection | Median microseconds | Source bytes | Source GB/s |
| --- | ---: | ---: | ---: |
| IQ3_S QKVZ only | 41.984 | 9,011,200 | 214.63 |
| IQ3_S QKVZ plus FP16 b/a, one launch | 41.984 | 9,256,960 | 220.49 |
| Native NVFP4 QKVZ, same shape | 27.648 | 11,796,480 | 426.67 |
| Native NVFP4 QKVZ plus dense b/a, two launches | 34.816 | 12,042,240 | 345.88 |

The quantized-only/mixed/mixed/quantized-only ABBA medians are all 41.984
microseconds. The floating tail therefore adds no measurable time in this
screen. The 20-microsecond target remains unmet; the shared decoder needs the
instruction-count work before this route is eligible for promotion. The NVFP4
plus b/a control is explicitly a two-launch comparator, not a measurement of
the NVFP4 line's fused verifier.

All 20,971,520 compact weight elements match official FP32 dequantization
exactly. QKV and Z projection maximum absolute errors against FP32 dense
products are both 0.001953125, with relative L2 errors 0.000321013 and
0.000325998. Both b/a outputs are bitwise equal to FP64 dense products rounded
to FP16 for this input. Fused and quantized-only QKV/Z outputs are bitwise
equal. Repeated graph replay completed without failure.

These measurements establish an operator boundary and local tail overlap.
They do not establish a model speedup or a per-round saving across all layers.
