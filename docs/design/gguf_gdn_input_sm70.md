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

## Signed and aligned decoder candidates

Two further research entries use the shared signed-codebook API and retain
the original two-level scales. The signed compact entry preserves the 224-CTA
geometry. The aligned entry uses two N32 tiles per quantized CTA, each with
eight warps. Each warp covers five K128 records, avoiding a K64 tail. It uses
64 quantized CTAs plus the unchanged 96 b/a CTAs, for a total of 160.

| Candidate | K-loop instructions per K16 | Registers | Shared bytes | Spills |
| --- | ---: | ---: | ---: | ---: |
| Signed compact | 128 | 40 | 49,152 | 0 |
| Signed aligned records | 93.25 | 52 | 49,152 | 0 |

The aligned loop contains 746 static instructions and 128 HMMA instructions
per K128. Both entries request the 96 KiB shared-memory carveout. The b/a
branch returns uniformly before signed-codebook initialization, so dense
CTAs do not populate the 32 KiB table. Static shared-memory allocation still
applies to those CTAs.

The shared CPU converter recovered every source byte of the real QKVZ
projection, including original scales, from its 9,011,200-byte aligned form.
The partition screen covers all 128 N32 tiles and all 40 K128 records per
tile. The aligned reduction uses eight FP32 partitions instead of sixteen;
its changed reduction order still requires numerical qualification.

These two entries have CPU compilation and SASS evidence only. They have no
GPU correctness, speed, or model evidence and are not admitted production
routes. The signed public API itself remains an experiment. No GPU tests are
run until its shared screening supports the next implementation decision.

## Fixed 13-bit signed-index candidate

The next CPU candidate consumes the shared lossless signed-index layout.
Each 13-bit ID contains the original nine-bit codebook index and four sign
bits. It reuses the shared constexpr record reader and
`fragment_signed<half, true>`; static table copies use uint4, the original
base half2 is cached per block, and small scales stay in their original u32
bitfield. No scale precision changes are introduced.

The mixed operator retains 160 CTAs and 512 threads per CTA. Its full K128
loop compiles to 564 static instructions, including 128 HMMA instructions,
or **70.5 instructions per K16**. It uses 52 registers, no spills, and
32,768 bytes of shared memory. The table and FP32 partials occupy a union;
a barrier after all MMAs protects the transition from table reads to partial
writes. The floating tail returns before table initialization.

The operator requests carveout 66. Its actual hardware shared-memory
configuration has not been measured; this request must not be reported as
proof of a 64 KiB configuration. The shared CPU converter independently
recovers every original QKVZ byte from its unchanged 9,011,200-byte payload.
Partition coverage remains complete. The subsequent GPU screen below
qualifies its numerical operands and local operator delta. It remains
research-only because the 20-microsecond target is unmet.

### Fixed-index mixed projection GPU screen

The shared fixed-index numerical gate passed before this matched operator
screen. On the same real layer-6 weights and cold-L2 graph workload, old/new/
new/old medians are **41.984 / 38.912 / 38.912 / 41.984 microseconds**. Every
point records 1290 MHz SM and 877 MHz memory clocks before and after timing;
the GPU temperature is 33 degrees Celsius.

| Projection | Median microseconds | Source bytes | Source GB/s |
| --- | ---: | ---: | ---: |
| Original compact mixed projection, 224 CTAs | 41.984 | 9,256,960 | 220.49 |
| Fixed-index mixed projection, 160 CTAs | 38.912 | 9,256,960 | 237.89 |
| Native NVFP4 QKVZ, same shape | 27.648 | 11,796,480 | 426.67 |
| Native NVFP4 QKVZ plus independent dense b/a | 34.816 | 12,042,240 | 345.88 |

The local improvement is 3.072 microseconds, or 7.32%. Applied only to the
eight admitted same-type GDN layers, it projects **0.024576 milliseconds per
round**. The 20-microsecond target is still missed. This route is not promoted,
and no model benchmark or trace is run.

Every reconstructed QKVZ F32 and F16 weight is bitwise equal to the official
reader. Against FP32 dense products, QKV/Z relative L2 errors are 0.000321087
and 0.000325754, with maximum absolute error 0.001953125. The changed eight-
partition reduction produces relative L2 differences of 0.000020029 and
0.000021586 against the old sixteen-partition kernel; this is a reduction-order
difference, with the same weight operands and FP32 accumulation. Both b/a
outputs remain bitwise equal to the old implementation and to the FP64 dense
reference rounded to F16 for this input. Graph replay remains stable.

## Exact HFMA unpack and N32 K64 ownership

The next candidate uses the shared exact HFMA2 unpack helper and lossless K64
records. The helper fuses biased-byte subtraction and the small-scale multiply,
while retaining the original separate d and unchanged F32 decode. Each CTA
owns one N32 output tile with sixteen K warps, so K5120 is divided into five
K64 records per warp. The quantized grid grows from 64 to 128 CTAs; the same
96 b/a CTAs bring the combined launch to 224. The original sixteen-partition
FP32 reduction order is restored. No prefetch or register pipeline is added.

Its K64 loop contains 272 static instructions and 64 HMMA instructions,
or **68 instructions per K16**, with 52 registers, no spills, and the same
32 KiB shared union. CPU inverse reconstruction covers every original byte
and all 80 K64 records per output tile.

A matched operator screen on another V100-SXM2-32GB host records stable
1290 MHz SM and 877 MHz memory clocks before and after every point. All
actual QKVZ F32/F16 weight bits match the official reader through both the
new unpack oracle and the K64 record reader. QKV, Z, b, and a outputs are
bitwise equal to the original 224-CTA implementation. Official-reference
projection errors therefore remain unchanged.

| Projection | Median microseconds | Source GB/s |
| --- | ---: | ---: |
| Original mixed 224 CTAs, first control | 41.984 | 220.49 |
| HFMA K64 mixed 224 CTAs, first candidate | 37.888 | 244.32 |
| HFMA K64 mixed 224 CTAs, second candidate | 37.888 | 244.32 |
| Original mixed 224 CTAs, second control | 43.008 | 215.24 |
| Previous K128 mixed 160 CTAs | 38.912 | 237.89 |
| Native NVFP4 QKVZ | 27.648 | 426.67 |
| Native NVFP4 QKVZ plus independent dense b/a | 34.816 | 345.88 |

The local saving is 4.096-5.120 microseconds against the original mixed
operator, and 1.024 microseconds against the previous fixed-index candidate.
Only eight same-type GDN layers are admitted, yielding a projection of
0.032768-0.040960 milliseconds per round against the original operator.
The 20-microsecond target remains unmet. The route stays research-only, with
no model changes, endpoint benchmarks, or traces.
