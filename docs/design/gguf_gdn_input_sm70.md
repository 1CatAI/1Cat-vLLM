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
Partition coverage remains complete. GPU numerical and speed qualification
are pending, so the entry remains research-only and contributes no claimed
model or per-round saving.
