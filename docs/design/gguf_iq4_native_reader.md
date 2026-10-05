# Original-byte IQ4_XS records and mixed pair reader

IQ4_XS contributes 25.696% of the Qwen3.8-27B IQ3_S GGUF payload.
IQ4_XS/IQ3_S gate/up pairs occur in eleven layers and cover 31.509% of
mixed-pair source bytes. This change prepares a lossless layout and reader
for those pairs. It does not select a new model or kernel route.

## Record layout

Each source block contains 256 values in 136 bytes: original FP16 `d`,
16 high scale bits, 32 low scale bits and 128 nibble bytes. A complete N32
tile with `B=K/256` blocks retains exactly `4352*B` bytes:

| Plane | Tile offset | Extent | Order |
| --- | ---: | ---: | --- |
| Nibble packets | 0 | `4096*B` bytes | `[B][8 K32 groups][32 columns][16 bytes]` |
| Original `d` | `4096*B` | `64*B` bytes | `[B][32 columns][2 bytes]` |
| Original scale high bits | `4160*B` | `64*B` bytes | `[B][32 columns][2 bytes]` |
| Original scale low bits | `4224*B` | `128*B` bytes | `[B][32 columns][4 bytes]` |

Within each 16-byte packet, adjacent nibbles represent adjacent logical K
values. The original low-16/high-16 nibble order is only permuted. Four
aligned uint32 packets each hold eight consecutive K values, allowing one
aligned uint4 load per column/K32 group. All block scales and scale bits
remain in their original encodings. No expanded coefficients or padding
bytes are added.

`pack_iq4_xs_records`, `unpack_iq4_xs_records`, and
`dequantize_iq4_xs_records` are in `gguf_iq4_native.py`. The inverse separately
unpacks logical nibbles and restores the original low/high planes, then
copies all metadata bytes back. Complete output-row N32 shards concatenate
to the full record stream. Partial N32 or K256 tiles are rejected here;
this reader does not introduce a policy for a TP cut through a source block.

## Numerical contract and public device interface

For group `g`, the six-bit source code is
`((scales_lo >> (4*g)) & 15) | (((scales_hi >> (2*g)) & 3) << 4)`.
Its signed small scale is that code minus 32. The reader computes
`scale = float(d) * float(small)` and then
`weight = scale * float(lut[index])`, with two separate round-to-nearest
FP32 multiplications. Half conversion occurs only on the final weight.
In particular, no intermediate coefficient is converted to FP16.

`Iq4XsNativeDecoder` exposes:

- `parameters(tile, blocks_k, block, col)`: original `d` converted exactly
  to Float, and the original high/low scale bits.
- `fragment<half|float>(parameters, group32, packet)`: eight final weights
  in logical K order. `packet` contains their original nibble indices.

The decoder calls the existing `Transform_HMMA_SM70_Lut4<0>` with a unity
scale to recover exact signed IQ4 integers. This reuses the existing LUT and
nibble conversion rather than defining another table or lookup formula.
All IQ4 integers are exactly representable in Half. The final weight
arithmetic uses FP32 and never uses a rounded Half source coefficient.
The existing LUT transform and raw lattice reader are unchanged.

## CPU evidence

The complete eleven full-width pairs were checked, with logical
N=17408/K=5120 and 942,120,960 original bytes. Both orientations are included:

| Gate/up | Layers |
| --- | --- |
| IQ4_XS / IQ3_S | 39, 48, 50, 52, 53, 54, 59, 60, 61 |
| IQ3_S / IQ4_XS | 42, 44 |

All 22 projections preserved their exact byte counts and recovered every
original byte. The existing IQ3_S converter performs its independent
inverse check; its device decoder is unchanged. The eleven IQ4_XS
projections comprise 520,847,360 source bytes and 980,418,560 weights.
Their record-based Float results and final Half results are bitwise equal
to official GGUF dequantization. Maximum absolute and relative Float errors
are zero. Maximum absolute weight is 0.46985626220703125; no final Half
weight is nonfinite.

Six CPU tests cover all six-bit scale codes, randomized nibble packets,
signed zeros, subnormal and extreme original scales, output-row sharding,
and rejected truncated/partial tiles. The standalone SM70 Float and Half
device reader instantiations compile with CUDA 12.8, each using 32 registers
and zero spills. Compilation establishes API and type compatibility; the
device reader has not yet undergone a GPU numerical or speed test.

Reproduce the full CPU oracle with:

```bash
python benchmarks/kernels/benchmark_gguf_iq4_native.py MODEL.gguf \
  --output iq4-native-oracle.json
```

The JSON includes all per-tensor shapes, original and record bytes, inverse
mismatches, Float/Half bit mismatches, maximum errors and Half finite counts.
It does not contain machine paths or model data.

## Mixed pair skeleton boundary

Reuse the measured N32 shared-activation gated pair with separate GateReader
and UpReader types. Both orientations share the same A tile and retain the
existing FP32 partials, reduction order and gated epilogue. Reader selection
is uniform for each projection's warp group. IQ3_S keeps its signed-record
packet reader; IQ4_XS loads four aligned K32 packets per K128 iteration and
selects the matching original scale group. No projection is concatenated
and no mixed coefficient is interpreted using the other reader's layout.

The next gate is device operand comparison followed by same-shape cold-L2
graph timing for each orientation and actual layer weights. Until that
passes, these eleven pairs and the other mixed pairs retain their existing
canonical implementation. No speed or model-level benefit is claimed here.

## GPU operand gate: canonical lane order

The initial native wrapper incorrectly assumed that the existing LUT
transform emitted eight consecutive logical values. The transform emits
`[q0, q4, q1, q5, q2, q6, q3, q7]`, arranging half2 pairs for its canonical
MMA layout. The native record stores logical `[q0, ..., q7]` packets.

The initial layer-39 gate rank-zero Float operand oracle found 15,536,553
bit mismatches among 22,282,240 values, with maximum absolute error
0.1352691650390625. All outputs were finite. The record inverse and CPU
dequantization had already passed; they did not exercise this device lane
mapping. This failure is retained as evidence for that separate test boundary.

The wrapper now maps logical lane `i` to canonical lane
`(i & 3)*2 + (i >> 2)` before the original FP32 scale operations. No source
bits, LUT values, scale representation, or arithmetic precision changes.
The operator remains unselected until the corrected device gate passes.

`gguf_iq4_native_oracle_sm70.cu` is a research-only Float/Half operand oracle,
registered only with `GGUF_IQ4_ORACLE_RESEARCH`. It performs no GEMM and is
not a production speed path. `benchmark_gguf_iq4_native_operand.py` tests
all 64 scale codes and 16 LUT entries crossed with signed zeros, the smallest
Half subnormal, non-dyadic scales and extreme finite Half scales. Its
`--stress-only` option isolates those inputs before real weight checks.

After this correction, the smallest GPU gate passes on V100-SXM2-32GB with
Torch 2.10.0+cu128 and CUDA 12.8. The 512x512 stress matrix crosses all eight
original d patterns with all 64 scale codes and all 16 LUT entries. Its
262,144 Float values and 262,144 Half values are bitwise equal to the
official reader. Finite maximum absolute error is zero. The extreme d cases
have 64,384 expected Half infinities; signs and bits match exactly, without
clipping. Both reader instantiations retain 32 registers and zero spills.

The corrected full layer-39 gate and layer-42 up rank-zero device checks
also pass: each has 22,282,240 weights, and both Float and Half comparisons
have zero bit mismatches and zero maximum absolute error. Every actual
weight is finite. The same complete stress point passes again. This proves
operand reconstruction; no mixed GEMM or model route is admitted by these
operand-only results. The initial failure and fixed stress results are stored in
`gguf_iq4_native_operand_oracle.json`; they contain no weight data or local
paths.
