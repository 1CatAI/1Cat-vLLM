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

## Numerical contract

For group `g`, the six-bit source code is
`((scales_lo >> (4*g)) & 15) | (((scales_hi >> (2*g)) & 3) << 4)`.
Its signed small scale is that code minus 32. The reader computes
`scale = float(d) * float(small)` and then
`weight = scale * float(lut[index])`, with two separate round-to-nearest
FP32 multiplications. Half conversion occurs only on the final weight.
No intermediate coefficient is converted to FP16. The CPU reader reuses
the official GGUF IQ4 table; it does not define another codebook.

A later device reader can use the existing TurboMind LUT conversion to
recover exact signed integers, then apply the original scales in FP32.
This change includes the CPU layout and oracle, with no new CUDA operator
or model dispatch.

## CPU evidence

The eleven full-width IQ4_XS projections in mixed pairs were checked,
with logical N=17408/K=5120. Both pair orientations are included:

| Gate/up | Layers |
| --- | --- |
| IQ4_XS / IQ3_S | 39, 48, 50, 52, 53, 54, 59, 60, 61 |
| IQ3_S / IQ4_XS | 42, 44 |

All eleven IQ4_XS projections preserved their exact byte counts and recovered
every original byte. They comprise 520,847,360 source bytes and
980,418,560 weights.
Their record-based Float results and final Half results are bitwise equal
to official GGUF dequantization. Maximum absolute and relative Float errors
are zero. Maximum absolute weight is 0.46985626220703125; no final Half
weight is nonfinite.

Six CPU tests cover all six-bit scale codes, randomized nibble packets,
signed zeros, subnormal and extreme original scales, output-row sharding,
and rejected truncated/partial tiles. Reproduce the full CPU oracle with:

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
