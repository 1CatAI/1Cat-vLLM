# Original-byte Q4_K pair records

IQ4_XS/Q4_K is the next mixed gate/up format set by source-byte coverage:
three layers account for 9.781% of mixed-pair bytes. Q4_K also pairs with
IQ3_S in two further layers. This storage preparation retains the complete
original quantization block for a future independent affine reader.

An N32 macro stores K128 payload records first. Each column has four
16-byte packets, interleaved across columns at 16-byte boundaries. One
aligned metadata packet per K256 follows: original d, original dmin and
all twelve packed scale/min bytes. Metadata is never expanded or rounded.
The record consumes exactly 144 bytes per original K256 block.

Packet offsets are `part * 2048 + packet * 512 + column * 16` within a
macro. Metadata starts at `blocks * 4096` and advances by 512 bytes per
K256 block, plus `column * 16`. K128 readers can cache the original
metadata over both halves, including an odd first part from split-K.

CPU checks invert the layout independently across N32/N64/N96, one,
three and twenty K256 blocks, and strided input rows. They compare every
byte and require unchanged input, source-sized output and rejected partial
blocks or tiles. This helper adds no GPU operator or model admission.

Ten CPU checks pass. Actual Q4_K gate/up tensors in layers 36,37,47,58
and 62 are sampled at 64 rows each with K5120. Independent inversion
preserves 921,600 source bytes, and official GGUF dequantization of the
restored blocks matches 1,638,400 original FP32 values bitwise with zero
maximum error. No GPU speed or additional model coverage is claimed.
