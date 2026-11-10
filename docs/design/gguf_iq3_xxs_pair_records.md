# Original-byte IQ3_XXS pair records

Seven layers in the Qwen3.8-27B IQ3_S checkpoint combine IQ3_S and IQ3_XXS
in gate/up: layers 2, 22, 35, 38, 41, 45 and 49. They account for 16.954%
of mixed gate/up source bytes, following the eleven IQ3_S/IQ4_XS layers.
This change prepares an aligned record layout; it adds no GPU operator,
model admission or performance claim. All seven pairs remain canonical.

`pack_iq3_xxs_records` permutes the original 98-byte K256 blocks into N32
macro records. Each K128 record contains two original 16-byte index packets
and one original 16-byte sign/scale packet per column. Packet planes
interleave columns at 16-byte boundaries. The original two-byte d plane
follows the payload, retaining one d for the two halves of each K256 block.
There is no sign expansion, scale rounding or reconstructed coefficient.

For B K256 blocks, an N32 macro contains `B * 3072` payload bytes followed
by `B * 64` d bytes: exactly `32 * B * 98` source bytes. Within a macro,
K128 part p and column c start at `p * 1536 + c * 16`; the other two packet
planes start 512 and 1024 bytes later. Original d starts at
`B * 3072 + (p // 2) * 64 + c * 2`. These cursors also describe odd split-K
starting parts. Partial N tiles and incomplete quantization blocks are
rejected; noncontiguous source-row views preserve their original bytes.

The future pair reader should reuse the existing IQ3_XXS decoder and its
FP16-operand proof from the original-byte lattice path. This storage helper
does not duplicate or modify dequantization formulas. Kernel admission must
wait for actual-weight numerical, CUDA graph and matched speed checks.

Nine CPU checks cover independent byte inverses across N32/N64/N96,
K256/K768/K5120, alignment, strided views and invalid inputs. Original input
buffers are unchanged. An additional CPU oracle samples the first 64 rows
of each of the seven actual IQ3_XXS projections, with K5120. Independent
record cursors reconstruct all 878,080 original bytes without mismatches;
official GGUF dequantization of the recovered blocks matches all 2,293,760
FP32 values bitwise, with zero maximum absolute error. These are storage
and decoding checks, not GPU GEMM or end-to-end quality evidence.
