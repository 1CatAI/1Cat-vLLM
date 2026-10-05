# Original IQ2_S shared-activation pair reader

The next two source-byte priorities pair IQ2_S with IQ3_S in layers5/9/17
and IQ3_XXS in layers7/18/33. All six are TP4 N4352/K5120. Both gate/up
orientations occur for each combination.

Preserve each original 82-byte K256 block as two K128 packet sets:
16-byte indices, 16-byte signs and eight bytes of high bits/scales per
column. Original d follows once per K256 block. Packet planes interleave
N32 columns with aligned 16/8-byte loads; no field expands or changes.
The reader caches original d across halves, including odd split-K starts,
and increments pointers in the K loop.

Reuse `LatticeRawDecoder<22>` codebook initialization and exact final FP16
operand formation, with the existing llama.cpp MIT attribution. No expanded
FP16 coefficient or second decode formula is introduced. The shared-A
skeleton, FP32 dot/reduction and gated epilogue are unchanged. Raw operators
compile both orientations; model capabilities remain canonical pending
actual-weight numerical and matched speed checks.

Ten CPU inverse/cursor checks pass, covering N32/64/96, complete K256 blocks,
strided inputs and all eight split-K starts at K5120. Six real IQ2_S tensors,
each sampled at N64/K5120, preserve 629,760 source bytes and 1,966,080 official
FP32 dequantized values bitwise. Native build, graph and speed checks are
pending; no end-to-end result is claimed.
