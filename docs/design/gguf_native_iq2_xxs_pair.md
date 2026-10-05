# Original IQ2_XXS shared-activation pair reader

Layers0/1 use IQ2_XS gate and IQ2_XXS up; layer14 uses IQ2_XXS gate and
IQ2_S up. All are TP4 N4352/K5120. Preserve each66-byte K256 block as two
aligned16-byte index/auxiliary packets per K128, followed by original d
once per K256. No field expands or changes.

Use the existing IQ2_XXS codebook and restore original8-bit indices,
7-bit sign indices with parity, and the scale nibble shared by32 values.
Its coefficient formula matches IQ2_S, so reuse exact final FP16 operand
formation. FP32 dot/reduction, shared-A skeleton and gated epilogue remain
unchanged. Only the two actual orientations enter raw dispatch; model
admission remains canonical pending numerical and matched speed checks.

Twelve CPU storage/cursor/operand checks pass. Both IQ2_XS and IQ2_XXS
codebooks/formulas preserve all finite original FP16 d, every scale/grid
integer and both signs bitwise in final operands:6,094,848 cases each.
Three real N64/K5120 IQ2_XXS samples preserve253,440 source bytes and
983,040 official FP32 dequantized values bitwise. The existing llama.cpp
MIT attribution for the codebook remains in place.

Native build, actual-weight GPU numerical/graph/cold-L2 comparisons are
pending. No end-to-end result is claimed.
