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

The normal CUDA12.8 SM70 extension and complete wheel build pass.
IQ2_XS/IQ2_XXS uses63 registers, IQ2_XXS/IQ2_S uses64; both use
33,792 shared bytes, zero stack/local memory. All three real numerical
and runtime M512/8/1/5/16/20/32/8 graph checks pass, with non-M8 outputs
remaining bitwise canonical. Native relative L2 is0.000491–0.000518.

At stable secondary1530/877MHz,300W, actual TP4 M8/N4352/K5120,16MiB
cold-L2 graph ABBA, layers0/1 native48.128us versus canonical76.800us;
layer14 native52.224us versus78.848us. Payloads12,185,600/12,881,920
bytes give253.2/246.7GB/s. Primary fixed-clock comparison remains queued
and will be reported separately. Model admission is prepared only for the two measured
SM70 FP16 M8/N4352/K5120 gate/up orientations.

The complete installed model-wiring wheel passes capability, official
numerical and runtime-M graph checks on all three prepared real layers.
All16 shipped native libraries are bitwise identical to the qualified
prototype wheel. The affected focused CPU suite has64 passing checks.
No end-to-end result is claimed.
