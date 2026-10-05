# Original IQ2_XS shared-activation pair reader

Layers11/16 use IQ2_XS gate and IQ3_XXS up; layer12 uses IQ2_S gate and
IQ2_XS up. All are TP4 N4352/K5120. Preserve the original74-byte K256
blocks as two aligned16-byte index/sign packets and four scale bytes per
K128, followed by original d once per K256. No field expands.

Use the existing IQ2_XS codebook and reconstruct the original9-bit index
and7-bit sign index, with parity restoring the eighth sign. The coefficient
formula matches IQ2_S, so reuse its exact final FP16 operand formation.
FP32 dot/reduction, shared-A skeleton and gated epilogue are unchanged.
Raw operators support only the two actual orientations; model admission
stays canonical until numerical and matched speed checks pass.

Eleven CPU storage/cursor/operand checks pass. Every finite original FP16 d,
every scale, all codebook integers and both signs produce bitwise identical
final operands to the original FP32 formula:6,094,848 values, including
subnormals and final-operand overflow. Three actual N64/K5120 samples preserve
284,160 source bytes and983,040 official FP32 dequantized values bitwise.
The existing llama.cpp MIT attribution for the codebook is retained.

Normal build, actual-weight GPU numerical/graph/cold-L2 speed checks are
pending. No end-to-end result is claimed.

The first GPU check rejected the new74-byte format in host length
validation before native GEMM. Host validation now uses each reader
format constant rather than a duplicate length table; the initial failure
is retained. The repaired normal SM70 extension and whole wheel pass.
A focused layer11 official numerical/runtime-M graph check passes, then
three actual layers pass numerical, fallback and graph checks.

At stable secondary1530/877MHz,300W, actual TP4 M8/N4352/K5120,16MiB
cold-L2 graph ABBA, layers11/16 IQ2_XS/IQ3_XXS native50.176/51.200us
versus canonical77.824us; layer12 IQ2_S/IQ2_XS native51.200–52.224us
versus82.944us. Payload14,970,880 bytes gives292.4–298.4GB/s for
IQ2_XS/IQ3_XXS; payload13,578,240 gives260.0–265.2GB/s for IQ2_S/IQ2_XS.
Native relative L2 is0.000491–0.000525. Primary fixed-clock comparison
is pending before merge. Model admission is prepared only for these two
measured SM70 FP16 M8/N4352/K5120 orientations.
