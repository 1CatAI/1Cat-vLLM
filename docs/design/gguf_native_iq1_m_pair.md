# Original IQ1_M shared-activation pair reader

Layer 13 pairs IQ1_M gate with IQ2_S up at TP4 N4352/K5120. Retain all
56 bytes per K256 block: aligned index/high packets are followed by the
original four scale words. Reconstruct the original FP16 d from their
four high nibbles once per block, including odd split-K starts. Preserve
all original bits, three-bit K16 subscales and per-octet delta signs.

Use the existing llama.cpp-derived 2048-entry IQ1 lattice codebook and
its retained MIT attribution. Decode d, subscale and grid/delta in FP32
before the final FP16 operand conversion. The shared-A skeleton, FP32
MMA accumulation/reduction and fused epilogue remain unchanged. Only
IQ1_M/IQ2_S enters raw dispatch; model admission remains canonical.

Ten CPU storage/cursor checks pass, including all eight split-K starts.
A real N64/K5120 gate sample preserves 71,680 original bytes and 327,680
independently decoded official FP32 values bitwise. All 2048 existing
codebook entries match the official reader.

Normal CUDA build and installed GPU numerical/graph/speed comparisons
are pending.
No end-to-end result is claimed.
