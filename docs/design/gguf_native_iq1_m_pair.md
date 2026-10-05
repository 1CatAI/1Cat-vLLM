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
IQ1_M/IQ2_S enters raw dispatch; model capability requires the measured
SM70 FP16 M8/N4352/K5120 shape.

Ten CPU storage/cursor checks pass, including all eight split-K starts.
A real N64/K5120 gate sample preserves 71,680 original bytes and 327,680
independently decoded official FP32 values bitwise. All 2048 existing
codebook entries match the official reader.

The complete CUDA12.8 SM70 wheel passes its installed native audit.
This pair uses64 registers,41,984 shared bytes and zero stack/local
memory. All real-weight official FP32 GEMM comparisons pass, relative
L2 0.000515–0.000522, as do runtime M512/8/1/5/16/20/32/8 and graph
checks; non-M8 remains bitwise canonical.

At stable secondary1530/877MHz,300W,16MiB cold-L2 graph ABBA, layer13
native64.512us versus canonical90.112us. Original12,011,520-byte payload
gives186.2GB/s versus133.3GB/s. The isolated single-layer saving is
25.600us; no end-to-end result is claimed. Admit only the measured
SM70 FP16 M8/N4352/K5120 orientation. The installed prepared-layer capability, official numerical and all
runtime-M graph checks pass. All16 native libraries are bitwise identical
to the qualified prototype wheel. Fixed1290MHz comparison remains queued
and will be reported separately.
