# Original Q2_K shared-activation pair reader

Layer 28 pairs Q2_K gate with IQ3_S up at TP4 N4352/K5120. Keep each
84-byte K256 block unchanged in size: aligned two-bit packets and original
scale/min nibbles precede the original FP16 d/dmin plane. No expanded
canonical weights or rounded metadata are introduced.

The reader increments aligned packet pointers, caches d/dmin once per
K256 block, forms both scale levels in FP32, and converts the decoded
weight once to the existing FP16 MMA operand. The shared-A skeleton,
FP32 dot/reduction and fused gated epilogue are unchanged. Only the actual
Q2_K/IQ3_S orientation enters raw operator dispatch. Model selection stays
canonical until actual-weight numerical, graph and matched speed checks
pass. The block definitions follow llama.cpp; its existing MIT license and
attribution are retained.

Ten CPU inverse-layout and split-K cursor checks pass, including odd
K128 starts for all eight splits. A real N64/K5120 gate sample preserves
107,520 original bytes and 327,680 official FP32 dequantized values
bitwise (maximum absolute error and relative L2 both zero).

CUDA build and actual-weight GPU validation are pending. No speedup or
end-to-end result is claimed.
