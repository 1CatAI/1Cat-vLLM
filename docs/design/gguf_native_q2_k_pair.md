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

The complete CUDA12.8 SM70 wheel passes its installed native audit.
The Q2_K/IQ3_S kernel uses64 registers,33,792 shared bytes and zero
stack/local memory. Official real-weight FP32 GEMM checks pass with
relative L2 0.000505–0.000514, as do runtime M512/8/1/5/16/20/32/8
graph checks; non-M8 remains bitwise canonical.

At stable secondary1530/877MHz,300W,16MiB cold-L2 graph ABBA, layer28
native58.368us versus canonical73.728us. Original16,885,760-byte payload
gives289.3GB/s versus229.0GB/s. The isolated saving is15.360us for this
single layer; it is not an end-to-end measurement. Enable model capability
only for the measured SM70 FP16 M8/N4352/K5120 orientation. Installed
prepared-route checks and fixed1290MHz comparison are pending.
