# Original Q4_K reader for shared-activation pairs

The next three IQ4_XS/Q4_K mixed pairs reuse the existing M8/N32 shared-A
kernel with an independent original-byte affine reader. The four K128
packet planes and original K256 metadata come from `gguf_q4_k_records.py`.
Metadata is cached across both halves, including odd split-K starts.

The reader restores the original six-bit scale/min fields and nibble order.
It multiplies original FP16 d/dmin and integer coefficients in FP32, then
converts only the final operand to FP16. For finite FP16 metadata,
`d * scale * q` has at most 21 significant bits and is exact in FP32;
FMA therefore preserves the official separate multiply/subtract rounding.
Dot products, split-K reduction and the gated epilogue are unchanged.

The original field interpretation follows llama.cpp's `get_scale_min_k4`;
the existing MIT license remains in the vendored source. No llama.cpp GEMM
or scheduling kernel is added. The raw operator compiles both orientations,
while model capability admission remains unchanged until actual-weight
numerical and cold-L2 graph comparisons pass.

An independent CPU cursor oracle covers all eight split-K starts, both
N32 tiles, all columns and sixteen fragments per K128 record at K5120.
It checks cached original metadata, scales/mins and unpacked nibbles.
Normal build, device-operand, graph and latency checks remain pending.
