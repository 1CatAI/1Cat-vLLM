# Shared-activation GGUF output projections

The GDN output and full-attention output share TP4 M8/N5120/K1536.
The prototype reuses the measured single-matrix QPN body and existing lattice
and IQ4 readers. It screens eighty N64 CTAs against one hundred sixty CTAs
with two K partitions. The single-partition version writes its ordered FP32
intra-CTA reduction directly; the two-partition version retains the last-CTA
reduction. Neither starts a separate reduction kernel.

Original GDN weights retain GGUF's three-head tiled K order. Each 128-wide
activation segment is mapped to its vLLM head order while filling shared A.
No activation permutation kernel, decoded weight cache, new coefficient
rounding or second reader implementation is introduced. Attention output
uses the same kernel with ordinary input order. FP16 operands and outputs,
original FP32 scale products and FP32 accumulation are preserved.

The benchmark selects one actual TP4 shard for every supported role/type
combination. It uses the adapter's existing GDN shard/layout logic and
compares both partition counts with official FP32 GGUF reconstruction,
three seeded inputs, one thousand graph replays and same-clock cold-L2 ABBA.
Canonical input layout restoration is recorded, and any required canonical
input permutation is included in the comparator. Affine formats and
unmeasured model shapes remain canonical. Compile and GPU results are pending;
this prototype does not change model dispatch.
