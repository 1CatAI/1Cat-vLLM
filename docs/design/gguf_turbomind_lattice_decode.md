# Canonical lattice grouped decode

Large expert counts often give each active expert only one or a few sorted
rows. The existing grouped mma884 schedule pads these rows into a batch tile.
This operator prototype instead performs warp-partitioned dot products from
the same canonical packed weights and metadata. It retains FP16 activations
and FP32 products, local accumulators and reductions.

All seven lattice formats use `LatticeCanonicalDecoder`, shared with canonical
dequantization. This preserves codebook indices, signs, IQ1 deltas and expanded
FP16 coefficients. A block initializes one codebook, handles sixteen output
columns, and partitions K among sixteen thread groups. Empty experts return
before table initialization. Offset/pointer inputs stay on the GPU for capture.

The initial operator is a measurement candidate. Existing grouped GEMM remains
the default until real TP4 timing establishes a useful interval. Tests cover
all formats, empty/distinct experts, FP32 reference output, CUDA graph replay
and full-graph tracing. Correctness and speed results are pending compilation.
