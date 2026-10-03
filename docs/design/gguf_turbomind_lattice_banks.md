# Replicated IQ3 shared-codebook candidate

The canonical IQ3_XXS decoder reads two four-byte shared codebook rows per
fragment. This candidate interleaves eight identical copies of each row and
selects a replica using the low three lane bits. The physical shared address
separates lane groups across banks; indices, signs, scales and reconstructed
values remain identical. Initialization retains aligned word copies and reads
the original row for every replica.

IQ3_XXS shared codebook storage grows from 1 KiB to 8 KiB. GPU weight storage
is unchanged. IQ3_S and other formats retain their original tables. The change
uses the same decoder in GEMM, grouped vector and dequantization, with FP16
activations/weights and FP32 accumulation. Descriptor and cache keys remain
unchanged because each existing kernel has one decoder implementation.

This is a measurement candidate. Runtime tracing places the IQ3 grouped gap
in the mainloop, but hardware bank-conflict counters are unavailable. Reduced
shared-bank contention is a hypothesis, not an established runtime cause.
Extra shared storage or address registers can reduce occupancy and must be
measured. All lattice GPU oracle/capture/tracing checks and matched real-weight
E512/E4 timings are required before keeping or rejecting the implementation.
