# Word copies for canonical lattice codebooks

Canonical lattice operators initialize an aligned shared byte table before
reading codebook rows as 32/64-bit words. Initialization previously used one
byte load/store per loop iteration. This change aligns source tables and copies
four unchanged bytes per iteration, preserving table order, values and size.
GEMM, grouped vector and canonical dequantization use the same initializer.

The weight representation, coefficients, FP16 activation/reconstruction and
FP32 accumulation are unchanged. No dispatch threshold, new weight allocation
or environment variable is added. Table alignment makes source word loads
explicitly valid; shared tables are already aligned by the kernel framework.

All seven formats must pass the canonical GPU oracle, grouped/empty expert,
graph replay and tracing checks. Matched IQ3_XXS TP4 expert timing will compare
M=1/128/512/8192 at E512, plus E4 M=8/16/8192, with the prior initializer and
native AWQ. The normal extension build and measurements are pending. The
previous packet-storage and native-tile experiments did not close the grouped
gap and remain excluded from this implementation.
