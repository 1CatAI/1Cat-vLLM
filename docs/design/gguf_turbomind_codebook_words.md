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
native AWQ. The normal extension build and all 40 focused GPU checks pass.
Source-table payload hashes match the six original byte arrays; IQ1_M retains
its IQ1_S table alias. The previous packet-storage and native-tile experiments
did not close the grouped gap and remain excluded from this implementation.

## Real-weight source timing

V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128, actual Flash-Next
IQ3_XXS TP4 N=160/K=2560, 100 ms warmup and 100 timing iterations.
All small-output CUDA graphs contain eight device invocations per replay.
M counts sorted expert rows; routing and the full FFN are excluded.

| Experts | M | Prior selected us | Word-copy selected us | Word-copy GEMM us | AWQ us |
| --- | --- | --- | --- | --- | --- |
| 512 | 1 | 17.39 | 16.88 | 52.62 | 19.52 |
| 512 | 128 | 76.97 | 76.16 | 134.55 | 72.51 |
| 512 | 512 | 264.50 | 260.40 | 303.54 | 186.40 |
| 512 | 8192 | 562.14 | 554.80 | 555.25 | 351.00 |
| 4 | 8 | 15.85 | 15.29 | 21.08 | 18.40 |
| 4 | 16 | 23.98 | 23.08 | 22.95 | 16.18 |
| 4 | 8192 | 299.86 | 296.54 | 296.43 | 227.77 |

This reduces initialization work and provides a modest operator improvement;
it does not close the grouped large-M gap. Output relative L2 remains
0.00039–0.00041. Source benchmark admission verifies the package origin and
loaded normal core digest before reading the checkpoint. The measured core
SHA256 is
`4910c47ab1aaed253001d5950bf44dd40a350b2b087202a8ea2b13f2c5457782`.
Ordinary installed-wheel checks remain pending.
