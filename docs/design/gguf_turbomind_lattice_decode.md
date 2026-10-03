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

The kernel framework enables vector decode only for measured descriptors and
M intervals. Other descriptors retain grouped GEMM with an explicit admission
reason. Tests cover all formats, empty/distinct experts, FP32 reference output,
CUDA graph replay and full-graph tracing. Static compilation uses 39 registers
for IQ2/3 and 48 for IQ1, with no stack/local memory reported; those counts are
not runtime performance evidence.

## Initial real-weight comparison

V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128. Actual Flash-Next TP4
expert N=160/K=2560, all 512 distinct experts. M counts sorted expert rows;
routing and the full FFN are excluded. All routes use 100 ms warmup and
20 iterations with graph replay. Times are microseconds.

| Type | M | Grouped GEMM | Grouped vector | AWQ |
| --- | --- | --- | --- | --- |
| IQ2_XS | 128 | 105.42 | 72.86 | 72.40 |
| IQ2_XS | 512 | 276.79 | 240.49 | 184.84 |
| IQ3_XXS | 128 | 134.78 | 78.39 | 72.49 |
| IQ3_XXS | 512 | 307.71 | 266.09 | 183.77 |

At M=128, vector decode nearly matches AWQ for IQ2_XS and is about 8%
slower for IQ3_XXS. M=512 retains a material gap. A full small-M sweep and
smaller expert counts are required before declaring default capability bands.

## Measured small-row dispatch

The kernel framework declares source format, local N/K, expert count and M
intervals. IQ2_XS/IQ3_XXS at K=2560/N=160 use vector decode for four experts
and M=1–8, or 512 experts and M=1–128/M=512. The unmeasured M=129–511
interval retains grouped GEMM. Other expert counts and source formats report
`grouped_vector_shape_has_no_calibration`. Missing operators, disabled policy,
unsupported activation dtype and local packing have explicit reasons. No new
environment variable is used.

| Type | Experts | M | GEMM us | Vector us | AWQ us |
| --- | --- | --- | --- | --- | --- |
| IQ2_XS | 4 | 1 | 23.89 | 10.70 | 15.20 |
| IQ2_XS | 4 | 2 | 23.00 | 11.02 | 15.30 |
| IQ2_XS | 4 | 4 | 22.97 | 11.23 | 16.01 |
| IQ2_XS | 4 | 8 | 23.18 | 17.14 | 18.46 |
| IQ2_XS | 4 | 16 | 23.47 | 28.99 | 16.19 |
| IQ2_XS | 4 | 32 | 25.51 | 53.26 | 17.39 |
| IQ2_XS | 4 | 64 | 27.67 | 101.03 | 21.60 |
| IQ2_XS | 4 | 128 | 31.93 | 196.76 | 28.00 |
| IQ3_XXS | 4 | 1 | 21.10 | 9.74 | 15.18 |
| IQ3_XXS | 4 | 2 | 20.28 | 10.18 | 15.31 |
| IQ3_XXS | 4 | 4 | 20.33 | 10.40 | 16.01 |
| IQ3_XXS | 4 | 8 | 20.50 | 15.84 | 18.45 |
| IQ3_XXS | 4 | 16 | 23.91 | 26.96 | 16.19 |
| IQ3_XXS | 4 | 32 | 24.71 | 48.90 | 17.36 |
| IQ3_XXS | 4 | 64 | 37.41 | 93.29 | 21.50 |
| IQ3_XXS | 4 | 128 | 27.90 | 181.69 | 27.99 |
| IQ2_XS | 512 | 1 | 45.79 | 18.40 | 19.54 |
| IQ2_XS | 512 | 2 | 56.32 | 18.28 | 28.69 |
| IQ2_XS | 512 | 4 | 59.62 | 18.37 | 26.48 |
| IQ2_XS | 512 | 8 | 64.93 | 18.71 | 28.66 |
| IQ2_XS | 512 | 16 | 69.10 | 21.49 | 33.43 |
| IQ2_XS | 512 | 32 | 77.61 | 30.61 | 57.33 |
| IQ2_XS | 512 | 64 | 87.17 | 46.33 | 63.82 |
| IQ2_XS | 512 | 128 | 105.28 | 70.54 | 72.34 |
| IQ2_XS | 512 | 512 | 276.61 | 240.73 | 185.42 |
| IQ3_XXS | 512 | 1 | 55.91 | 17.38 | 19.51 |
| IQ3_XXS | 512 | 2 | 66.09 | 17.30 | 25.10 |
| IQ3_XXS | 512 | 4 | 68.39 | 17.59 | 26.41 |
| IQ3_XXS | 512 | 8 | 74.23 | 18.04 | 28.63 |
| IQ3_XXS | 512 | 16 | 78.05 | 21.15 | 33.08 |
| IQ3_XXS | 512 | 32 | 87.25 | 31.56 | 56.85 |
| IQ3_XXS | 512 | 64 | 101.36 | 49.17 | 64.04 |
| IQ3_XXS | 512 | 128 | 135.19 | 77.06 | 72.27 |
| IQ3_XXS | 512 | 512 | 311.31 | 264.42 | 186.23 |

Four experts have a clear crossover: vector decode wins through M=8 but
repeats weight work as each expert receives more rows. With 512 experts,
empty or single-row experts benefit throughout the measured small-M range.
These distributions exclude routing and do not establish model throughput.
The ordinary installed wheel passes 104 GPU checks with one non-SM70 skip,
covering affine, bitplane, LUT and lattice operators, exact canonical
dequantization, graph replay and capability selection. Source, packaged and
installed core SHA256 values match, with no RPATH/RUNPATH; dependency checking
passes for all 210 installed packages.

## Installed selected-route measurements

The fresh environment imports vLLM from its installed wheel, outside the source
tree, using Python 3.12.3, Torch 2.10.0+cu128 and CUDA 12.8.
Wheel SHA256:
`bbdaa23d631a87b7f991a3a6cb15d38b9e90b1d615b0e5a7294f6084678ff3da`.
Core SHA256:
`913608cd66cf22ea9f670b046d3f582dab5317cf907dae5c28c52ac579f6f5bc`.

The corrected sweep uses actual TP4 N=160/K=2560 expert matrices, 100 ms
warmup, 100 iterations and eight device invocations per CUDA graph replay for
every small-output route. All earlier small-row measurements were rechecked
under this common capture contract; the table above reports the corrected
sweep. The measured default intervals are unchanged. Times are microseconds.

| Type | Experts | M | Selected | Selected us | Direct vector us | GEMM us | AWQ us |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IQ3_XXS | 4 | 8 | Vector | 15.85 | 15.84 | 20.50 | 18.45 |
| IQ3_XXS | 4 | 16 | GEMM | 23.98 | 26.96 | 23.91 | 16.19 |
| IQ2_XS | 512 | 128 | Vector | 70.50 | 70.54 | 105.28 | 72.34 |
| IQ3_XXS | 512 | 128 | Vector | 76.97 | 77.06 | 135.19 | 72.27 |
| IQ2_XS | 512 | 512 | Vector | 240.42 | 240.73 | 276.61 | 185.42 |
| IQ3_XXS | 512 | 512 | Vector | 264.50 | 264.42 | 311.31 | 186.23 |

The first E4/M=8 selected measurement was 311.35 us versus 16.54 us for the
identical direct call. Both out-operator callables returned `None`, causing
single-invocation capture while the GEMM wrapper exposed its output and
captured eight invocations. Returning the output buffer consistently removes
this measurement mismatch. The corrected selected/direct values agree; the
initial inconsistent point is excluded from calibration.

At E512/M=512, IQ2_XS remains about 30% slower than AWQ and IQ3_XXS about
42% slower. Output relative L2 remains 0.00039–0.00041. These projection
results exclude routing and the full FFN; model throughput and quality remain
unmeasured. No activation quantization or lower-precision accumulation was
introduced.
