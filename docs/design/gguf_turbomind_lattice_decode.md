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
| IQ2_XS | 4 | 1 | 23.97 | 11.42 | 15.26 |
| IQ2_XS | 4 | 2 | 23.03 | 11.72 | 15.39 |
| IQ2_XS | 4 | 4 | 23.01 | 11.98 | 16.08 |
| IQ2_XS | 4 | 8 | 23.18 | 17.87 | 18.31 |
| IQ2_XS | 4 | 16 | 23.54 | 29.70 | 16.28 |
| IQ2_XS | 4 | 32 | 25.64 | 53.91 | 17.40 |
| IQ2_XS | 4 | 64 | 27.80 | 101.73 | 21.67 |
| IQ2_XS | 4 | 128 | 31.92 | 197.48 | 27.96 |
| IQ3_XXS | 4 | 1 | 23.27 | 10.39 | 15.33 |
| IQ3_XXS | 4 | 2 | 20.39 | 10.85 | 15.44 |
| IQ3_XXS | 4 | 4 | 20.39 | 11.16 | 16.18 |
| IQ3_XXS | 4 | 8 | 20.53 | 16.64 | 18.13 |
| IQ3_XXS | 4 | 16 | 24.12 | 27.60 | 16.27 |
| IQ3_XXS | 4 | 32 | 24.84 | 49.66 | 17.42 |
| IQ3_XXS | 4 | 64 | 37.50 | 93.90 | 21.70 |
| IQ3_XXS | 4 | 128 | 27.97 | 182.32 | 27.96 |
| IQ2_XS | 512 | 1 | 45.98 | 18.94 | 19.79 |
| IQ2_XS | 512 | 2 | 56.68 | 19.00 | 25.51 |
| IQ2_XS | 512 | 4 | 59.55 | 31.23 | 26.94 |
| IQ2_XS | 512 | 8 | 65.42 | 27.80 | 28.71 |
| IQ2_XS | 512 | 16 | 69.42 | 23.40 | 33.27 |
| IQ2_XS | 512 | 32 | 77.72 | 33.23 | 57.80 |
| IQ2_XS | 512 | 64 | 87.52 | 46.64 | 63.69 |
| IQ2_XS | 512 | 128 | 105.11 | 71.01 | 72.31 |
| IQ3_XXS | 512 | 1 | 55.49 | 18.07 | 19.54 |
| IQ3_XXS | 512 | 2 | 66.32 | 17.87 | 25.10 |
| IQ3_XXS | 512 | 4 | 68.34 | 18.18 | 26.07 |
| IQ3_XXS | 512 | 8 | 74.20 | 18.43 | 28.14 |
| IQ3_XXS | 512 | 16 | 78.82 | 21.96 | 33.54 |
| IQ3_XXS | 512 | 32 | 88.20 | 32.10 | 56.61 |
| IQ3_XXS | 512 | 64 | 101.11 | 49.66 | 63.90 |
| IQ3_XXS | 512 | 128 | 136.51 | 77.62 | 72.23 |

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

All timings below use actual IQ3_XXS TP4 N=160/K=2560 expert matrices and
CUDA graph replay; times are microseconds. These points use 100 ms warmup and
20 iterations, with eight device invocations per graph replay.

| Experts | M | Selected | Selected us | Direct vector us | Grouped GEMM us | AWQ us |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 16 | GEMM | 24.63 | 27.75 | 24.00 | 16.22 |
| 512 | 128 | Vector | 78.23 | 78.18 | 135.91 | 72.37 |
| 512 | 512 | Vector | 264.24 | 264.50 | 307.40 | 184.63 |

The E4/M=8 selected route measured 311.35 us once, while the identical direct
vector call measured 16.54 us. This inconsistent point is excluded from route
calibration and awaits a targeted replay check. E512 selected and direct calls
agree, but M=512 remains about 43% slower than AWQ. Output relative L2 is
0.00040–0.00041. These projection results exclude routing and the full FFN;
model throughput and quality remain unmeasured.
