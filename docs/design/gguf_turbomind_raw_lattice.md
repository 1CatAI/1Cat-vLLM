# Original GGUF lattice blocks on SM70

## Storage and arithmetic contract

The first direct-block operators cover IQ3_S and IQ2_S. Persistent GPU
weights contain the original GGUF row bytes and zero to seven trailing
alignment bytes per row. Scales, signs, grid indices, and high index bits
remain in their original positions. There is no canonical weight cache.

The source layouts and formulas follow llama.cpp commit
`bed0a856606ee4a24a164066f73d2379447033f5`. Its MIT codebook attribution is
retained in `lattice_codebooks.h`. IQ3_S has 110 bytes per 256 values;
IQ2_S has 82 bytes per 256 values.

`RawGGUFProjection` accepts mmap-backed packed rows and slices TP shards by
complete source blocks. N slicing preserves every selected row. K slicing
requires a boundary divisible by 256. A boundary that cuts a source block is
rejected before transfer; this initial operator layer does not change the TP
layout or silently decode a full expert into persistent FP16 storage.

The FP16 block scale and original small-scale bits are multiplied in FP32
inside the decoder. FP32 dequantization is checked against the official
`gguf.quants` implementation. Vector dot products use FP32 FMA. MMA operands
and the temporary cuBLAS workspace are rounded to FP16 after reconstruction;
MMA and cuBLAS accumulate in FP32. cuBLAS explicitly disables reduced
precision reductions.

## Operators

| M | Initial candidate | Temporary storage |
| --- | --- | --- |
| 1 | Warp vector dot product, optional split-K | FP32 split partials |
| 2–64 | Original block staging and Volta m8n8k4 MMA | Shared blocks and FP32 split partials |
| 512 and above | Direct dequantization into transposed FP16 workspace, cuBLAS | FP16 workspace |

Each warp loads consecutive 64-bit words from a row. An unaligned block
start is aligned down and stitched in shared memory. Neighboring blocks can
therefore request an overlapping aligned word; the physical DRAM traffic
must be measured rather than inferred from storage size. Shared codebooks
are initialized per CTA. M8/16 tiles and N8/32 tiles are separate candidates.
Split-K partitions whole GGUF blocks and reduces partials in FP32.

The operators are registered in the normal `_C` extension. Capability
records distinguish source type and M interval. Model dispatch is not
switched until correctness and speed gates pass. M65–511 is not covered by
the initial capability records.

## Validation and performance status

All 12 focused checks pass, including official dequantization, split-K,
N tails, and changed-input CUDA graph replay. The four real projections
also have zero FP32 dequantization error. Across all tested candidates the
maximum output relative L2 error is 0.000217, including the final FP16 output
rounding. No model default is switched by these operator results.
The benchmark uses complete CUDA graph capture and replay for both raw and
canonical candidates; uncaptured calls only initialize kernels and handles.
Graph timings cycle distinct copies of the same projection, with an
automatic bank count exceeding twice V100 L2 for both layouts. The report
records this count; graph time is divided by the number of projections.
This avoids measuring only cache-resident small experts. Counter capture
evicts L2 with a 64 MiB fill before one selected projection replay; the
eviction is outside the profiled region. These remain isolated projection
results, rather than an expert-routing or model latency measurement.
It compares the existing canonical dispatcher, including its measured
shape-specific dequantization/cuBLAS bands.

The first real TP4 shapes are:

| Model / tensor | Type | N | K | Payload bytes/row | Alignment bytes/row |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B gate | IQ3_S | 4352 | 5120 | 2200 | 0 |
| Qwen3.8-27B down | IQ3_S | 5120 | 4352 | 1870 | 2 |
| Flash-Next expert gate | IQ3_S | 160 | 2560 | 1100 | 4 |
| Flash-Next expert gate | IQ2_S | 160 | 2560 | 820 | 4 |

M values are 1, 5, 8, 16, and 512. An individual expert projection is not a
MoE grouped or end-to-end result. Counter reports must distinguish original
weight payload, alignment, codebook initialization, activations, output,
and temporary workspace traffic. Estimated per-step savings must state the
projection multiplicities and remain separate from measured model latency.

## First real-shape graph results

Environment: V100-SXM2-32GB, 80 SM, CUDA 12.8, Torch 2.10.0+cu128,
driver 580.173.02. Installed wheel source `090c05b83b`; core SHA256
`28b1f2a19dfa8014e70cdc5dc0ec879e86c8afc6bcf69d54a48656f4ce1235e8`.
Benchmark source `0e1bea4402`. Activation/output FP16, accumulation FP32,
TP4 rank-zero projection shapes, complete CUDA graph replay. There are 100
replays with eight bank sweeps per replay. Independent addresses cover more
than twice L2: 2 banks for each dense shape, 72 IQ3_S expert banks, and 96
IQ2_S expert banks. Values below are microseconds per projection.

| Projection | M | Canonical | Raw best | Raw/canonical | Raw candidate |
| --- | ---: | ---: | ---: | ---: | --- |
| 27B gate IQ3_S | 1 | 31.895 | 48.296 | 1.514 | vec_split1 |
| 27B gate IQ3_S | 5 | 34.712 | 77.810 | 2.242 | mma_n32_split3 |
| 27B gate IQ3_S | 8 | 34.668 | 97.891 | 2.824 | mma_n32_split3 |
| 27B gate IQ3_S | 16 | 39.555 | 159.698 | 4.037 | mma_n32_split3 |
| 27B gate IQ3_S | 512 | 368.323 | 394.594 | 1.071 | raw_dequant_cublas |
| 27B down IQ3_S | 1 | 33.135 | 43.788 | 1.321 | vec_split1 |
| 27B down IQ3_S | 5 | 31.818 | 82.158 | 2.582 | mma_n32_split2 |
| 27B down IQ3_S | 8 | 32.001 | 92.973 | 2.905 | mma_n32_split2 |
| 27B down IQ3_S | 16 | 37.841 | 139.717 | 3.692 | mma_n32_split2 |
| 27B down IQ3_S | 512 | 340.628 | 369.133 | 1.084 | raw_dequant_cublas |
| Flash expert IQ3_S | 1 | 16.262 | 7.299 | 0.449 | vec_split8 |
| Flash expert IQ3_S | 5 | 17.358 | 11.515 | 0.663 | mma_n8_split10 |
| Flash expert IQ3_S | 8 | 17.614 | 12.909 | 0.733 | mma_n8_split10 |
| Flash expert IQ3_S | 16 | 17.287 | 14.521 | 0.840 | mma_n32_split10 |
| Flash expert IQ3_S | 512 | 69.002 | 31.736 | 0.460 | raw_dequant_cublas |
| Flash expert IQ2_S | 1 | 18.161 | 9.704 | 0.534 | vec_split8 |
| Flash expert IQ2_S | 5 | 19.038 | 14.294 | 0.751 | mma_n8_split10 |
| Flash expert IQ2_S | 8 | 19.352 | 15.181 | 0.784 | mma_n32_split10 |
| Flash expert IQ2_S | 16 | 19.019 | 16.365 | 0.860 | mma_n32_split10 |
| Flash expert IQ2_S | 512 | 64.060 | 34.041 | 0.531 | raw_dequant_cublas |

The small expert projections improve at every tested M. The dense projections
fail the speed gate: vector decode is slower, small-M MMA loses substantially,
and dequantization/cuBLAS loses about 7–8%. Neither a universal raw default
nor removal of canonical model storage is justified. Next collect DRAM and
stall counters, then evaluate load pipelining and the equal-byte GPU reorder
fallback for dense shapes. No end-to-end step saving is claimed by this table.
