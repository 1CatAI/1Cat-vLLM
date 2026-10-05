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

All 20 focused checks pass, including official dequantization, split-K,
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

## Equal-byte MMA permutation candidate

The dense speed failure motivates a GPU-only bit permutation. IQ3_S encodes
each eight K values as two nine-bit grid indices and eight signs (26 bits).
IQ2_S uses one ten-bit grid index and eight signs (18 bits). Thirty-two octets
therefore consume 104/72 bytes per source row block. Adding the original
FP16 d and 4/8 small-scale bytes retains exactly 110/82 bytes per 256 weights.
A 32-column macro-tile places these tightly packed packets before the d and
small-scale planes. The final N macro-tile uses its actual column count and
a continuous packet bitstream. Only the entire buffer receives up to seven
trailing alignment bytes.

GPU reordering consumes the original padded rows and writes that equal-byte
layout. A production loader would release the source buffer afterwards; the
comparison benchmark keeps separate candidate buffers for measurement. No
expanded scale or index cache belongs to the format. A warp reads 13/9
consecutive uint64 packet words per K octet and uses shuffle to extract its
26/18-bit packet. Original scales are loaded once per 256-value block and
multiplied in FP32. This removes per-block raw staging barriers from MMA.
The new operators are compiled in the same normal extension. The numerical,
bit-for-bit inverse and real-shape speed results follow below.

The first counter experiment exposed a harness issue: its canonical graph
captured a 32 MiB workspace initialization because warmup used a different
stream. Ordinary graph timings already warmed their actual capture stream.
Counter capture now also warms its explicit stream and pins the unprofiled
winning candidate. The earlier counter artifacts are retained as rejected
comparisons; their initialization writes are not application weight traffic.

## Vectorized activation loads

Source `3d89c06570` replaces eight scalar activation loads per fragment with
one aligned uint4 load, without changing the arithmetic. All 12 original
checks pass again. Whole-wheel SHA256:
`2008ed9c89af31501163ba5a5f548f8bd684ae403d5f5a7a10db2de0a78aa559`;
core SHA256:
`f5ba280ffdae2bac646120f3f4be0ba8800b23dec83f6daadafa0b8c1b3138e6`.
The graph/bank/environment contract is the same as the first table.

| Projection | M | Canonical µs | Raw best µs | Raw/canonical |
| --- | ---: | ---: | ---: | ---: |
| 27B gate IQ3_S | 1 | 31.809 | 40.883 | 1.285 |
| 27B gate IQ3_S | 5 | 32.932 | 66.426 | 2.017 |
| 27B gate IQ3_S | 8 | 33.206 | 68.804 | 2.072 |
| 27B gate IQ3_S | 16 | 39.823 | 84.152 | 2.113 |
| 27B gate IQ3_S | 512 | 366.879 | 393.140 | 1.072 |
| 27B down IQ3_S | 1 | 33.245 | 37.027 | 1.114 |
| 27B down IQ3_S | 5 | 31.674 | 72.490 | 2.289 |
| 27B down IQ3_S | 8 | 31.713 | 74.223 | 2.340 |
| 27B down IQ3_S | 16 | 38.063 | 90.863 | 2.387 |
| 27B down IQ3_S | 512 | 340.157 | 368.266 | 1.083 |
| Flash expert IQ3_S | 1 | 16.317 | 7.019 | 0.430 |
| Flash expert IQ3_S | 5 | 17.623 | 10.651 | 0.604 |
| Flash expert IQ3_S | 8 | 17.613 | 10.822 | 0.614 |
| Flash expert IQ3_S | 16 | 18.102 | 12.118 | 0.669 |
| Flash expert IQ3_S | 512 | 69.008 | 31.937 | 0.463 |
| Flash expert IQ2_S | 1 | 18.168 | 9.619 | 0.529 |
| Flash expert IQ2_S | 5 | 19.316 | 13.160 | 0.681 |
| Flash expert IQ2_S | 8 | 19.375 | 13.350 | 0.689 |
| Flash expert IQ2_S | 16 | 19.649 | 14.123 | 0.719 |
| Flash expert IQ2_S | 512 | 64.062 | 34.175 | 0.533 |

Vectorization helps dense decode/MMA, but dense MMA still loses about 2–2.4x
and prefill still loses 7–8%. The equal-byte permutation remains necessary
to evaluate; no default or storage-removal decision is made from these data.

## Equal-byte permutation measurements

Source `92d466ae58` (core
`fe0a70f7c0f21cfc65e95ccea7c5035a92833ce523c25e68a0a9bfbfad7821e7`)
passes all 20 focused checks, including independent reconstruction of every
source bit, exact FP32 dequantization and changed-input graph replay. Maximum
output relative L2 error is 0.000217. Dense M512 matches canonical; vector
and small-M dense MMA still fail the speed gate.

Source `89eb838d72` (core
`a1b1c877e3876634aa4334a009897f5d935f81f14697cfa9b95d0491a2e716cb`)
specializes complete N32 tiles and adds warp-parallel K vector decode. All
20 checks pass again (5.91 s). The same graph/bank contract now includes two
Flash-Next dense projections. Times are µs/projection; GPU permutation runs
once before timing.

| Projection (N×K) | M | Canonical | Best equal-byte | Candidate |
| --- | ---: | ---: | ---: | --- |
| 27B gate (4352×5120) | 1 | 31.925 | 38.378 | compact_row_vec_split1 |
| 27B gate (4352×5120) | 5 | 36.200 | 53.871 | compact_mma_split3 |
| 27B gate (4352×5120) | 8 | 33.274 | 55.404 | compact_mma_split3 |
| 27B gate (4352×5120) | 16 | 42.205 | 103.585 | compact_mma_split3 |
| 27B gate (4352×5120) | 512 | 364.116 | 363.092 | compact_dequant_cublas |
| 27B down (5120×4352) | 1 | 30.937 | 37.298 | compact_row_vec_split1 |
| 27B down (5120×4352) | 5 | 31.836 | 59.372 | compact_mma_split2 |
| 27B down (5120×4352) | 8 | 31.998 | 60.600 | compact_mma_split2 |
| 27B down (5120×4352) | 16 | 37.978 | 76.327 | compact_mma_split2 |
| 27B down (5120×4352) | 512 | 341.079 | 339.048 | compact_dequant_cublas |
| IQ3_S expert (160×2560) | 1 | 16.222 | 7.019 | compact_row_vec_split8 |
| IQ3_S expert (160×2560) | 5 | 17.472 | 10.801 | compact_mma_split10 |
| IQ3_S expert (160×2560) | 8 | 17.567 | 10.894 | compact_mma_split10 |
| IQ3_S expert (160×2560) | 16 | 17.263 | 12.218 | compact_mma_split10 |
| IQ3_S expert (160×2560) | 512 | 68.837 | 27.321 | compact_dequant_cublas |
| IQ2_S expert (160×2560) | 1 | 18.460 | 9.612 | compact_row_vec_split8 |
| IQ2_S expert (160×2560) | 5 | 19.388 | 12.939 | compact_mma_split10 |
| IQ2_S expert (160×2560) | 8 | 19.444 | 13.090 | compact_mma_split10 |
| IQ2_S expert (160×2560) | 16 | 19.438 | 12.016 | compact_mma_split10 |
| IQ2_S expert (160×2560) | 512 | 64.236 | 29.600 | compact_dequant_cublas |
| Flash gate (1536×2560) | 1 | 17.945 | 11.250 | compact_row_vec_split1 |
| Flash gate (1536×2560) | 5 | 18.331 | 17.845 | compact_mma_split7 |
| Flash gate (1536×2560) | 8 | 18.479 | 18.060 | compact_mma_split7 |
| Flash gate (1536×2560) | 16 | 20.209 | 22.179 | compact_mma_split7 |
| Flash gate (1536×2560) | 512 | 89.881 | 126.663 | compact_dequant_cublas |
| Flash output (2560×1536) | 1 | 14.969 | 9.902 | compact_row_vec_split1 |
| Flash output (2560×1536) | 5 | 15.617 | 16.782 | compact_mma_split4 |
| Flash output (2560×1536) | 8 | 15.848 | 17.024 | compact_mma_split4 |
| Flash output (2560×1536) | 16 | 17.301 | 20.812 | compact_mma_split4 |
| Flash output (2560×1536) | 512 | 81.837 | 87.971 | compact_dequant_cublas |

Flash gate M512 does not improve with the current dequantization/cuBLAS path.
No default or format is promoted across the remaining speed gaps.

## Counters and next optimization

Nsight Compute 2022.4 records original-row graph counters after warming the
actual capture stream and pinning the unprofiled candidate. Source
`92d466ae58`, M8, one projection, 64 MiB eviction outside the measured region:

| Shape | Canonical DRAM read bytes | Original-row DRAM read bytes |
| --- | ---: | ---: |
| 27B gate | 11,246,016 | 10,483,840 |
| IQ3_S expert | 268,224 | 246,912 |
| IQ2_S expert | 274,624 | 206,720 |

These are whole-graph counters including activations, codebooks and split
partials. Dirty output left in L2 can yield zero DRAM writes in the region.
Profiled durations are perturbed and do not replace unprofiled timings.

Node profiling of `89eb838d72` identifies the dense MMA bottleneck:

| Gate M | Equal-byte long-scoreboard stall | Canonical stall | Equal-byte active warps |
| --- | ---: | ---: | ---: |
| 8 | 65.42% | 12.90% | 30.05% |
| 16 | 67.60% | 12.94% | 28.67% |

The vector node has 63.99% active warps, 32.56% long-scoreboard stall and
19.25% math-pipe throttle. There are no register spills. The next candidate
separates packet fetch from extraction and prefetches the next packet window
and aligned activation fragment in registers before current decode/MMA.
It uses no cp.async, expanded scale or reduced-precision accumulation.

Static storage savings are 6,236,160 bytes per rank for the four 27B IQ3_S
projections, and 1,138,094,080 bytes per rank across Flash-Next IQ3_S/IQ2_S
tensors. For C1, ten experts per expert projection and one read of each touched
weight imply about 38,031,360 fewer weight bytes per rank, including dense
projections. This conditional estimate does not measure expert reuse,
collective overlap or end-to-end latency. Grouped and model measurements
are required before reporting a step speedup.

## Register prefetch measurements

Source `25573bc57d`, core SHA256
`e1cb495a03b29fdeffd6725ce0d8cc18810ea2de90d25a59ba79c308f3118f3c`,
whole-wheel SHA256
`3a1b2a1442d90e71615365bf21f31951c2d77f5ef58b4ffac38a78ba5fa2d803`.
All 20 focused checks pass again (6.40 s), including both prefetch choices,
N tails, split-K and changed-input graph replay. The matched six-shape graph
sweep below keeps the earlier cache/bank contract.

| Projection | M | Canonical µs | Best equal-byte µs | Candidate |
| --- | ---: | ---: | ---: | --- |
| 27B gate | 1 | 31.857 | 38.672 | compact_row_vec_split1 |
| 27B gate | 5 | 33.151 | 46.412 | compact_mma_prefetch_split3 |
| 27B gate | 8 | 33.191 | 47.525 | compact_mma_prefetch_split3 |
| 27B gate | 16 | 39.612 | 63.935 | compact_mma_prefetch_split3 |
| 27B gate | 512 | 367.236 | 366.102 | compact_dequant_cublas |
| 27B down | 1 | 30.917 | 37.236 | compact_row_vec_split1 |
| 27B down | 5 | 31.736 | 49.470 | compact_mma_prefetch_split2 |
| 27B down | 8 | 31.964 | 50.604 | compact_mma_prefetch_split2 |
| 27B down | 16 | 37.871 | 67.944 | compact_mma_prefetch_split2 |
| 27B down | 512 | 340.455 | 338.301 | compact_dequant_cublas |
| IQ3_S expert | 1 | 16.329 | 7.021 | compact_row_vec_split8 |
| IQ3_S expert | 5 | 17.706 | 9.803 | compact_mma_prefetch_split10 |
| IQ3_S expert | 8 | 17.600 | 9.741 | compact_mma_prefetch_split10 |
| IQ3_S expert | 16 | 17.803 | 10.714 | compact_mma_prefetch_split10 |
| IQ3_S expert | 512 | 68.832 | 27.337 | compact_dequant_cublas |
| IQ2_S expert | 1 | 18.528 | 9.607 | compact_row_vec_split8 |
| IQ2_S expert | 5 | 19.187 | 12.177 | compact_mma_prefetch_split10 |
| IQ2_S expert | 8 | 19.261 | 12.347 | compact_mma_prefetch_split10 |
| IQ2_S expert | 16 | 19.067 | 11.905 | compact_mma_split10 |
| IQ2_S expert | 512 | 64.144 | 29.617 | compact_dequant_cublas |
| Flash gate | 1 | 18.119 | 11.252 | compact_row_vec_split1 |
| Flash gate | 5 | 19.132 | 15.495 | compact_mma_prefetch_split7 |
| Flash gate | 8 | 18.498 | 15.763 | compact_mma_prefetch_split7 |
| Flash gate | 16 | 20.876 | 20.427 | compact_mma_prefetch_split7 |
| Flash gate | 512 | 89.973 | 126.816 | compact_dequant_cublas |
| Flash output | 1 | 16.455 | 9.894 | compact_row_vec_split1 |
| Flash output | 5 | 15.625 | 14.502 | compact_mma_prefetch_split4 |
| Flash output | 8 | 15.822 | 14.742 | compact_mma_prefetch_split4 |
| Flash output | 16 | 17.402 | 18.944 | compact_mma_prefetch_split4 |
| Flash output | 512 | 81.922 | 87.949 | compact_dequant_cublas |

Prefetch improves dense MMA, especially M16, but 27B dense M1/M5/M8/M16
still fail parity. Expert projections improve at all measured M. Flash gate
M512 and output M16/M512 remain gaps. Original-byte storage is proven; speed
parity across the required shapes is not. No model default or persistent
canonical storage removal is justified yet.

## Prefetch counters and rejected vector variant

Post-prefetch node counters use the same cold-cache capture contract and
`compact_mma_prefetch_split3`. Core SHA256 is
`ef268dee78f95e4e56dbe8510d6c257356ab83ea143d6199dcdf0c3928cf26fc`.
The additional vector change in this build does not alter these MMA kernels.
The counters measure the main MMA node, excluding the FP32 split reduction.

| 27B gate M | DRAM read bytes | Active warps | Registers/thread | Long scoreboard | Before prefetch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 9,674,720 | 30.10% | 56 | 48.17% | 65.42% |
| 16 | 9,758,432 | 30.48% | 72 | 62.81% | 67.60% |

Main-node reads are close to the 9,574,400-byte weight payload and remain
below canonical main-node reads. Remaining traffic includes activations and
codebooks; reduction-node traffic is additional. Prefetch reduces the stall
but leaves substantial global load latency, particularly at M16.

A separate next-block prefetch experiment for warp-parallel K vector decode
(`3a01698a25`) passes all 20 checks in 5.89 s but does not improve dense M1.
It is reverted rather than selected by default. Its matched full-graph M1
results are retained below. The existing original-row vector remains an
independent candidate.

| Projection | Canonical µs | Compact row-vector prefetch µs |
| --- | ---: | ---: |
| 27B gate | 33.775 | 40.539 |
| 27B down | 30.877 | 40.275 |
| IQ3_S expert | 16.233 | 6.952 |
| IQ2_S expert | 18.097 | 9.618 |
| Flash gate | 18.233 | 11.110 |
| Flash output | 15.288 | 10.220 |

The next MMA candidate stages a complete equal-byte packet block in shared
memory. It holds the next block's coalesced 64-bit words in registers while
computing the current block. Two CTA barriers protect reuse of shared packet
storage. This adds only temporary storage, retains the original d/scale
planes, and preserves FP32 reconstruction and accumulation. Full N32 tiles
are initially required; an unsupported N tail returns a specific reason.

## Shared packet staging measurements

Source `bb42559ab8`, core SHA256
`e5042c7779ecfda96bd6ebcab02a6eae64fea2a333a82e7f4dc0b868c30e7bd3`,
whole-wheel SHA256
`de20a18c209b0ed7f213bc3657f6ab331d4bd8e0ebb01a9391e4f589f2003a84`.
All 20 checks pass (6.17 s), including shared staging with split-K and
changed-input graph replay. The matched M5/8/16 sweep retains all original,
compact and prefetch candidates. Shared staging helps M16 dense shapes but
has no universal advantage. Times are µs/projection.

| Projection | M | Canonical | Best equal-byte | Candidate |
| --- | ---: | ---: | ---: | --- |
| 27B gate | 5 | 32.970 | 46.552 | compact_mma_prefetch_split3 |
| 27B gate | 8 | 33.385 | 47.491 | compact_mma_prefetch_split3 |
| 27B gate | 16 | 39.702 | 59.090 | compact_mma_staged_split3 |
| 27B down | 5 | 36.017 | 48.505 | compact_mma_staged_split2 |
| 27B down | 8 | 33.902 | 50.194 | compact_mma_staged_split2 |
| 27B down | 16 | 37.851 | 62.033 | compact_mma_staged_split2 |
| IQ3_S expert | 5 | 17.351 | 9.382 | compact_mma_staged_split10 |
| IQ3_S expert | 8 | 17.730 | 9.444 | compact_mma_staged_split10 |
| IQ3_S expert | 16 | 17.880 | 10.216 | compact_mma_staged_split10 |
| IQ2_S expert | 5 | 19.374 | 10.958 | compact_mma_staged_split10 |
| IQ2_S expert | 8 | 21.540 | 11.027 | compact_mma_staged_split10 |
| IQ2_S expert | 16 | 19.133 | 11.878 | compact_mma_split10 |
| Flash gate | 5 | 18.829 | 15.502 | compact_mma_prefetch_split7 |
| Flash gate | 8 | 19.364 | 15.728 | compact_mma_prefetch_split7 |
| Flash gate | 16 | 20.235 | 18.036 | compact_mma_staged_split7 |
| Flash output | 5 | 16.940 | 14.513 | compact_mma_prefetch_split4 |
| Flash output | 8 | 15.855 | 14.913 | compact_mma_prefetch_split4 |
| Flash output | 16 | 17.385 | 16.993 | compact_mma_staged_split4 |

Dense 27B parity remains unmet. The next narrow experiment increases CTA
coverage through whole-block split-K. Existing four-CTA-per-SM targets yield
only about 30% active warps in dense node counters. The benchmark exposes
CTA targets as explicit command arguments, retains the one-split candidate,
and records every resulting split count. This changes no arithmetic or
persistent weight storage; model dispatch remains gated.

## CTA coverage and remaining metadata loads

With CTA targets of 4/8/16 per SM, higher split counts reduce dense M5/M8
latency but do not improve M16. The core remains `bb42559ab8`; only benchmark
arguments change. Graph timings retain the same bank and FP32 contract.

| Projection | M | Canonical µs | Best compact µs | Candidate |
| --- | ---: | ---: | ---: | --- |
| 27B gate | 5 | 34.851 | 40.261 | compact_mma_prefetch_split10 |
| 27B gate | 8 | 36.592 | 40.906 | compact_mma_prefetch_split5 |
| 27B gate | 16 | 39.577 | 58.763 | compact_mma_staged_split3 |
| 27B down | 5 | 31.667 | 40.525 | compact_mma_prefetch_split4 |
| 27B down | 8 | 32.164 | 41.076 | compact_mma_prefetch_split4 |
| 27B down | 16 | 37.817 | 62.493 | compact_mma_staged_split2 |

A node capture of packet-only shared staging at gate M16/split3 records
9,759,456 DRAM read bytes, 30.68% active warps, 80 registers per thread,
52.29% long-scoreboard stall and 9.09% short-scoreboard stall. Barrier stall
is 4.31%. Temporary shared memory is 13,568 bytes per CTA. These counters
remain separate from unprofiled latency.

Packet-only staging still loads the original d/scale planes from global
memory once each block. The next candidate stages the complete equal-byte
block, including both metadata planes. Its register prefetch has the same
word count per thread because 110/82-byte blocks still fit within four/three
coalesced uint64 words per thread. Persistent storage and arithmetic are
unchanged. Numerical and speed validation of this extension is pending.

## Complete-block staging measurements

Source `3dc27805ef`, core SHA256
`a853192e62999c9aa4329c8c361d168da9d2efce98c5210b7613fa4719a8c71c`,
whole-wheel SHA256
`cc81a5175f850ee5a4fa4dbaecc722dbd878a718d78941a002ff922b9360518f`.
All 20 checks pass (5.61 s). Matched graph results use CTA targets 4/8/16 and
preserve every candidate. Times below are µs/projection.

| Projection | M | Canonical | Best equal-byte | Candidate |
| --- | ---: | ---: | ---: | --- |
| 27B gate | 5 | 33.041 | 40.437 | compact_mma_prefetch_split10 |
| 27B gate | 8 | 33.332 | 41.108 | compact_mma_prefetch_split5 |
| 27B gate | 16 | 39.439 | 57.315 | compact_mma_staged_split3 |
| 27B down | 5 | 32.692 | 40.563 | compact_mma_prefetch_split4 |
| 27B down | 8 | 32.141 | 40.989 | compact_mma_prefetch_split4 |
| 27B down | 16 | 37.866 | 60.609 | compact_mma_staged_split2 |
| IQ3_S expert | 5 | 17.787 | 9.090 | compact_mma_staged_split10 |
| IQ3_S expert | 8 | 19.772 | 9.151 | compact_mma_staged_split10 |
| IQ3_S expert | 16 | 17.870 | 9.811 | compact_mma_staged_split10 |
| IQ2_S expert | 5 | 19.321 | 10.689 | compact_mma_staged_split10 |
| IQ2_S expert | 8 | 19.391 | 10.799 | compact_mma_staged_split10 |
| IQ2_S expert | 16 | 18.990 | 11.872 | compact_mma_split10 |
| Flash gate | 5 | 18.817 | 12.270 | compact_mma_prefetch_split10 |
| Flash gate | 8 | 18.518 | 12.344 | compact_mma_prefetch_split10 |
| Flash gate | 16 | 20.185 | 15.074 | compact_mma_staged_split10 |
| Flash output | 5 | 15.938 | 12.031 | compact_mma_prefetch_split6 |
| Flash output | 8 | 15.772 | 12.151 | compact_mma_prefetch_split6 |
| Flash output | 16 | 17.317 | 14.901 | compact_mma_staged_split6 |

Including metadata modestly improves staged M16, while higher CTA coverage
remains the best dense M5/M8 option. Dense 27B parity is still not reached.

SASS for the staged IQ3_S M16 main loop contains 130 FMUL, 64 FSEL and 66
I2F instructions across its unrolled body. The next decoder reuses the
TurboMind PRMT/half2 technique to unpack biased grid bytes into exact integer
values in pairs, restoring signs by XOR rather than multiplying by ±1.
Every intermediate integer is exactly representable: 1024+b, 1152 and b-128
for all byte values b. After unpacking, original scale reconstruction and
weight multiplication remain FP32. No coefficient or weight rounding is
introduced before the existing MMA/workspace FP16 boundary. Exhaustive
codebook-entry/sign tests and the official FP32 oracle gate this change.

## Exact paired-grid decoder measurements

Source `2431c8cd3b`, core SHA256
`0bf22ad1ff3ccb02ca2abf30ffac336b088a807e321f55b90c54679767e96339`,
whole-wheel SHA256
`19ffe9b8fdeee54403d02b6f612fdda3bf8b2686014b8ff4057a41dcd0b1eecf`.
All 22 checks pass (5.61 s), including exhaustive IQ3_S/IQ2_S codebook indices
with both sign polarities. FP32 dequantization remains exact. SASS confirms
FMUL 130→66, I2F 66→2 and elimination of 64 FSEL instructions in the IQ3_S
staged M16 loop. There are no local-memory spills. Matched full-graph timing
covers M1/5/8/16/512; CTA targets remain 4/8/16.

| Projection | M | Canonical µs | Best original/equal-byte µs | Candidate |
| --- | ---: | ---: | ---: | --- |
| 27B gate | 1 | 33.836 | 36.883 | compact_row_vec_split1 |
| 27B gate | 5 | 33.081 | 34.069 | compact_mma_prefetch_split10 |
| 27B gate | 8 | 34.843 | 36.619 | compact_mma_prefetch_split3 |
| 27B gate | 16 | 42.455 | 51.898 | compact_mma_staged_split3 |
| 27B gate | 512 | 366.732 | 370.124 | compact_dequant_cublas |
| 27B down | 1 | 30.812 | 34.852 | vec_split1 |
| 27B down | 5 | 31.793 | 32.854 | compact_mma_prefetch_split4 |
| 27B down | 8 | 31.944 | 33.306 | compact_mma_prefetch_split4 |
| 27B down | 16 | 37.942 | 50.806 | compact_mma_staged_split2 |
| 27B down | 512 | 340.309 | 342.554 | compact_dequant_cublas |
| IQ3_S expert | 1 | 17.021 | 6.946 | vec_split8 |
| IQ3_S expert | 5 | 19.389 | 7.686 | compact_mma_staged_split10 |
| IQ3_S expert | 8 | 17.575 | 7.729 | compact_mma_staged_split10 |
| IQ3_S expert | 16 | 17.301 | 8.796 | compact_mma_staged_split10 |
| IQ3_S expert | 512 | 68.691 | 27.632 | compact_dequant_cublas |
| IQ2_S expert | 1 | 18.302 | 9.524 | vec_split8 |
| IQ2_S expert | 5 | 19.173 | 9.844 | compact_mma_staged_split10 |
| IQ2_S expert | 8 | 19.234 | 9.893 | compact_mma_staged_split10 |
| IQ2_S expert | 16 | 18.932 | 11.188 | compact_mma_staged_split10 |
| IQ2_S expert | 512 | 64.294 | 30.379 | compact_dequant_cublas |
| Flash gate | 1 | 17.760 | 10.773 | vec_split1 |
| Flash gate | 5 | 18.354 | 10.925 | compact_mma_prefetch_split10 |
| Flash gate | 8 | 18.478 | 11.008 | compact_mma_prefetch_split10 |
| Flash gate | 16 | 21.281 | 14.056 | compact_mma_prefetch_split10 |
| Flash gate | 512 | 90.025 | 126.685 | compact_dequant_cublas |
| Flash output | 1 | 15.237 | 9.127 | vec_split1 |
| Flash output | 5 | 15.695 | 10.874 | compact_mma_prefetch_split6 |
| Flash output | 8 | 15.767 | 10.931 | compact_mma_prefetch_split6 |
| Flash output | 16 | 17.778 | 13.618 | compact_mma_prefetch_split6 |
| Flash output | 512 | 81.793 | 87.810 | compact_dequant_cublas |

Dense 27B M5/M8 is within about 3–5% of canonical in this run, but M1/M16
still loses and Flash dense prefill gaps remain. These data do not justify
removing canonical model storage or claiming a model step speedup.

The column-owned warp experiment (`1763f30013`) passes all 22 checks in
5.58 s but gives no dense improvement. Relative to existing compact
candidates it loses about 4–21% on the individual experts. Dense Flash
changes are within about 2%. This candidate is reverted; its measurements
are retained as a rejected layout/scheduling choice.

The next narrow experiment selects an eight-row MMA tile at M16. This uses
already compiled TurboMind m8n8k4 variants, doubles M-direction CTA coverage
and reduces per-thread accumulator registers. Split-K still partitions
whole source blocks and computes FP32 partials. Repeated weight requests
from neighboring M tiles must be distinguished from physical DRAM reads;
new counter collection is required if this candidate wins. Numerical and
speed gates remain pending, with the original automatic row tile retained.

While the row-tile experiment waits for shared GPUs, an independent vector
candidate prefetches one coalesced uint64 source word per participating lane
from the next block. After the current FP32 FMA chain finishes, the warp
reuses its original-block shared buffer. Two warp barriers protect reuse.
This holds only the following original word in registers; it does not expand
metadata or prefetch a decoded coefficient/activation bundle. The existing
vector candidate and its accumulation order remain unchanged. Numerical
and speed validation of both candidates is pending.

## Original vector prefetch and row-tile measurements

Source `9a70e6b2f3`, core SHA256
`d5cff734aeeb692b062cfd71f4b3c5076b0a1809f40dd8b07e721f3d07d1d8ed`,
whole-wheel SHA256
`d5d8c7a689ab8f4c0079ffab5bc3caee53496273ef93bf5a14f6b16ba4319e31`.
All 22 checks pass (6.82 s), including original-word prefetch and changed-input
M16/eight-row graph replay. M1 and M16 are the only remeasured points.

| Projection | M | Canonical µs | Best original/equal-byte µs | Candidate |
| --- | ---: | ---: | ---: | --- |
| 27B gate | 1 | 31.956 | 33.012 | vec_prefetch_split1 |
| 27B gate | 16 | 39.530 | 51.731 | compact_mma_staged_split3 |
| 27B down | 1 | 30.740 | 31.379 | vec_prefetch_split1 |
| 27B down | 16 | 37.998 | 50.504 | compact_mma_staged_split2 |
| IQ3_S expert | 1 | 16.695 | 6.776 | vec_prefetch_split8 |
| IQ3_S expert | 16 | 17.457 | 8.524 | compact_mma_staged_rows8_split10 |
| IQ2_S expert | 1 | 18.156 | 9.496 | vec_split8 |
| IQ2_S expert | 16 | 19.537 | 10.952 | compact_mma_staged_rows8_split10 |
| Flash gate | 1 | 17.915 | 9.235 | vec_prefetch_split1 |
| Flash gate | 16 | 20.235 | 14.068 | compact_mma_prefetch_split10 |
| Flash output | 1 | 16.783 | 8.883 | vec_prefetch_split1 |
| Flash output | 16 | 17.097 | 13.620 | compact_mma_prefetch_split6 |

Original-word prefetch brings dense M1 within about 2–3% of canonical here.
Eight-row tiles help the individual M16 experts but lose on dense 27B.
Dense M16 remains a speed gap. Node profiling of the gate M1 vector records
9,734,688 DRAM read bytes versus 9,574,400 weight payload bytes, 54.82% active
warps, 38 registers/thread, 24.94% long-scoreboard stall and 17.55% math-pipe
throttle. The counter includes activation/codebook/instruction traffic;
its 41.024 µs profiled duration is not the unprofiled 33.012 µs graph time.

## Exact final FP16 operand formation candidate

IQ3_S grid magnitudes are 1/3/5/7/9/11/13/15; its small coefficient is an odd
integer at most 31. Their product is an integer no larger than 465, exactly
representable in FP16. IQ2_S grid magnitudes are 8/25/43 and its coefficient
is an odd integer divided by eight; their product is at most 1333/8 and is
also exactly representable in FP16. Original d remains in its source width.

For these two formats, the product of finite FP16 d and that exact factor
has at most 22 significant binary bits, so the official FP32 dequantization
product is exact. Forming the final FP16 operand with one half2 multiply
therefore has the same final rounding as FP32 dequantization followed by
conversion to FP16. This introduces no expanded/rounded scale coefficient.
FP32 dequantization, vector FMA and all accumulation remain in FP32.

A CPU exhaustive check covers all 63,488 finite FP16 d encodings, every small
coefficient, grid magnitude and sign: 16,252,928 IQ3_S and 6,094,848 IQ2_S
weights. Every exact-factor and final-output bit comparison matches. This
proof does not substitute for device validation: the new focused GPU checks
cover every finite block scale, signed zero, subnormals, overflow boundaries
and all small coefficients against the official reader. Numerical and speed
validation of this candidate remains pending.

## Exact operand device and graph results

Source `f2de1df460`, core SHA256
`01558584aeaaf91c23dd00a8434c77e82481f4ed63234c9d8c86384be6a2ca91`,
whole-wheel SHA256
`0364b5822501bb7a4e0e2f49415562f8635491886c98a874f963eb9540da467d`.
All 24 checks pass (7.84 s). The GPU final-weight bit comparisons match the
official FP32 reader conversion for every finite FP16 block-scale encoding,
all small coefficients and both signs, including subnormal/overflow boundaries.
Float output and vector FMA preserve the original FP32 path. The four changed
M points below retain the graph/bank/CTA contract. Times are µs/projection.

| Projection | M | Canonical | Best original/equal-byte | Candidate |
| --- | ---: | ---: | ---: | --- |
| 27B gate | 5 | 33.031 | 31.743 | compact_mma_prefetch_split3 |
| 27B gate | 8 | 33.174 | 33.433 | compact_mma_prefetch_split3 |
| 27B gate | 16 | 39.463 | 50.704 | compact_mma_staged_split3 |
| 27B gate | 512 | 366.340 | 364.399 | compact_dequant_cublas |
| 27B down | 5 | 31.849 | 28.706 | compact_mma_prefetch_split4 |
| 27B down | 8 | 34.416 | 30.010 | compact_mma_prefetch_split4 |
| 27B down | 16 | 37.903 | 51.207 | compact_mma_staged_split2 |
| 27B down | 512 | 340.191 | 337.450 | compact_dequant_cublas |
| IQ3_S expert | 5 | 17.354 | 7.420 | compact_mma_staged_split10 |
| IQ3_S expert | 8 | 17.735 | 7.452 | compact_mma_staged_split10 |
| IQ3_S expert | 16 | 17.326 | 8.640 | compact_mma_staged_split10 |
| IQ3_S expert | 512 | 69.081 | 26.942 | compact_dequant_cublas |
| IQ2_S expert | 5 | 19.915 | 9.567 | compact_mma_staged_split10 |
| IQ2_S expert | 8 | 19.942 | 9.605 | compact_mma_staged_split10 |
| IQ2_S expert | 16 | 20.097 | 10.954 | compact_mma_staged_split10 |
| IQ2_S expert | 512 | 64.113 | 30.441 | compact_dequant_cublas |
| Flash gate | 5 | 20.948 | 10.039 | compact_mma_prefetch_split10 |
| Flash gate | 8 | 18.515 | 10.317 | compact_mma_prefetch_split10 |
| Flash gate | 16 | 21.358 | 14.038 | compact_mma_prefetch_split10 |
| Flash gate | 512 | 90.160 | 126.530 | compact_dequant_cublas |
| Flash output | 5 | 15.870 | 9.900 | compact_mma_prefetch_split6 |
| Flash output | 8 | 15.855 | 10.232 | compact_mma_prefetch_split6 |
| Flash output | 16 | 17.606 | 13.625 | compact_mma_prefetch_split6 |
| Flash output | 512 | 81.825 | 87.589 | compact_dequant_cublas |

Dense 27B M5/M8 and M512 reach approximate parity or improve, but M16 is
still 28–35% slower. Flash dense M512 still loses. No model default or storage
removal is promoted from these partial results.

A current M16 node recapture is deferred after a shared GPU lock timeout;
that timeout is not a numerical failure. Existing staged-node counters show
the register limit admits six CTAs/SM while shared memory permits seven.
The current compiled staged kernel still uses 78 registers/thread. The next
candidate retains the same inlined computation body but constrains register
allocation to admit seven CTAs/SM. Its original entry point remains the
comparison. An eight-CTA budget produces an eight-byte stack allocation and
is rejected before GPU timing. The seven-CTA variant must pass resource,
FP32 numerical and matched full-graph performance checks before selection.

## Seven-CTA device and counter results

Source `7122dd3b11`, core SHA256
`c0733c0713737a81dc3fe1a7700ec992508a974508f1a4167617d62c504db6a2`,
whole-wheel SHA256
`9d0019a47a4340add14e2d11788278f7af92f185a3d1baf5fb9e543f820bf3b5`.
All 24 checks pass (8.70 s). Matched M16 graph timings improve every selected
shape, but dense 27B remains slower than canonical.

| Projection | Canonical µs | Best original/equal-byte µs | Candidate |
| --- | ---: | ---: | --- |
| 27B gate | 39.593 | 47.087 | compact_mma_occupancy7_split2 |
| 27B down | 38.052 | 42.632 | compact_mma_occupancy7_split2 |
| IQ3_S expert | 17.853 | 8.189 | compact_mma_prefetch_occupancy7_split10 |
| IQ2_S expert | 19.085 | 10.382 | compact_mma_staged_occupancy7_split10 |
| Flash gate | 20.202 | 13.684 | compact_mma_prefetch_occupancy7_split10 |
| Flash output | 17.384 | 13.148 | compact_mma_prefetch_occupancy7_split4 |

The gate M16 winner is the plain packet decoder with the seven-CTA budget
and split2. It uses 72 registers/thread with no stack allocation. Node
profiling records 9,757,152 main-node DRAM read bytes, 20.86% active warps,
32.55% long-scoreboard stall and 10.40% math-pipe throttle. Register occupancy
limit is seven CTAs/SM; actual occupancy remains lower with this grid.
The FP32 split reduction reads another 559,520 bytes. These counters include
non-weight traffic and remain separate from unprofiled wall time.

## Flash dense prefill attribution and TN candidate

A matched cold-cache node capture at Flash gate M512 (`N=1536, K=2560`)
identifies the current dequantization/cuBLAS cost:

| Node | Profiled µs | DRAM read bytes |
| --- | ---: | ---: |
| Canonical fused GEMM | 120.928 | 4,637,056 |
| Equal-byte dequantization | 20.384 | 1,696,640 |
| cuBLAS NN GEMM | 106.176 | 14,592,352 |
| cuBLAS FP32 split-K reduction | 16.128 | 6,295,328 |

The unprofiled complete paths remain about 90/127 µs. Profiling perturbs
node durations, so the rows above are attribution rather than a replacement
wall-time comparison. The NN GEMM selects a 64×64 Volta kernel plus split-K
reduction; dequantization is not the dominant gap.

The next candidate writes the temporary FP16 workspace in natural `[N,K]`
row order using aligned 128-bit stores and calls cuBLAS TN. It retains the
NN candidate and the same FP32 computation/reduction policy. Temporary
workspace byte size is unchanged; persistent weights remain equal-byte
packets. Changed-input graphs and exact workspace comparisons cover both
layouts before speed selection. Validation of TN remains pending.

## Strided natural-workspace result

Source `c64a8d7c2f` passes all 24 checks (7.87 s), including exact workspace
values and changed-input graphs for NN and TN. The initial natural `[N,K]`
workspace path is slower at every measured M512 shape; it is not selected.

| Projection | Canonical µs | Compact NN µs | Initial compact TN µs |
| --- | ---: | ---: | ---: |
| 27B gate | 367.676 | 363.468 | 1075.453 |
| 27B down | 342.257 | 339.610 | 982.836 |
| IQ3_S expert | 69.220 | 27.258 | 57.179 |
| IQ2_S expert | 64.382 | 30.288 | 59.517 |
| Flash gate | 91.674 | 126.169 | 172.969 |
| Flash output | 81.766 | 87.610 | 137.094 |

A node capture of Flash gate TN attributes 100.064 µs to dequantization
(4,540,096 DRAM read bytes) and 109.728 µs to its 128×128 cuBLAS GEMM
(10,513,248 read bytes). There is no separate split-K reduction. Natural
workspace dequantization uses 32 registers with no stack; the loss comes
from strided partial-sector writes rather than spills. cuBLAS has 250
registers/thread and 6.25% active warps in this capture.

The next candidate stages decoded FP16 values in a padded shared tile and
changes warp ownership for the output copy. Each warp then writes contiguous
K vectors with aligned 128-bit stores. Padding belongs only to temporary
shared storage; both workspace and persistent weight byte sizes remain
unchanged. NN is retained and TN numerical/performance checks run again.

## Coalesced natural workspace

Source `57c614fc33` passes all 24 checks (8.07 s). This includes official
FP32 dequantization equality, exact FP16 workspace values for both layouts,
and changed-input full CUDA graphs. The padded shared tile changes only
output-copy ownership; persistent packet and workspace byte sizes are unchanged.

| Projection | Canonical µs | Compact NN µs | Coalesced compact TN µs |
| --- | ---: | ---: | ---: |
| 27B gate | 366.867 | 363.066 | 386.360 |
| 27B down | 341.748 | 336.033 | 351.894 |
| IQ3_S expert | 69.209 | 27.254 | 48.237 |
| IQ2_S expert | 64.371 | 30.255 | 50.275 |
| Flash gate | 94.161 | 126.231 | 113.438 |
| Flash output | 81.841 | 87.671 | 81.270 |

These are unprofiled full-graph replay times with distinct weight banks
exceeding twice V100 L2, 100 replays and eight bank sweeps per graph. NN
remains preferable for the 27B projections and single-expert shapes. TN
reaches parity for Flash output but remains 20.5% slower for Flash gate.
This result does not satisfy the complete promotion gate.

Cold-cache node profiling of Flash gate records 25.696 µs and 1,697,664
DRAM read bytes for coalesced dequantization, versus 100.064 µs and
4,540,096 bytes for the earlier strided writer. The cuBLAS TN node records
109.472 µs, 10,512,480 read bytes, 250 registers/thread and 6.25% active
warps. Profiled node times are attribution, not the wall-time comparison.
The next bounded experiment compares cuBLAS GEMM algorithm choices with
FP32 computation and reduced-precision reductions disabled; no precision
change is proposed.

The installed wheel SHA256 is
`abe02a1520497f0dbfb634014d213ac70dca89b95b0aec67600cfd5cdb65c9e2`;
its installed core SHA256 is
`0b7e3ecde6192dd394761129adb6f20bc97d8a2a0b9f8de546f433155d34ba87`.
All 210 installed dependency packages are compatible.

## cuBLAS algorithm comparison with overflow gate

A same-precision algorithm probe found that explicitly requesting tensor
algorithm 10 with FP16 output can overflow intermediate split partials even
with reduced-precision reductions disabled. A cancellation workload with
FP16 all-one weights, K=2560 and activations +128 for the first half and
-128 for the second has exact final output zero. Default and tensor
algorithm 2 remain finite and exact; algorithm 10 produces infinity for
both NN and TN. FP32 output removes that overflow. The FP16 algorithm 10
candidate is rejected rather than changing the numerical contract.

Source `81700e4c74` exposes only default and algorithm 2 for calibration.
All 26 checks pass (7.59 s), including original-format IQ3_S/IQ2_S
cancellation workloads in both layouts and changed-input full graphs.

| Projection | Canonical µs | Compact default NN µs | Compact default TN µs | Compact algo2 NN µs | Compact algo2 TN µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 27B gate | 367.179 | 365.779 | 388.633 | 447.735 | 528.049 |
| 27B down | 340.546 | 335.970 | 350.843 | 396.105 | 445.426 |
| IQ3_S expert | 69.211 | 27.235 | 48.245 | 56.859 | 57.225 |
| IQ2_S expert | 64.406 | 30.248 | 50.274 | 59.478 | 58.852 |
| Flash gate | 91.896 | 126.308 | 114.272 | 97.719 | 107.057 |
| Flash output | 85.270 | 87.684 | 81.563 | 74.480 | 83.712 |

Complete-path full-graph timing narrows the Flash gate M512 gap to 6.3%
and makes Flash output faster. Algorithm 2 regresses the other shapes and
is not selected globally. The provisional winners retain the default NN
algorithm for 27B and experts and algorithm 2 NN for these Flash dense
shapes. Promotion still requires dense M16 and remaining storage/operator
coverage. Kernel-only synthetic GEMM timing is not used as that gate.

The installed wheel SHA256 is
`ab49a0b0bb3e88dc657eba658f4384fca463dcd0cdce72852ab3a174b154eb0f`;
its installed core SHA256 is
`1e25499da47451c85409831d01b8f4a8435d44baefe96773503d033d8f8356e6`.
The dependency check passes for all 210 packages. The [cuBLAS numerical
behavior documentation](https://docs.nvidia.com/cuda/cublas/index.html#gemm-algorithms-numerical-behavior)
explains why intermediate split reductions need explicit numerical checks.

## Narrow packet fetch candidate

The next candidate keeps the same equal-byte bit permutation but fetches
one 32-bit word per participating lane instead of 64 bits. A 26-bit IQ3_S
or 18-bit IQ2_S packet spans at most two such words. Two 32-bit shuffles
and a funnel shift replace four shuffles and 64-bit stitching. Warp reads
remain consecutive and cover exactly the original packet bits; FP32 scale
formulas and accumulator precision are unchanged. This candidate requires
the existing inverse, all-grid, boundary, graph and real-shape speed checks
before selection.

## Narrow packet fetch result

Source `44f2c91727` passes all 26 checks (7.32 s), including official
dequantization, bit-preserving permutation, N tails, coefficient boundaries,
FP32 partial cancellation and changed-input full graphs. The 32-bit fetch
retains the original payload bit layout and byte size.

| Projection | M | Canonical µs | Best equal-byte µs |
| --- | ---: | ---: | ---: |
| 27B gate | 5 / 8 / 16 / 512 | 33.016 / 33.315 / 39.381 / 366.328 | 29.757 / 31.468 / 46.931 / 366.653 |
| 27B down | 5 / 8 / 16 / 512 | 31.812 / 31.942 / 37.912 / 340.344 | 27.483 / 28.717 / 42.705 / 338.213 |
| IQ3_S expert | 5 / 8 / 16 / 512 | 19.291 / 18.268 / 17.384 / 69.054 | 7.514 / 7.743 / 8.149 / 27.067 |
| IQ2_S expert | 5 / 8 / 16 / 512 | 20.433 / 19.983 / 20.114 / 64.033 | 9.578 / 9.628 / 10.343 / 30.559 |
| Flash gate | 5 / 8 / 16 / 512 | 20.823 / 19.252 / 21.268 / 89.837 | 9.770 / 10.174 / 13.483 / 97.621 |
| Flash output | 5 / 8 / 16 / 512 | 16.082 / 15.855 / 17.065 / 81.533 | 9.652 / 9.987 / 12.907 / 74.387 |

The 27B M5/M8 improvement is retained. M16 remains 19.2% slower for gate
and 12.6% slower for down, so narrower extraction alone does not close the
dense concurrency gap. Flash gate M512 retains an 8.7% gap in this run;
Flash output is 8.8% faster. These are complete unprofiled graph paths with
FP32 accumulation, cold weight banks, identical shape controls and 100
replays. The previously selected M1 original-row and row-wise vector paths do
not use the changed warp packet-fetch path and are not repeated here. Grouped expert/model step gains remain unmeasured.

Installed core SHA256:
`450dcc11c47ab2823730ad241791945102f4fa8354533f83d6e68ec01ea9be2e`.
Whole-wheel SHA256:
`45e62f62fc57cd77087b84240a4ea4d66bc2702c6de382886037db5264d54e85`.
All 210 installed dependency packages are compatible.

## Complete-tile dequantization specialization

Source `2a745757b6` selects fixed N32 address arithmetic and aligned
original metadata loads when N is divisible by 32. Partial tiles retain
the general bitstream/byte-gather implementation. All 26 checks pass
(7.99 s), covering exact FP32 and FP16 dequantization, both layouts, tails
and FP32 partial overflow guards in changed-input full graphs.

| Projection | Canonical µs | Compact default NN µs | Compact default TN µs | Compact algo2 NN µs |
| --- | ---: | ---: | ---: | ---: |
| 27B gate | 365.559 | 367.524 | 381.910 | 447.516 |
| 27B down | 340.825 | 340.645 | 347.410 | 399.784 |
| IQ3_S expert | 69.218 | 26.885 | 47.526 | 56.395 |
| IQ2_S expert | 64.401 | 29.153 | 49.760 | 58.229 |
| Flash gate | 93.396 | 125.793 | 111.878 | 97.236 |
| Flash output | 84.994 | 87.721 | 79.579 | 74.245 |

This is the same M512 complete-path full-graph comparison. The direct
loading specialization has modest benefits on experts and natural workspace
writing, but does not close Flash gate: algorithm 2 NN remains 4.1% slower
than this run's canonical control. The canonical gate varies between
approximately 90 and 94 µs across recorded runs; the compact path remains
near 97 µs. That difference is retained rather than called a speed win.

A matched NCU capture of the earlier 32-bit packet M16 kernel records
9,756,896 main-node DRAM read bytes, 72 registers/thread, 20.57% active
warps and 33.17% long-scoreboard stalls. ALU instruction count falls from
4,368,320 to 4,172,480, but this same seven-CTA/split2 descriptor does not
speed up materially. Reduction reads another 559,520 bytes. Narrow packet
fetch therefore does not explain or solve the remaining dense M16 gap.

Installed core SHA256:
`f01a71c068c044ebbfbad1d0b827a861786725a0adbf39787961266fe4ac0c20`.
Whole-wheel SHA256:
`d071b5456f0f3773344e0a845a9795d653188cb1ff8ff8c41cc2c1414991cedf`.
The 210-package dependency check passes. A bounded FP32-output cuBLAS
comparison is the next prefill screening experiment; it must include final
FP16 conversion and provide enough headroom for dequantization before any
complete-path implementation is added.

## FP32 output workspace screening

A cold-bank Flash gate GEMM-only full-graph probe includes final conversion
to FP16. Default cuBLAS NN with FP32 output takes 79.782 µs, versus
88.879 µs for algorithm 2 with FP16 output and its matched output copy.
Both retain about 0.0002075 relative L2 error. This is screening evidence
only: the FP16 probe copy is additional work compared with the existing
complete-path operator, and dequantization is excluded. It provides enough
headroom to justify one complete-path comparison but is not a speed claim.

Source `790cf01988` permits FP32 output only in compact BLAS validation.
Vector/MMA validation continues to require FP16 output. The cuBLAS output
type follows the output tensor, with FP32 computation and reduced-precision
reductions disabled. The benchmark captures a temporary `[M,N]` FP32 result
and its final FP16 copy in the same graph as temporary dequantization.
Persistent weights retain the original equal-byte budget. Numerical checks
and complete-path Flash gate M512 timing remain pending.

## FP32 output workspace complete-path result

Source `790cf01988` passes all 28 checks (8.88 s), including FP32 output
with final FP16 conversion, tails, changed-input graphs and exact cancellation
for both output dtypes. The Flash gate M512 complete-path result is:

| Path | Unprofiled full-graph µs | Output relative L2 |
| --- | ---: | ---: |
| Canonical fused lattice GEMM | 93.852 | 0.0004092 |
| Compact default NN, FP16 output | 125.834 | 0.0002072 |
| Compact algorithm 2 NN, FP16 output | 97.277 | 0.0002072 |
| Compact default NN, FP32 output plus FP16 conversion | 97.105 | 0.0002072 |
| Compact default TN, FP32 output plus FP16 conversion | 103.958 | 0.0002072 |

The screening margin does not survive complete-path timing: FP32 workspace
output is effectively tied with algorithm 2 FP16 output and remains 3.5%
slower than this canonical control. No model/default switch or expansion to
other shapes follows this result. The FP32 result remains a correctness
control; there is no persistent storage change.

The measured core SHA256 is
`11f8495b6c40551302bd269ce84aa54401e478faf52c11354f048f3175910c0a`.
The wheel's distribution version and Python version string differed because
a documentation commit landed between packaging phases. The binary source
was unchanged. A clean fixed-commit repackaging must reconcile the version
strings and preserve this core hash before publishing the package.

## Fixed-commit package and dense activation traffic

The fixed-commit package reconciles both source-version strings at
`1.5.2.dev486+g018136e79`; the distribution adds the intended
`.precompiled` build-flavor suffix. Its whole-wheel SHA256 is
`6346edf6f6e3b336e922f7fd61fb8472d287204182b6dd304d53443bb2c9d9af`,
and the core retains
`11f8495b6c40551302bd269ce84aa54401e478faf52c11354f048f3175910c0a`.
The 210-package dependency check passes. Numerical checks are not repeated
for an unchanged binary and a version-only packaging correction.

A cold-cache NCU capture of the fastest measured 27B gate M16 descriptor,
packet prefetch with split3, records the following main-node counters:

| Counter | Value |
| --- | ---: |
| DRAM read bytes | 9,740,768 |
| L1 global-load sectors (32 bytes each) | 1,798,846 |
| L2 read sectors (32 bytes each) | 965,893 |
| L1 throughput / sustained peak | 70.60% |
| L2 throughput / sustained peak | 24.43% |
| Active warps | 30.43% |
| Registers / thread | 70 |
| Long-scoreboard stall | 65.48% |
| Short-scoreboard stall | 4.22% |
| Math-pipe throttle | 2.72% |

L1 requests total about 57.6 MB and L2 reads about 30.9 MB while the
original weight payload is 9.57 MB. These counters include activation,
metadata and other reads; they do not identify each byte by source. The
fragment loading pattern issues strided activation reads, so a contiguous
shared activation tile is the next bounded candidate.

Source `11d6db5e68` adds optional M16 activation staging. Warps copy
complete contiguous K256 rows in aligned vectors into a padded temporary
shared tile and load MMA fragments from that tile. It retains original
weight bytes, original small scales and FP32 accumulation. Weight staging
and the seven-CTA register-budget variant remain separate controls. Changed
input/split-K/tail checks and two 27B dense M16 comparisons must pass before
this candidate is selected. Numerical and performance validation is pending.

## Contiguous activation staging result

Source `11d6db5e68` passes all 28 checks (7.77 s), including activation
staging on both formats, N tails, split1/split3 and changed-input full graphs.
The initial run waited for the shared GPU lock and exited with resource
status 75 twice; those attempts did not execute GPU tests. The completed
run holds the common and four per-device locks throughout verification.

| 27B M16 projection | Canonical µs | Best prior path µs | Activation-prefetch winner µs | Split |
| --- | ---: | ---: | ---: | ---: |
| Gate | 39.528 | 46.990 | 43.323 | 5 |
| Down | 37.952 | 43.038 | 42.049 | 2 |

These are complete unprofiled full-graph paths at the same TP4 shapes.
Activation staging reduces gate time by 7.8% and down by 2.3% relative to
the same-run prior candidates, but retains gaps of 9.6% and 10.8% versus
canonical. It does not satisfy the full speed gate. Plain activation
staging without register prefetch is substantially slower; the two changes
are not selected independently. Other shapes are not promoted from these
two results. Matched split3 and winning split5 NCU captures will distinguish
request reduction from synchronization/occupancy effects.

The IQ3_S complete-tile prefetch variant uses 66 registers/thread, no stack
and 18,688 bytes of shared memory. The prior prefetch variant retains
70 registers, no stack and 10,240 shared bytes; unused activation storage
is eliminated when staging is disabled.

Installed core SHA256:
`ba8209ae4efe61ea74ff9d6a93fd8856380b919b6ffbc9f27ecfe9706c384dec`.
Whole-wheel SHA256:
`7ac4b557bfb0578097d78fbf3b0871b2582a5314a1017365760134ea0d280a88`.
The package/Python source versions agree and all 210 installed dependencies
are compatible. Root profiling disables Python bytecode writes to avoid
creating root-owned cache directories in the task runtime.

## Activation staging attribution

The matched gate M16 split3 capture confirms fewer memory requests with
staging, but shows that synchronous tile preparation offsets the benefit:

| Counter | Prior packet prefetch | Synchronous activation staging |
| --- | ---: | ---: |
| L1 global-load sectors | 1,798,846 | 1,137,935 |
| L2 read sectors | 965,893 | 766,095 |
| L1 throughput / sustained peak | 70.60% | 36.84% |
| Long-scoreboard stall | 65.48% | 47.28% |
| Active warps | 30.43% | 28.55% |
| Main-node profiled µs | 54.752 | 63.200 |

The staging split5 wall-time winner records 1,158,021 L1 sectors, 740,137
L2 sectors, 27.45% active warps, 47.47% long-scoreboard stall and 4.58%
barrier stall. Main-node DRAM reads are 9,747,296 bytes; the FP32 reduction
reads another 1,392,800 bytes. Node times remain separate from full-graph wall
time. The lower request count alone is not evidence of a speed win.

The next candidate preloads the first activation tile into shared memory
and keeps the following complete tile in registers while decoding and
multiplying the current one. CTA barriers protect shared reuse; SM70 does
not require an unavailable asynchronous-copy instruction. The storage
budget and every numerical operation remain unchanged. The same extended
checks and two dense M16 wall comparisons must validate this candidate
before it is selected.

## Register activation lookahead reaches dense M16 parity

Source `331e6234c9` passes all 28 extended checks (10.07 s). Preloading the
following activation tile into registers changes no weight bits, activation
values or FP32 accumulation operations. The two dense TP4 M16 comparisons are:

| Projection | Canonical µs | Register activation-prefetch µs | Split |
| --- | ---: | ---: | ---: |
| 27B gate | 39.633 | 38.769 | 3 |
| 27B down | 37.940 | 33.509 | 2 |

The gate path is 2.2% faster and down is 11.7% faster in complete unprofiled
full-graph replay. Earlier synchronous staging reached 43.323/42.049 µs.
The new prefetch variant uses 119 registers/thread, no stack allocation and
18,688 shared bytes, so its gain is demonstrated despite reduced CTA
residency. Register count alone is not used to reject the measured winner.
Other Flash M16 shapes and matched read-byte counters still require follow-up
before a broader default decision.

Installed core SHA256:
`c86213d5a23f44eee18d8a58e48b252b4d2caee678e0f90496ac926656fe687c`.
Whole-wheel SHA256:
`47f3683cbc85816effdd796bd0ce2a8e0dc0b5d64a403865d686ee0f3c7a12a0`.
The package/Python source versions agree and all 210 installed dependencies
are compatible. Persistent payload bytes and alignment budgets are unchanged.

## Register activation lookahead on remaining M16 shapes

The same installed binary completes all four remaining Flash M16
comparisons without repeating numerical checks:

| Projection | Canonical µs | Best equal-byte µs | Schedule |
| --- | ---: | ---: | --- |
| IQ3_S expert | 18.364 | 8.190 | activation prefetch, split10 |
| IQ2_S expert | 19.554 | 10.285 | activation prefetch, split10 |
| Flash gate | 22.259 | 13.619 | prior seven-CTA packet prefetch, split10 |
| Flash output | 17.403 | 12.392 | activation prefetch, split4 |

All six measured M16 shapes are now faster than their same-run canonical
control. This is a shape-dependent selection: the earlier Flash gate
schedule remains preferable, so activation staging is not enabled globally.

The matched 27B gate activation-lookahead split3 NCU main node reads
9,748,192 DRAM bytes, 1,122,209 L1 global-load sectors and 854,936 L2 read
sectors. Active warps are 21.15%, long-scoreboard stall is 36.83%, and
register count is 119 with no stack. The FP32 reduction reads another
835,744 bytes. Lower occupancy does not negate the demonstrated wall-time
gain; node durations remain separate from the unprofiled comparison.

The next prefill candidate separates packet fetch from extraction in the
FP32/natural-output and FP16 transposed dequantization loop. It reads the
following packet window into registers while reconstructing and storing
the current one. Persistent byte size, official scale formulas and output
layouts are unchanged. The full numerical suite and one Flash gate M512
complete-path comparison must validate it before broadening.

### Temporary packed FP16 workspace candidate

An additional prefill candidate decodes the equal-byte IQ3_S/IQ2_S packet layout
straight into TurboMind's existing FP16 operand layout, then invokes its SM70
FP32-accumulating GEMM with a single grouped descriptor. The workspace is
`[N/32, K/8, 32, 8]` with contiguous eight-element fragments. It is temporary
and does not alter persistent weight storage or expand stored scales.

The focused checks compare this layout against both the TurboMind converter and
an independent tensor permutation, replay graphs with changed activations, and
exercise cancellation where intermediate partial sums exceed the FP16 range.
Correctness and speed remain unvalidated until the packaged runtime passes
these checks and the complete M=512 graph comparison. This candidate has no
model dispatch registration.

The first packaged candidate passed all 32 focused checks. For Flash-Next
`blk.0.attn_gate.weight`, TP4 rank 0, N=1536/K=2560/M=512, its complete graph
was 102.531 us versus canonical 91.606 us and the best cuBLAS alternative
96.956 us. It therefore does not satisfy the speed gate. Exact dequantization
error was zero; output relative L2 error was 0.00020724.

NCU node counters show the packed-workspace dequantizer reads 1,695,744 bytes
for a 1,689,600-byte payload, while its FP16 GEMM reads 10,557,376 bytes. The
selected 128x128x16 GEMM uses 255 registers and achieves 11.07% active warps.
The cuBLAS algorithm-2 alternative reads 1,696,064 bytes in dequantization and
10,504,800 bytes in GEMM. These counters describe separate workspace passes,
not persistent compressed-weight expansion. Profiler timings are not used as
end-to-end speed measurements. Smaller FP16 tiles are the next candidate;
no model route is enabled by this result.

The smaller FP16 tiles passed all 32 checks but the measured complete path
remained 103.099 us versus 92.847 us canonical. They do not justify selecting
this workspace path. The next dequantizer experiment increases a CTA from four
to eight warps. For the Flash gate, the 480-CTA grid previously exposed only
24 resident warps per SM on average; its 75-83% long-scoreboard stall and low
DRAM throughput motivate more independent packet loads. Octet assignment uses
the actual warp count, preserving exact coverage and the original byte layout.

The eight-warp dequantizer passed 32 checks and reduced the fastest complete
Flash gate M512 path to 95.871 us, versus 91.890 us canonical. The TurboMind
workspace candidate was 101.795 us. A follow-up candidate partitions the 32
K-octets across two or four CTAs per original block, providing more active CTAs
without changing storage. Each CTA writes a disjoint set of octets. Original
metadata may be reread through cache; DRAM counters must establish its cost.
The natural-layout shared-transpose kernel retains its one-CTA contract.

The first partitioned build passed 36 checks and failed four random-weight
checks: the initial packet prefetch used octet `warp` instead of the partition's
first octet. Constant-weight cancellation tests cannot detect that indexing
error. The prefetch address is corrected before collecting any speed evidence;
the failed build has no timing result or model eligibility.

The corrected partition build passes 40 checks. Flash gate M512 reaches
94.991 us with two dequantization partitions, versus 91.551 us canonical;
four partitions regress to 96.219 us. The TurboMind workspace variants remain
slower at 101.732/103.567 us. No prefill default changes follow from these
results. The next independent M1 experiment prefetches the next FP16 activation
fragment alongside the next original weight word, retaining the identical FP32
FMA order and original-byte vector layout.

The M1 activation-lookahead candidate passes 40 checks but regresses both 27B
projections: gate/down 36.081/34.132 us versus 31.846/30.805 us canonical.
It is reverted, restoring the earlier original-word prefetch schedule. The two
additional FP16 registry tiles also provide no measured workspace improvement
and are removed. The experimental FP16 workspace operator remains a declared
compact-layout capability at M>=512, with an explicit output-pack alignment
rejection for N tails; declaring support does not select a model default.

The cleaned normal wheel passes 40 checks. NCU for the two-partition Flash gate
dequantizer reports 1,731,392 DRAM read bytes, 72.01% active warps and 42.73%
long-scoreboard stall, versus 1,696,064 bytes, 32.66% and 75.64% for the earlier
four-warp single-partition reference. Metadata/cache traffic increases slightly;
persistent storage remains 1,689,600 bytes. Its cuBLAS node reads 10,505,600 bytes
and remains at 7.67% active warps. Profiled service times are not TPOT.

The next M1 candidate factors the original block d out of each lane's
eight-element local dot product. Small-scale/grid coefficients, local FMA,
base-scale FMA and final reductions all use FP32. The alternative changes the
summation order, so it requires the same official-weight dot-product error gate;
FP32 dequantization and final MMA operand formation are unchanged. The existing
vector schedule remains the default until a measured candidate passes.

The d-only factoring candidate passes 42 checks but is slower: gate/down
34.745/32.748 us versus the retained original-word prefetch 33.076/31.409 us
and canonical 31.779/30.894 us. That branch is replaced by full-scale factoring:
compute original d and small-scale multiplication in FP32, accumulate eight
activation/grid products in FP32, then apply the combined FP32 scale by FMA.
This removes per-weight FP32 scale multiplies while retaining both scale levels
and FP32 accumulation. It changes grouping, so the same official-weight numerical
and changed-input graph gates apply. The original schedule remains the default.
The existing vector NCU trace reports 17.55% math-pipe throttle, supporting this
instruction-count experiment.

The first full-scale build retains 40 passing checks but fails the two enabled
vector-variant checks: its accumulation call still applied d alone after the
fragment stopped applying small scale. The missing factor is corrected by
calling the original FP32 `block_scale` helper. The failed build collected no
speed evidence; the numerical tolerances are unchanged.

Corrected complete-scale factoring passes 42 checks. Gate/down M1 are
32.803/30.890 us versus 32.659/30.772 us canonical in the six-route comparison.
Nine alternating-order pairs produce gate medians 32.614/33.581 us raw/canonical
and down 30.807/30.475 us. Different canonical launch selection and process
conditions affect the gate baseline; these results do not close every M1 gate.
Output relative L2 is approximately 0.000205.

NCU shows full-scale factoring reduces FP32-pipe instructions from 3,777,536
to 3,168,256, with DRAM reads 9,738,016 bytes, 40 registers and 55.08% active
warps. Integer-pipe instructions remain 3,355,392. The next vector candidate
expands only the fixed codebook to exact FP32 integers in shared memory once
per CTA. It removes repeated byte unpacking/conversion; original weights/scales
are unchanged. The larger shared table may reduce occupancy, so it is optional
and requires measured dispatch. Scale formation, local FMA and reductions
remain FP32.

The FP32 shared-codebook experiment passes all 44 checks but slows 27B gate/down
M1 to 36.131/34.335 us. Complete FP32-scale factoring remains faster at
32.814/30.931 us in that run. The expanded shared-codebook implementation and
its argument are reverted; numerical precision and persistent byte budgets
were unchanged throughout the experiment.

A second prefill candidate performs original-packet dequantization in shared
memory and reuses the decoded tile for 64 or 128 activation rows in a CTA.
Eight warps cooperatively decode N=128/K=64 slices, then run Volta mma884 with
FP32 accumulation. The final FP16 weights retain the exact official conversion
contract. Split-K stores and reductions use FP32, including cancellation when
partials exceed the FP16 range. Original persistent storage is unchanged, and
this candidate avoids the global FP16 workspace pass. It is declared alongside
the existing workspace candidates, with an N-pack alignment rejection. No
model default or speed claim is implied before the packaged checks and complete
M512 graph comparison.

The first fused prefill build passes 52 checks, including FP32 cancellation.
For Flash gate M512, its best complete path is 147.802 us (64 rows, split 1),
versus 93.665 us canonical and 94.264 us for the best workspace candidate.
Other fused tiles/splits are slower. It therefore has no speed eligibility.
NCU reports 4,497,888 DRAM read bytes, 128 registers, 14.99% active warps,
43.36% long-scoreboard stall and 10.42% short-scoreboard stall. Tensor-pipe
instruction count is 7,864,320. The next experiment queues activation vectors
in registers before independent IQ decoding, then commits the shared activation
tile before MMA, preserving all operand bits and FP32 accumulation.

Overlapping activation staging passes 52 checks and improves fused prefill
from 147.802 to 143.102 us, still well behind 92.500 us canonical. The fused
implementation and its capabilities are removed; the negative results remain.

The next workspace candidate uses cuBLASLt `CUBLAS_COMPUTE_32F` and filters
split-K algorithms to `CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE`. In-place and
output-type reductions are rejected, including at execution. Plans are opaque
CPU metadata prepared before graph capture, without persistent caches; FP16
dequantization and the library workspace remain caller-owned GPU scratch.
Numerical tests cover every returned algorithm with changed-input graphs and
partials above FP16 range. No precision reduction or model default is implied.

The first cuBLASLt build triggers an nvcc internal assertion in GCC 13
`stl_construct.h` while compiling the new host heuristic containers. Algorithm
preparation and library descriptor management are moved to a normal C++ source
in the same CMake extension; CUDA packet decoding remains in its CUDA source.
No private library or runtime overlay is introduced.

The packaged cuBLASLt candidate passes 46 focused checks. Its two heuristic
plans take 120.04/122.72 us versus 91.90 us canonical on the Flash gate M512
shape. A standard-library configuration probe enumerates 273 valid FP32 plans
with NONE or COMPUTE_TYPE reductions only. Best algorithm 21, tile 20, stage 0,
split 1 takes 94.064 us including packet dequantization, versus 89.785 us
canonical in the same process (4.8% slower). Distinct eight-bank full graphs
and official FP16-weight/FP32-dot numerical comparisons are used. The probe
does not establish a speed-eligible Lt default or model step savings.

Flash M1 follow-up with the retained source passes its numerical gate: IQ3_S
expert 6.756 versus 16.289 us, IQ2_S expert 9.027 versus 18.088 us, dense gate
9.232 versus 17.996 us, output 8.762 versus 15.317 us. Shape-dependent original
vector schedules remain necessary; full-scale factoring is not always best.

The next dense-vector candidate interleaves four copies of the IQ3_S byte
codebook across shared-memory banks. It changes only CTA-local lookup layout;
original payload, scale reconstruction, FP32 FMA, and persistent storage stay
unchanged. The single-copy schedule remains available for comparison.

Four-way interleaved IQ3_S vector codebooks pass 48 focused checks but regress
dense gate/down to 34.571/33.366 us against the same-run single-copy
32.683/30.844 us. Non-prefetched variants regress too. The four-copy candidate
is removed; existing raw vector schedules and storage remain unchanged. The
canonical controls in this comparison are 33.926/31.370 us; earlier paired
measurements still show process/tactic variability, so M1 is not declared
unconditionally closed from this one run.

Original-byte grouped MMA now prepares exclusive expert tile counts on GPU
and resolves each tile by binary search over that prefix. The graph launch
bound is ceil(total routed rows/16)+E-1; empty experts do not initialize a
codebook or run MMA. Valid tiles reuse the measured N32/M16 register-lookahead
body with FP32 accumulators and original scale bits. Routing offsets and
caller-owned prefix scratch may change between graph replays. No CPU routing
synchronization, per-token expert launch, persistent FP16 weight, or default
model dispatch is added. Numeric and grouped timing gates remain pending.

The normal grouped wheel (`d374319288`, core SHA256
`646784f2234a34e59121b183cb930be4a37d1e3da780777276b107ffd566d060`)
passes 54 checks in 12.57 s, including changed routing, empty experts, distinct
expert scale bits, multi-K-block lookahead, and 1/7/512/1024 experts. No hidden
library overrides are used; 210 installed dependencies are compatible.

Real Flash expert gate comparisons use TP4 N160/K2560, E512 and ten distinct
experts per input token. Graphs cycle ten distinct complete weight banks, so
C1 active payloads exceed twice 6 MiB L2. Sorting and the rest of the FFN are
excluded. Original-bit GPU packet conversion is performed before timing.

| Format | Input tokens | Routed rows | Raw grouped us | Canonical grouped us |
|---|---:|---:|---:|---:|
| IQ3_S | 1 | 10 | 29.324 | 72.639 |
| IQ3_S | 5 | 50 | 37.518 | 94.395 |
| IQ3_S | 8 | 80 | 54.609 | 113.043 |
| IQ3_S | 16 | 160 | 79.822 | 141.101 |
| IQ3_S | 512 | 5120 | 331.224 | 525.757 |
| IQ2_S | 1 | 10 | 30.800 | 61.159 |
| IQ2_S | 5 | 50 | 48.837 | 81.237 |
| IQ2_S | 8 | 80 | 57.563 | 94.022 |
| IQ2_S | 16 | 160 | 88.404 | 104.694 |
| IQ2_S | 512 | 5120 | 388.271 | 473.192 |

Output relative L2 is 0.0002036–0.0002122 against official FP32 reconstruction
rounded to the existing FP16 MMA operands, with FP32 dot accumulation. Full
model numerical/quality and step-time comparisons remain separate gates.

C8 NCU graph-node counters after L2 eviction show grouped main-node DRAM reads
of 14,215,232 bytes (IQ3_S) and 10,761,856 bytes (IQ2_S), against 13,552,000
and 10,102,400 active original payload bytes. The prefix node reads 9,184 bytes
for each format. Main nodes use 122 registers and 18,688/24,832 shared bytes,
19.99/16.40% active warps and 24.90/37.33% long-scoreboard stall. Activations,
routing and memory-sector effects are included in DRAM counts; the counters
do not prove one exact physical read per source byte. Profiler durations are
not used to derive model TPOT. Persistent packet storage remains 90,112,000
and 67,174,400 bytes per 512-expert projection, exactly the original TP payload.

`benchmarks/kernels/benchmark_gguf_compact_grouped.py` provides the reproducible
full-graph comparison, automatic weight-bank sizing, original-byte budget,
official numerical oracle and per-route times. Routing sorting and the complete
MoE layer must be measured after model integration before claiming step savings.

The committed grouped benchmark CLI is also exercised on the complete IQ2_S
E512/top-10/TP4 C8 shape. It selects ten banks automatically, passes the same
numerical oracle (relative L2 0.0002036) and measures 65.684 us raw versus
94.024 us canonical with 20 replays. Earlier 50-replay raw timing is 57.563 us;
that cross-process variation is retained rather than hidden. Both comparisons
improve over their matched canonical controls.

The next FP32 cuBLASLt comparison exposes the existing coalesced natural
[N,K] temporary workspace as well as [K,N]. Descriptor transpose and leading
dimensions change together; persistent weights and decoded operand bits do
not. The same NONE/COMPUTE_TYPE reduction admission remains enforced, and
natural dequantization uses its existing single-partition shared transpose.
Both layouts receive changed-input graph and FP16-overflow cancellation tests.
No model default or precision change is made from this unmeasured candidate.

Natural-workspace cuBLASLt passes 58 focused checks in 13.47 s, including
overflow cancellation and changed-input graphs for both formats/layouts.
Eighty valid FP32 configurations from algorithms 3/21/23 are tested on the
Flash gate M512 shape; best is algorithm 21, tile 15, stage 0, no split:
104.300 us versus 89.683 us canonical, relative L2 0.0002070. Natural-layout
Lt support is removed; the existing cuBLAS layout controls and original Lt
interface remain. No precision or persistent-storage change is introduced.

The complete Flash checkpoint header inventory contains 75,828,974,080 tensor
bytes (70.62 GiB) across fourteen storage types, including IQ3_XXS, IQ3_S,
IQ2_XXS/XS/S, IQ4_NL/XS, Q2_0, Q4_K/Q5_K/Q6_K and F16/BF16/F32. The BF16
source-bit scan covers 763,944,960 elements across 484 tensors: zero nonfinite
values and zero finite values above FP16 range. No clamping or conversion is
applied by the scan. This is an input-overflow check, not a model quality gate.
TP block-boundary inventory remains conditional on the actual adapter axis;
column-parallel projections must not be rejected because their unused K-split
would cut a source block.

A follow-up full-graph comparison of existing FP16/FP32 output workspaces,
cuBLAS algorithms 99/102, and dequantization partitions 1/2/4 times out waiting
for the shared GPU lock before executing. It contributes no speed or numerical
evidence. Recorded successful comparisons and unresolved prefill gates remain
unchanged.

The installed retained grouped artifact passes two additional E512 routing
graph checks. Existing workspace combinations retain FP32 computation: best
FP32 output/default cuBLAS/two DQ partitions is 93.983 us versus 89.888 us
canonical; FP16 output/algorithm 2/two partitions is 94.956 us. A standard
cuBLASLt support query finds no valid mixed FP32-C/FP16-D configurations on
this runtime, so no mixed-output operator is added.

The next DQ candidate stages the original d/small-scale planes once per CTA
instead of loading them separately in all eight warps. It publishes through
the existing codebook barrier and queues each warp's first original packet
before setup. Shared planes retain original bits; there is no expanded FP16
coefficient, persistent overhead, precision change, or model default.

Cooperative metadata passes 58 focused checks in 13.84 s, including exact
temporary FP16 weights, changed inputs and overflow cancellation with both
FP16/FP32 results. In full graphs on the real Flash gate M512 shape (eight
distinct banks), FP32 output/default cuBLAS/two partitions improves from
94.088 to 92.978 us; FP16 output/algorithm 2/two partitions improves from
96.244 to 94.404 us. Four-partition comparisons show little benefit. Same-run
canonical is 89.941 us, leaving a 3.4% speed gap. No model default is selected.

The normal compiler reports unchanged 32 registers and no spills for both
formats, with only 384 additional shared bytes. NCU of the FP32-output winner
shows DQ reads 1,727,552 bytes, 70.99% active warps and 35.90% long-scoreboard
stall (prior DQ comparison: 42.73%). Its cuBLAS kernel uses 138 registers,
14.21% active warps and reads 10,981,856 bytes; final FP16 copy reads 3,147,168
bytes. These nodes and byte budgets are reported independently; profiler
durations are not used as model step latency. The public dense benchmark now
includes cooperative metadata and FP32-output partition controls.

A CPU Q2_0 TP4 fragment prototype on three real down experts (N2560/K640)
produces exact original-byte inverses and FP32 reconstruction for local K160.
It uses 46 unaligned bytes per rank row, totaling 184 versus 180 original
bytes: four duplicated original scale bytes per complete row. This is not a
strict equal-byte implementation, has no GPU kernel or default, and remains
separate from the accepted IQ3_S/IQ2_S storage layouts.

The complete real-weight FP32 GEMM algorithm check times out before acquiring
the shared GPU lock; it provides no new algorithm performance evidence.

The real-weight FP32 GEMM algorithm probe completes after retry. Seventeen
standard cuBLAS algorithms pass exact cancellation with FP32 results when
positive/negative partials exceed FP16 range. On actual Flash gate TP4 weights,
algorithm 11 GEMM plus final FP16 cast is 78.982 us versus 80.347 us default
(eight-bank full graphs); dequantization is excluded. Algorithm 11 is admitted
only with FP32 output, with a native rejection for FP16 output and installed
changed-input/overflow regressions. The complete-path speed gate is pending.

The packaged algorithm-11 change passes 62 focused checks in 13.30 s. Its
FP32-output/two-partition/cooperative-metadata complete graph takes 92.286 us
versus 89.942 us same-run canonical on the real Flash gate M512 shape, with
relative L2 0.0002074. The remaining speed gap is 2.6%; no model default or
step savings are claimed. FP16-output algorithm 11 remains explicitly rejected.
The public benchmark includes this measured FP32-only candidate.

### Equal-byte TurboMind planes

An additional lossless GPU permutation puts low grid indices/sign packets in
TurboMind's existing U2 MMA carrier order. Separate original-bit planes retain
FP16 block d, packed scale nibbles, high index bits and IQ3_S signs. They total
110/82 bytes per block for IQ3_S/IQ2_S, with no expanded persistent coefficient
array. The ordinary SM70 mainloop reconstructs its metadata carrier only in
registers/shared memory. The original small factor times grid is exact in FP16;
the final FP16 operand matches official FP32 dequantization rounded to FP16.
MMA accumulation and split reductions remain FP32.

The normal wheel passes 68 checks, including independent byte-for-byte inverse,
comparison of U2 carriers with the existing converter, changed-input full
captures and official FP16-weight/FP32-dot oracles. Six real TP4 shapes at
M1/5/8/16/512 have relative L2 0.000202–0.000223. Eight distinct banks are
replayed in full CUDA graphs on V100-SXM2-32GB, Torch 2.10.0+cu128 and CUDA 12.8.

| TP4 projection | M512 planes (µs) | Same-run canonical (µs) |
|---|---:|---:|
| 27B IQ3_S gate N4352 K5120 | 531.431 | 496.522 |
| 27B IQ3_S down N5120 K4352 | 447.723 | 378.257 |
| Flash IQ3_S expert N160 K2560 | 37.710 | 68.123 |
| Flash IQ2_S expert N160 K2560 | 37.054 | 63.015 |
| Flash IQ3_S dense gate N1536 K2560 | 108.252 | 89.491 |
| Flash IQ3_S output N2560 K1536 | 99.157 | 81.673 |

Small-M plane GEMMs regress; the existing vector and compact grouped candidates
remain preferable. No model default selects this prototype. After cache
 eviction, Flash dense gate NCU selects M128/N256/K32 with 144 CTAs, 254 registers,
65,552 dynamic shared bytes and 12.47% active warps. It reads 4,509,440 DRAM bytes
versus the 1,689,600-byte source payload, including activation/output/reduction
traffic. These counters do not establish model step savings. Additional
N128/M64 candidates do not improve the gate: 108.208 versus 89.786 µs.

A warp transpose to replace eight scalar workspace stores with one vector
store passes 68 checks but leaves the full FP32-output path at 92.449 versus
90.070 µs canonical (2.6% gap). It is removed because it adds shared memory
without a confirmed speed benefit. The original compact workspace schedule
and precision contract are retained. End-to-end integration remains pending.

A bounded thread-layout comparison checks 128/256/512 DQ threads with valid
one/two/four-way partitions, keeping the original byte layout and exact
workspace values. The normal merged wheel passes 126 checks. Flash dense gate
M512 full graphs remain fastest with 256 threads/two partitions: 92.326 versus
90.060 µs canonical. The alternatives range from 92.338 to 96.293 µs and do
not close the speed gap. No thread-count default changes.

Packed-integer IQ3_S workspace factor formation is mathematically exact:
130,023,424 CPU weight comparisons and 126 installed checks show zero changed
FP16 weight bits. Its complete graph remains 92.268 versus 90.028 µs canonical,
so it is removed. The original shared-metadata path remains unchanged.

Plane capabilities admit only the measured faster N160/K2560 expert at M512.
Dense descriptors report measured regression, uncalibrated shapes report that
reason, and unavailable operators, hardware, types and dimensions explain
fallback. This is operator admission; model storage is not promoted by it.

Adjacent-row vector metadata loads pass 129 checks but do not improve the
Flash dense gate (108.183 versus 89.500 µs) and regress the 27B down projection
(487.469 versus 378.309 µs). This candidate is removed. A general M32/N128
candidate is evaluated next to reduce per-CTA resource demand; performance
eligibility is unchanged until it passes matched measurements.

The M32/N128 repertoire with exact integer IQ3_S factor formation passes
129 checks. Dense M512 descriptors remain slower and are excluded from plane
capability admission. Single-expert eight-bank prototype measurements do not
exceed twice L2; they are retained as cached controls, not cold-bank evidence.
Corrected captures use 72 IQ3_S banks (12,672,000 bytes) and 96 IQ2_S banks
(12,595,200 bytes), both above twice the 6,291,456-byte V100 L2. They measure
35.325/36.733 µs versus 68.631/64.234 µs canonical at N160/K2560/M512, with
relative L2 0.0002063/0.0002076. Activations remain FP16 and accumulation FP32.

NCU after explicit eviction reads 2,824,864/2,783,008 bytes for IQ3_S/IQ2_S,
versus 2,797,440/2,752,640 bytes of activation plus original weight payload.
Other reads are included; these totals are not attributed solely to weights.
The selected kernels use 141/120 registers, 16,400/14,368 dynamic shared bytes,
and 11.39/22.67% active warps. Profiler node times are not model step times.

Paired Nsight Systems full-graph node traces also retain the remaining dense
workspace gap: profiled means are 16.569 µs DQ, 91.450 µs FP32-output GEMM and
4.207 µs final cast, versus 108.704 µs canonical fused GEMM. Profiling changes
these absolute times; matched unprofiled comparisons remain about 92.3 versus
90.0 µs. Model defaults and end-to-end savings remain pending.

### Existing canonical scale rounding

A CPU comparison quantifies the existing canonical group-scale rounding; it
is not a new original-byte operator. Compared with official FP32 weights,
maximum relative error is 0.0487% in the measured IQ3_S/IQ2_S projections.
After final FP16 operand formation, weight relative L2 is 0.0352% for Flash
dense gate, 0.0347%/0.0346% for 27B gate/down and 0.0337% for the IQ2_S expert.
The corresponding maximum absolute FP16 weight differences are 0.0002441,
0.0001221, 0.0001221 and 0.0000610. All measured weights remain finite.

These rounding statistics are not output-quality or speed evidence for a
new decoder. The current original-byte operators retain exact final operands
and FP32 accumulation; no pre-rounded-scale original-byte path is implemented.
