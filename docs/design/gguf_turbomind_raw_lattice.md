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
