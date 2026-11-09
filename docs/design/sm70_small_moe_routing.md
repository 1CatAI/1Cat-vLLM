# Small-batch MoE routing on SM70

Small speculative verification batches spend more time launching sorting,
indexing and restoration operations than computing their route metadata.
The small-batch operator combines expert alignment and activation gathering
in one launch. A second launch restores expert outputs and accumulates their
weighted sum in FP32 before converting to FP16.

Admission requires SM70, contiguous FP16 activations, colocated integer route
IDs, M=1..32, hidden size 2560, 512 experts and top-k=1..16. An opaque custom
operator selects the route using the actual runtime M, including during CUDA
graph capture. Unsupported geometries retain the original Torch operations.
Expert parallel mapping and expert GEMM are outside this change.

Each gather CTA computes the same expert histogram and stable route positions
independently, then copies its activation tile. This avoids a second metadata
launch or cross-CTA synchronization. Offsets are int32; sorted expert IDs
retain the int64 ABI of the existing grouped expert operators. Output
restoration uses the inverse positions directly, without an inverse sort.

## Operator measurements

Hardware: V100 SXM2 32 GB, driver 580.173.02. Runtime: Torch 2.10 CUDA 12.8,
Python 3.12.3, ordinary package `1.5.2.dev491+g33f9dce54.precompiled`.
Source: `33f9dce54c`. The benchmark uses Flash-Next's actual expert geometry
with synthetic activations, route IDs and expert outputs. Timings are medians
of five samples of 100 CUDA graph replays after warmup. They include alignment,
activation gathering and weighted output restoration, but no expert GEMM.

| M | Active experts | Torch chain (µs) | Fused chain (µs) | Estimated saving over 48 layers (ms) | Maximum absolute output error | Relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 42.199 | 7.096 | 1.685 | 1.22e-4 | 6.70e-6 |
| 5 | 45 | 60.303 | 7.475 | 2.536 | 6.10e-5 | 1.47e-6 |
| 10 | 89 | 62.607 | 10.179 | 2.517 | 1.22e-4 | 3.10e-6 |
| 20 | 170 | 147.978 | 14.316 | 6.416 | 2.44e-4 | 5.99e-6 |
| 32 | 241 | 140.667 | 18.371 | 5.870 | 4.88e-4 | 5.64e-6 |

Route membership, offsets and inverse positions are checked against stable
sorting exactly. Weighted output differences reflect FP32 reduction order;
accumulation precision is unchanged. Twenty GPU tests cover random, single
expert and sparse routing, weighted restoration and bitwise graph replay.
Three CPU checks cover capability rejection, opaque fake outputs and fallback
restoration. Model throughput and acceptance require a separate C1/C4 run;
the layer savings above are estimates, not complete-round measurements.

Reproduce under an exclusive GPU lease:

```bash
python benchmarks/kernels/benchmark_sm70_small_moe_routing.py --output RESULT.json
python -m pytest tests/kernels/test_sm70_small_moe_routing.py
```

## Preserve the existing FP32 reduction order

The initial restoration loop sums weighted outputs sequentially. Torch 2.10's
`ATen/native/cuda/Reduce.cuh` uses four independent accumulators for this
short, strided reduction, then combines them in order. Source `4867f38844`
keeps that order inside the fused kernel, with separate FP32 products and
FP16 final output. Thirty-five checks pass, including exact output equality
for M=1..32 and top-k=1/2/3/4/8/10/16 with FP16 or FP32 routing weights.

The original single-invocation graph timing can include host replay gaps when
two short kernels finish before the next replay is submitted. An additional
measurement uses eight operator chains per graph and normalizes by eight.
Source `4a019a1573`, ordinary package `1.5.2.dev495`, retains the same kernel
as the exact-output check. The table uses five samples of 100 graph replays.

| M | Torch chain (µs) | Exact fused chain (µs) | Estimated saving over 48 layers (ms) |
| --- | ---: | ---: | ---: |
| 1 | 46.424 | 5.964 | 1.942 |
| 5 | 54.248 | 5.956 | 2.318 |
| 10 | 56.138 | 8.342 | 2.294 |
| 20 | 134.326 | 13.037 | 5.822 |
| 32 | 140.357 | 16.768 | 5.932 |

Maximum absolute and relative L2 differences are zero at every point. The
single-invocation result for this same kernel was 20.326 µs at M=5, versus
5.956 µs with the longer captured sequence; both measurements are retained.
These remain synthetic-route operator results. The model integration run
used the earlier sequential reduction and measured C1 30.932 ms/round,
78.803 tokens/s and C4 56.924 ms/round, 258.219 tokens/s. C4 remains 1.5%
below the original 262.269 tokens/s, despite normal natural outputs.
The exact reduction's effect on model acceptance has not been established.
