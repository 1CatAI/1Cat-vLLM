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
