# SM70 DeepSeek V4 MXFP4 Grouped Decode

## Scope

- Model: DeepSeek-V4-Flash with MXFP4 experts
- Runtime: TP8 on 8 x V100-SXM2-32GB
- Decode: batch one, top-k 6, CUDA Graph, no MTP, no eager execution
- Quantization backend: TurboMind only; Marlin is out of scope
- Dependency: active-expert route from PR #160
- Endpoint stack also includes sparse MLA split-K from PR #163

The goal is to reduce the remaining six active-expert launches without changing
expert selection, SwiGLU clamp, accumulation order, or output precision.

## Accepted Baseline

The same-contract baseline uses exactly 1024 input tokens and 256 generated
tokens, official `temperature=1.0` and `top_p=1.0`, FP8 MLA KV, and TP8.

| Metric | Value |
|---|---:|
| TPOT, three-run mean | 33.853 ms/token |
| Decode throughput | 29.539 tok/s |
| MXFP4 GPU service | 7.542 ms/token |
| MXFP4 launches | 516/rank/token |

The 516 launches are `43 layers * 6 experts * 2 stages`. Each expert has one
row, so independently launching every row leaves most of V100 idle.

## Compact Grouped Stage

The compact scheduler treats six one-row experts as six logical source groups
inside one TurboMind launch. It bypasses the generic grouped-offset search but
keeps the same prepared weights and GEMM arithmetic.

Exact CUDA Graph microbenchmarks:

| Stage | Six independent launches | One compact launch | Result |
|---|---:|---:|---|
| W13, K4096/N512 | 0.1156 ms | 0.0286 ms | Bitwise equal |
| W2, K256/N4096 | 0.0433 ms | 0.0117 ms | Bitwise equal |

The latest graph-node trace confirms that MXFP4 service falls from 7.542 to
1.957 ms/token and launches fall from 516 to 86. A temporary endpoint candidate
containing this route and a later-rejected FP8 split measured 28.961 ms/token,
or about 34.53 tok/s. Because the FP8 split regresses the real overlapped
timeline, 28.961 ms is evidence that the grouped route transfers end to end,
not a final release result.

## Exact Direct Top-6 Route

The generic batch-one route still sorts and permutes against 256 experts before
the compact W13 stage. The direct route performs a stable six-ID sort and input
replication in one CUDA kernel, then reuses the exact production kernels for:

1. compact grouped W13;
2. `silu_and_mul_with_clamp` with the checkpoint's clamp value 10.0;
3. compact grouped W2;
4. `moe_unpermute` and the original weighted reduction.

Five CUDA Graph seeds, including route IDs changed after capture, pass bitwise
checks for gate/up, clamped intermediate, sorted W2 output, inverse permutation,
and final output.

| Pipeline | Median per layer | Projected 43-layer service |
|---|---:|---:|
| Generic routing plus compact GEMMs | 0.101144 ms | 4.349 ms |
| Exact direct top-6 | 0.051226 ms | 2.203 ms |
| Saving | 0.049918 ms | 2.149 ms/token |

This is operator evidence. It is not yet an endpoint claim.

## Accumulated Candidate

The next endpoint gate combines:

- compact grouped MXFP4;
- exact direct top-6 routing;
- removal of the rejected FP8 projection split;
- SWA-only split-K from PR #163.

Based on measured operator deltas, the current projection is approximately
25.3 ms/token, or 39.5 tok/s. Only an unprofiled three-run TP8 result can replace
that projection.

## Latest Accepted Endpoint

The accumulated TP8 1024/256 run measured 25.663779, 25.605544, and
25.621125 ms/token. The accepted mean is **25.630149 ms/token**, or
**39.01655 tok/s**. Relative to the original 33.853 ms/token baseline, this is
a 24.3% TPOT reduction and a 32.1% decode-throughput increase.

The latest graph-node trace attributes the following aggregate GPU service per
token. These categories overlap across streams and are not additive wall-time
savings.

| Category | Service per token |
|---|---:|
| FP8 dense | 5.964 ms |
| TP communication | 4.192 ms |
| Sparse MLA | 3.448 ms |
| mHC | 2.977 ms |
| FP16 GEMV/compressor | 2.395 ms |
| Routing | 2.134 ms |
| MXFP4 | 1.979 ms |
| Q/KV preparation | 1.364 ms |

## Rejected Paths

| Path | Evidence | Decision |
|---|---|---|
| Generic grouped TurboMind dispatch | W13 0.1156 -> 0.1362 ms | Reject |
| CTA_N=32/64 tactic-only changes | W13 0.1136 -> 0.1165 ms | Reject |
| Split fused WQA/WKV into two FP8 GEMMs | Single-stream faster, but four-stream model overlap regresses about 0.556 ms/token | Removed |
| mHC `tile_n=1` | Bitwise equal, only about 0.035 ms/token projected | Stop at marginal gain |
| Parallel FP8 KV insert blocks | Bitwise equal, about 0.12 ms/token projected | Stop below 0.2 ms threshold |

The FP8 split result is the key scheduling lesson: service-time improvements
on an auxiliary stream are invalid unless the full overlap timeline also
shortens.

## FP8 Dense CTA Screen

The first exact-shape FP8 microbenchmark accidentally initialized the
TurboMind workspace during CUDA Graph capture. Its replay therefore included
workspace fill kernels and overestimated standalone FP8 service. The benchmark
now warms the operation on the capture stream before capture and can delimit a
single steady replay with `cudaProfilerStart`/`cudaProfilerStop`.

With the corrected method, the six dense FP8 shapes project to 3.906 ms/token
in isolation. The latest full-model trace reports 5.964 ms/token of aggregate
FP8 service; the difference is multi-stream contention and profiling overhead,
not extra model GEMMs.

NCU on the main `M1 K4096 N1536` projection showed only 60 CTAs, 6.21% achieved
occupancy, 82.76% scheduler cycles with no eligible warp, and 8.2-way average
shared-load conflict. Exact `CTA_N=64/32` candidates increased the launch to
120/240 CTAs and used the existing conflict-free `A[8,64]` shared layout. They
were tested with the baseline `split-K=5`; allowing the tuner to choose
`split-K=7` is not a quality-equivalent comparison.

| Exact route | Median | Delta from CTA_N=128 |
|---|---:|---:|
| CTA_N=128, split-K=5 | 21.212 us | baseline |
| CTA_N=64, split-K=5 | 22.558 us | +6.3% |
| CTA_N=32, split-K=5 | 23.541 us | +11.0% |

The smaller tiles duplicate activation/metadata traffic and add scheduling and
split epilogue work. Other model streams already consume the nominally idle
SMs, so increasing this kernel's CTA count does not recover end-to-end wall
time. The candidates are rejected before endpoint testing.

## Artifacts

```text
/home/fudanwl/v100-worktrees/runs/
  dsv4-mxfp4-grouped-decode-micro-20260802/
  dsv4-mxfp4-direct-top6-clamp10-micro-20260802/
  dsv4-fp8-dense-shapes-20260802/
  dsv4-fp8-dense-ncu-20260803/
  dsv4-sm70-kv-insert-parallel-micro-20260802/
  dsv4-tp8-stacked-mxfp4-fp8split-i1024-o256-20260802/
  dsv4-tp8-stacked-candidate-nsys-i1024-o128-20260802/
```

## Remaining Gates

1. Finish official-sampling text health and the model-specific quality gate.
2. Measure exposed, non-overlapped TP communication on the graph critical path.
3. Screen a topology-aware TP8 collective only if its projected wall-time
   saving exceeds 0.2 ms/token.
4. Move to sparse MLA or mHC when the TP path is below that threshold.
