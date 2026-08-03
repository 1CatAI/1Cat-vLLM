# SM70 DeepSeek V4 Sparse MLA Split-K

## Scope

- Base: `dd462e37f2552f3e038f1ed7128e62bd7b4ab0d7` (PR #159)
- Model: DeepSeek-V4-Flash, TP8 on 8 x V100-SXM2-32GB
- Decode: CUDA Graph, FP16 query/output, packed `fp8_ds_mla` KV
- Quantization: TurboMind MXFP4; Marlin is out of scope
- Route gates: `VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_SWA`,
  `VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_C4`, and
  `VLLM_SM70_DSV4_SPARSE_MLA_SPLITK_C128`

All three gates remain default-off until same-contract full-model speed and
quality checks pass. SWA, C4, and C128 use separate gates so any route can be
rejected without weakening the others.

## Trace Baseline

The accepted active-expert baseline is exact 1024 input / 256 output,
official `temperature=1.0`, `top_p=1.0`, no MTP, and CUDA Graph. Unprofiled
TPOT is `76.268 ms/token` (`13.112 tok/s`). The graph-node trace records
`78.087 ms/token`; use it for composition only.

| Sparse MLA layer type | Layers | Mean/layer | Per-token service |
|---|---:|---:|---:|
| SWA-only | 2 | 0.548 ms | 1.095 ms |
| C4 | 21 | 1.651 ms | 34.666 ms |
| C128 | 20 | 0.566 ms | 11.326 ms |
| Total | 43 | - | 46.920 ms |

Raw graph-node artifacts:

```text
/home/fudanwl/v100-worktrees/runs/
  dsv4-tp8-active-expert-b1-nsys-i1024-o256-retry1-20260802/
```

## Root Cause

TP8 leaves eight query heads per rank. The baseline uses `BLOCK_H=8`, so each
layer launches one CTA and serially scans every sparse KV block on one of the
V100's 80 SMs.

Exact C4 NCU evidence:

| Metric | Baseline |
|---|---:|
| Grid / block | `1 CTA` / `128 threads` |
| Duration | 2.11 ms |
| Registers | 32/thread |
| Dynamic shared memory | 51.71 KiB/CTA |
| Achieved occupancy | 6.25% |
| SM throughput | 0.20% |
| DRAM throughput | 0.03% |
| Scheduler cycles with no eligible warp | 92.61% |
| Long-scoreboard share of issue interval | 56.92% |

The bottleneck is insufficient CTA parallelism and exposed KV/dequant latency,
not saturated HBM or tensor-core throughput.

## Implementation

The candidate uses Flash-Decoding-style KV partitioning:

1. One CTA handles one 16-token sparse KV block and writes FP32 partial
   `(max, sum, weighted-value)` state.
2. A second kernel combines partial states in FP32 and applies the attention
   sink before writing FP16 output.
3. The C4 1024-token shape launches 40 stage-1 CTAs and 64 reduction CTAs.
   The SWA-only shape launches eight stage-1 CTAs instead of one serial CTA.
4. Scratch comes from the graph-safe worker workspace and is reused across
   layers; there is no hot-path allocation or host synchronization.
5. E4M3 normal values are decoded by exact IEEE-FP32 bit construction. The
   seven NOPE scales are loaded and decoded once per 64-element group, then
   broadcast instead of being redundantly expanded 64 times.

## Microbenchmark Evidence

Exact q=1, eight-head CUDA Graph measurements:

| Shape | Baseline | Candidate | Speedup | Max abs error |
|---|---:|---:|---:|---:|
| C4, main 128 + extra 320 | 1.957 ms | 0.101 ms | 19.4x | 1.53e-5 |
| C128, main 128 + extra 10 | 0.581 ms | 0.078 ms | 7.4x | 3.05e-5 |
| SWA-only, main 128 | 0.586 ms | 0.059 ms | 9.9x | 3.05e-5 |
| C128, extra 512 | 4.899 ms | 0.100 ms | 49.1x | 7.63e-6 |
| C128, extra 1024 | 9.265 ms | 0.119 ms | 77.9x | 7.63e-6 |
| C128, extra 2048 | 17.966 ms | 0.152 ms | 118.2x | 7.63e-6 |

Latest C4 NCU stage times are `87.74 us` for split-K and `5.22 us` for the
reducer. Stage-1 reaches 40 CTAs and 12.93% DRAM throughput. These are kernel
measurements, not an end-to-end claim.

Numerical and graph gates completed:

- all 254 finite E4M3FN byte encodings match the arithmetic decoder bitwise;
- both E4M3FN NaN encodings preserve the same NaN mask;
- realistic FP8 KV tests over multiple seeds have max absolute output error
  at or below `1.53e-5` for the main C4 target;
- q=1 and q=2 CUDA Graph capture/replay complete with finite output;
- SWA-only q=1 passes three seeds and q=2 graph replay stays within `3.05e-5`;
- C128 lengths through 2048 compressed tokens remain finite and within the
  recorded error bound.

Artifacts:

```text
/home/fudanwl/v100-worktrees/runs/
  dsv4-sm70-sparse-attn-micro-20260802/
  dsv4-sm70-sparse-swa-splitk-micro-20260802/
```

## Full-Model TP8 Result

The combined C4+C128 candidate was measured with the same accepted contract as
the trace baseline: exact 1024 input / 256 output, official
`temperature=1.0`, `top_p=1.0`, no MTP, FP8 MLA KV, and FULL CUDA Graph.

Three unprofiled runs used seeds 4201-4203. All consumed exactly 1024 prompt
tokens, emitted 256 tokens, and stopped at the length limit.

| Metric | Active-expert baseline | C4+C128 split-K | Change |
|---|---:|---:|---:|
| TPOT | 76.268 ms | 33.853 ms | -55.61% |
| Decode throughput | 13.112 tok/s | 29.539 tok/s | +125.29% |
| TTFT | 1789.254 ms | 1787.083 ms | -0.12% |

Candidate TPOT was 33.869, 33.873, and 33.818 ms across the three runs. The
mean interval p50/p90/p99 values were 33.845/35.104/36.928 ms. Basic output
health checks found no NUL, replacement-character, or single-character-repeat
failure. This is a text-health smoke, not the remaining deterministic
logit/token or semantic-quality gate.

The fresh graph-node capture contains 255 decode steps x 8 ranks. Aggregate
statistics use 247 middle steps and have 98.20% graph-node kernel coverage.
Node tracing raises request TPOT to 35.409 ms, so the following values are for
composition, not accepted absolute speed.

| Timing view | Mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| TP rank replay interval max | 35.734 ms | 35.577 | 36.225 | 38.887 |
| TP rank GPU service max | 35.217 ms | 35.073 | 35.726 | 38.350 |
| Replay minus GPU service | 0.517 ms | 0.561 | 0.871 | 1.141 |
| Replay-start skew | 0.740 ms | 0.606 | 1.102 | 3.379 |

GPU service by category:

| Category | Rank-average | p50 | p90 | p99 | Rank-max mean | Launches/rank/token |
|---|---:|---:|---:|---:|---:|---:|
| TurboMind MXFP4 MoE | 7.542 ms | 7.543 | 7.611 | 7.857 | 7.671 | 516 |
| TurboMind FP8 dense GEMM | 6.251 ms | 6.251 | 6.309 | 6.516 | 6.478 | 279 |
| SM70 sparse MLA attention | 4.392 ms | 4.392 | 4.422 | 4.535 | 4.471 | 84 |
| TP all-reduce/communication | 4.045 ms | 4.035 | 4.197 | 4.423 | 4.519 | 87 |
| Dense GEMV/GEMM and compressor | 3.380 ms | 3.379 | 3.406 | 3.542 | 3.442 | 252 |
| mHC TileLang | 2.980 ms | 2.980 | 3.011 | 3.107 | 3.040 | 174 |
| MoE routing/activation | 2.412 ms | 2.411 | 2.440 | 2.521 | 2.474 | 363 |
| Q/KV preparation and cache | 1.360 ms | 1.207 | 1.772 | 1.790 | 1.407 | 234 |

The sparse MLA service consists of 41 split-K stages at 3.125 ms/token, 41
reducers at 0.173 ms/token, and two unsplit SWA calls at 1.094 ms/token. Total
sparse service fell from 46.920 to 4.392 ms/token (-90.64%). The 42.527 ms
sparse reduction translated into a 42.415 ms unprofiled TPOT reduction, so the
endpoint realizes 99.7% of the measured kernel-family saving.

This endpoint result predates the SWA-only candidate. The remaining two unsplit
SWA calls contribute `1.094 ms/token` in that trace. The SWA microbenchmark
projects about `1.05 ms/token` additional service reduction, but that value is
not an endpoint claim until the accumulated full-model gate completes.

No output-position decay is visible in this 256-token trace:

| Emitted-token positions | Replay interval | GPU service | Sparse rank-max |
|---|---:|---:|---:|
| 5-64 | 35.738 ms | 35.201 ms | 4.474 ms |
| 65-128 | 35.589 ms | 35.101 ms | 4.461 ms |
| 129-192 | 35.802 ms | 35.281 ms | 4.469 ms |
| 193-251 | 35.813 ms | 35.290 ms | 4.481 ms |

Raw endpoint and graph-node artifacts:

```text
/home/fudanwl/v100-worktrees/runs/
  dsv4-tp8-active-expert-splitk-both-i1024-o256-20260802-2008/
  dsv4-tp8-active-expert-splitk-both-nsys-i1024-o256-20260802-2020/
```

## Rejected Variants

| Variant | C4 graph mean | Decision |
|---|---:|---|
| 2 stage-1 warps | 0.481 ms | Reject; too little intra-CTA latency hiding |
| 4 stage-1 warps | 0.174 ms before scale grouping | Keep |
| 8 stage-1 warps | 0.279 ms | Reject; extra warps increase cost |
| `BLOCK_H=4` | 0.278 ms | Reject; duplicates KV/HMMA work to fill 80 CTAs |
| `BLOCK_H=8` | 0.177 ms before scale grouping | Keep |

## Remaining Gates

1. Run the accumulated C4+C128+SWA endpoint speed and quality gate.
2. Compare deterministic tokens/logits and run a model-specific semantic
   quality gate with official sampling.
3. Sweep long-context decode before any route becomes default-on.
4. Run C4-only endpoint attribution only if a later regression requires C128
   to be isolated from the accepted combined path.
