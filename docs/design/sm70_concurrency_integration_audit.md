# Concurrent projection integration audit

AI-assisted integration of PR #476 onto main, preserving #548/#549/#550.
The old experimental results in the migration log retain their historical
attribution; the decisions below describe the integrated defaults.

## Accepted paths and fallback

Channel-FP8 QPN8 M9-16 and M17-32 projection paths default on for measured
hardware/layout/geometry contracts. Native M17-32 dense uses two ordered
split-K phases; gated projections use M16 chunks. Unsupported split-K plans
retain the old reconstruction/GEMM path instead of raising after promotion.
The three `VLLM_SM70_FP8_QPN8_M*` switches remain explicit rollbacks.
Block-scaled FP8 and unmeasured shapes retain their earlier routes.

`VLLM_SM70_NVFP4_QPN2_M16_NATIVE` defaults on. Dense split-16 and gated
split-8 use the validated two-row tile at M9-16. Larger rows retain their
existing kernel. There is no M64 FP8 or M32 NVFP4 promotion.

Request-major batched grouped verification and enlarged push-allreduce
messages remain opt-in: retained evidence includes short-context regressions,
so neither is promoted universally. Main's batch/sum2/HC collectives remain
available. Both collective kernels use the enlarged payload stride; HC
signals, payloads and generation words begin beyond that whole region.

## Fresh source validation

- Fresh CUDA 12.8 QPN native sidecar and Flash-V100 build.
- 247 kernel/routing/rejection tests passed, including 23 new QPN tests
  against FP64 same-weight references, rollback and graph replay.
- 266 policy tests passed; 13 GPU-required checks skipped in that CPU run.
- TP4 direct native push/sum2: seven message cases, seven changing-value
  graph replays each; correct outputs and unchanged HC-region canaries.
- No full-model end-to-end suite rerun, no greedy-identity requirement.

The fresh per-channel FP8 operator benchmark uses the retained mixed
checkpoint, not a block-FP8 model that cannot hit these routes. NVFP4 uses
actual layer-55 TP4-local weights. Activations are synthetic, paired graphs
alternate for five samples on one owned V100-SXM2-32GB. Environment:
Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8 compiler/runtime.

| Projection | M | Previous path (us) | Candidate (us) |
| --- | --- | --- | --- |
| FP8 down | 16 | 185.290 | 43.277 |
| FP8 down | 32 | 189.456 | 63.818 |
| FP8 output | 16 | 63.376 | 18.365 |
| FP8 output | 32 | 73.898 | 22.976 |
| FP8 gate/up + SiLU | 16 | 356.234 | 87.165 |
| FP8 gate/up + SiLU | 32 | 355.894 | 174.346 |
| NVFP4 gate/up + SiLU | 16 | 67.222 | 65.363 |
| NVFP4 down | 16 | 34.979 | 30.573 |

FP8 relative L2 versus the old path ranges from 0.00235% to 0.06636%;
independent FP64 reference tests also pass. NVFP4 paired outputs are exact.
These timings are operator evidence, not a fresh endpoint throughput claim.
The original PR's retained 44.9% B2 / 30.6% B4 endpoint improvements and
completed coherent-output checks remain separately attributed to that build.
