# Qwen3.8 concurrent decode: next-stage optimization

## Acceptance target, not a performance claim

Integration base: `1e90d17f2c75e443b2a85a576ed68fa04c5f9dd6` (`onecat/main`).
Scope: Qwen3.8 Flash-Next NVFP4, V100-SXM2-32GB, TP4, no MTP. Improve
C2/C4 and obtain at least **40% more aggregate decode throughput at both
C8 and C16**, without reducing arithmetic precision or regressing quality.

The retained source-built control from [PR #692](https://github.com/1CatAI/1Cat-vLLM/pull/692)
provides planning values, not a newly measured latest-main baseline:

| Concurrency | Historical aggregate decode tok/s | Step ms | +40% tok/s target |
| ---: | ---: | ---: | ---: |
| 1 | 96.85 | 10.325 | Protect against regression |
| 2 | 114.08 | 17.532 | Improve; report separately |
| 4 | 221.77 | 18.036 | Improve; report separately |
| 8 | 372.04 | 21.503 | 520.85 or higher |
| 16 | 557.38 | 28.706 | 780.34 or higher |

The C8/C16 targets correspond to at most 15.36/20.50 ms per batch step.
Forty percent higher throughput requires 28.57% less step time, not 40%.
Before endpoint acceptance, establish a matched high-precision control from
the actual source-built runtime. Do not compare different KV dtypes, DCP,
prefill contracts, model revisions, graph states or precision policies.

Historical contract: 8192 input tokens, 256 forced output tokens for timing,
max context 262144, chunk 8192, max sequences 16, no prefix cache or MTP,
FP16 activations/KV, FP32 GDN state, CUDA graphs, Torch 2.10.0+cu128, CUDA
12.8, driver 580.173.02. Quality tests use natural EOS separately. PLE was
disk-mmap prefill plus pinned-host UVA decode: not a disk-only/no-RAM route.

## Avoid repeating existing experiments

PR #692 already investigates grouped MoE expert reuse, fused ordered W2,
packed GDN input and small gate/collective fusions. Its approximately 9.8%
C16 endpoint candidate still lacks accepted output parity. Reuse that
evidence; neither call it 40% nor default-enable it without quality gates.

[PR #504](https://github.com/1CatAI/1Cat-vLLM/pull/504) covers HC TP output
sharding. The #692 experiments additionally tested pinned-arithmetic cuBLAS
shards and a slower CUTLASS up/mix/publish epilogue. The experiment here is
different: native register-level branch mixing and weight-fragment reuse
across both M8 halves of C16, not a duplicate sharding-policy change.

The fixed-width trace identifies C16 GPU-service costs of approximately
9.0 ms MoE, 4.7 ms HC and 5.8 ms non-HC dense projections. These overlap
in places and are not an additive wall-time budget. Shared/routed expert
stream overlap is already selected by the historical baseline.

## HC register-level up/mix experiment

The candidate packs checkpoint FP16 up weights into warp-contiguous branch
fragments. Four quad pairs calculate four HC branch gates for the same
hidden coordinates. It retains:

- Volta FP16 inputs with FP32 HMMA accumulation;
- the original increasing-K accumulation sequence;
- the FP16 gate materialization boundary;
- FP32 sigmoid and branch-ordered FMA, then the final FP16 result.

At C16, the paired variant reuses each loaded weight fragment for two
independent M8 accumulators. Mixing in registers removes the global gate
scratch and its reload. Both full and quarter-hidden projections are
screened. This does not implement or measure TP communication yet.

The hardware mapping follows NVIDIA's
[PTX MMA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).
Small dimensions can waste tile lanes; fewer instructions/loads are not
sufficient to claim faster execution. Retain only measured winners.

## Reproduction and gates

Use an isolated environment, CUDA 12.8 toolkit and caches, and an exclusively
owned idle SM70 card. There is no production dispatch or default change.
The benchmark builds its own extension directly from the accompanying source;
that extension is a **research-only microbenchmark**, not a production-sidecar
dependency or a reproducible full-model speed claim.

```bash
CUDA_HOME=/path/to/cuda-12.8 TORCH_CUDA_ARCH_LIST=7.0 \
TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions" \
.venv/bin/python benchmarks/kernels/benchmark_sm70_hc_batch_reuse.py \
  --build-only --out .artifacts/hc_batch_build.json

CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/path/to/cuda-12.8 \
TORCH_CUDA_ARCH_LIST=7.0 \
TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions" \
TRITON_CACHE_DIR="$PWD/.cache/triton" \
.venv/bin/python benchmarks/kernels/benchmark_sm70_hc_batch_reuse.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 --pairs 8 \
  --rows 2,4,8,16 --out .artifacts/hc_batch_screen.json
```

The harness disables FP16/BF16 reduced-precision reductions and FP16
accumulation. It checks six input scales on real distinct checkpoint weights,
mutates graph inputs, poisons output buffers, and compares both gates and
mixed outputs bit-for-bit. Quarter-hidden candidates must also match the
corresponding columns of the **replicated** cuBLAS projection: the smaller
GEMM's heuristic alone is not a valid runtime oracle.

Only exact candidates receive alternating-order graph timings. Report these
as microseconds per HC up/mix, never as end-to-end throughput. Admission
requires all 96 HC pairs, all TP shards, dynamic batches, four-card
communication, source-complete build, natural-output health/token checks,
matched unprofiled endpoint measurements and a confirming critical-path trace.

## First screen: exact but insufficient endpoint budget

On an exclusively locked V100, 8 distinct checkpoint HC weights, 40 schedule
configurations and six dynamic-input scales all matched gate and mix outputs
bit-for-bit. This includes the replicated-projection comparison for shards.
These are synthetic activation tests, not full-model output validation.

Representative non-paired, one-warp, unroll-4 graph results (microseconds per
up/mix pair; paired A/B measurements):

| Tokens | Replicated baseline | Fused | Quarter-hidden baseline | Fused |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 13.05 | 10.04 | 8.69 | 6.17 |
| 4 | 13.37 | 10.23 | 8.80 | 6.30 |
| 8 | 13.86 | 10.72 | 8.40 | 6.15 |
| 16 | 14.87 | 12.06 | 8.43 | 6.22 |

The quarter-hidden timings **exclude TP communication** and must not be
compared directly with the replicated timings as an endpoint speedup.
The local fusion savings extrapolate to only approximately 0.21–0.30 ms
across 96 up/mix pairs. That is insufficient for the 6.14/8.20 ms C8/C16
step-time reductions needed for the target.

The proposed C16 paired weight reuse did not help: at one warp the
quarter-hidden candidate regressed from 6.22 to 7.22 microseconds; replicated
12.06 versus 12.05 microseconds is neutral. Four-warps-per-CTA was also
slower. Reject these schedules rather than promoting "less traffic" without
measured benefit. `--selected-only` retains the non-paired one-warp candidate
for broader validation without repeating that search.

The measured source was the integration base plus the benchmark patch;
kernel source SHA256 `d0a88b63bc96579a67d98ca10d6246c0b76c75ba5c109900f59be85f36e51b00`,
extension SHA256 `2253c7079d14475e00ff7ad6561055325cd8eb435509c0a3020307040096c811`.
Raw measurements are retained task-locally as `.artifacts/hc_batch_v1.json`.
Build, CPU tests (11) and repository pre-commit gates passed. All-96-pair,
all-shard validation and full-engine integration remain pending. No default
changed and no new endpoint throughput is claimed. **The 40% target is not
achieved by this PR.**
