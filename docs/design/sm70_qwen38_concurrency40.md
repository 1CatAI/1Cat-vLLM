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

Current validation: SM70 source compilation succeeds without local-memory
spills; CPU layout/writer coverage tests pass (11 tests). GPU numerical and
performance results are pending. The 40% target is not achieved by this PR.
