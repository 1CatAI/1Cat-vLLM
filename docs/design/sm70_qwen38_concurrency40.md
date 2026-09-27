# Qwen3.8 concurrent decode: next-stage optimization

## Acceptance target, not a performance claim

Status update: the initial sections below record research-only screens.
An opt-in native runtime candidate is now being integrated; see
"C1-to-batch runtime candidate" below. It is **not yet an accepted endpoint
speed or quality result** and remains disabled by default.

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

## Next screen: preserve the high-precision down partition

A four-weight diagnostic trace with FP16/BF16 reduced-precision reduction
and FP16 accumulation disabled identifies the replicated HC down route as
`cutlass_70_wmma_tensorop_s161616gemm_f16_16x16_64x2_tn_align8`, grid
`(8, 3, 20)`, followed by a separate `splitKreduce_kernel` with FP32 partial
inputs. This is a projection-only trace, not another model startup or an
unprofiled throughput measurement.

The next research candidate preserves twenty K=512 partitions. Its second
kernel performs the ordered FP32 reduction, materializes the same FP16 GEMM
boundary, and fuses HC SiLU plus injection extraction. It does not change
the checkpoint, precision flags, expert selection or production dispatch.

Besides M8xN32 and its paired-M8 version, the screen includes M16xN16 within
a single warp: two quad pairs own each eight-token half. This shares weights
across both halves without doubling the per-lane accumulator array, while
exposing twice as many output tiles as the paired-M8xN32 schedule. It is a
layout/resource hypothesis to test, not an asserted speedup.

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/path/to/cuda-12.8 \
TORCH_CUDA_ARCH_LIST=7.0 \
TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions" \
TRITON_CACHE_DIR="$PWD/.cache/triton" \
.venv/bin/python benchmarks/kernels/benchmark_sm70_hc_batch_reuse.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 --pairs 4 \
  --projection down --rows 2,4,8,16 --out .artifacts/hc_down_screen.json
```

Admission checks projection, SiLU and injection bits separately across all
six scales, as well as replaying the actual timed graph with poisoned
outputs. Only exact candidates receive alternating A/B timing. CPU layout
tests and compilation are not GPU numerical/performance acceptance.

Packing also has a memory gate. Keeping all original tensors *and* every
replicated packed tensor would add 660 MiB/rank for down and 600 MiB/rank for
up. This microbenchmark is not permission to add those copies to production.
A whole-chain candidate must account for sharding/replacing packed weights,
fallback ownership and available KV capacity, not hide the allocation cost.

### Down screen result: exact, still a small component gain

Four real weights, 14 configurations and six input scales passed bit equality
for projection, SiLU and injection, including the timed graph's outputs.
Unprofiled alternating graph timings on an exclusively locked V100:

| Tokens | Replicated down + SiLU baseline (us) | Native (us) |
| ---: | ---: | ---: |
| 2 | 15.53 | 12.31 |
| 4 | 15.84 | 12.87 |
| 8 | 16.39 | 13.63 |
| 16 | 17.81 | 15.41 |

The winner is non-paired M8xN32, one warp. Neither paired-M8 (C16 15.71 us)
nor M16xN16 (17.09 us) wins. Fewer repeated weight loads alone again do not
establish a faster schedule. The extrapolated 96-pair saving is only
0.23–0.31 ms, **not endpoint acceptance**. Do not spend a model startup on
this isolated component.

Artifact: `.artifacts/hc_down_v1.json`; measured kernel SHA256
`c7214e9185088cb78eebd2e82ef71f1121533c13e496c51f0c2afc62d29d591c`,
extension SHA256
`f20a9fede35e192a57c34072291eee9dbd0c641807bc92960fd0d284b6f079f3`.
The subsequent TP4 extension below has a different hash and needs its own
validation; the earlier result is not silently relabeled as that extension.

### TP4 down/up/mix and communication screen

`benchmark_sm70_hc_batch_native_tp4.py` connects the winning local schedules to
quarter-output down/up weights, fusing the ordered down reduction and SiLU
inside the first gather. The second gather collects already mixed hidden
shards. Down and output transfers have independent, research-owned IPC
channels using the source tree's push-packet helpers. No opaque communicator
from a foreign extension or auxiliary runtime stream is reused.

```bash
CUDA_HOME=/path/to/cuda-12.8 TORCH_CUDA_ARCH_LIST=7.0 \
TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions" \
.venv/bin/python benchmarks/kernels/benchmark_sm70_hc_batch_native_tp4.py \
  --build-only --out .artifacts/hc_tp4_build.json

CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=/path/to/cuda-12.8 \
TORCH_CUDA_ARCH_LIST=7.0 \
TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions" \
TRITON_CACHE_DIR="$PWD/.cache/triton" \
.venv/bin/torchrun --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_hc_batch_native_tp4.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 --pairs 8 \
  --rows 2,4,8,16,2 --out .artifacts/hc_tp4_screen.json
```

The reported rank-maximum pair time **includes both gathers**, but excludes
combine/norm, the final mixer and the rest of the model. Check LoRA, injection
and mixed block outputs separately on every rank. Repeated C2 after C16
exercises payload-size transitions without resetting communication state.
The packed quarter weights would add 330 MiB/rank if originals are retained;
runtime integration and the memory/performance/quality gates remain pending.

### First TP4 result (8 distinct HC pairs)

On exclusively locked GPUs4–7, every rank passed bit equality for LoRA,
injection and mixed block outputs at all six scales and every tested batch
width, including the final C16→C2 transition. Each row below is the median
of six alternating A/B trials, using the slowest rank's time per trial.

| Tokens | Replicated chain (us/pair) | Native sharded chain including gathers (us/pair) | Median paired saving (us/pair) |
| ---: | ---: | ---: | ---: |
| 2 | 31.53 | 23.72 | 7.83 |
| 4 | 31.18 | 24.08 | 7.12 |
| 8 | 31.36 | 24.69 | 6.87 |
| 16 | 33.19 | 26.15 | 7.04 |
| 2, repeated after C16 | 29.35 | 21.07 | 8.28 |

The repeated C2 exposes some absolute-latency drift; use the paired samples,
not unmatched row-to-row timing. C16's paired savings range from 6.99 to
7.16 us across the six trials. **These are HC subchain measurements, not
engine decode or prefill throughput.** Their approximately 0.64–0.80 ms
96-pair extrapolation is still not enough to claim the +40% endpoint goal.
An all-96-pair follow-up waited for a TP4 lease and exited before creating
any CUDA context; do not describe it as completed validation.

Artifact: `.artifacts/hc_tp4_v1.json`. Kernel SHA256 values:

- Projection: `2961c36dcb770f62146cb75aee11152b8875d8aa7afa3378b083b65ea74b9456`.
- Gather: `089a769efcacd7c03c4c550b779d77a480d3af7d409875ca16185f39f1d3d19f`.

Both source-built research modules exited normally and released all four
cards. CPU contract tests: 21 passed. The original cuBLAS/all-reduce benchmark
`benchmark_sm70_hc_batch_tp4.py` remains unchanged; the new native screen has
its own filename. No production path/default, KV format or DCP work changed.

## C1-to-batch runtime candidate

One default-off admission switch, `VLLM_SM70_QWEN38_BATCH_FASTPATH=1`, now
groups the following source-built candidates. The existing checkpoint-FP16
GEMV, fused HC/GDN and TP4 push routes must also be admitted by the normal
Qwen3.8 runtime profile. The candidate targets SM70, TP4, FP16, M2–16, no
speculation and no microbatch overlap; batch-invariant execution is excluded.
C1 retains its existing kernels. Dynamic prefill retains the original
weights and path. Unsupported layouts fall back to the ordinary operators.

| Component | Batch implementation | Numerical contract |
| --- | --- | --- |
| HC down/up | Quarter-output TP4 projections, two dedicated gathers, fused SiLU and gate/mix | Twenty ordered K512 FP32 partials for down; original up K order; original FP16 boundaries |
| GDN input | Packed QKVZ and b/a projection with output splitting in the epilogue | Ordered FP32 QKVZ; four original b/a partitions and left-to-right FP32 reduction |
| Shared gate | Keep the original linear; fuse only sigmoid and multiply | FP16 sigmoid materialization before FP16 multiply |
| C2 communication | Admit 10-KiB ordinary and sum2 push collectives | Unchanged rank-ordered FP32 sum and FP16 result |

HC is new native integration of this PR's arithmetic screen. GDN and gate
reuse the exact candidates from PR #692, not its unqualified grouped-MoE
or altered gate-dot candidates. The code ships in the normal `_C` extension
and its owning `_custom_ar` namespace, with no private DSO dependency.

The HC transport differs from the research sentinel screen: each FP16
payload travels with an epoch tag in the same 32-bit word. Down/output
channels are disjoint from M1 HC and auxiliary-stream MoE collectives.
Consumed packets are cleared, so inactive lanes cannot retain a valid tag
when a graph changes batch size. The transport preserves every half bit
without reserving a floating-point value. Consequently, the old research
timings and correctness tests do **not** qualify the new native transport.
Native TP4 changed-input/changed-batch graph validation is mandatory.

Keeping the original fallback weights adds 330 MiB HC plus 725.625 MiB GDN
packed storage per rank (1055.625 MiB total), plus 364.25 KiB/rank of HC
communication buffers. This must be included in the model/KV memory budget.
Per-call workspaces are small FP32 partials and FP16 outputs, not another
model-sized allocation. Do not promote based on isolated hot-cache timing.

The admitted Qwen3.8/SM70 FP16 runtime now explicitly disables FP16 and BF16
reduced-precision GEMM reduction and FP16 accumulation for **both control
and candidate**, even if the GEMV switch is off. An older result collected
with different Torch precision flags is not a matched control.

Current integration validation: 63 CPU layout/admission/fake-export tests
passed; 28 GPU tests were deliberately skipped with CUDA hidden. This is
not CUDA arithmetic or performance evidence. Native source build, actual
TP4 runtime graphs, all-layer quality and matched endpoint measurements
remain acceptance gates. There is no new decode-speed claim or default-on
decision in this update.
