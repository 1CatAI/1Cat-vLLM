# Qwen3.8 concurrent decode: next-stage optimization

## Quality-qualified endpoint result (2026-09-28)

The repaired ordinary engine pair passes the previously failing token gate.
Control source: `7f27016c12465c9ac45d33914b6670c11fbcf45c`; candidate:
`4b2404f99588f8b1588f8a5e0f4bf35915852c8e`. Their production Python/CUDA
sources and native binaries are identical; the intervening commit adds only
optional benchmark long-context checks. Integration base is
`14abfc27ee4e13cd2a9d1a8a882f36a629e5889a`, including main's MTP and M48
changes. Both arms use the repaired native FP32 gated norm; the batch
fast-path switch is the A/B difference.

Contract: GPU0..3, V100-SXM2-32GB, TP4, driver580.173.02,
Torch2.10.0+cu128/CUDA12.8, the RadixArk Qwen3.8 Flash-Next NVFP4 checkpoint,
FP16 activations/KV, FP32 GDN state and GEMM accumulation/reductions,
no MTP or prefix caching, max context262144, chunk8192, max16 sequences,
FULL decode graphs. Both arms reserve exactly4160749568 KV bytes/rank;
this explicit budget overrides memory utilization and gives313030 tokens
of reported capacity in each arm. PLE remains mmap prefill plus11.92 GiB
pinned-host UVA decode per rank, **not a disk-only/no-RAM configuration**.

Pooled unprofiled fixed-width engine intervals, 8192 input/256 greedy forced
output tokens per request, two atomic-admission cohorts per width:

| Width | Control ms/step | Batch-on ms/step | Batch-on aggregate tok/s | Batch-on per-stream tok/s | Throughput change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 10.198848 | 10.272228 | 97.350 | 97.350 | -0.71% |
| C2 | 17.316287 | 13.401774 | 149.234 | 74.617 | +29.21% |
| C4 | 17.715166 | 15.040512 | 265.948 | 66.487 | +17.78% |
| C8 | 21.398201 | 18.847988 | 424.448 | 53.056 | +13.53% |
| C16 | 28.662908 | 26.464495 | 604.584 | 37.786 | +8.31% |

The repaired batch-off C1 is98.050 tok/s. Batch-on's observed C1 cost is
0.073380 ms/step; do not claim an exactly zero regression or infer its cause
from two repetitions. C2 minus C1 falls7.117440 ->3.129546 ms, still above
the3 ms target. C8/C16 +40% remains unmet. Separate one-output-token prefill
cohorts remain about6.83K tok/s for these repeated synthetic8K inputs; this
is not a natural-chat/dataset prefill baseline or a sum of overlapping TTFTs.

Quality checks:

- All62 cross-arm sequences, totaling15872 compared greedy output tokens,
  match exactly. All31 within-arm repeat comparisons pass in each arm.
- Each arm passes two single-request and16 concurrent natural-EOS answer
  checks with checkpoint sampling: temperature1.0, top_p0.95, top_k20,
  seed0. Concurrent natural traffic intentionally uses streaming admission;
  its text differs across arms and is **not token-parity evidence**. The
  fixed-cohort greedy comparisons above supply that separate evidence.
- Candidate natural middle-record retrieval passes twice at130559 and
  261631 input tokens, allowing513 output tokens within128K/256K contexts.
  Outputs stop naturally after163 and123 tokens respectively and repeat
  exactly at each length. The expected archive code is recovered.
- The exact262143-input-plus-one-output boundary passes with valid request
  metrics. This is a boundary check, not a one-token answer-quality score.

These are bounded regression/health checks, not a claim of universal output
identity or dataset accuracy. The earlier diagnostic failures remain below
as history; they are not the repaired acceptance result. Both reports now
have `complete=true` and `token_parity_passed=true`. Raw records:
`endpoint_{control,candidate}_canonical_v1.json` and logs;
`quality_canonical_summary.json` records the pooled comparison. All owned
workers shut down and released GPU0..3; no API was started.

The numerical repair defaults on for its admitted no-MTP contract. The
batch optimizations remain opt-in via `VLLM_SM70_QWEN38_BATCH_FASTPATH=1`:
their packed copies still cost1055.625 MiB/rank, so this result is not a
blanket default-promotion claim for other memory budgets or model routes.
The final merge audit also removed a duplicate GDN translation-unit entry;
the generated native build already compiled and linked it only once.

After a normal source build, reproduce in a clean, exclusively leased TP4
environment with owned compiler caches and no private DSO/preload overrides:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PWD" VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_QWEN4EXP_PLE_HOST_GIB=12 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for arm in control candidate; do
  export VLLM_SM70_QWEN38_BATCH_FASTPATH=0
  extra=()
  if [[ "$arm" == candidate ]]; then
    export VLLM_SM70_QWEN38_BATCH_FASTPATH=1
    extra=(--reference control.json --long-context)
  fi
  .venv/bin/python benchmarks/benchmark_sm70_qwen38_concurrency.py \
    --model "$MODEL" --mode nomtp --widths 1,2,4,8,16 \
    --input-len 8192 --output-len 256 --repeats 2 \
    --gpu-memory-utilization 0.94 --kv-cache-memory-bytes 4160749568 \
    --atomic-cohort --health --measure-prefill --out "$arm.json" \
    "${extra[@]}" || exit 1
done
```

## Output-quality investigation and repair

Same-input deep observation on the retained high-precision source
`581e69adb4751d3613ca1fdcc031b480b3dc2502` localizes the earliest C1
cross-arm difference to layer0's gated RMSNorm, before its output projection.
At decode step2, all four ranks have identical weights, projected inputs,
convolution states/outputs, FP32 recurrent states and attention-core outputs.
Only rank1's normalized coordinate117 differs: control
`-0.003421783447265625`, candidate `-0.0034198760986328125`.
The local projection then differs at11 elements, TP output at4, and HC
normalization at4. Do not misidentify HC or TP reduction as the origin.

An independent actual-tensor replay reproduces every input projection,
convolution, recurrent output/state and local output projection bit.
Native FP32 gated RMSNorm and its standalone compiled form reproduce the
candidate's value. FP64 evaluation gives `-0.0034208292707999443`, whose
nearest FP16 value is also the candidate's. Thus simply forcing the old
control token sequence would not establish greater arithmetic accuracy.
The full-model compiled normalization therefore required a stable, explicit
numerical contract before another endpoint acceptance attempt.

The diagnostic includes two40-step C1 repeats per arm, fixed KV bytes and
intrusive intermediate copies. Control matches the preceding observer at
all144 boundaries and logits for all40 steps; repeat state values also agree.
Physical cache-slot indices differ normally between requests. This is
localization evidence, **not** new throughput or broad quality qualification.
Raw deep captures and comparison/replay reports use `quality_v4_*` and
`replay_deep_control.rank1*` in the task artifacts.

Benchmark acceptance now includes single-stream `baseline_runs` parity
and refuses to accept a run with no actual parity observations. Thirteen
focused CPU regressions pass. This closes a reporting bug separately from
the normalization fix; it cannot qualify the optimization by itself.

### Native FP32 normalization repair

Source `7f27016c12465c9ac45d33914b6670c11fbcf45c` integrates
`onecat/main` at `14abfc27ee4e13cd2a9d1a8a882f36a629e5889a` and reuses its
native exact gated RMSNorm. The narrowly admitted SM70/TP4/Qwen3.8 NVFP4
no-MTP configuration now defaults `VLLM_SM70_RMSNORM_GATED_EXACT=1`, while
respecting an explicit override and leaving MTP defaults unchanged. The
operator preserves native FP32 reduction and activation arithmetic and the
FP16 output boundary; it does not lower precision or copy model weights.

The device-capability guard is a compile-time constant for the input device,
and route logging is inside the opaque operator rather than the traced
forward. A small full-graph test caught the old capability-query tracing
failure before a model launch. Its corrected version checks changing batch
shapes and neighboring reductions against native FP32 bits. The issue is
different arithmetic in different compilation contexts, **not random output
within one compiled graph**. The diagnostic has not identified one specific
PTX instruction as the cause.

Focused validation before endpoint acceptance:

- All four TP ranks and three retained decode steps replay exactly through
  the input projections, convolution, recurrent state/output and local
  output projection. The new gated norm matches native FP32 eager bits.
- 39 native gated-norm/router/shared-expert GPU checks and 116 GDN/dense/QSA
  GPU checks pass, including dynamic graph inputs and long selector shapes.
- Four-card HC checks on eight actual weight pairs and six activation scales
  have zero LoRA/output/injection bit mismatches. The existing fused HC
  microbenchmark gain survives the integration; it is not endpoint evidence.
- 39 configuration/default CPU checks and the explicit no-MTP versus MTP
  precision-policy checks pass. The MTP API argument order and its qualified
  routes are preserved by the integration.

The source-built `_C` SHA256 is
`2757c48735b4a6ee65889f93b45082be9b8a67b1572d3dd7edf19e157184e484`.
The ordinary paired engine result above passes token repetition, cross-arm
parity and natural-EOS text health. It uses identical KV bytes
(`4160749568` per rank) and atomic cohort admission;
neither the old unequal KV capacities nor host request-submission timing may
serve as a hidden A/B difference. Microtests at long selector shapes do not
substitute for a 256K full-model quality check.

## Acceptance target, not a performance claim

Historical chronology: the sections below preserve the research-only screens
and rejected endpoint attempts preceding the repaired result above. The
batch route remains disabled by default; its measured improvement still
does not meet the full performance targets.

Integration base: `1e90d17f2c75e443b2a85a576ed68fa04c5f9dd6` (`onecat/main`).
Scope: Qwen3.8 Flash-Next NVFP4, V100-SXM2-32GB, TP4, no MTP. Improve
C2/C4 and obtain at least **40% more aggregate decode throughput at both
C8 and C16**, without reducing arithmetic precision or regressing quality.

Priority update (2026-09-27): **C2 minus C1 must be at most 3 ms/step**,
with no C1 regression. The matched historical unprofiled trace campaign
measured C1 10.335855 ms and C2 17.545152 ms: a 7.209297 ms gap. If C1 holds,
C2 must reach 13.335855 ms or less (about 150 aggregate tok/s). This needs
4.209297 ms of recovery; a small component win does not satisfy the target.
The C8/C16 target remains open, not superseded or claimed achieved.

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
packed storage per rank (1055.625 MiB total), plus 684.875 KiB/rank of HC
communication buffers. This must be included in the model/KV memory budget.
Per-call workspaces are small FP32 partials and FP16 outputs, not another
model-sized allocation. Do not promote based on isolated hot-cache timing.

The admitted Qwen3.8/SM70 FP16 runtime now explicitly disables FP16 and BF16
reduced-precision GEMM reduction and FP16 accumulation for **both control
and candidate**, even if the GEMV switch is off. An older result collected
with different Torch precision flags is not a matched control.

Current integration validation: the combined CPU layout/admission/fake-export,
shared-gate and integration suite passed all 134 tests with the actual native
registrations loaded; 39 GPU tests were deliberately skipped with CUDA hidden.
The normal CUDA 12.8/Torch 2.10.0+cu128 source build completed, including `_C`,
Flash-V100 and bundled FlashQLA. The optional Rust frontend was not built
(no Rust compiler); no complete-wheel claim is made. The loaded `_C` SHA256 is
`3a3ae3ba1d6baf6a92c8c4854068c53d5c2a16c9f543ea14fda1497c9a799c6f`.
Its dynamic dependencies contain only standard Torch/CUDA/C++/libc libraries,
with no private DSO or task-cache RPATH. A fresh process confirmed native
HC/GDN/gate registration without initializing a CUDA context.

The first native TP4 queue expired without creating a CUDA context. A later
finite exclusive GPU slot completed the native component gates below.
Full-model quality and matched endpoint measurements remain acceptance gates;
there is no new model decode-speed claim or default-on decision.

The retained control at `gpu_memory_utilization=0.90` loaded 21.31 GiB/rank,
left 3.59 GiB for KV and reported 290671-token capacity. Its headroom above
262144 tokens is smaller than the new 1055.625 MiB/rank packed copies. This
is a capacity risk inferred from the old control, not a new model measurement.
Do not attempt endpoint qualification by silently reducing context length or
KV precision. Reduce retained storage or explicitly qualify both control and
candidate at a matched memory budget before treating this as a drop-in path.

Native-runtime reproduction after the normal source build (exclusive TP4
lease required):

```bash
VLLM_SM70_TP4_PUSH_ALLREDUCE=1 VLLM_SM70_QWEN38_BATCH_FASTPATH=1 \
  .venv/bin/torchrun --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_hc_batch_native_tp4.py --runtime \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 --pairs 96 \
  --rows 2,3,4,8,16,2 --out .artifacts/hc_runtime_tp4.json
```

`--runtime` does not compile/load the research modules. It checks the loaded
`_C` belongs to this source tree, hashes it, and uses the native communicator
and loader packs. It also reuses existing CUDA graphs in an odd-replay
shrinking/growing sequence instead of relying on fresh captures to reset
state. Any numerical mismatch makes the benchmark fail and prevents timing
of the failing case. The native `_C` and Flash-V100 package must be rebuilt
from source; no private extension override is part of reproduction.

The reused `benchmark_sm70_qwen38_concurrency.py` keeps per-step decode,
prefill union-wall throughput and natural-EOS text health separate, records
token IDs and checks repeated cohorts against a matched reference. Both
arms record an explicit FP32 accumulation/reduction contract. A failed or
nonrepeatable control cannot serve as an accepted speed/quality reference.

## Native component results, 2026-09-27

These measurements use the normal source-built extension, not the earlier
research gather. All three low-precision Torch matmul options were disabled.
Native HC covers all 96 real weight pairs, all four ranks, six input scales
and odd-count shrinking/growing graph replays. Every LoRA, mixed-output and
injection FP16 bit comparison passed. Timing is the median slowest-rank
micro-chain of 96 pairs, including both gathers but excluding combine/norm:

| Width | Control ms | Native candidate ms | Saving ms |
| ---: | ---: | ---: | ---: |
| C2 | 3.169493 | 2.521579 | 0.647915 |
| C4 | 3.221781 | 2.723115 | 0.498667 |
| C8 | 3.319211 | 3.017963 | 0.301248 |
| C16 | 3.559467 | 3.574272 | -0.014805 |

C16 has **no measured HC improvement** here. Do not replace these numbers
with the earlier, faster eight-pair research transport result.

Packed GDN passes all 36 layer weights, all four logical TP slices and six
scales. Rank0's 36-layer C2 chain is 1.924992 -> 1.118208 ms; C16 is
1.995520 -> 1.192256 ms. The separate 48-layer shared-gate micro is
0.449459 -> 0.281093 ms at C2, with the original linear unchanged; all finite
half sigmoid inputs were also checked. The TP4 C2 ordinary/sum2 collective
micro (96 operations) is 0.733773 -> 0.327721 ms, with all-rank equality.
These separate timings are **not an additive endpoint result**.

Native artifact for that suite:
`d67ea552106eec5b7d6e11a6c182527b6ffd57156c4db59db13a68da23cc47a6`.
Raw files in the owned worktree's `.artifacts/` are
`hc_runtime_tp4_native_20260927.json`,
`gdn_native_packed_rank{0,1,2,3}.json`, `gate_native.json`,
`c2_push_native.json`, and `native_gdn_gate_gpu_tests.log` (68 passed).

### C2 selector and memory screens

The native QSA selector now has an explicit, default-false `decode_batch`
argument. It extends the existing exact C1 algorithm with per-row offsets
and retains default dispatch and M1 behavior. It is **not admitted by the
model runtime yet**. The 72-test follow-up suite includes graph-replayed
Top-K, strided rows, mixed lengths, signed-zero ties, infinities, output
canaries, independent stable-sort checks and the 256K compressed boundary,
plus GDN tests. All passed after fixing a boolean-mask bug in the new test.

In a same-buffer 12-call graph screen, C2 Top-K at 8K improves
0.209736 -> 0.116879 ms. However, at 64K it regresses
0.693166 -> 0.733348 ms; at 256K it is approximately unchanged. This is not
an unconditional runtime winner. Raw: `qsa_batch_decode_v{1,2}.json`.
Stable native extension hash:
`bfcd2a344cd2c7e006a02273b74effc71e0334cef5cafec0b4d1e40bc646f3b2`.

Rejected experiments, not production policy:

- Simply extending the C1 32-column score tile changes FP32 score bits.
  Fixing the original four-head left-to-right sum restores exact scores,
  but slows C2 8K scoring from 0.109670 to 0.130867 ms/12 calls and also
  regresses long contexts. Lower register count alone is not a speedup.
- No-copy row-major GDN removes the 725.625 MiB/rank packed allocation but
  is slower than packed GDN and regresses C16 versus the original path.
  Pairing two M8 groups preserves bits but worsens C16 to 2.366272 ms,
  versus 1.988416 ms control. Do not enable that memory-saving path blindly.

Neither experiment changes the runtime default. The optional score-sum
argument and direct row-major native entry are retained only to reproduce
the rejected screens; the model continues to use its existing score tile
and the opt-in packed GDN route.

### Dense contract correction and guarded integration

The first native dense screen was rejected: router was exact but slower;
output had small bit differences. A fresh CUDA-graph-node trace with all
three low-precision Torch options disabled shows **four** output K384
partitions, not the older two K768 assumption. Do not recover the obsolete
tree by enabling FP16 reduction. The corrected native kernel gives each
independent m8n8k4 quad pair one original FP32 partition and reduces the
four values left-to-right. N8 tiles spread work across more CTAs, without
packing/duplicating weights or changing the reduction tree.

Twenty direct native GPU tests pass. All 48 router and 48 output weights,
all four logical TP slices and six scales pass exact FP16-bit comparisons.
Rank0 C2 48-layer graphs: router 0.556636 -> 0.503532 ms, output
0.802499 -> 0.610724 ms. Router M8/M16 and output M16 have no win; runtime
admission under the existing default-off switch is restricted to router
M2..4 and output M2..8. Loader permission excludes speculation/microbatching;
C1, unsupported roles, layouts and widths keep their existing dispatch.
Fifteen additional GPU route/fallback tests pass through the registered
opaque Python op, including unchanged C1 and rejected larger widths.

Raw: `dense_contract.{qdstrm,nsys-rep,sqlite}`,
`dense_batch_v1_rank0.json` (rejected), `dense_batch_quad_v2_rank0.json`,
`dense_batch_quad_rank{1,2,3}.json`, `dense_runtime_gpu_tests.log` (35 passed).
Rank0 measurements used GPU4; the remaining logical slices used idle GPU0.
These are component results, not four-rank model endpoint measurements.

### Native HC fusion continuation

The eight-pair native trace `hc_native_chain_v1.sqlite` identifies the two
gathers as a large remaining cost. Excluding the first pair of each graph
replay avoids attributing host rank-start skew to normal gather work.
Stable rank0 median per-pair C2 kernel durations are approximately down
7.92 us, reduce/down-gather 7.22 us, up/mix 6.50 us and output-gather
6.69 us. These are instrumented primitive durations, not token latency.

The admitted candidate now has three kernels: down partials, coalesced
ordered down reduction/gather, and fused up/mix/output gather. The down
stage assigns contiguous FP32 columns to adjacent lanes instead of eight
interleaved columns per thread. Up retains its ordered K320 MMA, FP16 gate
boundary and branch-ordered mix. Its MMA fragments are retiled in shared
memory into exact half+tag packets; a dedicated tile-major channel keeps
it independent of the old row-major gather and auxiliary-stream collectives.
The new channel adds 320.625 KiB/rank (included in the total above).

Same-process TP4 GPU0..3 A/B, 96 actual weight pairs, median slowest rank,
CUDA12.8/Torch2.10.0+cu128, V100-SXM2-32GB, no low-precision reductions:

| Width | Previous native four-kernel ms | New three-kernel ms | Saving ms |
| ---: | ---: | ---: | ---: |
| C2 | 2.506987 | 1.940011 | 0.566976 |
| C4 | 2.694144 | 2.001600 | 0.692544 |
| C8 | 3.013163 | 2.143125 | 0.870037 |
| C16 | 3.547691 | 2.342251 | 1.205440 |

This is the communication-inclusive 96-pair HC micro-chain, **excluding
combine/norm and the rest of the model**. It is not a model-level +40%
result. Repeated C2 at the end is 2.506133 -> 1.936875 ms.
Every rank passes LoRA/output/injection bit comparisons at all six scales.
Twenty-seven mixed-width/old-new-path transitions also pass. The benchmark
now executes one extra pair per transition: replaying a 96-pair graph once
still increments epochs an even number of times, so the prior "odd graph
replay" check alone did not test odd communication epochs.

Native `_C` SHA256:
`a971b81f891496d6742e29ae840b643815f167895904994755ca53222b4c0b1d`.
Dynamic dependencies remain standard Torch/CUDA/C++/libc; no private DSO/RPATH.
Raw: `hc_fused_chain_paired_v3.json`, with earlier staged results retained
in `hc_fused_up_native_v1.json` and `hc_fused_up_coalesced_v2.json`.
The v2 timing control already contained coalesced down; use **v3** to
compare the complete old/new native paths in the same process.

```bash
VLLM_SM70_TP4_PUSH_ALLREDUCE=1 VLLM_SM70_QWEN38_BATCH_FASTPATH=1 \
  .venv/bin/torchrun --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_hc_batch_native_tp4.py --runtime \
  --fused-chain --model /path/to/Qwen3.8-Flash-Next-NVFP4 --pairs 96 \
  --rows 2,3,4,8,16,2 --out .artifacts/hc_fused_chain.json
```

The runtime selects the qualified three-kernel HC implementation and the
guarded dense winners behind the **same default-off batch flag**, not new
user tuning switches. CPU admission/export/wiring tests: 79 passed.
No full-model startup or new endpoint speed/quality claim in this round.
The packed-weight 256K-capacity gate and matched C1/C2 model validation
remain open; C2 minus C1 <=3 ms and C8/C16 +40% are **not yet demonstrated**.

### Source-built endpoint control (2026-09-27)

The first complete measurement on this branch uses the same ordinary model
defaults with the batch switch off. Contract: V100-SXM2-32GB GPU0..3, TP4,
CUDA12.8, Torch2.10.0+cu128, FP16 activations/KV, FP32 GDN state and GEMM
accumulation/reductions, no MTP or prefix cache, 262144 maximum context,
8192 input tokens per request, 256 greedy output tokens, two repeats,
max16 sequences and 8192 batched tokens. Both A/B arms use explicit
`--gpu-memory-utilization 0.94`, preserving capacity despite the candidate's
1055.625 MiB/rank packed copies. This is not a reduced-context comparison.

PLE is the existing **hybrid** path: file-backed prefill and pinned-UVA
decode, with `VLLM_QWEN4EXP_PLE_HOST_GIB=12`. Actual placement is zero GPU
table rows, 11.92 GiB pinned host table per rank, and a separate 47.684 GiB
file mapping for prefill. Do not describe this contract as disk-only/zero-RAM.
The control loads 21.31 GiB/rank and has 393216-token KV capacity.

| Width | Control step ms | Aggregate decode tok/s | Per-stream tok/s |
| ---: | ---: | ---: | ---: |
| C1 | 10.400342 | 96.151 | 96.151 |
| C2 | 17.590719 | 113.696 | 56.848 |
| C4 | 18.114192 | 220.821 | 55.205 |
| C8 | 21.668120 | 369.206 | 46.151 |
| C16 | 29.044725 | 550.875 | 34.430 |

These are pooled, unprofiled, fixed-width engine intervals; they exclude
prefill and changing-width steps. C2 minus C1 is 7.190378 ms. Separately,
one-output-token prefill cohorts give aggregate6817.653/6816.154/6816.623/
6812.797/6824.260 tok/s at C1/2/4/8/16. These repeated synthetic 8K inputs
do not establish natural-chat, dataset or long-context throughput.

**Quality gate remains open:** C1/C2/C4/C16 repeated greedy tokens match,
but C8 streams0/1 first differ at zero-based token52/49. Both single-request
health cases and all16 concurrent natural-EOS health cases pass. Natural
text health is not a substitute for token parity; this control is a fully
measured diagnostic reference, not an accepted all-width quality baseline.
The candidate must not be promoted against this failed reference.

Environment bootstrap was completed before this measured run: numba0.67.0,
tilelang0.1.10, apache-tvm-ffi0.1.10 and xgrammar0.2.0. Newer unconstrained
TVM-FFI0.1.14.post1 aborts TileLang import; XGrammar0.2.8 uses a newer FFI
API despite metadata admitting the pinned FFI. The compatible imports and
a128-token original FlashQLA indexed-state execution pass before loading
the model. No precision flag, Torch version or model weight was changed.
In-place source builds now place Flash-V100/FlashQLA companion kernels next
to the source packages; wheel staging remains unchanged. Use a short owned
IPC/TMP directory because UNIX socket paths cannot exceed107bytes.

GPU source is the published1ae3403 kernel revision, rebuilt normally after
formatting; native `_C` SHA256:
`e4572643d7f61f5b968323c859ee3be456472bc6438dfc77407626009bc990b7`.
Raw: `endpoint_control_v4.json` and `.log`. Earlier v1/v2 failures occurred
before weight loading (long IPC path, missing numba); v3 failed graph capture
on the dependency mismatch and has no speed result. Preserve those records,
but do not repeat their resolved startup diagnostics.

### Diagnostic candidate endpoint, not a promoted baseline

The same-contract batch-on arm completed all cases on the same GPU0..3.
Actual worker logs confirm native batched HC, packed GDN input, guarded
small-batch dense and shared-gate epilogue dispatch. Model allocation is
22.35 GiB/rank; available KV3.97 GiB gives320740 tokens. Thus256K capacity
survives the packed copies, although KV capacity is not identical between
arms. Both use memory utilization0.94, not equal KV byte budgets.

| Width | Control ms | Candidate ms | Candidate aggregate tok/s | Per-stream tok/s | Throughput gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 10.400342 | 10.356449 | 96.558 | 96.558 | +0.42% |
| C2 | 17.590719 | 13.590531 | 147.161 | 73.581 | +29.43% |
| C4 | 18.114192 | 15.304173 | 261.367 | 65.342 | +18.36% |
| C8 | 21.668120 | 19.161013 | 417.514 | 52.189 | +13.08% |
| C16 | 29.044725 | 26.777536 | 597.516 | 37.345 | +8.47% |

C2 minus C1 falls7.190378 ->3.234083 ms, still0.234083 ms above the target.
C8/C16 are nowhere near+40%. Candidate prefill aggregate at C1/2/4/8/16 is
6827.845/6825.408/6818.820/6817.946/6826.257 tok/s, effectively unchanged
for these synthetic inputs. This is actual unprofiled engine measurement,
not a sum of separate microbenchmark savings. The+0.42%C1 difference is
small variation, not evidence of a new single-request optimization.

**This pre-repair v4 candidate was not quality-qualified.** All two single-request and16
concurrent natural-EOS health checks pass in each arm, but cross-arm token
parity fails at every width. In particular, candidate C1 first differs at
zero-based token36 in both repeats despite C1's intended unchanged decode
route. Candidate C2 repeat stream0 differs at25; C4 streams0/1 at12/64.
C1/C8/C16 repeat within the candidate is exact. Text health cannot override
these failures, and no precision reduction is enabled to obtain the speed.
Both reports finish with measurements_complete=true and complete=false;
the candidate explicitly used --diagnostic-reference against the failed
control. No default promotion or main merge is justified by these numbers.

Raw engine records expose an additional comparison confound: control C8
repeat0 begins before the entire cohort is queued and has an extra early
decode step versus repeat1. This is a scheduling difference, not proof of
a kernel arithmetic defect. It also does not explain the cross-arm C1
difference. The new optional --atomic-cohort mode pauses the scheduler
without clearing caches, enqueues the full cohort, resumes in finally and
then drains outputs. Its admission mode must match the reference; legacy
reports mean streaming admission. The new helper has four CPU ordering and
failure-cleanup tests (16tests pass with the baseline suite). It has not yet
been full-engine qualified and was **not** used retroactively in these v4
measurements. Production scheduling is unchanged.

Next gate: preserve these failures, fix cohort admission for repeated-token
tests, and localize the C1 difference with fixed inputs/teacher-forced
logits at the prefill-to-decode boundary. Consider equal KV capacity as an
isolation control; do not assert scheduling alone is the root cause. Only
after numerical qualification should the remaining C2 gap and C8/C16
scaling be accepted or further tuned using a new whole-model trace.

Raw: `endpoint_candidate_v4.json`, `endpoint_comparison_v4.json`, both arm
logs and complete token sequences. All benchmark processes exited and
GPU0..3 were released. No persistent API was started.
