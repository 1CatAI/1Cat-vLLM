# Qwen3.8 no-MTP batch decode: exact follow-up

## Expert-reuse follow-up: batched W2 reduction (2026-09-26)

This follow-up remains on the owned branch / Draft #692 and integration base
`fcf59f8e9ae50c186333e98e5cf6aae705f320de`. The grouped candidate is still
default-off. There is no precision, router, top-k, K-split or sampling change.

The retained C16 step across 48 layers has 160 routes but only 1.83 routes per
active expert on average; 61% of active experts receive one token. C2/C4 do not
become large GEMMs just by admitting grouping. Their direct batch paths remain
unchanged: 45 of 48 real C4 route prefixes regressed under unconditional
grouping, while every screened C8/C16 route improved with the selected chain.

The new W2 operator assigns each CTA a 32-column tile for **the whole batch**.
Thirty-two warps traverse the GPU planner's actual expert packs, reusing the
weights for up to eight routed rows. They materialize W2's original FP16
results in 10 KiB of CTA shared memory; one barrier then permits the unchanged
ten ordered FP32 FMA contributions per token. This removes global scatter
materialization and the separate reduction launch. It adds no weight replica,
host route-count readback, floating-point atomics or persistent GPU worker.
The old grouped W2 operator remains available as the microbenchmark control.

The Python opt-in requires the new native capability explicitly and logs
`batch-column W2 with ordered CTA reduction`; an older extension cannot
silently substantiate the new route. Native registration, fake implementation,
production dispatch and the public benchmark are shipped together.

### Measured native microbenchmark

Normal source `setup.py build_ext --inplace` exits 0. The optional Rust
frontend reports a missing compiler; it is not used by the Python-engine
tests. No Rust frontend or complete wheel release is claimed. Required native
SM70 extensions and Flash-V100 components build normally, with no private
DSO, preload or RPATH/RUNPATH dependency. `_C.abi3.so` SHA256:
`01e0d9ffb37d2471affa74bbfafb33b686e0c9ffb40cc8d24fb987cc66b94f25`.

Physical GPU4 V100-SXM2-32GB, Torch 2.10.0+cu128, CUDA 12.8, driver 580.173.02;
actual layer-0/rank-0 checkpoint weights, 48 retained real routing patterns
(prefixes for C8), synthetic activations. Five alternating samples, ten graph
replays/sample, 16 whole expert calls/replay; includes planner, W13/SwiGLU,
W2 and ordered merge. No full-model or per-token wall claim in this table:

| Width | Direct mean | New grouped chain mean | Reduction | Sum of paired savings |
| --- | ---: | ---: | ---: | ---: |
| C8 | 88.45200 us | 75.50280 us | 14.64% | 0.62156 ms |
| C16 | 176.73800 us | 106.67747 us | 39.64% | 3.36291 ms |

All 48 patterns improve at both widths. Reusing one layer's weights for 48
routes is not measuring 48 different layers. The earlier independent GPU1
three-arm screen isolates the **increment over existing grouping**: 5.159 us
at C8 and 4.375 us at C16, averaged across the same 48 route patterns. Most of
the direct-to-candidate gain is the existing exact W13/W2 expert reuse; do not
add that gain to the previous grouping estimate a second time.

Focused native tests: **78 pass**, including M8/M16, both W13 layouts, changed
activations/routes/top-k weights, invalid IDs, poisoned metadata/outputs,
canaries and changed-input CUDA Graph replay. W13 and final W2 match the direct
path bit-for-bit. The fused operator leaves poisoned global routed scratch
untouched, confirming no hidden scatter/reduce fallback. Kernel parity is not
a full-model quality result.

Public reproduction after normal native build:

```bash
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 \
  TORCH_EXTENSIONS_DIR="$TASK_CACHE" \
  .venv/bin/python benchmarks/kernels/benchmark_sm70_moe_packed_w13.py \
    --model "$MODEL" --layer 0 --rank 0 --tokens 8,16 --splits exact \
    --interleaved --packed-w2 --batch-reduce --samples 5 --repeats 10 \
    --route-glob "$ROUTE_GLOB" --out "$RESULT"
.venv/bin/python -m pytest -q \
  tests/kernels/quantization/test_sm70_moe_packed_w13.py \
  tests/quantization/test_sm70_nvfp4_grouped_decode_dispatch.py \
  tests/benchmarks/test_sm70_qwen38_concurrency.py
```

Raw local artifacts: `.artifacts/moe_batch_native_48.{json,log}`,
`moe_batch_native_tests.log`, `moe_reuse_w2_sweep.{json,log}` and
`build_moe_batch_reduce.log`. The retained local `moe_root_cause.md` separates
NCU instrumentation, real graph service and unprofiled engine wall contracts.

### Rejected schedules and next gate

- W13 register-only activation did not earn a win; retain the prior cached
  same-split implementation.
- Different experts per Volta quad-pair fragmented weight/scale access and
  made the full chain much slower; numerical equality alone does not admit it.
- Real SM70 software producer/consumer double buffering with 4/8/16 K groups
  per stage preserves arithmetic but increases the complete chain from about
  118 us to 132--142 us. Do not repeat this as simple register prefetch.
- Fused W2 eight-column variants regress to 148--161 us. A sixteen-column
  eight-warp variant is comparable to, not a demonstrated substantial gain
  over, the selected coalesced 32-column variant. No autotuning flag is added.

All research-only variant sources/results are retained locally in
`.artifacts/moe_batch_reuse*`; none of those DSOs is used in engine tests.
The next measurement uses exactly one control and one MoE-only candidate
initialization, same new native hash, GPU4--7, 8K/256 tokens, TP4, no MTP/prefix
cache, max context 256K, FP16 activation/KV and the existing hybrid PLE policy.
Other experimental HC, sum2, gate and GDN flags remain off in both arms.

The concurrency harness now reports p90 and runs concurrent natural-EOS
text-health checks after timed cohorts. An explicit `--diagnostic-reference`
can consume a fully measured nonrepeatable control, but propagates its failed
acceptance into the candidate and never marks it accepted. This avoids another
model reload just to collect the cases omitted by a quality-fail-fast harness.
Matched endpoint results and quality status follow after that measurement.

## Scope and baseline

Integration: `onecat/main`, base `fcf59f8e9ae50c186333e98e5cf6aae705f320de`.
This follows the September 26 fixed-width C1/C2/C4/C8/C16 trace. It does not
change checkpoint precision, recurrent state, KV precision or sampling.

The matched 8,192-input/256-output, TP4 V100, FP16-KV, full-decode-graph
baseline measured 10.336/17.545/18.251/21.692/28.941 ms per batch step.
These are complete engine timestamp intervals, not summed kernel service.
The 262,144 context limit is configuration, not an input-length claim.
PLE uses mmap for prefill and rank-local pinned-UVA for decode; this is not
disk-only decode. The baseline used the current Python tree with the earlier
source-built native artifact; a newly built extension requires its own control.

## First implementation candidates

- Admit the missing FP16 `[2, 2560]` sum2 payload to the existing graph-only,
  fully-connected SM70 TP4 push path. Five 128-thread CTAs cover its 10 KiB.
  Preserve the established FP16 local sum and rank-ordered FP32 reduction.
  `VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M2=1` opts into only this admission.
  The initial default-on proposal was withdrawn after the engine quality gate
  below failed; this does not attribute the failure to the collective.
- Keep the batched shared-expert linear unchanged. Fuse only its FP16 sigmoid
  and output multiply, preserving the intermediate FP16 rounding. The M1
  fused dot is unchanged. The new operator is registered in the normal `_C`
  extension, not an external runtime overlay. The new batch epilogue remains
  opt-in via `VLLM_SM70_QWEN38_SHARED_GATE_BATCH_EPILOGUE=1` during validation.
- Screen the existing HC combine/norm memory scheduling at small batch sizes
  before changing production dispatch.

Do not simply widen the M1 shared-dot or HC guards. Historical batched dot
fusion changed outputs. Earlier packed-HC and plain TP-sharded-HC experiments
failed the complete-chain performance gate; those implementations are not
being repeated here.

Overlap check: existing Draft #504 owns a separate batched HC projection /
private-channel candidate with outstanding numerical/model-quality admission.
This work does not duplicate or enable that route. Its initial changes retain
the existing batched projection arithmetic and address gate epilogue and C2
sum2 admission; the GDN split-copy optimization is already present in this base.

## Validation contract (in progress)

1. Focused CPU dispatch/capability tests, including older wheels and rollback.
2. Gate epilogue: all 65,536 FP16 logit encodings, finite bit equality including
   signed zero, special-value classification, changed-input/poisoned CUDA
   Graph replay and 48 actual checkpoint gate weights.
3. Sum2: mixed-size graph transitions including 10 KiB, changing inputs,
   poisoned outputs, canaries, rank skew, NaNs/infinities/subnormals and an
   independent rank-ordered reference. Time full 48-layer collective rounds.
4. Admit only exact and beneficial microbenchmarks to a matched engine A/B;
   require identical greedy token sequences against that same-build control.
   Short text-health checks supplement, but do not replace, kernel parity.
5. Report endpoint throughput separately from graph service and projected
   microbenchmark savings; release task-owned GPU workers after tests.

Microbenchmarks:

```bash
.venv/bin/python benchmarks/kernels/benchmark_sm70_qwen38_batch_gate.py \
  --model /path/to/checkpoint --out /path/to/gate.json
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/benchmark_sm70_tp4_mtp5_push_allreduce.py \
  --tokens 2 --json-out /path/to/sum2.json
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_tp4_small_message_push.py \
  --out /path/to/mixed-size.json
```

Use task-owned caches and idle reserved GPUs. Native changes are built using
the ordinary source `setup.py build_ext --inplace`/wheel pipeline, not by
loading a private communicator DSO. Results and admission decisions follow
below once measured. No endpoint gain or completed quality gate is claimed yet.

AI assistance: implementation and test drafting assisted by Codex. Human
review and acceptance remain required before promotion from Draft.

## HC combine/norm screen

The production Triton kernel was screened verbatim at 512/1024 tiles, with
and without early weight loads, 32 changed-input cases per width, four input
scales, and six alternating timing samples. Batch's existing 512 tile plus
prefetch remains bitwise equal. Timings for 96 repeated combine/norm calls:

| Rows | Existing (ms) | Prefetch (ms) | Saving (ms) |
| ---: | ---: | ---: | ---: |
| 2 | 0.24248 | 0.22268 | 0.01980 |
| 4 | 0.24783 | 0.22505 | 0.02277 |
| 8 | 0.24979 | 0.22832 | 0.02147 |
| 16 | 0.27026 | 0.25100 | 0.01926 |

This is a small pointwise-component gain, not the entire HC chain or endpoint.
The 1024-tile variants change reduction association and fail bitwise parity
(up to 0.00390625 in these batch cases); they are rejected. This does not
change the established M1 policy or by itself prove a model-quality defect.

Only weight-load scheduling is added behind
`VLLM_SM70_QWEN38_HC_BATCH_NORM_PREFETCH=1` for FP16 H2560/HC4, M2--M16.
M1, other architectures, shapes and large-prefill dispatch remain unchanged.
Raw task artifacts: `.artifacts/hc_combine_screen.{json,log}`. The standalone
screen extracts the unchanged JIT function to avoid needing unrelated native
operators during the first source build; full runtime gates remain pending.

The public `benchmarks/benchmark_sm70_qwen38_concurrency.py` retains the prior
measurement's complete engine-timestamp criterion. It records complete token
sequences and the native hash. `--reference` requires same-contract per-request
greedy token equality; `--health` separately uses official sampling/natural EOS.
`--measure-prefill` reports a separate one-output cohort using the union of
scheduled-to-first-token intervals, not a sum of overlapping request TTFTs.

## Native microbenchmark results and priority correction

The initial candidates pass their local tests but are **not sufficient** for
the user's substantial concurrency-speedup objective. Do not start a full
model simply to measure these small components:

- Shared-gate epilogue, 48 actual gate weights: saves 0.158--0.164 ms at
  M2/M4/M8/M16. All 65,536 FP16 logit encodings pass the arithmetic check;
  11 targeted GPU tests pass.
- C2 sum2 admission: 48 ordinary plus 48 sum2 collectives fall from
  0.70780 to 0.31534 ms in the paired TP4 microbenchmark (0.39246 ms saved).
  Both arms already use ordinary small-message push. Mixed-size/poisoned
  graph checks pass for random, signed-zero and special-value inputs, 64
  cycles per family on all four ranks.
- HC combine/norm prefetch: only about 0.02 ms, as above. The remaining HC
  projection work is not solved by this pointwise change.

The next screen therefore targets MoE, which accounts for about 7.825 ms of
the C1-to-C16 increase in the retained trace's rank-average service view.
This is hotspot selection, not an additive endpoint prediction.

### Hardware evidence and rejected variants

One NCU sample of the production C16 W13 kernel, layer-0/rank-0 checkpoint
weights and a retained real routing pattern, reports:

| Counter | Value |
| --- | ---: |
| DRAM throughput | 50.19% (416.15 GB/s) |
| SM throughput | 33.08% |
| Achieved occupancy | 30.22% |
| L2 hit rate | 32.37% |
| Long-scoreboard share of warp cycles between issued instructions | 64.1% |

This is a single instrumented kernel sample, not a whole-model utilization
claim. It justifies investigating memory latency and instruction scheduling;
there is no evidence here of a saturated HBM or compute ceiling.

Source-derived research screens retain the original ordered HMMA sequence and
all FP16 boundaries. Results, including negatives, are retained in task-local
`moe_iteration_v1/v2/v3` and `moe_group_v1/v2` JSON/log artifacts:

- Removing masked input loads or changing unroll factors alone is neutral or
  slower. Explicit 2/4/8-group staging does not earn integration.
- Read-only caching plus suitable launch bounds saves about 1.85 ms projected
  over 48 C16 calls, without grouping. The same direct-kernel edit regresses
  C4 and gives only a small C8 gain; **it is not installed in direct dispatch**.
- Applying scheduling/cache changes to the existing grouped implementation,
  while keeping C16 at **split1**, earns the larger result below. This reuses
  the already-main grouping and W2 code; it does not duplicate Draft #504's
  HC private-channel work.

### Source-built exact C16 grouped MoE

The final implementation is in the normal
`csrc/sm70_turbomind/ops/nvfp4_grouped_decode_sm70.cu` build:

1. Reuse packed expert weights for rows routed to the same expert, using the
   existing integer planner and original-route scatter.
2. Use the read-only cache and 64-thread/eight-resident-block launch bounds
   for the single-split W13 kernel. This permits latency-hiding instruction
   scheduling without introducing more K splits.
3. Remove the redundant split reduction and one barrier when there is only
   one partial, retaining the direct path's FP16 rounding (including zero
   sign). W2 and its ordered FP32 top-k reduction are unchanged.
4. The existing experimental opt-in now selects M16 **split1**, not split8;
   M8 retains split4. No new flag is added. The default remains off until the
   matched full-model gate; C1, direct C2/C4, MTP and prefill are untouched.

Normal source CMake `_C` rebuild/install succeeded. No research DSO is loaded
in the following native benchmark or GPU tests. Measured `_C` SHA256:
`9230c386d485a29ed2f6571ba88b7b2bd9064ded4f94d40fdddbe0a0581e3787`.
It has no RPATH/RUNPATH or private-library dependency. Subsequent source-only
formatting/braces do not change the algorithm. Full package build had an
unrelated missing-`patchelf` failure while packaging Flash-V100; no complete
wheel or full-engine run is claimed for this task.

Native screen: Torch 2.10.0+cu128, CUDA 12.8, physical V100-SXM2-32GB GPU2,
FP16 activations/prepared scales, unchanged NVFP4 weights and FP32 accumulation,
actual layer-0/TP4-rank-0 weights, 48 retained C16 routing patterns. Five
alternating samples, ten graph replays/sample, 16 complete MoE calls/replay.
Timing includes planning, W13/SiLU, grouped W2, scatter and ordered reduction.

| Complete MoE call across 48 routing patterns | Result |
| --- | ---: |
| Mean direct control | 175.81693 us |
| Mean exact grouped candidate | 111.40787 us |
| Time reduction | 36.63% |
| Sum of paired microbenchmark savings | 3.09164 ms |
| Per-pattern saving range | 44.448--80.275 us |
| Changed-input W13/W2 maximum absolute difference | 0 |

All 48 improve. **This reuses one layer's weights with different real routes;
it is not a measured 48-layer model round or a new endpoint tokens/s result.**
It clears the small-gain screen and warrants a matched engine A/B, not default
enablement or a claimed C16 step of `28.94 - 3.09` ms.

Reproduce with the public benchmark and local checkpoint/retained route paths:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
  TORCH_EXTENSIONS_DIR="$TASK_CACHE" \
  .venv/bin/python benchmarks/kernels/benchmark_sm70_moe_packed_w13.py \
    --model "$MODEL" --layer 0 --rank 0 --tokens 16 --splits 1 \
    --interleaved --packed-w2 --route-glob "$ROUTE_GLOB" \
    --samples 5 --repeats 10 --out "$RESULT"
```

Without retained routes, omit `--route-glob` to screen distinct/shared/random
routes; those synthetic-route numbers must be reported separately.

Validation:

- 57 tests pass: 24 grouped GPU tests plus the then-current 33 dispatch tests.
  New C16 tests exercise 32 changing-input graph replays for each W13 layout,
  eight routing patterns, four activation scales, output canaries and poisoned
  metadata/output. Both W13 output and complete W2 result are bitwise equal
  to the existing direct kernels. Full same-split M4/M8 tests also pass.
- After adding the M8/split4 and M16/split1 production-dispatch assertions,
  the grouped-dispatch plus NVFP4 integration suites pass **76 tests**. These
  suites overlap the preceding run; do not sum the counts.
- Targeted Ruff and whitespace checks pass. This is not a full pre-commit,
  long-output-quality or token-sequence acceptance claim.

Raw native artifacts: `grouped_native_48.{json,log}`,
`grouped_native_pytest.log`, `grouped_dispatch_pytest.log`, and
`build_grouped_incremental.log` under the task's `.artifacts` directory.
The public benchmark now additionally enforces bit equality for same-split
paths, rather than treating positive/negative zero as equal.

No full model/API was started. Task GPU workers exited. An attempted extra
layer/rank sweep yielded its cooperative GPU lock when another task acquired
GPU0--3, before launching any new CUDA process. Next: finish the extra-weight
screen when cards are free, then one matched native-engine control/candidate
test with endpoint timing and greedy/text-health gates. Keep Draft.

## Broader batch-benefit screen (2026-09-26 afternoon)

The 3.09-ms MoE estimate above is still not sufficient evidence of a substantial
**whole-engine** concurrency benefit. Reuse the retained trace instead of
reprofiling a full model for every candidate. C1 to C2 increases rank-average
HC service from 2.043 to 4.016 ms and other dense projections from 3.687 to
5.765 ms. Those are priority-selection numbers, not additive endpoint savings.

### Fixed-arithmetic HC output sharding

Simply allowing a new GEMM shape to select its own cuBLAS heuristic changes
the K partition. The new **benchmark-only** screen pins the replicated
projection's algorithm, K partition, reduction storage type and stages while
reducing each rank's output columns. No accumulation precision is reduced.
In the declared CUDA 12.8 runtime, the matched down configuration is algorithm
21 / tile 5 / stages 14 / split 22 / reduction 4; up is the same algorithm and
tile with split 1 / reduction 0. These IDs are runtime-specific, not a portable
production default.

The fused SiLU/gather and mix/gather research code is adapted from
[Draft #504](https://github.com/1CatAI/1Cat-vLLM/pull/504), revision
`5a049230cda498087a537efd292fe7fa386f8e81`. Unlike that draft's ordinary local
GEMM heuristic, this screen retains the original replicated GEMM's arithmetic.
Signed zero is not canonicalized through a disjoint-output sum. The fixture
passes raw IPC buffer addresses from its own dedicated channel, never an
opaque communicator object across extension ABIs. It does not install an HC
model wrapper, enable a runtime flag, or load a research library in serving.

TP4 V100 GPUs 4--7, Torch 2.10.0+cu128, CUDA 12.8, FP16 checkpoint weights and
activations, FP32 accumulators; all **96 different checkpoint HC weight pairs**.
Fixed-pointer graphs, six alternating trials, 24 replays/trial. Maximum rank
time per trial is retained. Includes down/up, SiLU, gate mix and both gathers;
excludes combine/norm, the final mixer, attention, MoE and scheduler work:

| Width | Replicated control | Exact sharded chain | Saved | Chain reduction |
| --- | ---: | ---: | ---: | ---: |
| C2 | 3.16211 ms | 2.54441 ms | 0.61771 ms | 19.53% |
| C4 | 3.21811 ms | 2.58321 ms | 0.63490 ms | 19.73% |
| C8 | 3.29751 ms | 2.66477 ms | 0.63275 ms | 19.19% |
| C16 | 3.44267 ms | 2.81483 ms | 0.62784 ms | 18.24% |

All four ranks, four widths, 96 pairs and six input scales
`0/0.001/0.03/0.1/1/3` have zero bit differences for block and injection outputs,
including changed-input graph replay with poisoned outputs. These are kernel
checks, not a model-quality score. The speedup is a component result, **not
19% whole-engine improvement**. Additional packed-up storage would cost about
150 MiB/rank across these pairs; production integration remains deferred until
the extra complexity earns a sufficiently large complete-step benefit.

Portable reproduction (research-only JIT is built from these shipped sources):

```bash
CUDA_VISIBLE_DEVICES='' TORCH_CUDA_ARCH_LIST=7.0 \
  .venv/bin/python -m benchmarks.kernels.benchmark_sm70_hc_batch_exact --build-only
CUDA_VISIBLE_DEVICES=4,5,6,7 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  VLLM_SM70_TP4_PUSH_ALLREDUCE=1 TORCH_CUDA_ARCH_LIST=7.0 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --module benchmarks.kernels.benchmark_sm70_hc_batch_exact \
  --model "$MODEL" --rows 2,4,8,16 --out "$RESULT"
```

Use a declared CUDA 12.8 `CUDA_HOME`, task-owned Torch/Triton caches and GPU
locks. The cuBLAS benchmark helper now supports the versioned standard CUDA
libraries supplied by Torch wheels, without requiring private library paths.
Raw initial results: `.artifacts/hc_shard_tp4_v1.{json,log}`; projection-only
screens: `hc_dense_v1`, `hc_shard_fixed_v1` JSON/log pairs in the same directory.

### Rejected additional schedules

- Plain cuBLASLt heuristic changes produced no useful exact winner. Fixing the
  K partition while varying output tiling saves only about 1 us per router,
  b/a or output projection. This does not justify new production dispatch.
- Six CUTLASS WMMA small-M schedules preserve the exercised outputs but are
  slower: GDN QKVZ C2 best 51.63 us versus 40.02 us; HC up C2 best 11.84 us
  versus 11.27 us. Eight distinct real-weight allocations rotate through each
  timed graph, avoiding a one-weight hot-cache claim.
- Narrowing grouped MoE output tiles preserves all 24 changed-input/route
  checks, but increases total call cost at C16: 117.33 us for the current exact
  grouped path versus 152.57 us with half-width tiles, or 205.21 us with quarter
  width. More resident work alone is not an efficiency win. The full-width
  one-warp variant is neutral at C16 and saves only about 0.24 ms projected
  across C8's 48 calls. No variant is added to production.

Raw artifacts: `dense_v1`, `dense_schedule_v1`, `dense_wmma_v1`,
`moe_narrow_v1` JSON/log pairs. These are research screens, not native-engine
speed claims. GPU microbenchmark workers exited after each run.

The source packaging failure noted above was resolved by installing `patchelf`
in the task uv environment and resuming the same incremental build. Normal
source build now completes, without copying another tree's kernels. Rebuilt
`_C` SHA256 is
`ac11029627cc046e9e06d3392a1ce5de376fdc6faa3524ee7af9074ca3592a03`;
no RPATH/RUNPATH or private DSO dependency is present. The subsequent native
engine A/B uses the already-implemented grouped MoE/gate/norm/sum2 candidates,
**not** the research-only sharded HC route.

### Whole-engine check: not accepted

Exactly two model initializations were used: one control, then one candidate.
Both use the same source-built `_C` hash above, physical GPUs 4--7,
TP4 V100-SXM2-32GB / driver 580.173.02 / CUDA 12.8 / Torch 2.10.0+cu128,
8,192 input tokens, 256 forced greedy output tokens, two repetitions per
width, no MTP or prefix cache, 262,144 configured context, FP16 activations/KV,
the ordinary dual-compile/full-decode-graph defaults and memory utilization
0.9. PLE uses mmap prefill plus **12 GiB/rank pinned-UVA decode**. This is the
97-tok/s contract, not disk-only decode. Generation and prefill are separate.

The control reproduces the prior performance. The following are complete
engine intervals, **not** kernel-service sums or accepted candidate gains:

| Width | Control aggregate decode, two repetitions | Control step |
| --- | ---: | ---: |
| C1 | 96.824 / 96.808 tok/s | 10.328 / 10.330 ms |
| C2 | 114.077 / 113.996 tok/s | 17.532 / 17.544 ms |
| C4 | 221.655 / 221.571 tok/s | 18.046 / 18.053 ms |
| C8 | 371.203 / 371.417 tok/s | 21.552 / 21.539 ms |
| C16 | 556.707 / 556.771 tok/s | 28.740 / 28.737 ms |

Control prefill is 6,872--6,915 input tok/s across the ten separate cohorts.
Both natural-output checks pass and stop at EOS in both arms. However, the
candidate's first C1 case (96.640 tok/s) differs from both control C1 repeats
at **zero-based token 63**. Both outputs are readable; readability is not
token-parity admission. The then-fail-fast harness stops there, so candidate
C2/C4/C8/C16 and prefill were **not measured**. Do not claim that the projected
3.09-ms MoE saving has become a measured whole-engine improvement.

Offline comparison of already-saved data also finds control C2 request 0
differs between its own repetitions. C1/C4/C8/C16 control repeats match.
Consequently the observed difference does not by itself identify a new kernel
as the cause. C1's dispatch guards exclude the new batched arithmetic paths;
graph initialization/state or existing reproducibility must be localized
before acceptance. No logit-margin or per-layer activation comparison was
captured, so **no root cause or harmlessness claim is made**.

Decisions:

- Keep Draft; no merge, default promotion or end-to-end speedup claim.
- New C2 sum2 admission is also made explicitly opt-in, matching the other
  experimental candidates. Existing C1/C4/C8/C16 default collectives are not
  changed. The native default-off rebuild is separate from the measured hash.
- Fix benchmark efficiency: preserve first-difference positions and within-run
  repeatability, collect the remaining planned cases after token differences,
  then fail the final gate. `measurements_complete=true` is distinct from
  `complete=true`; a parity failure leaves the latter false and exits nonzero.
  Health/structural failures still abort. No third model restart this turn.
- Before another engine run, localize the first differing greedy prefix with
  fixed-input/logit and layer-boundary checks; do not combine four unknown
  switches and blindly restart. Existing saved control evidence is retained.

Reproduction uses the public concurrency harness. Set the four switches
`VLLM_SM70_QWEN38_SHARED_GATE_BATCH_EPILOGUE`,
`VLLM_SM70_QWEN38_HC_BATCH_NORM_PREFETCH`,
`VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M2`,
`VLLM_SM70_NVFP4_MOE_GROUPED_DECODE` to 0 for control and 1 for candidate.
Keep other task environment variables/caches explicit and identical in
meaning, without private DSO/preload overrides:

```bash
export PYTHONPATH="$PWD/flash-attention-v100:$PWD"
export CUDA_VISIBLE_DEVICES=4,5,6,7 CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_QWEN4EXP_PLE_HOST_GIB=12 VLLM_PLE_OFFLOAD_PREFAULT=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1
.venv/bin/python -m benchmarks.benchmark_sm70_qwen38_concurrency \
  --model "$MODEL" --mode nomtp --widths 1,2,4,8,16 \
  --input-len 8192 --output-len 256 --repeats 2 --health --measure-prefill \
  --out "$CONTROL"
# The candidate adds --reference "$CONTROL" and writes a distinct output.
# Do not reuse a reference whose quality/repeatability gate has failed.
```

Artifacts: `.artifacts/control_gpu4567.{json,log}` and
`.artifacts/candidate_gpu4567.{json,log}`. The control was generated before the
new repeatability gate, so its old `complete=true` must not be interpreted as
passing that newly added gate. All engine workers exited after the failure.

### Deeper HC fusion and remaining work

A subsequent isolated microbenchmark fuses the local up projection, FP16 gate
materialization, exact sigmoid/mix and IPC gather into one CUTLASS WMMA CTA
epilogue. It uses a separate IPC channel and four-branch output tiling; it
does not change FP32 accumulation or FP16 rounding points. This research
variant is **not** installed in the model or added to production dispatch.

All 96 real HC pairs, four ranks, six changed-input scales and M2/4/8/16 have
zero block/injection bit differences. The portable exact-sharding fixture
also passes in this three-arm run. Maximum-rank median component timings:

| Width | Replicated | Exact shard | Up/mix/gather fused |
| --- | ---: | ---: | ---: |
| C2 | 3.16806 ms | 2.55388 ms | 2.60736 ms |
| C4 | 3.22458 ms | 2.58223 ms | 2.65658 ms |
| C8 | 3.29758 ms | 2.66507 ms | 2.72378 ms |
| C16 | 3.44213 ms | 2.81583 ms | 2.86933 ms |

Fusion alone loses 0.053--0.074 ms relative to the exact sharded chain and is
rejected. It removes a kernel boundary but makes the compute/communication
schedule worse. Raw research: `.artifacts/hc_up_fused_96.{json,log}`. The
two-pair smoke is used only for correctness, not a cold-layer throughput claim.

CPU analysis of the 48 retained C16 routing patterns used for the MoE screen
finds 160 routed slots spread over 87.29 distinct experts on average (53--126
depending on layer), for 1.83x potential weight reuse. Actual 8-row grouping
needs 87.85 groups on average: only 22.76% of row slots are occupied. These are
**one retained step's route statistics, not measured SM utilization or a
hardware throughput ceiling**. Prefixes of that batch give reuse ratios of
1.02/1.12/1.39 at C2/C4/C8. More thread blocks did not help the earlier screen;
future MoE work should target small-group load/dequantization pipelining, while
HC/other dense projections still dominate the C1-to-C2 jump. Do not assume
either direction will earn a large endpoint gain before measurement.

Raw CPU analysis: `.artifacts/route_reuse_stats.json`. The new benchmark
bookkeeping tests pass (11 tests); no full-model quality acceptance is inferred.

The default-off guard's ordinary source rebuild exits successfully; `_C` SHA256
is `863e344fb063cf7d0d383fba188ac2b4f3a9c65570ab1918b744b17b0657bd5e`,
with no RPATH/RUNPATH/private DSO dependency. The optional Rust frontend is
not built (no Rust compiler); this does not fail the Python/CUDA build.
A finite native TP4 smoke captures both C2 sum2 flags 0 and 1, interleaves
five graphs across 14 message sizes, and passes random, signed-zero and
special-value checks with poisoned outputs/canaries, four cycles per family.
This narrowly validates the opt-in change; it does not repeat the full earlier
64-cycle matrix or excuse the model token difference. Artifacts:
`.artifacts/build_native_optin.log`, `.artifacts/optin_native_mixed_size.{json,log}`.
All owned GPU workers are released; no API is left resident.

## Follow-up: packed batched GDN input, 2026-09-26

Continue on the same owned branch/PR and integration base, from source
`9b7fdd40072e0c81280a5271f344278ebc3bb0b0` plus this change. No additional
model initialization was used. The whole-engine token/repeatability issue
above is still unresolved; this section does **not** supersede that gate.

### Native candidate and numerical contract

Add `qwen38_gdn_input_batch_sm70_out` to the normal SM70 CMake build, selected
only with `VLLM_SM70_QWEN38_GDN_INPUT_BATCH=1`. The existing exact-model,
TP4, no-MTP, checkpoint-FP16/fused-GDN admissions must also pass. The new
runtime guard admits contiguous/aligned FP16 M2..16 only; M1, prefill,
batch-invariant mode and unsupported layouts retain the previous path.

The loader copies, rather than quantizes, FP16 weight bits into N32/K16 tiles.
One kernel writes QKV, Z, b and a directly, replacing two projections, the
small projection's reduction and output-layout copies. QKVZ retains a single
ordered K reduction; b/a retains four contiguous 640-element partitions and
the original left-to-right FP32 partial sum. Balanced b/a sums are rejected:
they produce FP16 bit differences. There is no precision or rounding-point
reduction. Original weights remain available for M1/prefill; packed buffers
are nonpersistent and rebuilt after weight reload.

Extra allocation is **725.625 MiB per rank across 36 GDN layers**. This is a
real capacity tradeoff, not a free optimization. Production promotion still
requires checking available KV-cache capacity and model-level equivalence.

### Complete 36-layer component benchmark

V100-SXM2-32GB, driver 580.173.02, CUDA 12.8, Torch 2.10.0+cu128. Real
Qwen3.8-Flash-Next-NVFP4 checkpoint FP16 projection weights, synthetic FP16
activations, FP32 accumulation. Both arms have split-copy fusion enabled.
Each graph runs all **36 different GDN weight pairs**, not repeated layer0
weights. Seven alternating trials, 16 graph replays/trial. Physical GPU0
tests TP weight ranks 0/1/2 separately; GPU2 tests rank3. These are independent
single-device component tests, **not simultaneous TP4 engine wall time**.
Clocks/caches are not fixed by a profiler.

Rank0 median whole-input-chain results:

| Width | Previous chain | Packed fused chain | Saved | Reduction |
| --- | ---: | ---: | ---: | ---: |
| C2 | 1.810624 ms | 1.098304 ms | 0.712320 ms | 39.34% |
| C4 | 1.795968 ms | 1.095168 ms | 0.700800 ms | 39.02% |
| C8 | 1.812864 ms | 1.120512 ms | 0.692352 ms | 38.19% |
| C16 | 1.851520 ms | 1.171072 ms | 0.680448 ms | 36.75% |

Ranks1/2 have C16 pairs 1.850944 -> 1.168320 ms and
1.851520 -> 1.168960 ms; rank3 on physical GPU2 has
2.005376 -> 1.191552 ms. Do not attribute a device-dependent timing
difference to rank arithmetic. All 36 layers, all four rank weight sets,
M2/4/8/16 and six changed input scales (0/0.001/0.03/0.1/1/3) have **zero
FP16 bit differences for all four outputs**, including fixed-pointer graph
replay and poisoned candidate outputs.

This saves approximately 0.68--0.71 ms of component work on GPU0. It does not
establish a new endpoint tok/s result, and must not be added to previous
MoE/HC service sums as if they were non-overlapping engine wall time.

Reproduce after building this worktree's ordinary native extension:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 \
  VLLM_SM70_QWEN38_GDN_INPUT_BATCH=1 VLLM_SM70_GDN_BATCH_SPLIT_COPY=1 \
  PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/kernels/benchmark_sm70_gdn_input_batch.py \
  --model "$MODEL" --layers all --rank 0 --rows 2,4,8,16 --out "$RESULT"
```

The benchmark uses installed native registration, no JIT sidecar. Use the
task GPU lock and explicit task caches. Repeat `--rank 1/2/3` to cover the
other checkpoint slices. It exits nonzero on any bit mismatch and retains
the mismatch counts instead of timing a rejected candidate.

### MoE/QSA rejected variants and confirmed limits

- NCU on the actual retained C16 layer0 route (99 distinct experts) measures
  current grouped W13 DRAM throughput **72.81%, 621.82 GB/s**, SM throughput
  32.62%, achieved occupancy 18.72%. Grid (5,160), 64 threads, 78 registers.
  These counters are from one instrumented kernel, not the whole MoE chain
  or an absolute hardware ceiling. The earlier synthetic 160-distinct-expert
  counter sample is not substituted for this real-route sample.
- Explicit cross-iteration MoE prefetch variants and W2 read/prefetch
  variants preserve tested bits but regress: current grouped chain about
  118.4 us; best new W13 variant about 122.19 us. No new MoE schedule is
  shipped in this follow-up; the prior grouped candidate remains separate.
- Batched QSA physical-index resolution saves only about 0.1 ms across
  twelve selected-attention calls, insufficient to justify another route.
  Register limits, more warps and small tile variants are neutral, slower
  or numerically different.
- Capacity-sized QSA scoring launches many inactive CTAs at short contexts.
  Bounded/strided Triton scheduling is exact and improves an isolated C16
  8K score call (about 69 -> 48 us), but regresses the 256K boundary. No
  static-policy switch is admitted.
- Inspection of this Triton 3.6 SM70 score kernel finds ordered FP32 SIMT
  FMA, not Tensor Core MMA, despite padding four heads to sixteen. Native
  MMA experiments change FP32 score bits and are rejected, even where the
  error is small. This is a statement about this compiled kernel, not all
  V100 attention implementations.
- A native SIMT rewrite preserves the original 128-term FMA and
  left-associative head sum, passing all score/visibility bit checks through
  256K. It is nevertheless slower. Hoisting page arithmetic and fully
  unrolling the reduction still gives C16 8K about 99 us versus 65 us,
  and 64K about 765 us versus 299 us. Neither native scorer is integrated.
  All experimental QSA dispatch/scheduling edits were removed.

Raw results are retained under the owned worktree's `.artifacts/`:
`gdn_input_all36_r{0,1,2,3}.{json,log}`, `gdn_input_packed_v1`,
`dense_packed_v1`, `moe_pipeline_v1`, `grouped_real_ncu.csv`,
`qsa_batch_indices_v1`, `qsa_schedule_v1`, `qsa_score_capacity_v1`,
`qsa_score_native_v1`, and `qsa_score_simt_v{1,2,3}`. Research experiments
used isolated source-built JIT libraries; none are runtime dependencies.

The focused native GDN/previous split-copy suite passes **72 tests**,
including M2..16, opaque-op graph replay, M1/M17 fallbacks, unaligned-storage
fallbacks, output canaries, invalid output geometry, exact weight packing,
reload/nonpersistent buffers, and disabled/batch-invariant admission.
These are kernel/dispatch tests, not full-model output-quality acceptance.

The source-complete build exits successfully; the optional Rust frontend is
not built in this environment. The timing-confirmation `_C` SHA256 is
`5af5b4310919d039dc9a31f7b129023bef8b8dbdc2466c35544e134b2ccb1c0b`.
`readelf -d` has no RPATH/RUNPATH or private DSO dependency. A fresh-process
36-layer C16 native check with `LD_PRELOAD`/`LD_LIBRARY_PATH` explicitly unset
passes all six-scale bit comparisons and measures 1.883712 -> 1.194368 ms
(0.689344 ms saved). The all-rank
matrix above used the earlier GDN build
`6ea6685d43b3bb44d1340f2156d0887982eb50d217bf276c4ea2c8567d7017b5`.
The rebuild removes an unrelated research-only registration guard; GDN
arithmetic is unchanged. Artifacts: `build_gdn_final.log`,
`gdn_input_final_r0_m16.{json,log}`, `gdn_batch_tests_final.log`.

After final formatting and a local-variable spelling fix, the final native
build hash is
`c22e3d18ccc71f9f8943afed89fd1e900afd3f701c0e9afb129a638d7c199404`.
It retains only standard CUDA/Torch dependencies, with no RPATH/RUNPATH.
Declared packages include cuBLAS 12.8.4.1 and CUDA runtime 12.8.90.
The focused fresh-process final-build smoke passes four tests: M2/M16 opaque-op graphs,
M9 output canaries and packed-weight reload. No whole-engine rerun or
whole-model quality approval is claimed. All changed-file pre-commit checks
pass; no repository-wide sweep was needed. All owned GPU workers exited;
no API or automatic GPU queue is left running.
