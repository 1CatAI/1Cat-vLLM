# Qwen3.8 no-MTP batch decode: exact follow-up

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
  `VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M2=0` rolls back only this admission.
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
