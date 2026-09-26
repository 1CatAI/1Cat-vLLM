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
