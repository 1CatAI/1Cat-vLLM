# Flash-Next MTP4: default 15-ms complete-round campaign

The owner authorized changing FP32 accumulation association on 2026-10-03.
Bitwise identity is no longer an acceptance requirement for this campaign.
Calculation precision remains FP16, with existing FP32 accumulators and SSM
state. This authorization does not admit lower-precision weights, KV, or state.

## Frozen acceptance contract

- Integration: `onecat/main`; initial base
  `7eee270267d36c9648bad6286ede54c04a031046`.
- Owned branch: `agent/v100-flash-next-mtp4-15ms-20261003-033830`.
- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4`; TP4, one request, four
  V100-SXM2-32GB GPUs. Checkpoint expert quantization remains NVFP4.
- FP16 activations and main KV, FP16 convolution state and FP32 SSM state.
  PLE stays in host memory under the normal model memory policy.
- Exactly 8192 input tokens, four speculative tokens, greedy draft sampling,
  CUDA graphs enabled. Record actual resolved engine configuration and routes.
- No user-supplied `VLLM_*` optimization/diagnostic variables, profiler,
  private extension overlay, or per-iteration synchronization. Build/cache
  directories and standard CUDA toolchain selection are infrastructure only.
- Acceptance latency is endpoint steady decode time divided by completed
  draft-round counter increments. It includes all four drafts, target
  verification, sampling, state handoff, scheduler, and next-round preparation.
  Do not substitute target-forward time or divide by emitted tokens.
- Freeze the historical 8K+513 deterministic length fixture for regression.
  It contains forced output beyond EOS and therefore qualifies timing only.
  Also freeze natural-answer 8K prompts before observing candidates; measure
  with checkpoint sampling T1/top-p .95/top-k20 and normal EOS.
- Warm up first, reset prefix cache before measured requests, run at least
  three repeated requests and confirm in a fresh default engine. Report each
  observation and aggregate time/rounds. Every repeat must be <=15.0 ms;
  workload coverage, TTFT, acceptance and emitted-token speed remain explicit.
  No dropped outliers or profiler-adjusted numbers qualify.

The historical 21--23-ms results are useful evidence but are not a fresh
baseline after the intervening configuration/operator changes. Freeze the
current default baseline before altering production routes.

## Initial numerical gates

Use the same fixed prompt tokens, continuation tokens, position IDs and
checkpoint for control and candidate. Evaluate target and draft separately,
including decode/verification batch shapes; a prefill-only comparison cannot
qualify a decode-only kernel. Keep complete-vocabulary logits before sampling
transforms and precision unchanged. Compute metrics offline in FP64 so the
metric itself does not conceal small changes. Compare at least 2048 positions
across math, code, Chinese text and retrieval, including late recurrent state.
Also repeat a control to establish the ordinary numerical noise floor.

| Metric | Initial admission limit |
| --- | ---: |
| Mean forward KL from control to candidate, natural logarithms | <=1e-4 |
| P99 per-position KL | <=1e-3 |
| Maximum per-position KL | <=1e-2 |
| Top-1 token agreement | >=99.5% |
| Top-1 agreement where control top-two margin >=0.1 | >=99.9% |
| Maximum absolute raw logit error | <=0.125 |
| Maximum logit error after removing each row's common offset | <=0.125 |
| P99 of per-position maximum absolute raw logit error | <=0.03125 |
| Nonfinite logits/state or invalid snapshot writes | Zero |

These are engineering starting limits, frozen before candidate results, not
universal model-quality constants. Pinsker's inequality gives total variation
<=sqrt(KL/2): KL 1e-4 permits at most about 0.707% probability-mass movement
per position at that divergence; mean KL bounds mean total variation through
concavity, not every individual position. P99/max gates bound the tail.
FP16 spacing at magnitude 16 is 0.015625, so 0.03125 and 0.125 correspond to
two and eight ULP there. Actual logit magnitudes, ULP errors and common shifts
must be recorded; this scale explanation does not excuse passing large errors
at smaller magnitudes. High-margin top-1 agreement protects confident choices
while permitting limited changes at near ties. Report margin distributions.

Report errors by prompt and early/middle/late position segments to expose
recurrent drift. Passing aggregate limits cannot override a NaN, corrupted
state, unexplained growing tail, or a failed behavioral gate. Any threshold
revision needs a recorded rationale independent of making a candidate pass.

## Behavioral and acceptance gates

Reuse the frozen seed-42 GSM8K64, MATH-50064 and HumanEval64 fixtures and
graders from the prior evaluation. Run a fresh control under the same runtime,
normal EOS, output budgets, checkpoint sampling and seeds. Keep raw outputs.

- Correct answers may decline by at most one example per 64-example dataset
  and two examples across all 192. Record paired wins/losses and uncertainty;
  this small sample alone does not establish statistical noninferiority.
- Output-budget truncations may increase by at most one per dataset and two
  overall. Require no new request timeout, nonfinite output, or endless output
  beyond the declared budget; keep length/EOS/repetition diagnostics explicit.
- Report acceptance as accepted/proposed draft tokens, separately per dataset
  and natural 8K prompt. Decline must be <=2 percentage points AND <=5% relative
  against the matched control; require enough proposed tokens and repeated
  sampling for a stable comparison. A marginal/noisy result needs more evidence.
- Supplemental saved-prefix continuations are diagnostic and never replace
  primary scores. Preserve known reference inconsistencies and baseline failures.
- Before promotion, check long-context retrieval and the 256K boundary because
  recurrent changes can accumulate. Those checks do not replace the fixed-8K
  latency gate.

## Candidate priority and retained evidence

The calibrated historical cycle was target 15.534 ms, four drafts 4.575 ms,
sampling/handoff 0.781 ms and preparation 0.652 ms. These numbers belong to
their original source/runtime and are planning evidence only. Reaching 15 ms
requires several milliseconds, not a sum of unverified microbenchmark savings.

BV16 GDN is reopened for numerical and timing screening. It previously
changed a handful of FP16 outputs and many low FP32 state bits through LLVM
FMA contraction; the old harness stopped before timing because of bit checks.
Retest the ordinary unpinned arithmetic, including changing acceptance and
state snapshots. The separately pinned exact BV8/BV16 versions saved only
about 0.05 ms per 36-layer chain; do not assume the reopened candidate is a
large full-model speedup.

Inspect HC projection/mix fusion, MoE dispatch/materialization and four-step
draft execution next. Keep the target verifier on batch decode. An optimization
that merely changes acceptance or shrinks the speculative width does not
qualify the requested MTP4 latency. Candidate selection belongs in normal
capability/configuration dispatch; final users must not set tuning variables.

## Worklog

- 2026-10-03: inspected current main and open PRs. No open PR implements this
  complete-round campaign; existing QSA request-splitting and PLE placeholder
  PRs have separate scope. All four visible GPUs were idle at preflight.
- Created an isolated worktree and task-specific compiler caches. Native
  source build uses Torch 2.10.0+cu128 and CUDA 12.8; no wheel
  packaging is requested. All speed/quality admission remains pending.

- Reopened ordinary BV16 recurrence component (M5/M10, 36-layer chain):
  0.643029 -> 0.633856 ms and 0.777557 -> 0.762027 ms. Finite FP16 output
  and FP32 state, maximum observed output error 7.6294e-6 and state error
  1.9074e-6 on this synthetic changing-state screen. Differences are allowed
  to proceed numerically, but only 0.009/0.016-ms savings do not justify a
  model promotion or a new quality run. This is not teacher-forcing evidence.
- Real-checkpoint projection screen uses eight pairs (two layers x four TP
  slices), seven input scales and alternating graph timings. Splitting QKV
  K2560 across four warps gives M5 1.139328 -> 1.095840 ms per 36-layer chain,
  but M10 1.171104 -> 1.277600 ms. Reject as a performance candidate. The
  task-local research DSO is not loaded by the model/default benchmark.
- Native build completes after materializing the task's Triton source directory;
  the first install attempt failed copying a directory symlink. Preserve both
  build logs. Optional Rust build is unavailable; CUDA/Python model inference
  uses the normally built native extensions. No wheel was packaged.
- The first benchmark attempt exits before engine construction because its
  environment audit ran after platform import, which sets two framework
  defaults. Commit `fd39475e8d` moves the audit before vLLM imports. The launch
  contract has zero supplied VLLM variables; framework defaults are retained.
- Current default engine startup proceeds on leased GPU0--3 with MTP4 and V2,
  FULL_AND_PIECEWISE graphs and an exact M5 capture shape. Actual speed remains
  pending; do not report configuration diagnostics as measured request hits.

- Default post-capture warmup stalls before any request timing. All four CPU
  stacks stop at `GPUModelRunner.sample`'s hidden-state indexing. The PLE
  cascade implementation unconditionally waits for remote rows whenever
  `_is_cpu_offloaded` is set, including hybrid decode's complete local table.
  FULL replay deliberately submits no hybrid offload request: after prefill
  resets the semaphore, decode cannot proceed. Restrict the remote merge to
  cascade placement; hybrid decode gathers locally. Two focused behavior
  tests pass, covering both local decode and preserved remote cascade waits.
  Full native/default startup validation is pending. Both failed engines were
  stopped by verified task PID/cwd/start time; no foreign service was stopped.
- Current visible GPU0--3 topology has NV1/NV2 links but two SYS pairs. The
  custom all-reduce topology guard disables CUSTOM and selects PYNCCL. This
  is not the previous fully interconnected four-card benchmark topology;
  record it with new timings rather than comparing old wall times as matched.
- The next default run isolates vLLM's compile cache through the normal
  `CompilationConfig.cache_dir` infrastructure setting, in addition to owned
  Inductor/Triton/extension caches. No user performance environment flags.

- With the PLE condition repaired, a normal default engine completes graph
  capture, kernel warmup and a full 8K+513 warmup/request. The benchmark's
  first counter admission fails because `LLM` disables statistics by default,
  and its cleanup calls an obsolete engine method. Enable standard endpoint
  statistics explicitly and use `llm_engine.engine_core.shutdown()`. Preserve
  this failed harness run; it supplies no complete-round measurement.
- `CompilationConfig.cache_dir` does not isolate AOT artifacts or the separate
  decode compiler. The benchmark now sets only the internal cache-location
  root after auditing supplied variables. This is infrastructure, not an
  operator/performance override; no environment setup is required from users.

- Full-local HC split-K plus fused up/gate/mix screening is rejected: the
  96-module M5 control is 3.633 ms versus best candidate 14.586 ms; M10 is
  3.717 versus 14.582 ms. FP16 checkpoint intermediates remain finite, but
  neither numerical relaxation nor fusion alone implies lower latency.
  Retain the component JSON and source; no model promotion/quality reload.
  Its initial launch resolved the inherited installed vLLM instead of the
  owned source and lacked HC registration, producing no timing. The corrected
  run explicitly verifies the owned source import before timing.
- Audit discovers the earlier BV16 script also resolved the inherited installed
  module (different source hash). Its 0.009/0.016-ms result is therefore an
  exploratory component observation, not current-source dispatch evidence;
  do not promote it or use it in an endpoint speed estimate.
- Owner supplies a remote development host. Its four V100 GPUs are idle and
  fully connected by NV2 links. Keep the desktop/display GPU and all foreign
  source trees untouched. Runtime is Python 3.12.14 / Torch 2.10.0+cu128;
  transfer the task source and normal in-tree native build with SHA verification,
  no private overlay and no wheel. Freeze the host's 185-W GPU power cap.
