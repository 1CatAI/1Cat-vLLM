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
  PLE defaults to disk mmap; only bounded results/control buffers are pinned.
- Exactly 8192 input tokens, four speculative tokens, greedy draft sampling,
  CUDA graphs enabled. Record actual resolved engine configuration and routes.
- Service maximum remains 262144 tokens.
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

## Shared numerical acceptance

Use the [shared Flash-Next distribution contract](sm70_qwen38_distribution_acceptance.md)
and `benchmarks/qwen38_distribution_probe.py` for both no-MTP and MTP. That
single contract defines the mean/p99/maximum KL, top-1 and raw-logit error
limits; this document maintains no separate threshold or ULP gate. FP16
calculation and existing FP32 accumulators/state remain unchanged.

Compare complete valid-vocabulary logits on identical frozen teacher-forcing
prefixes, separately for target and draft. Repeat the default arm to establish
noise. MTP's alignment adapter reuses the shared row metrics, summarization
and thresholds; it also checks actual accelerated top-1 decisions. Raw and
centered error, tail values and high-margin agreement remain visible as
diagnostics. No forced/capture request qualifies latency or natural acceptance.

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

The owner-approved implementation order is:

1. M5 TP4 HC down/inject and up/mix sharding, using the existing FP32-partial
   fused operator. The logical weight-traffic saving is approximately 1 GB
   per rank per target round; verify actual dispatch and complete-round gain.
2. A resident MoE kernel spanning grouping, W13/SwiGLU, W2 and weighted
   reduction, preserving every selected expert and FP16 arithmetic boundaries.
3. A reduced draft vocabulary with LM-head work overlapped with drafting,
   measuring coverage and acceptance separately from target distributions.
4. Adaptive draft length with actual proposal-width histograms and emitted-token
   speed, alongside a fixed MTP4 control. Shorter rounds are not relabeled as
   four-draft verifier measurements.

Reuse PR #831's bounded mapped result staging on disk-backed PLE and prefetch
future draft positions without publishing speculative results as committed
rows. Request identity, ngram history, rejection and cancellation remain
correctness constraints. BV16 and four-warp micro-optimizations are closed;
retain their negative/small results without repeating them.

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

- The owner confirms 256K service admission remains mandatory on the remote
  host; speed inputs remain exactly 8192 tokens. Do not shrink context or KV
  capacity to hide a failed startup. The first remote default run is killed
  by the kernel OOM handler while PLE tables materialize (zero timed cases).
  Its logical placement is 9.04 GiB pinned host plus 2.88 GiB device per rank.
  Torch 2.10's CachingHostAllocator rounds 9.04 GiB to 16 GiB; four backing
  allocations exceed the host's 62.7-GiB RAM despite the 15.67-GiB reserve.
- Add a normal CPU-dispatched native factory using `cudaHostAllocMapped` at
  the requested byte size, with shared ownership retained by the existing
  UVA alias. PLE uses it without changing placement, dtype or table contents.
  Native source build succeeds; six GPU allocation, mapping and lifetime
  checks pass, including unchanged Torch host-cache allocation counters.
  No wheel, preload, private DSO or new user setting is required.
- The corrected owned-source BV16 recurrence component is 0.642517 ->
  0.632661 ms (M5) and 0.777301 -> 0.764160 ms (M10) per 36 layers; finite
  output/state checks pass, with maximum observed output error 3.0518e-5.
  This confirms a tiny component gain and does not justify promotion.
- Next numerical candidate: the worker's standard FP32-reduction policy
  prevents the legacy MTP HC half-partial schedule from dispatching. Reuse
  the already-shipped FP32-partial batch operator under that policy, honoring
  explicit MTP/batch disables and retaining the legacy half-partial selection
  when reduced reductions are enabled. Loader/dispatch CPU gates pass 26
  tests. Full-model numerical, quality, acceptance and speed gates are pending;
  this candidate is separate from the startup/memory repair.

- Add post-speed teacher-forcing capture in the same engine, preserving the
  target M5 graph and running only the diagnostic draft eagerly. Restore
  original runner methods even when a dump fails. Two alignment/restoration
  CPU tests pass; real-model capture remains pending.
- Freeze sixteen 8K teacher-forcing prompts (four GSM8K, four HumanEval,
  four Chinese explanations, four exact-key retrieval cases), each with
  160 reference continuation tokens plus padding. Reference tapes are
  independent of model-generated outputs. Manifest SHA256:
  `44ccb3ad6bea479c6b6c33d3c2bc44985283039e04ec41d2de05006fafeb4062`. Retain it under
  `.artifacts/mtp15/teacher_forcing_manifest.json`; never use forced requests
  for latency or natural acceptance.
- Remote GPUs were subsequently acquired by another task in the
  `qwen38-nomtp-20261003` worktree. Preserve that task and wait for its existing
  common TP4 lease before launching this campaign. No current default
  complete-round measurement has passed admission yet.

- Add a default-admitted draft local-argmax candidate for FP16 Qwen4Exp MTP
  on SM70/TP4, greedy and serial drafts only. An explicit true/false remains
  authoritative; probabilistic and other model/hardware routes keep full
  logits. Both legacy and V2 proposers consume the resolved normal config.
  Existing native LM-head kernels and padded-vocabulary reduction are reused.
  Candidate selection and graph hash are source-complete, without a new
  environment variable. Together with HC admission, 49 CPU route tests pass.
- A fused top1 kernel does not materialize its full logits. The diagnostic
  records its actual selected token separately and compares it with control
  full-head top1, using the same overall/high-margin thresholds. Identical
  diagnostic logits alone cannot qualify a changed accelerated decision.
  Model numerical, speed, quality and acceptance results remain pending.

- Owner steering on 2026-10-03 supersedes the initial ULP-based thresholds:
  use the no-MTP distribution document/tool as the sole authority. PLE remains
  disk mmap by default, reusing PR #831's bounded mapped result transport.
  The waiting memory-PLE launcher was stopped before it acquired GPUs; no
  foreign process was stopped. Retain exact pinned allocation for optional
  resident tables without making whole-table pinning the default.
- Stop BV16 and four-warp projection work. Prioritize M5 TP4 HC sharding,
  persistent MoE, reduced draft vocabulary/head overlap and adaptive draft
  length. Draft width changes need their own speed/acceptance comparison;
  retain fixed MTP4 complete-round reporting to prevent denominator changes
  from hiding verifier latency. Extract the HC/argmax admission fixes for
  the separately authorized main merge while the structural campaign stays
  draft. PR #831 remains a pending dependency, not a claim of MTP admission.

- Initial shared numerical-tool checks pass 17 cases. Disk default/late IPC
  initialization passes its focused test. Mapped publication/PLE lifecycle
  checks pass 39 cases with one skip; one distributed-initialization test
  fails only under an intentionally empty CUDA-visible GPU set (world-size
  validation), and is retained rather than counted as a pass. M5/M10 delayed
  producer graph cases are added; their GPU execution remains pending.
