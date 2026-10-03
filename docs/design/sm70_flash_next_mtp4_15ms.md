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
  source build is in progress using Torch 2.10.0+cu128 and CUDA 12.8; no wheel
  packaging is requested. All speed/quality admission remains pending.
