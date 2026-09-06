# FlashInfer SM70 batch integration probe

Purpose: combine operator-screen winners, then perform one consolidated
control/candidate model comparison. No release default is changed.

Integration base: `onecat/main` at `755baae1d075ee04fa9096b23fc0225b23589a86`.
This owned branch combines #504 at `5a049230cd` (existing batch HC/projection
and TP baseline), #513 at `17fa36e35a` (native FlashInfer QSA), and #515 at
`9ea54c19f4` (fused gate/conv/GDN prototype). Combination commit `f8e822c335`.
The runtime bridge is new, not a duplicate of those component-only PRs.

## Decisions and contracts

- Retain existing M1, HC projection/TP and HC norm paths. Both FlashInfer HC
  staging variants passed numerical tests but lost to the existing Triton
  component. On GPU 4, FP16 B16 baseline 3.164 us, register variant 3.779 us;
  no HC replacement is admitted.
- `VLLM_SM70_FLASHINFER_BATCH=0` by default. Explicit prebuilt GDN and QSA
  library paths are used only in the probe; no compiler or JIT during capture.
- GDN preserves FP16 activations/projections and FP32 recurrent state. Fuse at
  the existing opaque input/core boundary to remove, not duplicate, the BA
  GEMM. Ordinary QKVZ projection and output norm/projection remain unchanged.
- GDN dispatch uses actual non-spec, uniform-decode metadata and compatible
  tensor geometry (2..64 rows). M1, larger shapes, prefill/mixed/MTP and other
  state/activation layouts fall back. No global TP, scheduler budget or KV
  dtype binding. Initial model validation is no-MTP; spec engines are not armed.
- QSA accepts the proven small-batch FP16 sparse-page family (4..16 rows),
  preserving index order, repeats, invalid-page masking and the FP16 output
  boundary before the existing output gate. E4M3 and other shapes fall back.
- Scratch allocations are call-local, never a cross-stream global mutable
  workspace. The zero page is persistent/read-only. Derived-weight reloads
  copy in place and reject changed geometry rather than replacing graph pointers.
- Do not change selector, expert routing, sampling, state precision or output
  length to manufacture a throughput gain.

## Validation and measurement

The component worklogs retain micro timings and reference/source/binary hashes.
On GPU 4, #515's updated suite passed **20 tests**, including register-HC
cases (66.92 s including rebuild). GDN targeted memcheck passed one B16 case
with **0 errors**; racecheck passed one B4 case with **0 errors/0 warnings**.
These are targeted operator checks, not whole-engine sanitizer coverage.

Initial bridge suite: **21 passed** on GPU 4 (11.67 s), including actual QSA
output gating, GDN input/core state evolution and eager/graph equality. A CPU
no-op initially imported the GDN module before its guard; fixed by guarding
first, so unsupported devices do not load native libraries.

The next consolidated run uses the unchanged 8192-input/256-output fixed-live-
width benchmark, C1/4/8/16, no MTP, NVFP4 FlashNext, TP4, 256K capacity, prefix
cache + Mamba align, FP16 QSA KV, FP32 GDN state. CUDA Graph, GPU 4--7; both
arms use identical baseline flags and NUMA-node-1 CPU affinity. Keep automatic
GPU boost; do not compare absolute timings with prior GPU 0--3 runs.

Use full EngineCore timestamp intervals, not client receive blocking time.
The fixed denominator remains 70 tok/s: C4/C8/C16 targets 238/420/728 tok/s.
The deterministic performance workload remains unchanged and is **not** quality
evidence. Its historical forced-length policy does not apply to quality cases.

Quality: reuse the fixed GSM8K first-16 health screen (temperature 1, top-k20,
top-p.95, natural EOS, max16384), plus the retained 80-case BFCL/schema manifest
as an offline generation/parser screen. Explicitly distinguish offline model
outputs from HTTP/SSE transport testing and official benchmark scores. Report
scores, incomplete outputs and regressions; do not claim broad quality from
a short smoke. No candidate model result has been produced yet.

Artifacts are task-private under this worktree's `.artifacts/`: native build
caches, `bridge-tests-gpu4-v2.log`, `env.sh`, `run-e2e.sh`, `run-model.py` and
`e2e/`. Frozen native vLLM/Flash-V100/HC libraries are pinned reference binaries;
this is not a freshly built release wheel. No shared checkout is modified.

The user freed GPU 4--7. The old **local** `1cat-qwen38-flash-next.service`
was stopped and disabled; remote delivery services were not changed. Stop all
task-owned engine workers after each comparison. Keep this PR Draft until
the model and quality gates pass. AI-assisted (Codex), DCO sign-off and human
review required before merge.

## First model comparison and compiled-boundary correction (September 6)

Draft integration PR: #523. Both arms used the frozen speed runner and
environment above on GPU 4--7. The control reproduced the historical baseline:

| Concurrency | Control tok/s | First candidate tok/s |
|---|---:|---:|
| 1 | 87.750 | 87.864 |
| 4 | 217.026 | 219.474 |
| 8 | 364.521 | 375.581 |
| 16 | 587.789 | 612.823 |

These first candidate numbers are **QSA only**, not combined GDN + QSA.
Although all four ranks prepared 36 GDN layers, no fused GDN route was hit.
The `1 < num_tokens <= 64` test was outside the opaque input/core boundary:
the initial large-prefill trace specialized that branch away. Move all shape
and metadata selection inside the runtime op. Preserve the existing fused
FP16 input projection on fallback, including M1, instead of substituting
separate QKVZ/BA GEMMs. The boundary has an extra Z output copy; its M1 cost
and full-model behavior still need the corrected combined comparison.

Added an export regression using the actual boundary predicate: one large
example must support the full 1..2048 row range. Reinstating the legacy shape
guard demonstrably fails export's shape constraints. The corrected predicate
and original-projection fallback checks pass: **24 CPU tests**, 5.63 seconds.
The first test attempt had a missing CPU-only op stub; fixed the fixture,
not the model implementation. The native operator binaries are unchanged.

Quality is **not admitted**: GSM8K is 15/16 with no truncation in both arms,
but the fixed offline BFCL screen is **52/64 -> 50/64** (irrelevance 13 -> 11).
JSON Schema remains 16/16. Retain all per-case records; a small stochastic
screen cannot establish noninferiority, and this negative signal cannot be
ignored in exchange for C16's 4.3% gain. Do not enable by default.

Artifacts: `.artifacts/e2e/control-*`,
`.artifacts/e2e/qsa-only-190e5f0225/candidate-*`,
`.artifacts/control-e2e-v1.log`, `.artifacts/candidate-e2e-v2.log`,
and `.artifacts/compiled-boundary-cpu-v2.log`.
Candidate attempt v1 exited at the GPU lease gate (75), before a model run;
it produced no performance result. Per-device external campaign leases are
also respected by the launcher now. All first-comparison workers exited;
GPU 4 was subsequently acquired by an unrelated sanitizer job, which is not
terminated by this task.

The next combined attempt at `45711b5e21` passed compilation but failed during
FULL graph capture, before any quality or speed request. AOT compilation
preceded KV allocation, so the input/core op received empty 1-D state
placeholders. The new bridge tried to transpose them. The existing standard
recurrent core already resolves this case from the scheduler-bound layer
cache; reuse that contract inside the new runtime bridge. Explicit nonempty
state inputs must never be replaced. Add both placeholder and explicit-cache
graph/state tests, including a different layer cache to catch accidental
replacement, and guard malformed/unbound states before transpose.

Failure log: `.artifacts/candidate-e2e-v3.log`. It is a startup failure, not a
performance or quality result. Its workers exited. Run the expanded GPU
bridge suite under the same GPU lease before the next model launch.

While waiting for the unrelated MTP job, candidate v4 exited at the lease
gate. Candidate v5's preflight rejected the test launcher's four-visible-GPU
setting (the component suite intentionally requires one visible GPU); fixed
the launcher to expose GPU 4 only to pytest, then 4--7 to the model. Candidate
v6 passed all four GPU bridge cases, including both state-binding cases, but
the default-off test caught a test-order contamination: monkeypatching an
`envs` module attribute restored a materialized `True` attribute, shadowing
dynamic environment lookup. Change the test fixture to patch the environment
variable with the cache disabled instead. Neither v5 nor v6 launched a model.

## Corrected combined result: faster, not admitted

Model source `f45c673898` (runtime fix `2273c24d7d`), log
`.artifacts/candidate-e2e-v7.log`. Expanded preflight: **29 passed**, including
four GPU cases, 8.30 seconds. Then one corrected combined model run on the
same GPU 4--7 and unchanged workload. Worker logs confirm native fused GDN at
B2/4/8/16 and native FlashInfer QSA at B4/8/16. All ranks prepared 36 GDN
layers. FULL graph capture completed; no MTP or state-precision change.

| Concurrency | Control tok/s | GDN + QSA tok/s | Gain | Fixed-70 efficiency | Target |
|---|---:|---:|---:|---:|---:|
| 1 | 87.750 | 88.556 | +0.92% | n/a | n/a |
| 4 | 217.026 | 232.169 | +6.98% | 82.92% | 85% |
| 8 | 364.521 | 390.046 | +7.00% | 69.65% | 75% |
| 16 | 587.789 | 630.996 | +7.35% | 56.34% | 65% |

Complete engine-step means (ms): C4 **18.431 -> 17.229**, C8
**21.947 -> 20.510**, C16 **27.221 -> 25.357**. These are unprofiled complete
decode intervals, not kernel sums, API receive blocking time or MTP rounds.
All three throughput targets remain unmet. This is one campaign, not a
cross-run confidence/stability study. Do not generalize its speed to other
contexts, model quantization, concurrency or serving modes.

| Fixed quality subset | Control | QSA only | GDN + QSA |
|---|---:|---:|---:|
| GSM8K health screen | 15/16 | 15/16 | 15/16 |
| BFCL simple Python | 14/16 | 14/16 | 13/16 |
| BFCL parallel | 11/16 | 11/16 | 11/16 |
| BFCL multiple | 14/16 | 14/16 | 14/16 |
| BFCL irrelevance | 13/16 | 11/16 | 11/16 |
| BFCL total | 52/64 | 50/64 | 49/64 |
| JSON Schema | 16/16 | 16/16 | 16/16 |

GSM8K and tool/schema outputs have no length truncations. Quality remains
**unadmitted**: five BFCL control successes become failures and two failures
become successes in the combined run. Identical prompts are retained. The
two irrelevance regressions recur in both candidates (`irrelevance_7` and
`irrelevance_10`); this is a localization lead, not proof that a specific
kernel or rounding operation is causal. Small stochastic screens cannot
establish either broad degradation or noninferiority. No HTTP/SSE transport,
coding, long-context quality or PPL admission is claimed here.

Decision: keep the integration default **off**, PR #523 **Draft**, existing
HC norm and production source defaults unchanged. Do not count these faster
numbers as accepted production performance. Next isolate the QSA arithmetic
and output-gate drift on retained real trajectories, then GDN independently;
do not rerun until a narrower check can distinguish a cause. Preserve this
manifest and sampling, and do not select favorable seeds to clear the gate.

Final artifacts: `.artifacts/e2e/candidate-{speed,gsm8k,offline-tools}.json`
and the corresponding control files, environment snapshots and GPU metadata.
The E2E FlashQLA reference binary SHA256 is
`c8bd7650444ec56cfe2576c044d8f5f438b0a352877064bbb51ac0510dc2ea2c`;
it is the same pinned library as the frozen HC model baseline. Component
native libraries remain task-private prebuilt probes, not a release wheel.

All task model workers exited normally. Cleanup check: GPU 4--7 each **7 MiB**,
no compute processes; the old local API unit is inactive/disabled (MainPID 0).
Unrelated GPU 0--3 processes and remote services were not touched. Engine
shutdown logs also retain Python resource-tracker shared-memory cleanup
warnings seen in the control; do not mislabel these as persistent GPU usage.

## Quality root-cause follow-up: same-input shadows

Continue the owned #523 scope at `25fd594d9c`; do not duplicate component PRs
or change production precision to fit the small task-score sample. Inspecting
raw responses excludes a parser-only explanation: new failures contain no
call or an actual irrelevant call. However, the QSA-only comparison changes
the **first generated token in 11/80 cases**, including both recurring
irrelevance failures, despite identical prompts. Prefill/admission variation
must be separated from decode arithmetic before assigning causality.

An artifact-only eager shadow run follows the original path for all actual
outputs/state updates. Rank 0 also computes new QSA on identical Q/K/V and
new GDN on cloned indexed initial states. Eight retained prompts and up to
48 output tokens are diagnostic input acquisition, **not quality scores**.
Other ranks do not run shadows/collectives. No production sampler, weights,
state dtype or output-selection policy is changed.

The first attempt stopped at callable-RPC serialization restrictions before
requests. Replaced that diagnostic trigger with recognition of the retained
request seeds; no insecure-serialization flag is enabled. Successful run:
`.artifacts/quality-shadow-v2.log`, with records/captured QSA inputs under
`.artifacts/quality-shadow-v2/`. All workers exited and GPU 4--7 returned to
7 MiB. An unrelated job owned the cards while this task waited for the lease.

Collected 144 GDN and 48 QSA comparisons (B8/B6/B5), plus 50 sampler records:

- GDN Z projection and convolution state: **bitwise equal** in all records.
- Maximum GDN output relative L2: `4.8673613e-5`; maximum recurrent-state
  relative L2: `1.7772339e-5`. Early B8 records alone had state errors around
  `1e-8`; do not report that as the worst of the completed run.
- Maximum QSA candidate-vs-FP64-oracle relative L2: `1.9871883e-5` versus
  reference-vs-oracle `1.8209650e-4`. The candidate is closer to the oracle
  in **all 48** matched comparisons. Preserve ordered/duplicate selections
  and the FP16-output/FP32-gate boundary in the oracle.
- No future or sequence-out-of-bounds selected positions, sampled-logit NaNs,
  or positive infinities in this diagnostic scope.

This does not establish whole-model noninferiority, CUDA Graph long-history
quality, or a causal explanation for the 52 -> 49 task-score result. It does
argue against blindly reducing new QSA precision or blaming GDN state
corruption without further evidence. Next isolate schedule/prefill and
sampling-trajectory confounders using matched conditional probabilities and
an unchanged-data A/A control. Preserve the old adverse scores.

### Fixed-cohort and conditional-probability diagnostic

Artifacts: `.artifacts/run-quality-cohort.py`,
`.artifacts/run-quality-cohort.sh`, `.artifacts/compare-quality-cohort.py`,
and `.artifacts/quality-cohort-v1/`. This retains all original 80 rendered
prompts, per-case seeds, 16K maximum output, natural EOS, schemas, parsers and
scorers. It changes only admission for a diagnostic: five fixed groups of
16, cold prefix reset per group, pause scheduling before enqueue and resume
after all group members are queued. The original continuous-admission
negative screen is NOT replaced by this diagnostic.

Two aborted diagnostic attempts are retained. The first teacher manifest
was corrupted by truncated tool output; validate JSON before constructing
the engine (`teacher-manifest-valid.json` is the valid input). The second
completed the first group but failed in artifact-side output association:
`enqueue()` returns randomized internal IDs whereas `RequestOutput` uses
external IDs. The corrected runner snapshots the engine's own mapping
while scheduling is paused; a CPU fake-engine test covers this association.
Do not disable request-ID randomization or insecure-RPC serialization guards.

Successful control log: `quality-cohort-v1/control-ids-fixed.log`. No new
FlashInfer operators are selected in this control. The two natural-EOS
repeats score BFCL **51/64 and 52/64**, JSON Schema **16/16 both**, with no
length truncations. **22/80 full outputs and 8/80 first tokens differ even
within this unchanged control engine.** This is direct evidence that a
single matched-seed score is not a deterministic attribution test. It does
not establish whether residual variation comes from prefill cohorts after
admission, numerical reduction/order, state lifetime or another mechanism.

The separate teacher probe follows 16 retained historical-control sequences
(2,225 tokens). It is diagnostic conditional NLL, not a task score, PPL
benchmark or speed measurement. All four control ranks report identical
349-step seed/position cohort traces, and CPU/GPU request seeds agree.
25 CPU routing/compiled-boundary tests pass.

Candidate log: `quality-cohort-v1/candidate.log`; comparison:
`quality-cohort-v1/comparison.txt`. Both native FlashInfer routes are observed
in CUDA Graph capture. Natural results (not release admission):

| Path | BFCL repeat 1 | BFCL repeat 2 | JSON Schema both | Truncated |
| --- | ---: | ---: | ---: | ---: |
| Unchanged control | 51/64 | 52/64 | 16/16 | 0 |
| GDN + QSA candidate | 50/64 | 51/64 | 16/16 | 0 |

The candidate A/A changes 23/80 full outputs and 8/80 first tokens. Preserve
these adverse comparisons and the original 52 -> 49 result; repeated tests
are neither independent extra dataset items nor grounds for picking the
best score. The original pair has five newly failed and two improved BFCL
cases; its exact paired two-sided test gives p=0.453125. Lack of significance
is NOT evidence of noninferiority.

The candidate-minus-control conditional NLL mean is `+0.00310631` over 2,225
tokens (positive is worse on the retained reference continuation); maximum
absolute delta is `1.93978739`. For first tokens alone, mean absolute delta
is `0.54163724`, versus `0.00885871` at offsets >=64. The four-rank
seed/position traces match exactly across arms. This points toward the
prefill/first-token portion, but needs a conditional A/A floor before
attributing the change to the new compiled boundary or kernels.

### Conditional control A/A floor and current decision

An additional control-only load runs the exact same 16 continuations twice,
with no intervening natural generation and no new FlashInfer operators.
Runner: `.artifacts/run-quality-teacher-aa.py`; command:
`FI_TEACHER_AA=1 bash .artifacts/run-quality-cohort.sh control`.
Log: `.artifacts/quality-teacher-aa-v1.log`; raw probabilities and four-rank
metadata: `.artifacts/quality-teacher-aa-v1/`. All 2,225 forced tokens match;
all four 698-step traces match, and their 349-step halves match each other.

| Diagnostic | Mean NLL delta, all tokens | First-token mean absolute delta | Maximum absolute delta |
| --- | ---: | ---: | ---: |
| Candidate vs control | +0.00310631 | 0.54163724 | 1.93978739 |
| Control repeat 2 vs repeat 1 | +0.00435804 | 0.80130252 | 3.39355850 |

Thus the unchanged control itself exhibits conditional-probability variation
at least as large in these aggregate measures as the candidate comparison.
This prevents assigning the task-score loss specifically to FlashInfer.
It does NOT prove all variation has the same cause, that state handling is
correct, or that the candidate is quality-equivalent. Investigate shared
prefill/first-token computation and state initialization/reuse first. Next
use a fixed-trajectory prefill boundary/state capture to locate the earliest
divergence; do not randomly change sampling, precision or decode fusion.
The AWQ FP16 atomic weighted epilogue found by source search is not on this
NVFP4 prefill path; do not claim it as the cause without a route-hit.

No production math, kernels, sampler or defaults changed in this follow-up.
The earlier ~7% unprofiled speed gain remains an experimental result for the
same implementation, not a new speed measurement or quality admission.
Keep #523 Draft and the parent switch default-off. All diagnostic workers
exited; GPU 4--7 returned to 7 MiB each, with no compute processes. No remote
API was changed. Loader/shared-memory teardown warnings are retained in the
logs; no persistent GPU allocation remained.
