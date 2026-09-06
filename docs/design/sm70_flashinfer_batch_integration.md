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

### Device-planned MQA implementation (2026-09-06, not quality-admitted)

Upstream head was rechecked once and remains
`6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f`. The SM70 adaptation follows the
official attention-score scheduler's live-tile prefix scan and contiguous
balanced worker assignment; it does not compile the SM100 kernel for Volta.
The new scorer uses FP16 WMMA with FP32 accumulation, 64-column tiles,
vectorized eight-half loads and padded shared leading dimensions. A one-warp
device planner runs on every invocation, including graph replay. Caller-owned
schedule/output buffers retain stable graph addresses. No host length readback
or capacity-sized empty CTA grid is required.

The native implementation supports int32 and int64 positions. Initial runtime
admission is the measured H4/D128, R4--16 geometry, independent of model name,
TP degree and configured maximum batch. Other shapes fall back locally.
The parent `VLLM_SM70_FLASHINFER_BATCH` remains **default off**. Runtime imports
the packaged `_sm70_flashinfer_C` fragment, not a development JIT library.
CMake, setup.py, manifest and source/license notice now include that fragment.
This is not yet a full built-and-installed wheel acceptance test.

Formal CMake component artifact:
`.artifacts/wheel-native-stage/vllm/_sm70_flashinfer_C.abi3.so`, SHA256
`1b2a2336d6dc22d207111008ed8c83cc97eac67f6ff0917dcb7e1ad456634edf`.
It passes **31 GPU tests** (`mqa-wheel-tests-v1.log`): FP64 oracle, both position
dtypes including overflow boundaries, strided layouts, graph length changes,
empty/padded rows, invalid pages, two-stream separate workspaces, and an audit
that counts every logical tile exactly once. CPU routing/compiled-boundary
tests pass **29/29** (`fi-cpu-v2.log`).

The five alternating A/B blocks in `mqa-selector-v1.json` include the existing
exact top-k and page/index expansion, but **exclude final sparse attention and
the model**. Inputs are synthetic at real indexer geometry; this is not a
captured-activation or end-to-end result. Eight changing graph replays per case
produce identical selected indices.

| Rows | Context | Existing index chain, us | New index chain, us | Reduction |
| --- | --- | ---: | ---: | ---: |
| 4 | 8K | 49.17248 | 40.56064 | 17.51% |
| 8 | 8K | 61.15328 | 42.27072 | 30.88% |
| 16 | 8K | 92.03712 | 53.99552 | 41.33% |
| 4 | 64K | 161.86369 | 124.08832 | 23.34% |
| 8 | 64K | 216.81152 | 149.88288 | 30.87% |
| 16 | 64K | 367.85152 | 237.88544 | 35.33% |

Within-run paired log-ratio 95% intervals have positive lower bounds for these
six cases; they are not cross-run or model-quality confidence intervals.
`sm70_paired_stats.py` requires five paired blocks and uses Student-t(df=4).
Its seven CPU tests pass. Do not multiply layer service savings and report
the sum as measured end-to-end gain.

Preserved rejected attempts: v1 had an incorrect lexicographic worker-end
condition, redundantly executing later rows while writing the same values;
numeric equality alone missed it. The tile-visit audit now detects this.
Corrected scalar/shared-unskewed v2 remained slower; vector loads and shared
padding were necessary. Logs `mqa-screen-v1.*`, `mqa-screen-v2.*` and
`mqa-native-trace-v2.nsys-rep` retain the failures. NCU counters were denied
(`ERR_NVGPUCTRPERM`); no counter-based occupancy or bandwidth claim is made.

GDN's separate BA/conv prepare and state-update prototype preserves all outputs,
BA partials, conv and recurrent states exactly across 256 changing-history
steps at R1/4/8/16/32/64 (`gdn-phases-v1.*`). Its first timing run accidentally
retained a padded last row; those timings are invalid for full-width claims.
The benchmark now restores all live rows before timing; rerun is pending.
No two-phase GDN runtime route has been admitted. The HC 96-module full-chain
benchmark has been extended to B4/8/16 and five alternating blocks; GPU
validation is pending. MoE device compaction and TP overlap remain unimplemented.

### First-prefill boundary localization (control only)

Artifacts: `.artifacts/run-prefill-boundary.py`, corresponding shell/bootstrap,
`prefill-boundary-v2.log`, and `prefill-boundary-v1/boundary-rank*-step*.pt`.
One control-only engine, no FlashInfer experiment, executes two cold passes
over the same 16 teacher prompts. Maximum output is one forced reference token
for diagnosis only, not the registered quality battery or a performance run.

On **all four ranks**, step 0 versus step 4 has bitwise-identical input IDs,
positions, query boundaries, request seeds and PLE ngram context. All six
requests are cold prefills. Layer 0 and layer 1 GDN projection inputs, conv
outputs, zero initial states and recurrent outputs are all bitwise identical.
Physical state IDs differ as expected after fresh allocation. Nevertheless,
the final hidden tensor [2048,2560] differs on 4,826,852 elements, max absolute
delta 31.966796875 and relative L2 0.2170600146; first-token logprob variation
also reproduces. This is **not** evidence of quality safety or a proven cause.
It localizes the first divergence downstream of the captured early GDN work;
inspect the next layers, first QSA/PLE, HC and MoE boundaries next. Do not blame
the new native scorer, which was disabled throughout this run.

All diagnostic model workers shut down normally; no remote service changed.
The adverse BFCL results remain unresolved and the new whole-model performance,
full wheel, expanded quality battery and default promotion are still pending.

### Root-cause closure and reuse of existing PR #494

The follow-up capture `prefill-boundary-later-v1/` (log
`prefill-boundary-later-v2.log`) localizes the first difference to layer 3
QSA on all four ranks. Its input hidden states, Q/K/V, gate, positions,
selected token IDs, and effective logical K/V read back after the cache update
are bitwise identical. The first QSA output alone differs (rank 0 relative
L2 `9.7149867e-5`, max absolute `0.00048828125`), followed by progressively
larger downstream differences; final hidden relative L2 is `0.22358379`.
CPU FP64 causal attention on the captured cold-prefix keys confirms both
outputs have small local numerical error, not corrupted K/V. This local
oracle does not establish model-quality noninferiority.

An isolated diagnostic alternative used logical request/page hash identities
and ordered collision resolution. Twelve relocations of the *same real Q/K/V*
changed 162,113--303,033 output elements in the old planner, versus zero in the
diagnostic alternative. The latter preserves grouped attention and FP16/FP32
types but costs about 451 vs 434 us for captured 2048-row
planner+attention+gate; this is a stability repair, not a speed win.
Six planner tests plus 31 MQA tests pass (37 total); its six planner cases pass
memcheck with zero errors. The first sanitizer attempt lacked an injection
library, so it is not counted. Successful log: `canonical-memcheck-v2.log`.
Proof binary: `.artifacts/mqa-plus-canonical-proof/vllm/_sm70_flashinfer_C.abi3.so`,
SHA256 `23a202e5f4d7c5a9bccb73e4eba577b85fe090ead5e672ac039a1223adc521c1`.

Changing only that planner in one additional cold-prefill engine eliminates
the observed A/A instability: all four ranks, all four scheduler-step pairs,
and their final hidden tensors match bitwise. All 16 first-token logprobs match
exactly (previous diagnostic mean absolute delta 0.60679578, maximum 1.85863316).
Artifacts: `prefill-boundary-canonical-v1/`, its launch log and
`prefill-boundary-canonical-compare-r*.log`. This is a causal diagnostic for
the allocation-order defect, **not a new task score or a validation of #494's
binary**.

The subsequent overlap review found existing open **#494**, reviewed source
`5fa8a605dab12cc9ee15459d9ac6b88d95c7be3a`, already fixes this same defect.
It additionally preserves cross-request physical-page deduplication and fixes
the separate XQA tail. Reuse that reviewed patch rather than publish a competing
implementation. The new alternative planner, binding and tests were removed
from build/runtime/source delivery and retained only in
`.artifacts/canonical-prototype-source/`. Its results above are independent
NVFP4 root-cause evidence, not a claim of authorship of the existing repair.
The frozen performance binary did not include #494. Integration/build/testing
of #494 is the next dependency step; do not mix its forthcoming results with
the retired alternative's proof.

The corrected full-width GDN phase screen is complete on GPU 0
(`gdn-phases-v2.json`). All 256 history steps and output/conv/state/BA partials
remain exact at B1/4/8/16/32/64. Splitting the phases regresses B4/8/16 by
9.43%/15.80%/3.81%; keep it out of their runtime. B1/B32/B64 improvements are
4.22%/0.51%/3.91%, not admissions or justification to replace the existing M1
route. No new end-to-end score has been measured. CPU routing, statistics and
compiled-boundary tests now pass 36/36 (`fi-cpu-v4.log`).

### Adopted #494 and reduced its stable-plan overhead

The reviewed dependency is cherry-picked as `3e0f7a40c1`, retaining the
original author and sign-offs. Its rebuilt Flash-V100 library passes 78
QSA/MQA tests in this integration (`pr494-gates-v1.log`), including the 31
MQA tests above. These are overlapping suites, not 78 additional MQA tests.

Real layer-3 NVFP4 Q/K/V replay also confirms #494 allocation invariance.
Unlike the retired diagnostic prototype, its fixed 8192-entry radix sort
costs 653.26 us for captured 2048-row planner + attention + output gate,
versus 433.00 us in the unsafe frozen reference. Neither timing includes
the QSA indexer or the complete model. Artifact: `pr494-real-replay-v1.json`.
Reference #494 library SHA256:
`a99cce1f5fe32d61ef42525435401c4d24ddff53894eafe64cc534efa937ea23`.

The follow-up retains #494's physical-page union, minimum logical owner,
category padding and exact sorted order. After loading the hash entries into
registers, a device block scan compacts valid entries, then chooses a
512/1024/2048/4096/8192-entry sort. All shared memory is reused within the
original 96-KiB budget; no host length readback or captured pointer changes
are introduced. The planner remains 128 registers/thread with zero local
spills according to `cuobjdump` (static resource data, not measured occupancy).

Current adaptive binary SHA256:
`1c6fc18851e551950885c48ba0d108be919efcb6f4e0f6ca27e69ecb6e8fcd33`.
Validation:

- 80 tests pass, including new empty-to-maximum live-union transitions across
  every sort-size boundary inside one captured graph and in reverse order.
- Twelve physical relocations of the captured real input are bitwise exact
  against #494's fixed-sort output, including gate materialization.
- Five alternating A/B blocks: **653.21 -> 458.50 us**, 29.81% reduction,
  within-run paired 95% interval **[29.76%, 29.85%]**. This recovers most of
  the stability-fix overhead; it is not an endpoint speed claim and remains
  slower than the allocator-dependent unsafe reference.
- Artifacts: `pr494-adaptive-gates-v1.log`,
  `pr494-adaptive-real-replay-v1.{json,log}` and both build logs/libraries.
  The two changing-sort-size graph cases also pass Compute Sanitizer memcheck
  with zero errors (`pr494-adaptive-memcheck-v1.log`); wider prefill performance
  and racecheck remain pending.

### GDN shared-parameter screen: reject for C8/C16

An isolated compile-time variant computes the unchanged gate reduction and
Q/K normalization once per CTA, sharing the results among its eight warps.
It preserves all FP16 round trips and FP32 state and introduces no global
workspace. The default kernel does not select this variant.

All B1/2/4/8/16/32/64 cases preserve output, conv, recurrent state and BA
partials exactly over 256 changing-input graph steps, including slot
permutation and negative padding. Full-width timing is restored after the
padding history. Actual checkpoint weights, synthetic hidden inputs:

| Rows | Frozen cooperative us | Shared-parameter us | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 9.820 | 9.779 | 0.42% |
| 2 | 10.424 | 10.557 | -1.28% |
| 4 | 12.923 | 12.728 | 1.51% |
| 8 | 22.180 | 22.477 | -1.34% |
| 16 | 45.926 | 46.326 | -0.87% |
| 32 | 84.306 | 85.627 | -1.57% |
| 64 | 163.308 | 165.827 | -1.54% |

The extra CTA barriers/shared accesses erase the eliminated computation in
the target batch range. This is consistent with the measurement, not an NCU
stall attribution. Register count is 123 versus 122, no spills, 1036 extra
shared bytes. Do not enable at C8/C16 or use the tiny M1 difference to replace
the established M1 route. Artifact: `gdn-shared-v1.{json,log}`. The separate
split-phase experiment also remains rejected; neither is a runtime default.

### HC gate and repaired quality control

The real 96-module HC graph screen stops at B4's numerical gate; no B4/8/16
speed result is admitted. Four-rank artifacts localize the first over-envelope
output to **HC module 9, normalized input (output index 37)**. The same two
elements exceed `atol=rtol=3e-3` on every rank. The failure itself is retained
and the tolerance has not been relaxed. Artifacts:
`hc-full-b4-v1.log`, `hc-full-b4-v2.log`, and
`hc-full-b4-v2.rank{0,1,2,3}.failure.pt`. A communication-free arithmetic
replay is prepared to separate GEMM rounding propagation from IPC errors.
This numerical rejection alone does not quantify task-score regression.

The expanded quality set was frozen before observing any expanded results:
`expanded-quality-preregistered-v1.json`, SHA256
`72ad0a68931252ade99ac4b4ed042c8de39a33a7f175c6e085c1850efdd8903d`.
It retains the original 80 cases, registers BFCL four categories x 128,
64 schema cases, GSM8K 128, full HumanEval 164, and three fixed seed bases.
HTTP/SSE, long multi-turn fixtures and the isolated coding evaluator remain
pending. It is a registration artifact, not completed quality evidence.

A repaired 80-case control now uses the same original prompt IDs, natural
EOS and 16K limit, with **batch HC and grouped MoE disabled** as well as the
FlashInfer parent switch. Both arms use the same adaptive #494 library.
This removes experimental components from the quality truth; do not compare
its speed to the frozen performance control. On GPU 0--3 the control scores
51/64 BFCL (15/11/14/11 by category) and 16/16 schema, zero truncations.
Candidate testing is pending; prior negative BFCL results remain on record.
Artifacts: `repaired-quality80-v1/control/`, its log and
`run-repaired-quality80.{py,sh}`. The model's normal shutdown retains one
resource-tracker shared-memory cleanup warning; this is not a leak-free
runtime admission. Model workers exited, remote services remain untouched.

### HC arithmetic attribution and compact MoE rejection

The communication-free HC replay now **exactly reproduces both tensors** in
the saved four-rank failure, using only the same sharded versus replicated
GEMMs and existing pointwise operations. Common-input probes first differ at
HC module 0's down projection: 696/1296 FP16 values differ, relative L2
0.00038058. The independent chains then diverge through injection and
normalization, reaching the recorded gate failure at module 9. This isolates
that failure to projection-arithmetic association, not IPC visibility or
stale buffers. It does not rule out unrelated communication bugs or prove
model-quality noninferiority. Artifacts: `localize-hc-arithmetic.py`,
`hc-arithmetic-prefix-v1.{json,log}`. Do not relax the failed gate.

MoE work consumption reused the existing device group table and native
W13/activation/W2/reduction math. On actual checkpoint layer-0/rank-0 weights
and captured routes, full-chain microseconds were:

| Rows / groups | Existing | Compact W13+W2 v1 |
| --- | ---: | ---: |
| 4 / 40 | 65.338 | 76.832 |
| 8 / 71 | 97.446 | 140.448 |
| 16 / 99 | 129.696 | 164.890 |

The first W2 mapping also colocated adjacent N tiles, confounding compaction
with the previously tested locality idea. A second screen restores the exact
original four-expert/common-N-tile CTA mapping and tests **W2 only**, without
rerunning the rejected W13 variants. It still loses: captured C16 W2
**43.302 -> 53.523 us**, complete MoE **131.411 -> 141.158 us**, paired
within-run full-chain reduction interval **[-7.64%, -7.35%]**. All seven
changing-route graph cases preserve W13 and final output exactly, including
duplicate, invalid and empty-expert work. This rejects these fixed-grid loops,
not device planning in general; an empty-CTA count alone is insufficient
evidence of a net win. Artifacts: `moe-compact-w4-v{1,2}.{json,log}` and
the build logs/binaries. Test data are real weights/routes with synthetic
activations, not a model-quality result.

The rejected scheduling code is retained **only as a benchmark source** in
`benchmarks/csrc/sm70_moe_compact_tasks.cu`. The existing production
`nvfp4_grouped_decode_sm70.cu` was restored unchanged. No CMake or serving
dispatch selects the benchmark namespace. The pre-retirement source delta is
retained in `moe-compact-before-retirement.patch` for exact binary provenance.

### Repaired original-80 combined result: still not admitted

The candidate runs on the subsequently freed GPU 4--7; the control used GPU
0--3, same V100 host, weights, sampling, prompt IDs, cache/state contract and
fixed cohorts. This is a quality localization run, **not a same-GPU speed
comparison**. All four candidate workers select GDN, MQA and sparse QSA;
batch HC and grouped MoE remain disabled in both arms.

| Original subset | Production control | Combined candidate |
| --- | ---: | ---: |
| BFCL simple | 15/16 | 15/16 |
| BFCL parallel | 11/16 | 10/16 |
| BFCL multiple | 14/16 | 14/16 |
| BFCL irrelevance | 11/16 | 11/16 |
| JSON Schema | 16/16 | 16/16 |

No truncations; all 80 first tokens match. Twelve complete outputs differ.
Only `parallel_1` changes pass/fail: the candidate emits one valid tool call
instead of the required two. Total BFCL is **51/64 -> 50/64**. The paired
score delta is -1.5625 percentage points; a conservative 95% interval from
simultaneous exact-binomial improvement/regression bounds is
**[-9.5612, +6.5981] pp**. The interval is reported, not used to dismiss the
negative result or declare noninferiority. This original-80 offline screen
is not the registered expanded or HTTP/SSE suite.

Artifacts: `repaired-quality80-v1/{control,candidate}/`, both logs,
`comparison-v1.{json,log}`, and `compare-repaired-quality80.py`. The original
adverse results remain intact. A scorer-only ablation uses the same fixed
80 cases and no new seed selection to distinguish MQA from GDN/sparse-QSA
effects. Full wheel, extended quality and endpoint targets remain unmet;
keep #523 Draft and `VLLM_SM70_FLASHINFER_BATCH=0` by default.

### Original-80 ablations and packaged GDN follow-up

The same repaired control and original cases now have three additional
ablations. No seed, prompt, output limit, or failed case was changed:

| Arm | BFCL / 64 | Schema / 16 | New BFCL failures vs control |
| --- | ---: | ---: | --- |
| Production control | 51 | 16 | n/a |
| Device-planned MQA only | 51 | 16 | none |
| GDN only, prototype library | 52 | 16 | none |
| MQA + wheel-component GDN, no native sparse QSA | 52 | 16 | none |
| MQA + GDN + native sparse QSA | 50 | 16 | `parallel_1` |

MQA alone preserves **all 80 complete output token lists**, not just scores.
GDN-only and MQA+GDN improve `irrelevance_4` and have no newly failing cases;
their 80 complete output token lists also match each other exactly.
Their paired BFCL difference is +1.5625 pp, conservative 95% interval
**[-6.5981, +9.5612] pp**. All arms have zero truncations and identical first
tokens. These small offline ablations narrow the adverse signal to native
sparse QSA and its interaction; they do not admit GDN, prove broad
noninferiority, or substitute for registered multi-seed/HTTP/SSE tests.
Artifacts: `repaired-quality80-v1/{mqa,gdn,mqa_gdn}/` and
`comparison-v3.{json,log}`. Earlier comparisons remain intact.

The sparse arithmetic audit identifies a concrete contract difference to
investigate next. Production Triton attention rounds softmax probabilities
to FP16 before its PV dot while retaining an FP32 denominator; the pinned
FlashInfer decoder accumulates FP32 probabilities against converted values.
Its tile/split reduction order also differs. This is a source-level lead,
**not causal proof that one cast caused the missing tool call**. Blindly
adding that cast does not reproduce the other reduction boundaries; a more
accurate FP64 operator comparison does not clear the observed model failure.

GDN now has a formal `_sm70_flashinfer_gdn_C` CMake/setup component, with
separate C++ namespaces for H2560/Q4/V12, Q8/V24 and Q16/V48. The benchmark
and package reuse one binding/header rather than diverging implementations.
The loader resolves the installed component before capture; explicit
preloaded prototypes prevent duplicate registration, and missing geometry
falls back locally. No external path or worker-side compilation is required
for this component. The source attribution now names the exact upstream
experimental GDN path. Other shapes remain unsupported, not globally gated
by TP degree or model name.

The staged CMake component was loaded through normal package discovery in
the MQA+GDN model arm above, without the external GDN-library override.
Its SHA256 is
`4003e0393021456924606ebfe93eb62f4d1dfbbfabedbccb373d859b170295c1`.
Seven GPU graph tests pass for B1/2/4/8/16/32/64: all three head partitions
give bitwise identical outputs, conv states and FP32 recurrent states across
eight changing-input steps, including slot permutation, negative padding,
SD conv layout and non-dense pool strides. This tests operator geometry
isolation, **not distributed TP communication**. Artifact:
`native-gdn-gates-v1.log`; build/install/import provenance is retained in
`native-gdn-*-v1.log` and `native-gdn-stage/`.

The final formatted source rebuild is staged separately in
`native-gdn-stage-v2/`, SHA256
`7cae0a12e1018c44e7a9d5f62a7feec85cd85bbeba92c805d18aa72c2cf2988c`.
It imports all three namespaces without initializing CUDA. All 30,537 lines
of disassembled SASS match the tested v1 component exactly; debug/source
metadata changes the library hash. Artifacts: `native-gdn-build-v2.log`,
`native-gdn-install-v2.log`, `native-gdn-v{1,2}-real.sass`. The initial
disassembler lookup under the build CUDA shim failed because that shim has
no `cuobjdump`; the successful comparison uses `/usr/bin/cuobjdump`, not
the empty outputs of that failed lookup. Do not relabel the v1 model run as
a new v2 wheel-install test.

A reload audit also found that an already prepared layer could retain its
old derived BA weights if a later reload changed to an unsupported dtype,
layout or geometry. Initial unsupported layers still fall back, but changing
an already captured contract now fails explicitly and requires graph
rebuilding; silently leaving `_sm70_fi_ready` with stale weights is forbidden.
The CPU routing/statistics/compiled-boundary suite now passes **42 tests**
(`fi-cpu-v6.log`), including absent/preloaded component and reload guards.

Coverage status for this follow-up:

| Hot chain | Implemented / evidenced | Remaining gate |
| --- | --- | --- |
| QSA scoring/selection | Device work plan; exact production top-k/index chain wins | Wider routing, long-context and expanded quality |
| QSA prefill plan | Reused #494; adaptive stable sort recovers overhead | Wider prefill timing and racecheck |
| Sparse attention | Native prototype reaches actual graph execution | Retained BFCL failure; arithmetic/interaction isolation; packaging |
| GDN | Native component; projection/BA/conv/state boundary; staged model and geometry tests | Expanded quality, full-chain B-shape admission, full wheel |
| HC / TP | Full-chain B4 numerical failure localized to projection association | Preserve arithmetic before further overlap work; multistream/IPC gates |
| MoE | Existing grouping retained; two compact-loop designs measured and rejected | New critical-path evidence before another scheduler change |

The split-phase/shared-parameter GDN variants and compact MoE loops are not
production selections. The full installable wheel, clean-environment
acceptance, expanded quality, long-context capacity and new fixed-contract
end-to-end targets remain unfinished. No 20--30% model-throughput gain or
release default change is claimed. Keep the Draft scope and parent default
off; do not alter the remote deployment.

Publishing preflight: `onecat/main` advanced to
`4366d9d5fe80eeaf79575b51ec36a6a032673df0`. The measured branch is not
silently rebased onto that changing integration line; merge compatibility
and a fresh integration gate remain required before promotion. Applicable
source checks pass (`hot-chain-precommit-v8.log`). All model, benchmark and
waiter processes owned by this follow-up have exited. Subsequent GPU 0--3
and 4--7 allocations belong to other task leases and were not interrupted;
there is no task-owned resident API or new remote service.

### Numerical isolation follow-up: EOS boundary and HC projection stages

The last measured E2E comparison remains C4/C8/C16
**232.169/390.046/630.996 tok/s**, +6.98%/+7.00%/+7.35% versus the original
performance control. It is still not quality-admitted. The newer MQA and
stable-planner savings have **not** been measured in a new E2E campaign.

Inspecting the retained repaired-quality outputs further localizes
`parallel_1`: MQA+GDN and the sparse-QSA combined arm share their first
**51 output tokens** exactly. The first tool call is complete and identical.
At zero-based offset 51, MQA+GDN emits newline token `198`, then a second
`calculate_em_force` call with `d_time=10`; native sparse QSA emits EOS
`248046`. Completion lengths are 105 versus 52 tokens. This is premature
generation termination, **not a parser dropping a generated second call**.
It does not identify which numerical change shifted the EOS draw.

The new CPU counterfactual retains Q/K/V, index order, duplicate/invalid
selections, reference 16-column split boundaries and the output/gate casts.
Everything else is evaluated in FP64, changing only the probability
materialization before PV. On all **36** previously retained QSA inputs,
restoring the FP16 probability cast is closer to the recorded production
output: median relative L2 **1.27622e-4 -> 3.82219e-5**, median per-case error
reduction **70.34%**. This supports that rounding as a substantial source of
the *operator* difference. These inputs came from the earlier shadow run,
not the repaired `parallel_1` EOS step; CPU arithmetic is not an exact CUDA
emulator. Do not interpret the result as permission to blindly round the
native kernel, reduce state precision, or declare model noninferiority.

Reproducible numerical tool and sanity tests:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  benchmarks/kernels/benchmark_sm70_qsa_rounding_isolation.py \
  --captures .artifacts/quality-shadow-v2 \
  --out .artifacts/qsa-rounding-isolation-v1.json
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/kernels/test_sm70_qsa_rounding_isolation.py \
  flashinfer-sm70/tests/test_batch_routing.py \
  flashinfer-sm70/tests/test_compiled_gdn_boundary.py \
  flashinfer-sm70/tests/test_paired_stats.py
```

The combined CPU suite passes **45 tests** (`arithmetic-routing-cpu-v1.log`),
including three new tests for duplicate weighting, invalid/NaN padding and
keeping the denominator unrounded. The counterfactual artifact is
`qsa-rounding-isolation-v1.{json,log}`, JSON SHA256
`05d6b8e1704feaf39503cbbfdbb52bcef673098719d428ffd46679c2546d1db9`.

For HC, `benchmark_sm70_hc_arithmetic_isolation.py` now separates full
sharding, down-only sharding, and up-only sharding across the 96-module
arithmetic chain on one GPU. It must first reproduce the saved four-rank
failure exactly and retains the same 3e-3 absolute/relative envelope.
It does not measure collective performance or replace the distributed/full
model gate. Retaining the reference down projection while sharding only up
is a test hypothesis, **not an implemented or validated serving repair**.

A focused, diagnostic-only teacher manifest also freezes the original
16-case parallel cohort through offset 51, using the existing per-case seeds
and trajectories. It will compare raw/processed logits and the stateless
EOS-versus-newline draw on all four ranks, with A/A repeats, before another
natural quality run. Manifest `parallel-eos-teacher-manifest-v1.json`, SHA256
`2f179c9940d28bd9a631821bd669926e5b11d5f29423f6f2a47f2092f9be01d2`.
Launchers/hooks stay in task artifacts and require explicit diagnostic
environment variables; they are not serving code. The spawn import check
passes without CUDA initialization. **This model diagnostic has not run.**

The HC launcher waited its ten-minute lease window and returned code 75;
a fresh attempt was also blocked by live foreign GPU processes, even after
one lease owner exited. No HC GPU result file was produced and no model
diagnostic was launched. The waiter has exited; no task-owned GPU process
or resident service remains. Do not preempt other tasks or describe these
queued checks as passed. Resume HC single-GPU isolation and then the focused
four-rank EOS diagnostic after resources are actually free. Source runtime
defaults remain unchanged; fixed quality and E2E admission are still pending.
