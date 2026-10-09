# Phase C: execution, policy and state ownership

Phase C starts from B's merged main `ec5f5b7eac702bfc93fa29368c0de16cd5581764`.
The scope is C1–C4 below; DDTree remains deferred. Neither runner selection nor
kernel defaults, qualification, arithmetic, weight layouts or supported shapes
are expanded. Validation uses operators and synthetic state on the authorized
54633 host, without model-weight loading or model-throughput claims.

## C0: parameter and path ledger

[The frozen source ledger](sm70_phase_c_parameters.md) records every detected
parameter consumer, its getter/default/parser, enclosing source function and
native call conditions at the B boundary. Generate the complete machine-readable
inventory (including arguments and function sizes) with:

```bash
.venv/bin/python tools/sm70/path_inventory.py --phase c --ref ec5f5b7eac702bfc93fa29368c0de16cd5581764
.venv/bin/python tools/sm70/path_inventory.py --phase c
```

This extends the B inventory, not the execution selector. Static source entries
are not evidence of a native launch. The table below supplies ownership and
fallback contracts; subsequent deliveries update their actual implementation
and evidence here instead of silently dropping old paths from the ledger.

| Family / entry and flow | Eligibility and parameters | Resources / fallback / destination |
|---|---|---|
| V1 target execution → sampling; V2 execute → sample; base proposer → draft | CUDA, existing speculative-method gates; V2 last PP stage and real draft rows only. Profile defaults off, interval 16 clamped to >=1. | Each consumer owns its totals; step context carries events across the existing boundary. Disabled contexts create no events. Common `StepProfiler`; original log labels and PP reporting rank retained. |
| Worker profile/capture → V1/V2 auxiliary warmup | Auxiliary default on and capability exactly 70. MTP concurrency default off. V1 zero → slot mapping → Mamba copy → aligned postprocess → proposer → deferred tree sampler → model conv; V2 bound conv → zero → speculator. | Runner/model/proposer keep resources; an ordered task executor owns sequencing only. Original best-effort task exceptions and allocator restoration retained; unsupported tasks leave normal runtime paths intact. |
| V1 input preparation → staged H2D → execute | Existing async, CUDA70, one request/token, previous sample, no speculative/encoder work and no accepted-count event. Current explicit staged-input default off. | C1b transfer owner retains pinned-source lifetime, prepare-event ordering and exception restoration. Ineligible steps use current ordinary copies. No V2 eligibility expansion. |
| V2 sample → specialized verifier → proposal | Existing DFlash2 qualification, draft rows, grammar/LoRA restrictions and sparse candidate support. Ordinary DFlash remains V2; DDTree remains V1. | C1b extends existing speculator contract: handled sample, cached dense logits, or ordinary fallback. Preserve RNG and avoid recomputing fallback logits. |
| GDN layer preparation → projections → conv → recurrence → norm/output | Existing backend request, device, head dimensions, dtype, layout and prefill/decode/verify shapes. Auto/explicit FlashQLA fallback rules remain unchanged, including SM75 and non-FP16 rejection. | C2a extends current selector with a stage plan; SM70 bindings live under FLA ops. Preserve fused-stage coverage and FP32/FP16 boundaries. Triton/other current backend remains the fallback; no forced common arithmetic. |
| GDN metadata builder → hybrid model state → state commit | Current ordinary MTP/DFlash request classification and accepted-token mapping. Non-decode draft-count sentinel is -1, not 0. | C2b reuses `GDNSpecDecodeStateContract`; builders own metadata buffers, model state owns conv/SSM movement. Capture and replay retain their respective preparation order and addresses. DDTree state remains a documented exception. |
| VllmConfig/speculative init → model qualification → platform defaults/validation | Existing explicit options, legacy aliases, model qualification, device and TP conditions. Early model verification and later platform checks retain their order. | C3 uses current model config map/platform hooks. Move each rule with its consumers, replace environment writes with engine-local values; preserve safety overrides and error/fallback behavior. |
| Shared embedding / LM head → packed candidate or ordinary logits | Current sharing relationship, weight layout, dtype, shape, TP and model qualification; existing packed/top-k opt-ins and defaults. | C4a model adapter owns sharing/qualification, provider owns packed preparation and compute. Preserve sharding, weight identity, gather, ties and FP32 logits; ordinary logits remain fallback. |
| Norm / residual / linear → selected provider | Current kernel selection, shape/dtype, graph state, Gemma weight and residual semantics. | C4a retains B linear selector and IR/CustomOp registration. Preserve cast and accumulation order, fake schemas and registration order. No duplicated weights or replacement algorithm. |
| CUDAGraph dispatcher → bucket/key → capture/replay | Current model/platform qualification, context buckets and partition rules. | C4b consumes prepared graph policy; dispatcher owns capture/replay and required key dimensions. Unsupported graphs retain current eager/piecewise fallback. |
| Communicator → collective selection → native op; HC composition | Current peer topology, TP, shape/dtype and native availability. TP8 hardware evidence is not implied by TP4 tests. | C4b communicator retains IPC registration, buffers and destruction; provider owns capabilities, model adapter owns HC composition. Existing collective fallback retained. Fusion pass consumes capabilities. |
| Scheduler and DDTree consumers | Existing generic Mamba alignment/retention rules stay in scheduler. Tree scheduling/verification is deferred by explicit scope. | No new scheduler framework. Deferred paths remain reachable and separately listed, not presented as eliminated debt. |

All parameter values are resolved at their owner's initialization. Typed values
win over compatibility input. Existing legacy-to-legacy precedence remains
binding: explicit `VLLM_SM70_DEBUG` overrides the old MTP profile flag, including
an empty channel list. Original integer parsing, errors and interval clamping
are preserved. Different engines do not communicate configuration through
process-environment writes. Computation-affecting policy belongs in graph hashes;
diagnostics and warmup-only selection do not.

## Deliveries

| Batch | Implementation boundary | Status |
|---|---|---|
| C1a | C0 ledger; three profilers share collection/aggregation; ordered warmup; typed runtime and diagnostic configuration | Merged as #1130 (`c31d98d333d90`) |
| C1b | Staged input resources and ordinary speculative sampling boundary | Merged as #1131 (`dfaeb1ba19ca98`) |
| C2a | GDN compute plan, providers and shared stages | Merged as #1133 (`ec535b1e69b6`) |
| C2b | GDN metadata and state preparation/commit | Merged as #1135 (`ad19d7ad1166`) |
| C3 | Ordered defaults, model qualification and engine-local effective values | Merged as #1137 (`51c0f97201a7`) |
| C4a | Shared embedding, LM-head, norm and linear providers | Merged as #1139 (`c73eb2d1d100`) |
| C4b | Graph/communication/fusion boundaries and final explanation report | Pending |

Each PR is based on merged main and is reviewed and merged before the next
PR is published. Existing experiments remain available; default-off alone does not imply
deprecation. Compatibility modules forward to the single owning implementation.
Generic executors neither call back into the old format/model module nor accept
an entire runner as an untyped execution context.

## C1a implementation and acceptance

`KernelConfig.sm70_runtime` resolves auxiliary/concurrency warmup once.
`ObservabilityConfig.step_profiler` resolves enabled/interval once. Both retain
parameter-source information; diagnostics and warmup-only fields are excluded
from compiled-model hashes. Their initialized values survive worker
serialization without rereading the environment.

The three independent event collectors and aggregate/report state machines now
use one `StepProfiler` with immutable report layouts. In-flight contexts remain
with their original execute/sample/propose owner, preventing cross-runner state.
The two auxiliary warmup loops use one ordered task executor. Resource-dependent
alignment initialization is itself an ordered task, not performed while the
plan is assembled. The proposer kernels and V1/V2 cache layouts remain distinct.

Tests cover report ordering and interval totals, PP report rank, disabled event
allocation, configuration precedence/hash/serialization/isolation, warmup order
and failure boundaries, allocator restoration, and CUDA graph replay with changed
inputs. Existing GDN/Mamba state and MTP safety tests are run as baseline/candidate
pairs. Results and source/artifact identities will be appended after validation.

Completion is measured by independent flows, repeated stages, policy read sites
and resource owners as well as coupling counts. New config type references are
reported honestly; no layering whitelist is expanded. C is not complete until
all seven deliveries have merged with evidence. Whole-repository D/E and DDTree
remain subsequent work.

### C1a validation record

Against the frozen B source: the 54633 GPU suite passed **205 tests**; the final
collector follow-up passed **10 tests**. Local CPU coverage passed **36 tests**
with one CUDA-only skip. The baseline's two failures were stale `__new__` fixtures
missing device/speculator initialization; the same repaired fixture file passed
all three tests against unchanged baseline source. No correctness tolerance or
production gate was relaxed to obtain these results.

Both sides used Python 3.12, Torch 2.10.0+cu128, CUDA 12.8 and a
V100-SXM2-32GB (driver 580.173.02), with device 0 visible. Native code was unchanged;
the installed Phase B `_C` and `_moe_C` artifacts had SHA256
`70cfdcc88eb7b8f2c86db0b78b93c59852f55c7911a8c8371d89bdcb0c624881` and
`4c671824b05e69741be29c39d70c0285062facc4ed9b90d0efba2536a23c7fe3`.
No model weights were loaded. Replay coverage used changed synthetic inputs;
Mamba tests also checked that warmup preserves live state.

Cumulative/interval log text and totals matched the actual baseline reporting
functions for all three layouts (six calls each, including repeated stage names
and alternating speculative steps). Nine runs of 20,000 disabled five-stage
synthetic CPU steps measured these medians:

| Consumer | Baseline µs | Shared collector µs |
|---|---:|---:|
| V1 runner | 1.001 | 0.886 |
| V2 runner | 0.960 | 0.888 |
| Proposer | 1.101 | 0.888 |

These are profiling-glue costs, not GPU or model throughput. A first V2 wrapper
added 0.75 µs; it was replaced by direct aliases to the common collector before
promotion. GPU-enabled tests also exposed cached legacy getter values leaking
between newly constructed configs; initialization now calls the existing raw
registry parser, while already constructed configurations stay frozen. Unified
debug/legacy conflicts and serialization are covered explicitly.

Independent profiling state machines: **3 → 1**. Auxiliary task executors:
**2 → 1**, retaining their separate ordered task lists. Source-inventory parameter
read sites: **381 → 370**. Initialized alias declarations preserve visibility of
migrated controls; the unified debug alias is now visible too. Repository-wide
coupling: platform **3880 → 3756**, model **2325 → 2322**, raw environment **271 →
271**. This includes the new typed-config reference and report-layout owner;
the layering ownership whitelist is unchanged.

Validation commands use the affected runner/MTP safety, GDN metadata, Mamba state
and new `test_step_profiler.py` suites under `tests/v1/`, followed by the normal
pre-commit checks (including mypy and layering). The C0 ledger and per-batch
status remain the handoff for C1b–C4b; none of those batches are claimed complete.

## C1b implementation and acceptance

The V1 runner binds `InputTransferSession` once through the platform hook. The
session owns preparation mode, event and trace counters; each `CpuGpuBuffer`
owns exactly one `StagedCopyOwner`. This preserves the original per-buffer
64-source bound, pinned-source lifetime and copy-stream ordering. It does not
expand the opt-in, single-request/token, non-speculative CUDA70 gate. The old
prepare-event property and staged-copy methods forward to these sole owners.
Synchronization or event-record failures now restore preparation mode; if a
copy may already be in flight when recording fails, the owner retains its
source until the stream confirms completion. No buffer/event owner is global.

`BaseSpeculator` now exposes target preparation, optional sampling and target
trace hooks. The V2 runner consumes three explicit outcomes: `SamplerOutput`,
`ComputedTargetLogits`, or `None`. Completed logits can themselves be `None` on
non-gather ranks: this still means projection has already run. The DFlash2
implementation owns sparse verification; its old fallback import aliases the
same outcome type. Grammar application, rejection sampling and accepted/rejected
counts retain their order. Ordinary DFlash/Eagle use the default fallback;
runner-generation selection is unchanged. Deferred tree tracing stays at its
existing V1 boundary.

`KernelConfig.sm70_runtime.staged_input`, `ObservabilityConfig.runtime_trace`
and `ObservabilityConfig.spec_decode_trace` capture legacy compatibility input
once, with explicit typed values taking precedence. Unified `VLLM_SM70_DEBUG`
retains priority over the event-trace alias. The target trace preserves the
legacy exact-string `1` parser and ignores malformed thresholds when disabled.
These host-transfer/diagnostic settings do not alter model graph hashes. Event
trace counters are per engine; existing unmigrated standalone trace functions
retain their compatibility entry points.

Local affected CPU coverage: **85 passed, 2 CUDA skips**. On 54633, the unchanged
C1a baseline passed **29 tests**; the candidate passed **77 tests**, including
source mutation immediately after staged enqueue, changed-input graph replay,
exact synthetic state, both initialization orders with cached legacy getters,
event/copy failure ownership, three sampling outcomes, and cached-logit reuse.
The existing sparse/logit tests retain their original arithmetic tolerances;
new transfer/replay tests require exact tensor equality. No weights were loaded.

C1b uses the same Python/Torch/CUDA/driver/native artifacts as C1a. GPU 0 was
reserved and checked idle before running the pair. Neither source nor tests
modify other tasks' runtimes or caches. Full local pre-commit, including mypy,
passes. The previous #1130 final CI run also completed successfully.

Runner input-preparation lifecycle branches now share one context manager;
source leases remain independently owned per buffer. The V2 runner's two
DFlash2-specific preparation/sampling dispatch sites become feature hooks,
while the verifier remains one implementation. On the expanded C inventory
(which includes the pre-existing buffer, event-trace and speculator files on
both sides), parameter read sites decrease **376 → 370**. Alias declarations
remain visible in the generated explanation. Repository coupling decreases
from platform **3756 → 3727**, model **2322 → 2315**, raw environment **271 → 269**;
no ownership whitelist changes are used. Sparse-verifier alignment dumps and
DDTree tracing remain with their feature owners and are not claimed migrated.

### Integration with concurrent main changes

Before C1b merged, main advanced through #1007 and #1028 to `465da8a45`.
C1b was rebased on that main: its host-backed KV allocations and newly added
greedy verifier remain intact. The sampler protocol fixture now supplies the
real kernel configuration/device used by that call site. The new greedy
provider still exposes a whole-runner adapter in main; C4a will narrow that
pre-existing boundary together with packed LM-head selection. It is not counted
as a C1b decoupling result. Updated main-to-C1b coupling is platform
**3747 → 3718**, model **2322 → 2315**, environment **271 → 269**. The rebased
CPU integration suite passed **76 tests**, with seven CUDA-only skips.

### C1b final performance and source follow-up

The rebased GPU integration suite passed **83 tests**. A final source-owner
follow-up covers restored enqueue ordering and recovery when a stream handle
cannot be obtained (retain an unfenced source until device synchronization).

The larger staged copy initially added about 4 µs of host time. Moving the
current-stream lookup back after copy enqueue restores the original overlap;
the normal buffer method also calls the owner directly. Both failure protections
remain. The final same-process A/B uses the real baseline class/method bodies,
15 alternating-order rounds, 10,000 context entries or 100 copies per round,
with the legacy getter cache enabled:

| Operation | Baseline median µs | Shared owner median µs |
|---|---:|---:|
| Ordinary input-preparation context | 4.916 | 2.604 |
| Staged input-preparation context | 3.092 | 2.460 |
| 1 KiB staged copy, host | 54.852 | 55.157 |
| 128 KiB staged copy, host | 69.635 | 69.833 |

Paired copy deltas are 0.479/0.069 µs, inside the observed baseline ranges
(54.347–56.378 and 68.995–73.641 µs). The earlier unexplained copy regression is
not carried into the final change. Separate stable-address H2D graph controls
measured 2.270/2.403 µs for 1 KiB and 22.668/23.111 µs for 128 KiB; these are DMA
controls, not model kernels. Peak GPU allocation was identical (2,048 and
132,096 bytes including the control buffers). Separately instrumented host
allocation peaks were 451/419 and 451/662 bytes; tracing was disabled during
timing. No new per-copy device workspace or model-performance claim is made.

The final trace-config/lease CPU suite passed **59 tests** (two CUDA skips).
Disabled legacy diagnostics retain their old behavior for malformed, unused
numeric options; enabling the diagnostic or retained tree worker override still
raises the original parser error. Source explanations mark the inactive default
instead of presenting the malformed input as its effective value. This does
not change typed validation or computation hashes.

## C2a implementation

The C2a source baseline is C1b's merge `dfaeb1ba19ca98c4648edfab4bf297ca61f68dfa`.
C1b's final queued input-owner GPU follow-up completed with **40 passes**;
its final CI pre-commit check also passed (run `37925367488`).

The existing prefill selector now supplies `GdnExecutionPlan`. Its declarations
also feed the C0 explanation catalog: backend, actual operator entry, state and
output layout, external/in-kernel Q/K normalization, and gate conversion order.
The report explicitly labels selection as static evidence, not a native launch.
The existing auto-candidate ordering is retained, including SM75 prefill
rejection, FP16 admission, CUDA/package checks and dtype fallback. No second
backend selector is introduced.

`KernelConfig.gdn` captures backend/compute policy; `gdn.schedule` captures FLA
launch geometry and autotune candidate lists. Typed values win, the historical
FlashQLA alias order remains, and invalid legacy numeric tuning continues to be
ignored according to the old parsers. Initialization captures values using raw
legacy getters, without inheriting the worker's getter cache. The original
platform checkpoint now supplies GDN defaults directly instead of writing seven
environment keys. Effective backend/compute policy enters graph hashes;
profiling and decode-warmup controls do not. Inactive GDN does not affect another
model's cache. FlashInfer/CuteDSL plans ignore unused FLA chunk-tuning controls.

Both ordinary and explicit packed decode call the same convolution and mixed-QKV
recurrence stages. Their original validation modes, caller-owned output views,
and allocating versus preallocated-output semantics remain explicit. FlashQLA
bindings live under `fla/ops/sm70`; the native verifier receives a head contract
and current tensor operands instead of the entire model layer. The binding does
not retain an early parameter address, allowing later weight replacement/reload.
The three externally normalized prefill providers share Q/K normalization while
retaining their different conversion order: FlashInfer's half-precision log,
FP32 cast and exp round-trip is unchanged. CuteDSL normalizes before converting
the gate; native FLA normalizes internally. These differences are part of the
stage declarations rather than silently replaced arithmetic.

The old layer remains the registration/model-lifecycle boundary and orchestrates
compute stages. Its ordinary metadata gathering/commit and deferred DDTree flow
are C2b's remaining work, not counted as removed by moving provider functions.
Old imports and CustomOp registration locations remain available. Common compute
modules do not import the old model layer, accept it as context, or read legacy
environment policy during execution.

Prefill autotuners are created once per engine and shared by that engine's layers;
JIT arithmetic is reused without sharing mutable winner caches between engines.
The worker binds these runtime resources after configuration transfer. GDN
prefill profiling similarly has one engine-owned budget, preserving per-stage
limits and capture/compile guards. Its serialized policy belongs to
`ObservabilityConfig.gdn_profile`; compatibility-only helpers retain the old
standalone import behavior.

### C2a validation ledger

Validation remains operator/synthetic only. The first candidate run exposed a
missing compatibility export and fixtures that bypassed initialization. The new
replay comparison also initially reused convolution's in-place input between
lanes; independent input storage and explicit replay resets correct the harness.
Padding rows have unspecified allocating-output values in the original kernel:
compare all live output rows and every conv/SSM state slot, retaining the `-1`
index semantics. No arithmetic tolerance was relaxed.

Native validation uses the same artifact for both lanes. Its `_C` SHA256 is
`d258385a58f891a5b5bb926ee2050d5fb1e3399ccca9edb37f1bb922a993ba81`;
installed source identity is `eca6e7370`. GDN verifier source, `ops.h` and Torch
registration source match this batch's baseline exactly. That artifact also
contains unrelated QSA additions, which these tests do not invoke. Python,
Torch, CUDA, V100 and driver settings remain those recorded for C1a. Results and
same-process alternating A/B measurements are recorded before promotion.

The new configuration/provider suite passes **27 CPU tests**. The existing
backend/device/projection integration selection passes **49 CPU tests**, with
61 CUDA-only skips. The configuration utility suite also passes. Full pre-commit,
mypy, environment-registration and layering checks pass without a new whitelist.
Six modified FLA JIT bodies/decorators are AST-identical to the baseline.

On 54633 the corrected candidate selection passed **107 GPU tests**, with four
remaining launch-metadata expectations. Running the old source reproduced the
same four failures: TP4 single-request q8 already selects BV2, while its old test
expected BV8. Correcting only that expectation passes all four on both sources;
output/state equality and canary checks are unchanged. Two additional bound
native-verifier cases pass, including parameters supplied after provider binding.
After the host-call refinement, all **10 affected shared-stage/native-provider
GPU tests** pass again (the two already-passing native-prefill cases are not
repeated). Counts from successive targeted runs overlap and are not summed.

The final paired microbenchmark alternates lane order over nine rounds. Both
lanes use the same two kernels, inputs, states and native artifact; all outputs
and conv/SSM states are exactly equal. CUDA graph replay isolates device time;
eager Python launch timing and allocation tracing run separately.

| Rows | GPU baseline/shared µs | Eager host baseline/shared µs | Extra GPU bytes baseline/shared |
|---|---:|---:|---:|
| 1 | 6.932 / 6.902 | 214.179 / 216.978 | 0 / 0 |
| 4 | 8.970 / 8.950 | 235.128 / 240.101 | 0 / 0 |

An initial approximately 10 µs host increase was localized to temporary argument
dictionaries in the shared wrappers. Fixed argument calls reduce it. Residual
paired host median deltas are +5.194/+8.689 µs; baseline round ranges are
206.625–223.111 and 223.450–240.026 µs. This is a small eager-call cost, not a
claimed host speedup; captured replay adds no stage dispatch. Python allocation
peaks are 15,822/18,563 and 15,822/15,846 bytes, with no additional device allocation.
GPU medians remain inside baseline variation. No model performance is inferred.

Scope accounting: the standard speculative/non-speculative and explicit packed
convolution sites now share one stage; allocating/out mixed-QKV dispatch has one
entry. Three external-normalization prefill providers share the normalization
stage. Four prefill algorithms remain because their layout/cast contracts differ;
FlashQLA decode and the native sequential verifier likewise remain distinct
algorithms. Prefill selection still has one selector. Layer-owned execution no
longer reads the migrated compute/tuning controls; import-time standalone FLA
compatibility defaults and deferred state/diagnostic consumers are explicitly
remaining C2b/C3 work. With the expanded inventory scanned at both revisions,
legacy read sites decrease **374 → 358**; the catalog now detects 265 names
(including previously indirect tuning aliases), not 265 active per-token reads.
The generic coupling ratchet decreases environment **269 → 267** and platform
**3718 → 3708**, with model **2315 → 2315**. No claim is based on file size.

Reproduction artifacts are retained in the task-owned `phase-c2a-20261009`
validation directory: source patches/checksums, per-lane JUnit results,
`host-profile.log`, `hostfix-gpu.log`, `hostfix-ab.json` and native hashes. The
rebase onto main's QSA deliveries changes no GDN kernel source or registration.

## C2b: metadata and state ownership

The baseline is C2a merge `ec535b1e69b6d1b2e6e4e75ce5bd24fc7302972a`.
C2a's final CI pre-commit job passed (run `37933537976`).

GDN's existing state contract and common request metadata now live below both
backend and runner. The per-group and shared preparation paths call the same
token-order/query-offset stage. Stable sorting, the non-decode **-1** sentinel,
accepted-count versus selected-slot distinction, align-mode authoritative state
IDs and padded state rows are retained. The grouped pointer-table kernel and
DDTree arithmetic are unchanged. Capture padding is applied before consumption;
prepared metadata remains keyed to the exact builder and descriptor addresses.

The process-global layer-name registry previously allowed identical names in
separate engines to overwrite one another. Each engine now owns its registry
and reusable common buffers. Forward contexts borrow that exact owner; missing
engine registrations cannot fall back to another engine or the compatibility
registry. Capacity changes retain the old capture buffers while allocating the
new capacity. Builders continue to own their state-index tensors, grouped
metadata descriptors own pointer tables, and `ModelState` owns accepted counts
and conv/SSM movement. No state tensors are copied into a compatibility owner.

`KernelConfig.gdn.state` captures metadata routing and speculative-core policy;
`ObservabilityConfig.gdn_state` captures shadow checks, assertions, metadata
profiling and state-table dumps. Active consumers use the captured values.
State-table budgets are independent and filenames add an engine suffix so two
engines writing into the same directory cannot overwrite one another. Historical
standalone imports and dictionaries forward to a separate compatibility owner.
Serialized worker policy retains resolved choices; runtime owners bind after
configuration transfer and are not computation-hash inputs.

Four handwritten metadata save/replace/restore protocols now use one borrowed
view that restores on normal return, exceptions and nested use. Explicit tensor
operands and CustomOp schemas/fakes remain unchanged. The ordinary explicit
speculative commit also uses C2a's convolution stage. Backend builders declare
their model-state inputs through the existing metadata boundary: generic hybrid
state no longer switches on GDN/Mamba/PLE/Flash-V100 implementation classes.
Flash-V100's verification field mapping stays with its existing speculative
metadata owner; the generic builder gains no model-specific knowledge.

### C2b validation

- Existing metadata/provider CPU selection: **116 passed, 27 CUDA skips**.
- New ownership/configuration selection: **35 CPU passes, two CUDA skips**,
  including 15 owner/contract tests and 20 configuration utility cases. Swapped
  engine construction order, replacement registrations, neutral accepted counts,
  reordered/empty requests, nested failure restoration, disabled diagnostics and
  worker-policy serialization are covered.
- On 54633: **22 baseline grouped-metadata tests passed** and **214 candidate
  tests passed**, including metadata replay, mixed speculative execution and
  exact conv/SSM state comparisons. A final ownership/backend-input/PP follow-up
  completed **26 tests**, including actual CUDA graph capture/replay with changed
  indices, accepted lengths and state data. The subsequent unrelated PP runner
  initialization blocked in a network socket and was interrupted; it is not
  counted as passing. No weights were loaded and that test is not a C2b gate.
- Both lanes' cache-group fixtures now use one real configuration per engine;
  the old fixtures created separate configurations and relied on global buffers.
  This preserves the intended within-engine grouped test and independently tests
  cross-engine rejection/isolation. Numerical assertions are unchanged.
- All three metadata Triton JIT functions/decorators are AST-identical. Full
  pre-commit, mypy, environment and layering checks pass with no new whitelist.

The paired benchmark runs on the same 54633 environment/native artifacts as
C2a. It executes the exact old entry-function bodies and new entries in one
process, alternates order, checks state/metadata equality and traces allocations
separately from timing. Host classification uses synthetic CPU request tensors;
grouped preparation additionally uses real V100 CUDA graph replay.

| Operation | Baseline median | C2b median |
|---|---:|---:|
| Pure-spec request classification, host µs | 142.140 | 142.873 |
| Mixed request classification, host µs | 285.390 | 281.623 |
| Two-group metadata preparation, host µs | 115.154 | 99.924 |
| Two-group metadata graph replay, GPU µs | 4.116 | 4.127 |
| Two-field metadata borrowing, host µs | 0.887 | 2.009 |
| Nine-field metadata borrowing, host µs | 4.358 | 5.431 |

The grouped host paired median improves 15.731 µs; GPU time remains inside the
baseline 4.024–4.372 µs range. Classification paired differences are +0.613 and
-0.315 µs, inside observed variation. The shared borrowing owner costs about
1.1 µs per eager invocation, localized to its object/dictionary lifecycle; it is
not a claimed speedup and does not execute per replay in a captured graph.
Classification/grouped Python allocation peaks are unchanged (1,128/1,416/4,837
bytes). Borrowing peaks change 0→264 and 560→472 bytes. Extra grouped GPU
allocation is **zero in both lanes**. These are operator/host-step observations,
not model throughput or TTFT results.

Scope accounting: request token preparation **2 → 1** implementations; temporary
metadata restoration **4 → 1** protocols; runtime metadata registries **one
process-wide → one per engine**. Lower state/compute stages do not import runner
or the old GDN model layer. Expanded C inventory legacy-read sites decrease
**358 → 340**; generic coupling changes environment **267 → 266**, platform
**3708 → 3694**, model **2315 → 2313**. Remaining old-layer projection diagnostics,
model eligibility, graph policies and communication owners belong to C3/C4;
DDTree-specific fast-build/trace algorithms remain deferred.

Task-owned `phase-c2b-20261009` artifacts retain source/base patches, baseline and
candidate JUnit records, follow-up results, `bench-state.json` and JIT parity.

## C3: ordered defaults and engine-owned policy

The comparison is merged C2b (`ad19d7ad1166`), with B's operator selectors
unchanged. Model qualification/defaults now belong to
`model_executor/models/runtime_defaults.py`, reached through the existing
`MODELS_CONFIG_MAP` adapter. Device participation, compilation policy and
resource defaults belong to `platforms/runtime_defaults.py`, invoked through a
platform hook at the original early checkpoint. The existing late platform
validation checkpoint remains in place. Legacy imports forward to these owners;
model exports are lazy to avoid config/distributed/model import cycles.

This batch replaces the process environment as the carrier of automatic
settings. It does not merge distinct algorithms or change the model eligibility
contracts. Model-specific PLE checkpoint layout and BFLA shape qualification
are separate from device/storage/worker capability checks.

### Configuration and consumer ledger

`config/execution_policy.py` declares 36 legacy aliases with typed fields.
`config/policy_defaults.py` binds ordered default writes to those fields plus
existing B native policies, runtime warmup and DFlash fields. The declarations
and `runtime_policy_report` are the parameter-to-owner source of truth.

| Owner | Fields / retained consumer | Lifecycle and fallback |
|---|---|---|
| `CompilationConfig.runtime` | AOT/mega-AOT, breakable graph, compile/no-compile decode, capture size, no-op elimination, dual compile, split MTP graphs, memory estimator | Resolved before compilation; decorators, wrappers, cache keys, runners and graph utilities borrow the owner. Existing graph modes and eager fallbacks remain. Dynamic token dispatch is unchanged. |
| `KernelConfig.layer_execution` | Batch layouts, FP16 GEMV, fused GDN input/HC, Gemma compile semantics, LM-head top1, shared expert overlap, GLM projection and mHC choices | Layers/providers capture the policy; local dtype/shape/device/native admission remains. No weight/buffer copies are added. |
| `ParallelConfig.communication` | TP4 push, TP8 hierarchy/push, MoE add/reduce, message-queue capacity, PP partition | PP partition is captured early enough for model/speculative validation. Collective selection consumes the prepared value. C4b still owns provider extraction and remaining native controls. |
| `OffloadConfig.ple` | Hybrid, CPU and disk placement | Executors and connector receive engine-local placement. PLE child resets only its own PP partition; it no longer removes a process variable. Existing IPC ownership remains. |
| `AttentionConfig.flash_v100` | BFLA keep ratio, grouped verify enable/minimum context, small-query bound | Bound by the existing backend/spec policy. Attention arithmetic, masks and A3 admission remain unchanged. |
| `SpeculativeConfig.sm70_dflash2` | BF16 emulation and proposal temperature/top-p, alongside existing verifier fields | Draft initialization resolves values once; original validation/errors retained. BF16 affects the model graph; sampling parameters do not salt the model-computation hash. |
| Existing B native policies / `sm70_runtime` | Dense tune limits, grouped expert rows, GLM W13 choice, AWQ warmup limit | Platform defaults update existing owners, with typed overrides retained independently for linear and MoE. Native workspace/address ownership remains B's. |

Resolution order is explicit typed value, explicit legacy input, ordered
model/platform defaults, then the original unset default. The original early
collective, breakable, prefill/projection and graph-cache checkpoints remain in
order. Batch invariance retains the mandatory AOT disable. Model checkpoint
provenance is scoped so later platform decisions are not mislabeled as model
defaults. Reported values and source history are diagnostic, not execution hits.

Consumers' actual parsers are preserved where they differed from registration
metadata: BF16 emulation defaults on and accepts stripped truth words; two GLM
projection flags use `raw != "0"`; native thread/tuning integers retain `atoi`
prefix parsing and their fallback/clamping. mHC accepts 128/256/512/1024 threads,
otherwise 256, while M1 always uses 128 threads.

The original `sm70_glm_mhc_pre_norm_out` schema remains as the compatibility
entry. `sm70_glm_mhc_pre_norm_configured_out` supplies the captured thread value
to the same launcher/kernel. Engine execution uses the configured entry, which
requires the rebuilt normal `_C` extension; it never silently reverts to a
process environment launch. Fake registration and old import/schema remain.
No CUDA arithmetic changed.

Prepared policies serialize with their config into workers. Active execution
borrows the engine's policy map from its existing forward-context resources;
standalone component initialization retains legacy compatibility. Explicit
`get_pp_indices(..., partition=None)` means automatic partitioning and cannot
inherit another engine's partition. Omitting that keyword retains the legacy
standalone/current-engine API.

Effective computation is hashed by the typed owners. Migrated aliases are
removed from compilation's additional environment factors once resolved, so
explicit typed and equivalent legacy input share cache identity. Resource-only
settings, source provenance and inapplicable graph/communication/PLE controls
are excluded. Changing the process environment after initialization does not
change a prepared engine's policy or its migrated cache factors.

### Structural evidence and retained scope

On the same expanded C inventory, C2b to C3 source parameter reads decrease
**341 → 269**, across **57 → 61** inventoried files. This includes the new
configuration/model/platform owners, so moving a consumer does not erase it
from the ledger. The 36 newly captured aliases have **zero direct Python env
reads outside initialization/configuration**. This is a static source audit,
not a native-hit or latency claim. C3 adds no new independent calculation
pipeline; its change is ordered policy ownership and per-engine isolation.

C2b's Python 3.13 metadata-builder annotations are corrected here. Remaining
LM-head/norm provider boundaries, graph context/bucket ownership, communicator
native controls and fusion capabilities are explicitly C4a/C4b work. DDTree,
external NCCL/CUBLAS process-global determinism setup and D/E are not described
as removed by this batch.

### C3 validation record

CPU checks cover ordered/default/conflicting settings, worker spawn transfer,
two-engine initialization order, PP auto/explicit isolation, legacy parser
edges, unused-policy hashing, typed/legacy cache equivalence and report/API
serialization. No model weights are loaded.

- Configuration/graph/DFlash/PLE suite: **203 passed**.
- Report/platform/unused-policy/ownership suite: **96 passed** (overlaps the
  ownership cases above; not an additional unique-case total).
- Pipeline and policy-focused regression: **113 passed**.
- New native launch CPU/ownership subset: **18 passed, 12 CUDA skips** before
  GPU execution. Runtime compatibility guard: **6 passed**.
- Full pre-commit and Python 3.13 typing pass. The compatibility guard now
  derives migrated aliases from declarations and rejects execution-time reads;
  no layering whitelist is expanded.

An initial report integration had a premature return, and an old test slice
omitted the now-lazy qualification imports. Both were corrected before
acceptance. Native CPU/GPU build configuration initially lacked three unrelated
CMake source files; the matching source files were supplied, then the normal
`_C` target was rebuilt. These failures are not passing validation evidence.

Standalone fullgraph admission is also covered: configuration construction is
constant-folded outside tensor tracing only when no engine owner exists. Active
engines bypass that compatibility path. A fullgraph CPU test alternates two
engine forward contexts and verifies the selected policy does not leak.

C3 GPU acceptance on 54633 is complete. The normal `_C` CMake target was built
for SM70 with CUDA 12.8, Torch 2.10.0+cu128 and driver 580.173.02; GPU 0 is a
V100-SXM2-32GB. Native SHA256 is
`4df50cb5cc140449a80705b4f7e846dcb4f393b14fb3a9a6b3e82446b0d9ddd1`.
Its dependencies are standard CUDA/Torch/system libraries. Fresh Python
processes loaded and exercised it without preload or private library overrides.
The build's three changed native files match the final source token-for-token;
only clang-format whitespace differs in the CUDA source.

- Existing mHC subset: **7 baseline + 7 candidate passed**.
- Configured native threads, updated-input graph replay, Gemma norm and GDN
  stages: **46 passed**, including bitwise legacy/configured comparisons.
- Seven alternating paired rounds tested M1/M8 at 256/1024 configured threads
  (M1 still launches 128). All 56 direct-ABI output digests and all 56 actual
  provider output digests match their paired baseline. Both use the same
  source/model-less input contract, with no model weights loaded.

Actual `sm70_mhc_pre_norm_from_staging` provider measurements, in microseconds:

| Rows / threads | GPU baseline → candidate | Host baseline → candidate | Paired host delta |
|---|---|---|---|
| 1 / 256 | 16.583 → 16.584 | 17.226 → 18.506 | +1.285 |
| 1 / 1024 | 16.572 → 16.571 | 16.933 → 18.449 | +1.110 |
| 8 / 256 | 14.506 → 14.506 | 19.125 → 18.535 | -1.620 |
| 8 / 1024 | 12.802 → 12.801 | 19.042 → 18.269 | -1.461 |

Entries are medians; paired deltas are medians of matched-round differences.
Baseline host ranges were respectively 16.164–18.878, 15.959–19.221,
18.184–21.645 and 18.730–21.300 microseconds. M1 has a small measured host
increase inside that variation; this is not a speedup claim. GPU changes are
within the paired baseline variation. Direct ABI medians likewise remain
16.621/16.610/14.596/12.888 → 16.623/16.614/14.596/12.886 microseconds.
**No additional temporary GPU allocation** was measured in either harness.

Raw evidence: the task-owned `phase-c3-20261009/artifacts` directory on 54633,
including `baseline-v3.xml`, `candidate-v3.xml`, `candidate-stages-v3.xml`,
`bench-v3.jsonl`, `bench-provider-v2.jsonl`, native dependency records and build
log. Local retained copies and the source-token comparison are in
`/home/ymzx/arch-ws/tmp/phase-c3`. No model throughput, TTFT or 35B conclusion is
made by this acceptance.

## C4a implementation and acceptance

Three draft-weight binding flows now share `models.shared_weights`: legacy
EAGLE/MTP, Runner V2 EAGLE/MTP, and ordinary DFlash. The model contract retains
PP=1 qualification for the first two, final-stage embedding replication for
DFlash, the existing memory-dependent weight comparison, and MTP shared-head
and top-k buffer identity. The runner owns execution order and local-argmax
validation; it no longer implements the binding lifecycle.

Gemma native/CUDA entries use one ordered provider dispatcher for fixed-width
normalization, fused FP32 residual, and long-prefill normalization. Their
arithmetic implementations remain independent because reduction and casting
order differ. Custom-op names, schemas, fake functions and registration order
are unchanged. The three moved Triton kernel ASTs are unchanged.

Prepared LM-head state owns packed FP16/QPN8 weights and scratch, while the
embedding owns its original parameter and sharding. Dense FP16 projections use
the same ownership boundary. Compatibility attributes borrow that state; weight
replacement drops it. Packs are nonpersistent buffers and are not duplicated in
state dictionaries. Packed top-one, raw GEMV, FP32 logits and QPN8 candidate
rerank keep their layout, accumulation and dense-vocabulary tie contracts.
Shadow comparison returns a typed request to the layer adapter, which computes
ordinary logits once; the provider has no quant-method or runner callback.

B's input flattening/output restoration is shared by quantized, FP16 and LM-head
providers. These views now live outside the kernel selector package initializer,
so importing the generic linear layer cannot trigger a registration cycle.
Preparation binds the native policy only after device/layout admission; a CPU
or unsupported-device fallback does not require a CUDA ABI. The compiler still
uses the original-weight operator while eager execution uses prepared weights.
Dynamic M admission remains inside the existing execution boundary.

| Parameter family | Initialization owner and compatibility |
|---|---|
| Dense enable, suffix allowlist, max M, LM-head dense/top-one, Gemma eager/long-prefill | `KernelConfig.layer_execution`; typed values precede legacy aliases. LM-head truth-word admission and deferred integer-parser logging errors remain distinct. |
| Native FP16 max M and tuning | B's `Sm70NativeConfig` nested under the layer policy; two conflicting typed max-M requests raise rather than silently disagree. |
| DFlash FP32 logits, QPN8 rerank and shadow | `SpeculativeConfig.sm70_dflash2`; explicit/model-resolved choices reach native FP16 policy. Shadow remains explicit and affects computation hashes because it returns dense results. |
| DeepSeek FP13 / exact FP16 fallback | Layer execution policy consumed by the existing model provider; tensor-value qualification and FP13-to-FP16 fallback are unchanged. |
| Dense/unquantized/profile/greedy-token/Qwen trace | `ObservabilityConfig.runtime_trace`; borrowed by engine resources and excluded from calculation hashes. |

The expanded same-scope source inventory measures **277 → 251** legacy read
sites (62 → 70 source files, including the destination providers). These are
source counts, not kernel-launch counts. Draft binding lifecycles decrease
**3 → 1**, and Gemma stage dispatchers **2 → 1**. The retained independent
arithmetic/layout branches are listed above; no layering whitelist was enlarged.
Layering counts are env 248, platform 3447, model 2137.

Local operator/configuration/state coverage passed **174 tests**, with **14
CUDA-only skips**. A separate fresh-import/consumer selection passed **87 tests**
(overlapping coverage; not additive). Tests include explicit/legacy conflicts,
engine initialization order, stable configuration after environment changes,
weight/pack release, missing native ABI rejection, cached greedy-verification
results, and row bounds on eager/compiled dispatch. A fresh-process import test
caught and eliminated the registration cycle before promotion.

On 54633, baseline and candidate each passed **64 GPU tests** (Gemma norm,
DeepSeek exact/packed projections, fused auxiliary projections and greedy
verification). The new LM-head replay suite passed **10 tests**. Final follow-up
passed **37 tests**, including loaded FP16 projection fullgraph views, changed
inputs, empty inputs and all FP16 gated-norm payloads. The same repaired
native-norm fixture passed **26 tests** on baseline. These suites overlap.
The old fixture changed an environment variable after the kernel policy had
already resolved to disabled; it now explicitly enables the typed policy on
both sides. No numerical tolerance was relaxed.

Five alternating process pairs measured the following medians. GPU figures use
CUDA events over repeated captured calls; host figures measure enqueue cost
separately. Both use Torch 2.10.0+cu128, CUDA 12.8, driver 580.173.02 and one
V100-SXM2-32GB. LM-head geometry is N1024/K512, normalization width is 5120.

| Operator / M | GPU old → new µs | Host old → new µs |
|---|---:|---:|
| Raw top-one / 1 | 5.231 → 5.218 | 77.210 → 69.650 |
| Packed top-one / 1 | 18.971 → 18.743 | 75.066 → 71.034 |
| Packed top-one / 8 | 27.845 → 26.903 | 74.121 → 71.828 |
| Dense FP16 / 1 | 10.209 → 10.146 | 43.006 → 41.077 |
| Dense FP16 / 8 | 8.481 → 8.471 | 44.152 → 41.484 |
| FP32 logits / 1 | 5.861 → 5.859 | 37.541 → 32.250 |
| FP32 logits / 8 | 5.997 → 5.999 | 40.318 → 36.313 |
| QPN8 rerank / 1 | 40.714 → 40.727 | 295.055 → 280.077 |
| QPN8 rerank / 8 | 45.531 → 45.472 | 302.103 → 272.619 |
| Fixed Gemma / 1 | 3.482 → 3.492 | 51.157 → 50.613 |
| Fixed Gemma / 8 | 3.527 → 3.531 | 51.726 → 49.386 |
| FP32-residual Gemma / 1 | 3.516 → 3.508 | 75.716 → 74.643 |
| FP32-residual Gemma / 8 | 3.600 → 3.604 | 76.459 → 75.708 |

GPU timing shows no regression beyond observed baseline variation; temporary
allocation bytes are unchanged for all 13 cases. Twelve cases have identical
output digests in every pair. Default dense M1 autotuning produced the same two
digests on both sides across independent processes; it is not a deterministic
cross-process baseline. Three further pairs with the same explicit
`VLLM_SM70_F16_DENSE_TUNE_MAX_M=0` on both sides produced one identical digest.
This diagnostic setting does not change production defaults.

Raw artifacts are under the task-owned `phase-c4a-20261009/artifacts` directory
on 54633, including `bench-v2.jsonl`, `bench-fixed-v1.jsonl`, baseline/candidate
JUnit and `followup-v2.xml`. The unchanged packaged C3 extension has SHA256
`4df50cb5cc140449a80705b4f7e846dcb4f393b14fb3a9a6b3e82446b0d9ddd1`.
No model weights, throughput, TTFT or 35B acceptance are claimed.

## C4b graph and communication ownership

This delivery is based on merged C4a, `c73eb2d1d100`. It preserves the
existing graph dispatcher and collective selection order. The dispatcher binds
one immutable `GraphExecutionPlan`; live batch sizes, request counts, context
lengths, partition keys and capture/replay tables stay in their original owners.
Model geometry is declared by `GraphModelContract`, and the platform policy
resolves its existing capability gates. Explicit empty buckets still disable
auto buckets. Explicit partition overrides still suppress automatic variants.
The backend-priority cache includes the resolved Flash-V100 enable flag, so two
engines cannot reuse another engine's candidate list.

The existing custom communicator still allocates/exports/imports CUDA IPC,
registers eager and graph buffers, restores allocator settings and disposes the
native pointer. `NativeCollectiveBindings` binds its owning extension once for
initialization, launches and destruction. `CollectiveCapabilities` supplies one
shared admission implementation for ordinary reductions and the three Gemma
collective/norm variants. Their differences remain explicit: TP2 accepts FP16 or
FP32 residual; TP4 requires FP32 residual; long prefill requires divisible rows
and its separate registered output capacity. The fusion pass consumes these
capabilities and retains its missing-op and missing-buffer fallbacks.

HC row/width/operator contracts and fused-chain CTA requirements belong to the
model adapter. The communicator's old HC methods remain forwarding entrypoints.
The native provider handles optional schema arguments without knowing the model.
There is one native pointer owner and one registration/destruction path.

### C4b configuration and explanation ledger

| Owner | Migrated controls | Preserved behavior |
|---|---|---|
| `CompilationConfig.runtime` | MTP/compressed-index/FP8 buckets, FP8 batch variants, decode partition, E4M3 auto/wave/threshold, decode-only capture | Legacy names are read at initialization. Positive-bucket validation, explicit-empty disable, partition error and eager/piecewise fallback remain. Metadata builders retain their engine policy outside the forward context. |
| `AttentionConfig.flash_v100.enabled` | Flash-V100 platform/attention qualification | Existing model/device/backend gates stay in place; cache entries include this effective value. |
| `ParallelConfig.communication` | Top1 provisioning, symmetric memory, FlashInfer, AWQ tile/overlap/grids, long-prefill norm | Typed values precede aliases; batch invariance still forces symmetric memory off, now on this engine's configuration. Existing invalid tile mode returns the ordinary fallback. |
| `ParallelConfig.communication.native` | Eighteen native selector/launch inputs, including block limits, algorithms, TP4 push controls, TP2/TP4 norm threads and TP8 hierarchy | The shared native configuration adapter encodes raw strings once; native `atoi`, `strtol`, exact-string and unset/empty semantics stay distinct. Parser errors remain at the original operation checkpoint. |
| `KernelConfig.layer_execution` | Batch fastpath and HC MTP/cooperative/full-unroll stages | Consumers use resolved fields. Explicit shared batch policy is reconciled with the native collective input; contradictory typed owners fail initialization. |
| `ObservabilityConfig.runtime_trace` | TP all-reduce tracing and native profile tracing | Seen sets belong to each communicator/layer; diagnostic choices do not alter calculation hashes. |

The native configuration lives alongside B's native vector adapters in
`config/sm70_native.py`; `config/collective.py` forwards the import. The ordinary
packaged extension adds `init_custom_ar_configured` and keeps the old constructor
schema. The Python compatibility initializer also supplies the historical two
push defaults without writing them to `os.environ`. Missing configured native
ABI disables custom AR before IPC allocation and keeps ordinary communication
fallback. No sidecar, preload or private extension is required.

The C0 source ledger reads aliases directly from these declarations. The existing
runtime explanation report includes resolved values, sources, effective hash
fields, the initialized graph plan and optional collective observations.
Configuration selection and host dispatch observations are separately labelled;
observations during graph capture are not described as replay counts or as a
GPU kernel timing trace. Typed policies serialize with the engine. Runtime plans,
IPC pointers, buffers and trace sets stay in the worker resource owner.

### C4b verification in progress

The affected local suite passed 191 cases with five hardware skips. Follow-up
layer/configuration coverage passed 66 cases, and provider/trace/guard/graph
coverage passed 82 cases (overlapping subsets, not additive unique totals).
The CUDA backend-priority test requires the native extension and is assigned to
54633. The native CUDA kernel bodies compare identically for all 19 kernels in
`custom_all_reduce.cuh` and `custom_all_reduce.cu` after removing whitespace and
comments. Selection inputs and per-owner diagnostic flags are the changed native
host code.

The C4b normal CMake `_C` build and TP2/TP4 operator A/B are recorded under
`/home/ymzx/arch-ws/phase-c4b-20261009` on 54633. The GPU job waits for the shared
ownership lock; no result is claimed until that job completes. Final paired
GPU/host/allocation results and extension provenance will replace this pending
paragraph before merge. TP8 remains a contract-only test on this four-V100 host.

## Phase C structural accounting and retained boundaries

The following counts describe independently maintained protocols, not source-file
sizes or the number of numerical kernels. The baseline is merged B
`ec5f5b7eac70`; the per-batch acceptance sections above retain the exact successive
baselines and do not claim a single end-to-end model benchmark.

| Protocol | Before → after | Remaining real differences |
|---|---|---|
| Step event collection/aggregation | 3 → 1 `StepProfiler` | Runner V1, Runner V2 and proposer keep their report layouts and in-flight step owners. |
| Auxiliary warmup scheduling | 2 loops → 1 ordered executor | Model/proposer kernels and cache layouts remain separate tasks; allocator restoration and capture boundaries are preserved. |
| Staged input transfer | Runner-managed admission/buffers/events/recovery → 1 transfer owner | Single-request admission and normal synchronous fallback remain. |
| Ordinary DFlash sampling | 2 runner-specific decisions → speculator result contract | Handled result, existing dense logits and ordinary fallback remain distinct; dense logits are not recomputed on that fallback. |
| GDN common computation | Shared convolution and allocating/out recurrence entries; 3 external-normalization providers share one normalization stage | 4 prefill algorithms, FlashQLA decode and sequential verification retain distinct layouts, casts and arithmetic. There remains one prefill selector. |
| Request metadata preparation | 2 token-order/query-offset implementations → 1 | Grouped pointer-table kernel, padding and DDTree arithmetic remain distinct. |
| Metadata override/restore | 4 manual protocols → 1 borrowed view | Builders own tensors; the view restores references on normal return, nesting and exceptions. |
| Shared draft weight binding | 3 lifecycles → 1 model adapter | PP qualification, final-stage replication and MTP shared-head relationships remain explicit. |
| Gemma layer dispatch | 2 dispatch implementations → 1 | Fixed-width, FP32 residual and long-prefill arithmetic remain separate providers. |
| Collective admission | 3 Gemma admission blocks → 1 parameterized capability contract | TP2/TP4 residual dtypes and long-prefill capacity differ. IPC allocation/registration/destruction still has one owner. |
| Graph selection | 1 dispatcher before and after; static qualification becomes one initialized plan | Dynamic context buckets, partitions and graph tables remain in the dispatcher. No parallel graph scheduler is added. |

Run `tools/sm70/path_inventory.py --phase c --summary` at each reference using
the final inventory implementation. Its expanded source scope records **461 →
260** legacy read sites across **37 → 80** files; C4a to C4b alone is **307 →
260** across **75 → 80** files. Destination owners are included. The final catalog
also exposes 146 initialization alias declarations. These are static source
counts, not per-token execution counts; the parameter-name total is not a
before/after removal metric because the final declarations reveal indirect
aliases that the original consumer scan did not enumerate.

The native collective launch files have **27 → 0** direct `getenv` sites.
Eighteen distinct native inputs are captured once by the configured constructor;
the legacy constructor captures its compatibility snapshot once. The migrated
Python policy guard checks consumers against the configuration declarations.
Generic layering totals at these snapshots are env **271 → 243**, platform
**3880 → 3431**, model **2325 → 2136**. No whitelist was expanded. These aggregate
snapshots include intervening main deliveries; the protocol changes and each
batch's tests, rather than aggregate counts alone, establish C's changes.

| Mutable resource | Sole production owner | Borrowers / lifetime |
|---|---|---|
| Timing events and report totals | Each `StepProfiler` | Runner/proposer step context; disabled mode allocates no events. |
| Pinned staging and copy events | Input transfer component | Keeps source storage alive through completion; restores stream/event state after failure. |
| GDN layer registrations and common metadata buffers | Engine runtime resources | Forward contexts borrow the same registry; another engine never supplies missing registrations. |
| State-index tensors and pointer descriptors | Metadata builder / grouped descriptor | Captured descriptors retain the buffers they address across capacity growth. |
| Accepted counts and conv/SSM movement | `ModelState` and existing state contract | Prepared views borrow live state; compatibility forwarding does not copy state. |
| GDN tuning winners and profile budgets | Engine GDN runtime resources | JIT arithmetic is shared; mutable winners and budgets are not process-global. |
| Prepared LM-head/linear packs and scratch | Prepared provider state | Original parameters remain with the layer; replacement invalidates the pack. |
| Graph objects and replay tables | Existing dispatcher/runner owners | Initialized graph plan is immutable policy, not a second graph owner. |
| CUDA IPC and collective registrations | `CustomAllreduce` | Bound native namespace receives initialization, execution and destruction for the same pointer. |
| Collective diagnostic seen sets | Each communicator/layer | Explanation reports borrow records; capture observations are not replay counts. |

Retained entries are deliberately visible in the source ledger:

- DDTree's runner/proposer/scheduler branches, tree-specific GDN algorithms and
  their controls remain deferred. Ordinary DFlash uses the existing Runner V2
  selection; the generic Mamba scheduler is retained.
- The old GDN model adapter retains private projection/core-boundary experiments,
  standalone legacy helpers and optional tensor/graph dump hooks. Their legacy
  controls are not described as removed: the new common convolution, recurrence,
  prefill and metadata components do not depend on these helpers. They remain
  model-layer maintenance debt, separately identifiable from the shared stages.
- HC-specific experimental projection kernels and diagnostics remain with the
  HC model provider. The common communicator consumes its declared tensor and
  operator contract without owning that composition.
- External allocator settings, P2P probing and upstream batch-invariant,
  NCCL/CUBLAS process controls retain their existing initialization contracts.
  C removes the migrated policy's environment propagation, not every upstream
  process-global setting in the repository.
- TP8 topology and non-SM70 capabilities have contract coverage only where the
  required hardware is absent. Full-repository D/E and model performance
  acceptance remain separate work.
