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
| C1b | Staged input resources and ordinary speculative sampling boundary | Implemented; operator and synthetic validation recorded below |
| C2a | GDN compute plan, providers and shared stages | Pending |
| C2b | GDN metadata and state preparation/commit | Pending |
| C3 | Ordered defaults, model qualification and engine-local effective values | Pending |
| C4a | Shared embedding, LM-head, norm and linear providers | Pending |
| C4b | Graph/communication/fusion boundaries and final explanation report | Pending |

Each batch is reviewed and merged directly into main before creating the next
branch. Existing experiments remain available; default-off alone does not imply
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
