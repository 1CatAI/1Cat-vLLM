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
| C1a | C0 ledger; three profilers share collection/aggregation; ordered warmup; typed runtime and diagnostic configuration | Implemented; validation in progress |
| C1b | Staged input resources and ordinary speculative sampling boundary | Pending |
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
