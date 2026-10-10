# Phase D: configuration lifetime and retained deprecations

Integration baseline: `8cbaf7edda7a0148152b9731a013faf710a76b76` (Phase C).
All six deliveries target `onecat/main`, in order, without stacked branches.
Defaults, algorithm admission, numerical semantics and existing fallbacks are
preserved. DDTree is deferred. GPU acceptance uses operators and synthetic
state on the authorized 54633 machine; no model throughput claim is made.

## Parameter and consumer ledger

The reproducible, individual-parameter ledger is generated from the actual
registry, typed alias declarations and consumer source:

```bash
.venv/bin/python -m tools.config_inventory --json > /tmp/phase-d-parameters.json
.venv/bin/python -m tools.config_inventory
.venv/bin/python -m tools.config_inventory --check
.venv/bin/python tools/pre_commit/check_layering.py --report
```

Each JSON row includes the unchanged getter expression, declared/effective
defaults, automatic conditions, associated paths, deprecation evidence,
existing typed fields and every detected consumer's file, line, lexical scope
and read kind. The existing environment registry remains the metadata source;
the ledger does not execute getters, import CUDA, or establish another runtime
selector. `destination` is a migration grouping, not proof of an implemented
owner. `typed_declarations` distinguishes implemented bindings from that target.

The inventory includes raw Python reads, import aliases, registered attributes,
the Flash-V100 wrapper getters, constant native aliases, native env helpers and
related `TM_*`/`FLASH_QLA_*` inputs. Dynamic names that cannot be resolved are
listed separately. Source references are neither runtime operator hits nor a
proven call graph: initialization helpers and deferred code are not counted as
executed inference. Each later delivery must resolve its remaining dynamic
readers and record admission, fallback and resource lifetime with the owner.

| Family | Consumer chain and destination | Contract and lifetime to preserve |
| --- | --- | --- |
| GDN | Model adapter → GDN plan/provider; `kernel_config.gdn` | Projection, convolution, recurrence and norm retain layout, rounding and state contracts; per-layer conv/SSM state and graph buffers keep their owners. |
| Ordinary speculation | Speculative config → proposer/speculator → rejection sampler | General sampling policy belongs to `sampling_policy`; DFlash2 keeps `sm70_dflash2`. Preserve RNG, acceptance, dense-logit fallback and dynamic M boundaries. |
| Flash-V100 | Attention config plus existing graph fields → backend plan → Python package → native binding | Preserve candidate order, dtype/layout/partition admission, workspace capacities and capture/replay addresses. Shared graph/native settings have one resolved source. |
| MoE/linear/HC | Existing kernel config → selector/provider → prepared native binding | Keep B's weight preparation, codec, workspace and fallback contracts; do not duplicate the common execution stages. |
| Sparse/QSA/indexer | Sparse policy → qualified model adapter → provider | Import-time environment snapshots become per-engine policy; preserve top-k order, context thresholds, buffer geometry and fallback. |
| Diagnostics | `observability_config.runtime_trace` → engine-owned observation state | Capture paths/filters/budgets at init; retain dynamic enable-file existence checks and original observation points; flush outside capture. |
| Graph/communication/runtime | Existing C policy owner → graph/collective/runtime component | Preserve graph buckets, capture ownership, topology admission, allocator restoration and input-copy lifetime. |
| Loading/build | Process loader and build configuration | Native registrations and loaded libraries are process-scoped; do not present them as independently replaceable per-engine state. |
| DDTree | Existing deferred compatibility entry | Retained unchanged and reported separately, never silently included in completion counts. |

Legacy precedence is preserved per parser: explicit typed values win, legacy
aliases retain their original order, model/platform defaults retain their
original checkpoints, and forced safety restrictions still apply. Values and
provenance travel with worker configuration. Only effective computation enters
the corresponding graph hash; legacy aliases are excluded only when replaced
by that effective policy. Diagnostics and inactive formats do not invalidate
unrelated artifacts. Numeric parser errors and short-circuit behavior remain
part of the contract.

## Delivery status

| Delivery | State | Evidence / remaining work |
| --- | --- | --- |
| D1 registry and ledger | Merged [#1141](https://github.com/1CatAI/1Cat-vLLM/pull/1141); CI passed | Static inventory, structured deprecation metadata, shared registration scanner and explicit-input warn-once support. This establishes visibility; it does not claim execution consumers migrated. |
| D2 GDN and speculation | Merged [#1143](https://github.com/1CatAI/1Cat-vLLM/pull/1143); CI passed | CPU isolation/compatibility, 17 GPU operator cases and matched A/B passed. |
| D3 diagnostics | Merged [#1146](https://github.com/1CatAI/1Cat-vLLM/pull/1146); CI passed | Shared diagnostic owner, 74 initialized parameters, legacy typed MoE bridge, CPU isolation and 7 GPU cases plus matched operator A/B. |
| D4a attention package | Merged `d4ce51399`, CI passed [#1148](https://github.com/1CatAI/1Cat-vLLM/pull/1148) | Backend/package/versioned native policy, graph projections, diagnostics and Python workspace isolation; evidence below. |
| D4b FA2/79T resources | Merged [#1150](https://github.com/1CatAI/1Cat-vLLM/pull/1150), `b14c2ab0a`, CI passed | Native 79T policy, cuBLAS/stream/event/workspace ownership and normal FA2 build; evidence below. |
| D5 remaining providers | D5a merged [#1151](https://github.com/1CatAI/1Cat-vLLM/pull/1151), `e2a52d519`, CI passed; D5b in validation | Provider/native resource lifecycle is merged. Marlin, remaining event-trace consumers and loader-aware native scope coalescing are under validation. |
| D6 closure | This delivery, based on merged D5b | Full ownership gate, retained boundary inventory, FlashQLA binding and final evidence below. |

Baseline layering report: 243 literal raw environment reads in counted generic
modules. This excludes some registered reads, helper indirection and native
consumers. It is not the completion denominator. The full source inventory and
its unresolved-reader list are the D baseline; completion requires classifying
and migrating actual active consumers, not reducing this one regex count.

## Deprecation contract

`EnvVarMetadata` now distinguishes `alias`, `experiment` and `historical`
deprecations, with a reason, evidence and optional replacement. New structured
deprecations require evidence; aliases require a replacement. D6 completes the existing category-only entries; the metadata check now
rejects a deprecated category without structured reason and evidence. Default-off controls are not automatically deprecated.

Explicit legacy settings (including `0`) warn at most once per process and
name, independently of Python warning filters. Metadata inspection and report
generation never warn or evaluate a getter. The shared initialization resolver
also warns for an explicit legacy setting overridden by typed configuration,
without evaluating the overridden parser. Compatibility inputs remain usable;
this change removes no implementation and adds no removal deadline.

The initial negative-evidence entry is the AWQ single-kernel reducer CTA
experiment. Its counts 1 and 4 regressed the documented two-GPU 512-input /
32-output screen by 32.84% and 9.25% against its same-run tail-worker control.
That conclusion applies to that experiment and geometry, not to every overlap
implementation. See [the retained measurements](../sm70_tile_runtime_exploration.md).
The DFlash2 verify-fastpath alias illustrates the separate alias case: its
implementation stays active under the existing qualified typed policy.

## Acceptance record

Each delivery records relevant CPU contracts, source layering checks and any
affected GPU operator A/B results here. Final closure must include the number
of parameter sources, active execution reads, duplicate parser/diagnostic
flows, mutable global states and resource owners before/after; explicit
exceptions must identify a loading, standalone compatibility or DDTree entry.
Phase E owns the later repository-wide documentation/workflow restructuring.

### D1 validation

- 39 metadata/registration/inventory tool tests passed.
- 108 configuration, environment and existing policy regression tests passed.
- After the duplicate-warning self-review fix, 42 focused metadata, inventory,
  typed-override and DFlash2 policy cases passed again.
- All applicable pre-commit hooks passed, including Python 3.10 mypy,
  registration/reference checks and the existing layering/policy guards.
- No GPU computation or native schema changed; GPU testing is reserved for
  subsequent consumer migrations. No numerical or performance conclusion is
  inferred from these configuration-only tests.

### D2 policy and consumer migration

Base: `5dc280bb7f073f76c514a4696c5f434a0080c156` (merged D1).

| Initialized owner | Migrated consumers | Retained boundary |
| --- | --- | --- |
| `kernel_config.gdn.projection` (13 controls) | GDN projection/core wrappers, layout materialization and gated norm | Existing dtype/TP/shape admission; paired QPN8/one-pass requirement, missing-op error and deep-MTP safety guard. Explicit input-core disable still short-circuits its legacy input. |
| `speculative_config.sampling_policy` (17 controls) | Proposer sampling, static/dynamic vocabulary preparation, rejection sampler and async accept-count setup | Greedy/stochastic admission, temperature/top-p validation, RNG ordering and dynamic request/token metadata stay with existing execution. |
| `speculative_config.sm70_dflash2.lookup` (9 controls) | Runner V2 lookup controller initialization | Resolve only when assistance has a nonempty tail; retain legacy clamp bounds and request-state owner. |
| Existing `sm70_dflash2` (2 additional controls) | Fused GDN TP2 and QKV-pack admission | Same platform, speculation and shape qualification. |
| Existing GDN state policy | Old-runner synthetic capture metadata | The warmup checkpoint consumes the same `spec_core` decision as the layer. |

The engine uses a new explicitly configured gated-norm custom op; its old
name, schema and fake implementation remain available for independent legacy
callers. Registration stays at the old import checkpoint. Both entries invoke
the same numerical implementation, including the 12x128 one-pass branch and
its unchanged general-norm fallback. No CUDA/C++ source or native ABI changes
in this delivery.

Two historical parsers for draft top-p are deliberately distinct: the dense
proposal accepts only the exact string `1`; the fused proposal uses `bool(int)`.
Their results are captured together, including a fused-only malformed-value
error raised at its original consumption checkpoint. Explicit typed booleans
override both. The bonus switch retains `!= "0"`, token matching retains
`== "1"`, and the empty top-p override retains each caller's previous behavior.

Configuration is serialized with provenance. Effective policy hashes replace
migrated aliases in environment cache factors. Proposal-only controls do not
invalidate greedy paths; unrelated vocabulary controls do not invalidate
non-MTP paths. Dynamic target-candidate capture remains represented even for
non-MTP callers. Initialized engines pass their policy explicitly; only the
retained independent helper entry points capture legacy inputs themselves.

Validation:

- 128 focused configuration, model-adapter and proposer regressions passed.
- 16 focused policy cases subsequently passed, including five added checks for
  override short-circuiting, missing operators and poisoned getters during
  compile-cache factor generation.
- All applicable changed-file pre-commit hooks and PR CI passed.
- 17 GPU operator cases passed on 54633 GPU 2: norm schema/fake/AOT dispatch,
  changed-input capture/replay, projection-tail layouts and rejection paths.
- Three alternating source-lane A/B rounds produced identical output hashes
  for every compared case; temporary allocation peaks were unchanged. The
  post-freeze source delta only added initialization short-circuit handling
  and its CPU checks; numerical providers and execution consumers were frozen.

Norm results below are medians across three rounds. GPU time measures graph
replay per norm; host time measures eager Python enqueue, not end-to-end model
latency. The 24-row case rejects the 12-row one-pass shape and retains fallback.

| Shape / request | GPU before → after (µs) | Host before → after (µs) | Extra allocation, both |
| --- | --- | --- | --- |
| 12x128, ordinary | 1.5744 → 1.5675 | 153.45 → 143.65 | 3584 B |
| 12x128, one-pass | 1.3859 → 1.3846 | 72.18 → 64.58 | 3072 B |
| 24x128, ordinary | 1.5771 → 1.5827 | 151.87 → 143.99 | 6656 B |
| 24x128, one-pass requested | 1.5962 → 1.6003 | 161.98 → 152.42 | 6656 B |

GPU differences (-0.44% to +0.36%) are within the observed small measurement
variation. CPU proposal host time was 180.57 → 172.90 µs with identical seeded
outputs. These are operator/configuration measurements only. Torch 2.10.0+cu128,
CUDA 12.8, V100-SXM2-32GB, driver 580.173.02, TP1; no model loading. The same
normal compiled libraries served both complete Python source lanes; no native
source changed, no new private DSO or preload was introduced.
[Raw rounds and artifact contract](phase_d2_operators.json)
include extension identity and source-freeze details.

Diagnostics remain D3 scope. Provider warmup controls and the paused
`EMPTY_CORE_OUT` warning remain D5/D6 scope; the latter does not change current
allocation behavior. Graph/piecewise attention policy belongs to D4. Existing
DDTree execution and upstream Mamba scheduler controls are not claimed as
migrated here.

### D3 diagnostic policy and resource lifecycle

Base: `eb5a87b6ae03ad448d687a4dc133b3fc38b17e8a` (merged D2).

`observability_config.runtime_trace` now owns tensor channels under `dumps`,
ordinary sampler observations under `sampling`, and family-specific observations
under `dflash`. Its serialized provenance includes the legacy input or typed
field, parsed filters and deferred parser errors. Channel declarations also
feed the execution-read guard and compile-cache alias filtering. Reports use
these captured values; generating a report never evaluates an environment getter.

| Boundary | Shared behavior and retained differences |
| --- | --- |
| Initialization | Resolve paths, filter dialects, budgets and flags once. B's `sm70_moe.awq/fp8.diagnostics` fields forward into the same owner; explicit observability fields take priority over those old typed inputs, then legacy environment/defaults. Invalid unused format inputs retain their qualified error checkpoint. |
| Layer observations | Qwen and MoE use one capture/record implementation. The Qwen custom op still clones its result; the MoE custom op retains its aliasing schema. GDN projection payload preparation stays in the FLA adapter and uses the shared storage/output machinery. |
| Capture resources | One engine diagnostic owner contains independent channel counters, budgets, buffers and metadata. Replaced buffers stay alive for previously captured graphs. GDN prefill warmup keys are also engine-owned; immutable shared code does not own them. |
| Graph output | Qwen, MoE and GDN use one graph flush implementation. Existing runner filter precedence, paired Qwen/MoE flush checkpoint and enable-file behavior remain intact. |
| Sampler/runner output | MTP step, sample tensors, classic logits, compile inputs, ordinary speculative and DFlash selector dumps use the common output manager. Payload fields and log labels remain unchanged; engine outputs append a unique engine suffix to prevent same-directory overwrites. Independent legacy helpers retain their filenames. |
| Qualification/hash | Ordinary diagnostics are excluded from computation hashes. DSpark alignment requires extra confidence logits, so its effective output contract participates in DSpark's hash. An already-required confidence output does not gain another hash variation. |
| Legacy helpers | Historical custom-op names, fake implementations, imports and helper names remain. Module dictionaries forward to independent standalone owners only; engine execution never uses those owners to reread environment inputs. DDTree-specific diagnostics remain deferred. |

Original parser distinctions remain explicit: reverse ranges for Qwen/graph
filters, strict nonnegative sampler steps, comma-only GDN comparisons, empty
AWQ comparison sets, exact `1` flags versus integer booleans, and per-channel
zero-budget behavior. Trigger paths are fixed at initialization; file existence
is still checked dynamically. Disabled GDN diagnostics do not query CUDA or
allocate a diagnostic tensor. Rejection timing aggregates are per engine;
existing model/layer-local reference-comparison counters retain their owners.

The scoped structural counts are:

- 74 diagnostic inputs now have an initialization owner. In the changed execution
  modules their literal/registered source-read sites decrease from 112 to 0.
  These are source references, not a claim that all 112 ran on each token.
- Two layer capture/record flows become one; three graph flush loops become one.
  The runner, rejection sampler and selector share their filter parser with the
  layer observers, with dialect differences declared at initialization.
- 27 previously shared mutable diagnostic/warmup state containers or counters
  no longer serve engines. Each engine has one diagnostic owner plus its GDN
  warmup-key set. Historical standalone aliases remain explicitly separate;
  per-layer reference arithmetic and deferred DDTree probes are retained.
- The existing generic layering census decreases from 241 to 158 raw reads,
  3422 to 3326 platform references, and 2134 to 2132 model references. No owner
  exclusions or whitelist entries were added. This census remains narrower
  than the full Phase D inventory.

Validation: 121 related CPU cases passed; after the final resource/parser and
static-guard review, 55 focused ownership/default/guard cases passed. GPU tests
on 54633 passed 7 cases, including alternating engine capture/replay with changed
inputs, replacement-buffer lifetime, empty/dynamic shapes, non-aliasing custom-op
schema/fake behavior and AOT values. Small final changes to comparison-filter
edge cases, standalone parser forwarding and JSON reporting do not alter the
GPU tensor implementation; their CPU cases are recorded separately.

Three alternating process rounds compare the merged D2 source with the complete
candidate source on V100 SXM2 32GB GPU 2, TP1, Torch 2.10.0+cu128, CUDA 12.8,
driver 580.173.02. The normal native extensions are unchanged. The observation
operator uses FP16 `[8, 128]` tensors; direct file I/O is disabled in the timed
capture case. Values below are medians of the three per-process medians.

| Observation mode | GPU µs old → new | Host enqueue µs old → new | Admission µs old → new | Peak additional bytes, both |
| --- | --- | --- | --- | --- |
| Diagnostic off | 1.0803 → 1.0795 | 38.220 → 32.716 | 0.569 → 0.432 | 2048 |
| Capture copy on | 2.1488 → 2.1504 | 64.798 → 60.203 | 2.229 → 0.644 | 2048 |

All output hashes match and measured allocations are identical. GPU changes
are below 0.1%; the main measured reduction is repeated host policy parsing.
This is an operator result, with no model throughput or TTFT conclusion.
[Raw rounds and source/native hashes](phase_d3_operators.json) accompany the
change. Full logs, the benchmark script and the initial stale-archive collection
failure are retained under `/home/ymzx/arch-ws/tmp/phase-d3/`; the corrected GPU
run required no source-side numerical change. The remote task is
`/home/ymzx/arch-ws/phase-d3-20261009/` and has released its GPU lock.

### D4a attention configuration and worker ownership

Base: `649dbd63a3d45384a8225252336747318ba210a0` (merged D3).
The independent attention package and FA2/79T have distinct native builds, so
D4 is delivered in two consecutive main-based changes. Neither the retained
79T environment reads nor its native workspace maps are counted as complete
in this first delivery.

`attention_config.flash_v100.options` captures 105 remaining computation and
resource controls. `observability_config.runtime_trace.flash_v100` captures 23
diagnostic controls. Existing graph fields remain the sole owners of shared
partition/context policy; attention receives their projections after ordered
model/platform defaults. The existing five parent attention fields remain
compatible. Declarations also feed the inventory, explanation and read guard.

| Boundary | Implementation / preserved contract |
| --- | --- |
| Initialization | Typed values override legacy inputs. Alias precedence, exact-string versus first-character booleans, integer clamps and qualified errors are retained. Serialized provenance and deferred errors travel with workers. |
| Backend and graph | Consumers use resolved fields. Decode and MTP partition parsing occurs once; token count, request order, context length and prefill/decode choices remain dynamic at the original checkpoints. |
| Independent package | Thirty public/internal functions pass an optional explicit runtime through their existing calls. Engines bind the runtime once; no-config functions use a separate independent compatibility adapter. |
| Native package | ABI 1 adds a `PreparedPolicy` and 20 configured bindings; all existing exports remain. Its 66 parsed projections represent 65 legacy names, including one historical scalar alias. Calls borrow immutable parsed values and per-owner observations without parsing strings or reading the environment. An old binary fails clearly during engine initialization. |
| Hash | Effective backend, package and native choices participate, including cases where old parser dialects disagree. Resource sharing, ordinary diagnostics, inactive formats and disabled feature sub-options are filtered. Migrated aliases no longer additionally salt the environment hash. |
| Resources | Six package caches and eight backend/grouped-attention caches or warmup records belong to two worker owners. Shutdown closes only that engine. Captured bridge/gather buffers survive subsequent eager growth until graph teardown; existing eager OOM/fallback and capture-growth rejection remain. |
| Observations | Native counters, backend route/fallback counts, dtype observations and prefix-dump budgets are engine-owned. Comparison arithmetic and per-layer quotas keep their original owners. JSON and tensor writers share the diagnostic output manager; payloads/log labels remain and filenames gain the existing engine suffix. |
| Compatibility | Public package calls, legacy module aliases, custom-op schemas, fake registrations and native loading checkpoints remain. Standalone caches/counters serve independent no-config callers only. |

The common policy base and field resolver have no reverse dependency on an
owning attention/runtime module. Existing import paths re-export their public
classes/functions. Runtime resource objects are created after worker transfer;
no native handle or workspace address is stored in serialized configuration.
Native observation counts describe host dispatch/capture, not graph replay hits.

Scoped source census: the changed Python backend/package files contained 118
literal, registered or wrapped read references; 9 remain: two compatibility
getter definitions, one dynamic-library loader and six deferred DDTree sites.
The configured paths for the other 109 sites consume initialization results.
This count is a source census, not a per-token call count. Native package
getters now share the policy projection; direct-call fallback still preserves
their historical parsing. The 98 similarly named helpers copied into the
grouped/scalar FA2 sources are not reached by their exported operators; they
remain historical source, with three distinct defaults, rather than being
misreported as active reads migrated in this change.

The work replaces policy interpretation and shared resource ownership while
retaining the existing algorithm tree. Dense/paged layouts, FP16/E4M3/E5M2,
scalar/XQA/grouped and experimental BFLA branches remain for their actual
layout, numerical or algorithmic differences. Defaults and qualification are
unchanged; disabled experiments are not deprecated merely for being disabled.

Validation evidence is recorded in [the operator rounds](phase_d4a_operators.json).
On 54633 GPU 2 (V100 SXM2 32GB, Torch 2.10.0+cu128, CUDA 12.8, driver
580.173.02), three alternating source-lane rounds give the following medians:

| Operator | GPU µs before → after | Host enqueue µs before → after | Extra allocation, both |
| --- | --- | --- | --- |
| FP16 XQA decode, sequence 1537 | 47.4624 → 47.3702 | 91.105 → 81.760 | 0 B |
| E4M3 scalar decode, sequence 1537, partition 1024 | 146.6675 → 146.4422 | 86.140 → 82.776 | 0 B |
| Causal dense prefill, Q32/K128/GQA6/D256 | 42.2810 → 42.2400 | 176.035 → 182.337 | 295936 B |

Every output hash matches. GPU changes are below 0.2%. Dense-prefill host
samples overlap the baseline range; this does not establish a host-prefill
speedup. These are operator measurements, with no model throughput or TTFT
conclusion. Both standalone native packages were built through normal setup
from their source lanes, without private kernel overlays or preloads.

Initial capture/operator validation passed 18 cases; the added runtime release,
old-binary rejection and backend capture-growth cases subsequently passed too.
The broader GPU-host policy suite initially stopped because its task directory
lacked the unchanged normal FA2 library; that test setup failure is retained
in `gpu-final.log`. CPU source review also exposed older test fixtures patching
retired getters or omitting the new observability owner; those fixtures now
exercise resolved policies. Final validation passed 227 operator, routing and synthetic metadata cases on
54633, including all 22 bound native/runtime cases. CPU checks passed 198
configuration/default/worker cases, 69 prefill/resource cases, 58 focused
configuration/report cases and 24 final policy/metadata cases; these overlapping
suites are not added together. All changed-file pre-commit hooks passed.

A subsequent worker-transfer check passed 20 focused cases after adding the
resource-map transfer rule: live owners are omitted from serialization, so a
worker creates fresh handles and budgets from the transferred configuration.
This check includes an intentionally unserializable parent handle and verifies
that neither the handle nor parent diagnostic counts reaches the worker.
The final GPU run used the normal rebuilt attention extension plus the unchanged
normal FA2 dependency; its log/XML and source archives remain in the task
artifact directory. A legacy atexit route-summary logger writes to pytest’s
already closed captured stream after the successful suite; this did not affect
assertions or process exit status.

### D4b FA2/79T policy and native resource ownership

The normal `_vllm_fa2_C` target now ships prefill policy ABI 1 and the
`Sm70PrefillRuntime` owner. Existing Q8000/Q8192 schemas and registrations remain
available for independent legacy callers. A qualified engine binds the explicit
owner during attention initialization. A binary that supplies the qualified
legacy operator but lacks the binding fails initialization; missing operators
or FP32 accumulation qualification retain the existing dense fallback.

| Compatibility input | Canonical field | Retained parser / default |
| --- | --- | --- |
| `PREFIX_QK_CUBLAS_ALGO_RUNTIME` | `flash_v100.options.prefill_qk_algorithm` | Optional native `atoi`; unset retains the build's cuBLAS algorithm. |
| `VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS` | `flash_v100.options.prefill_score_block_tokens` | Whole-string `strtol`, multiples of 8192 in [8192, 131072]; unset uses the build default. Invalid legacy input raises at the original workspace checkpoint. |
| `PREFIX_TORCH_SERIAL_TAIL` | `flash_v100.options.prefill_serial_tail` | Unset or any string other than `0` enables serial execution. Memory-headroom qualification remains dynamic at allocation. |
| `PREFIX_TORCH_EXACT_TAIL` | `flash_v100.options.prefill_exact_tail` | Presence, including empty string; return-affecting experiment remains in the calculation hash. |
| `PREFIX_TORCH_DIRECT_TAIL` | `flash_v100.options.prefill_direct_tail` | Presence; return-affecting experiment remains in the calculation hash. |
| `PREFIX_TORCH_DUMP_TAIL` | `runtime_trace.flash_v100.prefill_dump_tail` | Presence; original observation points and stderr format, excluded from the calculation hash. |

Explicit typed values take precedence without rewriting the environment.
The seven-field native projection distinguishes an absent algorithm override
from algorithm zero. Native scalar parsing now also preserves ASCII whitespace
and libc overflow behavior; the config/report path never invokes an environment
getter after initialization. Inactive 79T parameters do not perturb other
attention paths. `PREFIX_*` declarations and native constant-array readers are
included in the existing inventory and policy checker.

Three active process caches (the shared score tensor and the two query-family
workspaces) become one worker owner containing the same three cache slots per
device. The owner also accommodates the two historical generic-score slots,
which are inactive in the normal recipe. Buffers, cuBLAS handles, streams,
immutable host metadata and dispatch observations belong to this owner. Its
lifetime is retained through capture/replay and ends after graph destruction.
Closing it drains device work, including a dispatch that failed before recording
its final event; this synchronization is confined to shutdown.

One physical-device execution gate deliberately remains shared. The unchanged
kernels bind device-global pointers, so independent workspace tensors alone
would introduce races between engines. The original event wait/record points,
including external capture events, remain in place. The shared gate coordinates
both query families and all block sizes; it owns no calculation policy or score
tensor. The two algorithms and their rounding sequences remain unchanged.

Scope counts: six active alias consumers previously interpreted compatibility
inputs in the native path; configured execution now performs zero environment
reads and no string parsing. The independent legacy adapter and benchmark-only
`PREFIX_BATCHED_TAIL_QK_ALGO_RUNTIME` / overlap-wave controls remain explicitly
outside engine execution. The latter occur under `!PREFIX_TORCH_EXTENSION`.

Validation: 63 focused CPU cases pass, including config/provenance/hash,
missing capabilities and ownership checks. On 54633, 203 operator/routing and
synthetic-metadata cases pass without loading a model. Bound Q8000/Q8192 outputs
match legacy exports bit for bit; two owners with distinct score-block policies
retain correct outputs across interleaved stream replay, changed inputs, later
query-family workspace allocation and independent shutdown. The normal FA2
artifact builds successfully. Eleven additional native parser projections pass
without allocating GPU tensors. Changed-file pre-commit and layering checks pass.

Three alternating process A/B rounds use the same GPU 2, TP1, Torch 2.10.0+cu128,
CUDA 12.8 and driver 580.173.02, with FP16 Q/Hq6/Hkv1/D256 and KV32768. Median
results (GPU events around five-call graphs; host enqueue measured separately):

| Query | GPU µs baseline → candidate | Host µs baseline → candidate | Temporary bytes |
| --- | --- | --- | --- |
| 8000 | 28920.83 → 28980.84 (+0.21%) | 306.65 → 309.05 | 0 → 0 |
| 8192 | 28972.65 → 28950.12 (-0.08%) | 293.98 → 298.52 | 0 → 0 |

Output digests match across all six processes at both shapes. GPU differences
are within observed event-sample variation; host samples overlap. No speedup or
model-performance conclusion is claimed. Raw samples, artifact SHAs and the
workload contract are in [phase_d4b_operators.json](phase_d4b_operators.json).

Artifacts are under `/home/ymzx/arch-ws/phase-d4b-20261009` on 54633. The first
baseline extraction retained old timestamps and Ninja reused candidate objects;
that run was rejected before benchmarking. The corrected build explicitly
refreshes source timestamps and produces a different baseline artifact. The
local policy/metadata suite also needs a GPU-capable platform for nine existing
fixtures; those fixtures pass in the 203-case remote run.

### D5a: provider configuration and resource lifecycle

QSA/indexer, GDN/FLA schedules, quantized loaders, MTP projections, ordinary
sampling, fallback attention, TurboQuant and the remaining HC/PLE provider gates
now consume their initialized owners. Deferred compatibility inputs travel with
worker configuration; malformed dormant inputs still raise at their original
admission checkpoints. Capturing a format does not activate it. Explicit typed
values take precedence without changing the process environment.

- QSA's two workspace maps, online-QPN8 address pools, sampler scratch and
  TurboQuant caches borrow the worker resource map. Captured allocations survive
  growth; AOT resolves new addresses by the existing layer prefix/owner slot.
- Native runtime ABI 1 owns TurboMind packing, tuning, scratch and observation
  counters in both normal extensions. Policy ABI 61 parses scalars and target
  lists once. Existing custom-op schemas remain compatible. An explicit engine
  configuration fails initialization against an insufficient binary.
- Six FLA modules no longer snapshot launch policy on import. GDN and KDA reuse
  `GdnScheduleConfig`; the model adapter declares the applicable hash fields.
  Live autotuners belong to worker resources and are omitted from serialization.
  Independent operator calls retain separate lazy compatibility caches.
- Diagnostic budgets and buffers use engine owners. The paused empty-output and
  replaced coarse GDN gates retain their original notices, including presence-only
  parsing for the latter. No empty-allocation experiment is reactivated.
- Async queue depth remains in the scheduler. The old output-token repair switch
  also affects ordinary async runs, so its canonical owner is `sm70_runtime`.
  Both batch constructors bind it once; request-history repair reads no env.
- Library paths, capabilities and upstream loading controls remain loading inputs.
  The six native `SM70_MARLIN_*` launch overrides are completed in D5b. DDTree
  remains deferred; the final common-counter audit is recorded in D6.

**Validation.** Focused CPU suites cover worker serialization, hash qualification,
two-engine initialization/execution order, malformed inputs, original golden
route/call-order snapshots, forbidden getters after initialization and resource
release. Logs reside under `/home/ymzx/arch-ws/tmp/phase-d5`. Final notice/default
coverage passes 32 tests. The remote provider suites initially stopped after
170 and 197 passes on fixture/package failures; all seven corresponding cases
pass after supplying the unchanged standard extension and using production
initialization scopes. The fallback-attention suite passes 34 tests. The final
FLA/GDN/native-owner/async-state suite passes 35 tests (8 unrelated deselections),
including changed-input/state graph replay and release of one of two owners.

Normal `_C`, `_moe_C` and `_C_stable_libtorch` components were built from source;
both arms use the same unchanged FA2/Flash-V100 components. Fresh-process ABI and
loader checks passed. Artifact hashes, raw samples and workload contracts are in
[phase_d5a_operators.json](phase_d5a_operators.json).

| Operator | GPU/event median change | Host median change | Output/state and allocation |
| --- | --- | --- | --- |
| QSA M512/K2048 | -0.16% | +4.90% | Exact |
| MTP router M5/N512/K2560 | +0.32% | +7.65% | Exact |
| FP16 M16/N512/K1024, fixed selector | -0.06% | -1.84% | Exact |
| QPN8 M8/M17/M32 | -0.16% / 0% / 0% | +8.64% / +7.69% / -1.73% | Exact |
| Triton attention M1/M33 | -0.21% / -0.21% | -16.83% / -11.97% | Exact |
| TurboQuant prefill M33, eager events | +0.43% | +0.67% | Exact |
| KDA/GDN prefill M64/H4/K64 | +0.11% / 0% | +0.65% / -0.24% | Exact |

All entries use three alternating A/B rounds. GPU differences are small; the
host overhead is **not** uniformly unchanged. Entering/exiting both native owners
adds about 25–27 microseconds per host boundary after caching ScriptObject method
wrappers (previously about 35 microseconds). It is paid at the runner boundary,
not per captured GPU node. QPN8's eager wrapper costs about 1.6–1.8 microseconds
and QSA about 5.5 microseconds in these samples. D5b will retain this measurement
and narrow the owner transition cost. No model-performance conclusion is made.

Rejected evidence is retained: TurboQuant prefill cannot use the attempted graph
capture because the original path calls `.item()`; its final A/B uses eager events
on both sides. FP16's first autotuned baseline changed its own digest across
rounds, so deterministic parity uses fixed selection on both sides. Initial CPU
fixtures that inferred a GPU or re-registered custom ops are not counted as
passes; corrected tests exercise the actual initialized policies.

### D5b: Marlin compatibility policy and shared native scope

Six `SM70_MARLIN_{DENSE,MOE}_{CTA_GEOMETRY,SPLIT_K,METADATA_CACHE}` inputs
now belong to `KernelConfig.sm70_marlin`. The normal native policy ABI grows
append-only to 67. Initialization parses the original geometry, `strtol` and
metadata dialect once; native shape qualification, defaults, error priority and
old operator schemas stay unchanged. Workspace preparation binds the policy;
runner execution borrows its stable slot. Independent no-config calls retain
the original compatibility adapter. Explicit policy against an older binary
fails before execution.

Both normal extensions now expose their active context identity. Initialization
coalesces handles only after proving that entering `_C` also changes `_moe_C` to
the same nonzero identity. Builds with separate TLS domains or without this
optional probe keep two owners. On 54633 the proof selects one handle. Event
tracing's remaining old-runner and graph-wrapper call sites now borrow the engine
tracer and its diagnostic budget; disabled tracing adds no CUDA timing or sync.

Validation passes 47 GPU/operator and lifecycle cases on 54633, including two
Marlin configurations, changed-input capture/replay, independent release, native
AOT slot reload, E8M0 subnormals and retained invalid-override errors. CPU policy,
worker/hash and loader-domain tests pass; the full changed-file hooks pass.
Source/build identities and raw A/B samples are in
[phase_d5b_operators.json](phase_d5b_operators.json).

Three alternating A/B rounds preserve every deterministic output digest and
allocation count. GPU time differs by -0.08% to +0.19%. Existing FP16/QPN8 host
calls including the native boundary improve by 4.6–8.9 microseconds (9–15%).
Marlin's eager operator host differences range from -3.9% to +6.5%; samples
include ordinary host scheduling variation. Its newly required engine boundary
adds about 20–26 microseconds compared with the old standalone call. That cost
is once per enclosing runner boundary, shared by its layers, and is reported
separately from GPU time. No model speed claim is made.

The first GPU run had eight failures and is retained. A focused repeat probe
proved the legacy Marlin split-K=8 path itself changes low bits on random FP16
inputs; the largest observed MoE spread was 0.00048828125. The exact policy oracle
uses the existing fixed-E8M0 fixture with binary-exact activations for both arms;
random-input split-K error tests remain enabled. Two FP16 owner tests also needed
the same fixed selector on both arms instead of comparing independent autotuning
winners. Neither fix changes production computation or defaults.

### D6: final ownership, deprecation and native closure

Base: merged D5b `22c4d22f4b2f0e5bb794a3d4c457afead19c920d`. The unrelated
serving change in #1152 is retained. The final gate is
`python -m tools.config_inventory --check`, installed as an always-run hook.
It inspects all tracked Python/native sources in vLLM, csrc, Flash-V100,
FlashQLA and the bundled lmdeploy tree. No directory was added to an exclusion.

The common reader resolves import/assignment aliases, registered attributes,
subscripts, `getattr`, presence tests and proven forwarding helpers. Dynamic
names need a declared input domain and exact consumer scope. Ownership comes
from reachable configuration annotations, inheritance and actual alias tables;
a destination guessed from a name is not accepted as an owner. The JSON ledger
links resolver, admission and hash implementations without evaluating getters.
Static predictions remain separate from native observations and graph replay.

The final pass also completes these consumers:

- PLE budgets belong to `offload_config.ple`, including the early EngineArgs
  capacity check. Automatic/empty values and deferred invalid-budget errors are
  preserved. Graph dense-capture and Gemma TP2 communication gates use their
  existing owners. Inactive model/provider choices do not salt unrelated hashes.
- Ordinary DSpark target profiling consumes `runtime_trace`, including the
  historically DDTree-named aliases. The primary profiler-step name wins even
  when empty; the two NVTX names retain their original OR semantics. Typed
  values win. Captured raw inputs and typed overrides appear in the explanation.
- Native FlashQLA column groups belong to `kernel_config.gdn`. ABI 1 binds an
  immutable policy in the normal bundled extension before execution; no new
  workspace is introduced. The existing native bodies and exports remain.
  Unset/empty input retains the dynamic token/head heuristic, libc `atoi` and
  invalid-value errors are preserved, and an old binary fails at initialization.
  The original-TileLang provider explicitly bypasses its standalone env adapter. Legacy negative/overflow `atoi` results remain invalid instead of
  colliding with the binding's internal automatic-selection marker.

The final hash audit also closes GDN/state, profiler, event, warmup, native
shared-stage and inactive-speculation aliases. Parsing and legacy-hash filtering
consume the same owner declarations. Poisoned-getter tests cover both named
overrides and dormant features; standalone no-config hashes retain compatibility.

The same expanded scanner was run on the C baseline and final D source. Raw
JSON and counting definitions are retained in
[phase_d_closure.json](phase_d_closure.json).

| Measure | C baseline | Final D |
| --- | ---: | ---: |
| Individually enumerated parameters | 736 | 757 |
| Parameters without typed owner or retained boundary | 361 | 0 |
| All legacy source read positions, including dynamic and retained entries | 914 | 355 |
| Positions with a statically resolved name (subset of the previous row) | 790 | 234 |
| Unclassified parameter-to-consumer edges | 563 | 0 |
| Unclassified consumer scopes | 291 | 0 |
| Dynamic-name readers without declared input domain | 53 | 0 |
| Bound native policy references | 68 | 154 |
| Incomplete registry metadata entries under the final rules | 60 | 0 |

The extra enumerated names come from complete alias/domain declarations; this
is not a count of new algorithms. These are source counts, **not per-token
execution counts**. The 355 remaining positions include initialization and
compatibility; the native edge count can increase when a dynamic helper's
supported names become explicit. No unclassified engine execution, forward or
capture reader is accepted by the final gate. Engine-isolation and poisoned-
getter tests validate the configured paths separately from this static census.

Of 757 parameters, 693 have reachable typed owners. The other 64 have explicit
retained destinations: 39 DDTree, 10 library/build loading, five standalone
benchmark controls, four early EngineArgs MTP defaults, three process logger
controls, one standalone FlashQLA override, one registered dormant attention
control and one historical no-op notice. The ledger lists their exact names,
consumers and reasons. It also exposes all 125 dynamic reader records (121 distinct positions).

Retained consumer edges comprise 171 standalone compatibility, 61 DDTree,
16 library loading, six standalone benchmark, three process logger, four early
EngineArgs and eight initialization edges. Another 98 are copied FA2 native helper
references unreachable from registered exports; the checker follows local
references from those exports and will reject a newly reachable legacy reader.
These helpers and the two unbuilt `paged_to_contiguous_old.cu` /
`paged_to_contiguous_fixed.cu` variants are retained with their history. The
normal Flash-V100 recipe builds `paged_to_contiguous.cu`; suffixes such as `_vN`
alone do not establish deprecation.

All 62 deprecated registry names now carry kind, reason, evidence and a
replacement where applicable (51 are in this acceleration inventory). Alias
retirement is distinct from an algorithm's negative result: the QPN8 PP2/TP4
alias does not condemn QPN8, and the AWQ reducer measurement applies only to its
recorded geometry. Default-off experiments remain accessible. Explicit settings
warn once per process/name, including when a typed value overrides them. The
warn registry is intentionally process-owned; diagnostic budgets are not.

| Scoped structural result | Before | After |
| --- | --- | --- |
| Layer capture/record implementations | 2 | 1 shared flow |
| Graph dump flush implementations | 3 | 1 shared flow |
| Dense/MoE Marlin override parsers | 2 copies | 1 common native parser, prepared once |
| Engine native env consumers: Marlin / 79T / FlashQLA groups | 6 / 6 / 1 | 0 / 0 / 0 |
| Audited shared diagnostic/warmup and attention cache containers | 27 + 14 + 3 | 0 serve engines through process globals |
| Independent owner groups for those 44 containers | Process lifetime | 5 engine owner groups |

The five groups are diagnostics, GDN warmup keys, Flash-V100 package resources,
backend attention workspaces and native prefill resources. Their individual
names are in the closure artifact. This is a scoped, reproducible census, not a
claim that every unrelated vLLM global was removed. D5 additionally assigns QSA,
QPN8 address pools, sampler scratch, TurboQuant caches, FLA autotuners and native
packing/tuning/scratch to worker resources. FlashQLA's new owner is immutable.

Shared process boundaries remain: DSO/custom-op registration, immutable compiled
code and prepared-policy registries, restored TLS/context references, the native
prefill device execution gate, warn-once names, standalone compatibility caches
and deferred DDTree state. The device gate is necessary for the unchanged native
device-global pointer contract; it owns neither policy nor score tensors.
Worker serialization excludes live owners, and close/release is engine-scoped.

**Validation.** The integrated CPU suite passes 162 cases, covering the
reader/ownership checker, alias precedence, metadata, deferred errors, hashes,
worker transfer, event/native owners and GDN plan compatibility. After the
cache-alias review, 129 related configuration/plan tests pass, followed by 31
parser/provenance cases; these overlap and are not added together. Final schema,
registration and layering checks require reductions without increasing limits.
The broad pre-existing PLE suite has five failures also reproduced on clean
D5b: four stale environment/field fixtures and one CPU device-inference fixture;
these are not reported as passing. The new placement contracts pass.

On 54633 V100 GPU 1, the normal FlashQLA build passes 24 operator/configuration
cases: empty/prefill input, every retained column-group choice, mixed decode,
reordered state including `-1` rows, two owners and changed-input graph replay.
Three alternating A/B rounds cover nine prefill/decode cases. Every output and
state digest matches exactly; temporary allocation is unchanged. GPU medians
vary from -1.01% to +0.09%, host medians from -5.46% to +8.65% with overlapping
sample ranges. No speedup claim is made. Native hashes, measurements and build
contract are in [phase_d6_operators.json](phase_d6_operators.json).

After source integration, 35 configuration/plan cases also pass on the free
54633 GPU 4 (Quadro P400, SM61); all 16 SM70 operator cases correctly skip.
The busy V100 workload was not interrupted, and this run does not replace the
24-case V100 evidence. Four additional native host-side admission checks pass
against that normal extension, including negative and overflow legacy inputs.

The first build lacked `patchelf`; the same normal bundler succeeded after
installing that build utility, and the failed log is retained. The final source
integration changes import organization and initialization qualification only;
the measured native numerical source is unchanged. D5b's measured 20–26 µs
native host boundary remains disclosed above. No model, TTFT, throughput, 35B
performance or untested topology conclusion is made. DDTree and the full Phase E
documentation/workflow reorganization remain separate follow-up work.
