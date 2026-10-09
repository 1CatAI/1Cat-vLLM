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
| D5 remaining providers | Pending | Model/provider import snapshots, remaining native knobs and loading boundaries. |
| D6 closure | Pending | Complete evidence audit, remaining-name ownership, report and execution-time read guards. |

Baseline layering report: 243 literal raw environment reads in counted generic
modules. This excludes some registered reads, helper indirection and native
consumers. It is not the completion denominator. The full source inventory and
its unresolved-reader list are the D baseline; completion requires classifying
and migrating actual active consumers, not reducing this one regex count.

## Deprecation contract

`EnvVarMetadata` now distinguishes `alias`, `experiment` and `historical`
deprecations, with a reason, evidence and optional replacement. New structured
deprecations require evidence; aliases require a replacement. Existing category-
only entries remain visible in the report with missing details until audited
in D6. Default-off controls are not automatically deprecated.

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

### D5 provider migration (in progress)

The first provider change binds nine QSA controls, two GDN projection controls,
two MTP batch projection controls, four layer provider controls, five
unquantized MoE controls and two GLM diagnostic controls. The existing configuration
owners and initialization checkpoints remain authoritative. No algorithm,
precision mode, shape bound or native schema changes in this part.

- QSA no longer snapshots computation policy on module import. Its two mutable
  workspace maps borrow engine storage, retain captured allocations across
  growth, and keep standalone compatibility storage separate. The QSA library
  path and grouped ABI capability remain process loading/capability concerns.
- GDN batch packing and MTP router/shared projection consumers use the same
  initialized policy as weight preparation. The ordinary top-k20 sampler captures
  its layer policy at construction; static helper calls retain a standalone
  compatibility adapter. Invalid dormant settings are recorded at initialization
  and raised at their original admission checkpoints.
- Unquantized MoE warmup and execution use one configuration. The temporary
  legacy-tile warmup override is context-local and restored on exceptions.
- GLM KDA finite/trace admission uses runtime diagnostics. Seen keys, armed
  prefixes and token indices belong to the engine diagnostic owner.
- Online QPN8 workspace pools are engine-local. Its ordinary and fused HC
  dispatch resolve workspace addresses by layer prefix inside opaque operations,
  preserving B's export/reload mechanism. Existing native operators are unchanged.

Focused CPU evidence is under `/home/ymzx/arch-ws/tmp/phase-d5`. The remote
baseline is main `b14c2ab0a`; normal `_C` and `_moe_C` builds are owned by
`/home/ymzx/arch-ws/phase-d5-20261010`. GPU validation and A/B results are pending.
The next candidate also binds DeepSeek/indexer controls, ordinary sampling
scratch, long-attention opt-out/manifest selection, top-token/GLM/DFlash diagnostics
and the remaining QPN8 native selectors. Prepared native policies parse scalar
and target-list inputs once. Native runtime ABI 1 owns TurboMind scratch, packing,
tuning and trace state per engine. Existing custom-op schemas remain unchanged;
new host-only runtime classes and an explicit capability query are built into
both normal extensions. Stable configuration-owner slots rebind AOT policy inputs
without embedding engine addresses or diagnostic values in compiled graphs.

CPU validation, native-header compilation, normal CUDA rebuild, and operator
A/B evidence are recorded separately. The candidate policy/provider/diagnostic
regression suite passes 294 tests (3 GPU-dependent skips); native owner isolation
and retained diagnostic checkpoints pass 55 focused tests. Both the standalone
C++ policy header and Torch class registration template compile successfully.
The same layering counter records 147 -> 129 raw environment references,
3191 -> 3186 platform references and 2114 -> 2111 model references relative to
the first D5 candidate's ledger. These counts are source references, not a claim
that all remaining readers execute per token. D6 must compare the full campaign
with one unchanged counter and classify its remaining consumers. The new native owner is still under
validation. Other provider/loading boundaries and the full D6 inventory are
unfinished; this section does not claim completion of D5 or D.

The next Python-only candidate captures eight Triton fallback-attention schedule
inputs and four quantized-warmup controls. The schedule is validated once with
its original deferred error gate; shape-dependent prefill/decode tiles remain
dynamic. Effective warp choices replace redundant alias inputs in the hash.
TurboQuant provider choices and comparison policy now share the engine lifecycle:
comparison/dump budgets use the common diagnostic channels, continuation reserve
records and Hadamard tensors use engine attention caches, and each layer retains
its initialized upstream workspace manager. The upstream manager installation
API remains a separate compatibility boundary.

The same candidate migrates both runners' greedy admission, the retained
piecewise/profile graph controls, MTP projection allowlists/weight-sharing rules, shared-expert gates and QSA
calibration inputs. Calibration destinations are fixed; the existing `COLLECTING`
marker still changes corpus shards dynamically. The six quantized loader families
and their remaining provider-selection aliases are still under review.

The focused CPU suite passes 166 tests with three CUDA-only skips and one
CPU-Triton fixture deselection (`attention-cpu-v6.log`). An additional FP16 Triton
operator test covers independent schedules and changed-input capture/replay; its
GPU result is pending. No tensor algorithm or numerical acceptance was broadened.
The unchanged D5 counter records 129 -> 119 raw environment references,
3186 -> 3130 platform references and 2111 -> 2111 model references for this
candidate. Ledger updates accept only total reductions, not additional exclusions.
