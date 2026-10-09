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
| D2 GDN and speculation | Validated [#1143](https://github.com/1CatAI/1Cat-vLLM/pull/1143) | CPU isolation/compatibility, 17 GPU operator cases and matched A/B passed; merge after final self-review. |
| D3 diagnostics | Pending | Consolidate filters/budgets/dumps and remove engine state from global dictionaries. |
| D4 attention | Pending | Connect backend, standalone package and versioned native policy; isolate workspaces. |
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
[Raw rounds and artifact contract](../../../benchmarks/results/phase_d2_policy_operators.json)
include extension identity and source-freeze details.

Diagnostics remain D3 scope. Provider warmup controls and the paused
`EMPTY_CORE_OUT` warning remain D5/D6 scope; the latter does not change current
allocation behavior. Graph/piecewise attention policy belongs to D4. Existing
DDTree execution and upstream Mamba scheduler controls are not claimed as
migrated here.
