# Flash-V100 attention ownership

`FlashAttnV100Impl` is the vLLM integration and composition boundary. It captures
construction policy, loads operators and assembles executors. Executors consume
configuration, native callbacks and owned workspaces; none imports or receives
an entire Impl. The legacy methods delegate to these owners, including instance
overrides used by FlashInfer-SM70.

```text
backend -> impl (assembly and forward entry)
               -> decode.DecodeExecutor
               -> prefill.PrefillExecutor -> prefill_candidates
               -> spec.VerificationExecutor
               -> debug_compare.ComparisonExecutor

executors -> plan/routing, config, workspace, kv_layout, masks, native ops
spec      -> plan, config, workspace, kv_layout, feature-owned metadata
plan/events -> synchronous subscribers in debug and spec/diagnostics
```

`tools/sm70/flash_v100_audit.py` checks every package import, including local
imports. Its small, explicit `COMPOSITION_EDGES` list describes constructor
wiring and the `verify` compatibility facade; it is printed separately, not
silently omitted. Execution/Spec imports cannot acquire new reverse edges.
There are no package import cycles. Dense-prefill primitives are operators in
the dependency rules. Debug formatting/configuration lives in `plan/diagnostics`;
logging consumes synchronous events. The two prefix-comparison subscribers keep
their order and propagate exceptions at the original execution point.

## Owners and public operations

- `config.V100AttnConfig`: frozen per-layer policy. `ConfigField` preserves old
  attribute reads/writes without a second policy store. Engine execution borrows
  `AttentionConfig.flash_v100.options` and diagnostic policy captured at
  initialization. No-config calls retain a separate legacy adapter; the D
  parameter inventory classifies its reads separately from engine execution.
- `workspace`: per-layer decode cache, per-builder persistent graph buffers,
  mixed-row plans and shared allocation helpers. The three original prefill
  invalidation sites remain at the same branch boundaries.
- `kv_layout`: paged cache splitting, contiguous views, gathers and dequantizing.
  Public operations such as `split_paged_kv_cache` retain their previous
  signatures. Historical underscore names are live aliases of the same owner.
- `ops` and `dense_prefill`: native ABI probes, loading, shape admission and
  dense/paged attempts. An attempt returning `None` still precedes the next
  candidate; preparation can allocate or gather.
- `spec`: feature policies, verifier, draft/small-query metadata and diagnostics.
  Common request metadata owns a `spec_state` packet. The builder owns a separate
  `SpecMetadataState` and injects immutable inputs plus explicit common callbacks.
  Neither common class inherits a speculative mixin.
- Runtime resources and diagnostics: engine-owned caches, workspaces and
  observation budgets are borrowed by forward contexts. `state.py` remains a
  compatibility proxy; independent standalone calls retain their legacy owners.
  Multi-part diagnostic completion stays at the original checkpoint. Do not
  share mutable capture buffers or budgets between engine instances.

The functions promoted to public owner operations are the actual execution
bindings. `compat.OwnerAliases` forwards old-name reads, writes and deletes;
copying a function binding would break external monkeypatches. The deprecated
`flash_attn_v100` module emits `DeprecationWarning` and forwards to these owners.
New code and tests should inject `DecodeOps`, `PrefillDriverOps`, `PrefillOps`,
`VerificationOps` or metadata callbacks. The shim remains covered by strict
owner/patch-consumption tests for external compatibility.

## Inputs, numerical semantics and fallback

Executors receive prepared query/KV tensors, metadata, policy and injected
operations. Storage codecs, converted arithmetic tensors and destination buffers
are distinct contracts. Existing strides, page/partition limits, scale semantics
and FP16/FP32 accumulation/reduction boundaries remain binding. Native policy
and resource ABI probes run at initialization; unsupported explicit policy on an
old binary must fail rather than be silently ignored.

An admission predicate does not prove an operator ran. Missing operators,
unsupported layout and native attempts that decline retain the existing ordered
fallback. `ROUTE_SPECS` declares accounting and structural admission; its table
order is not executor candidate priority. Resource growth must preserve captured
address lifetimes and the existing AOT reload/address-resolution contract.

## Adding a candidate

1. Define the route in `routing.ROUTE_SPECS`, including stage, codec and shape
   contracts. Preserve published names; diagnostics are separate from selection.
2. Implement `admit(request)` as a pure predicate and `run(request, record)` as
   execution. A preparation/native attempt that can decline belongs in `run`.
3. Add it at the intended position of the relevant candidate tuple. Decode
   priority is paged-prefill, dense-cache, dense-reference, scalar/XQA dispatch;
   per-sequence prefill keeps BFLA, split-D, contiguous BHMD/dense, bridge,
   split-KV and paged fallback order.
4. Inject its operations from assembly and test the actual executor. Prove
   decline/fallback effects, capture restrictions and output destination before
   changing the immutable behavior contract in a separate behavior-change PR.

## Adding a codec or feature

`vllm/v1/attention/kv_codecs.py` is the single Python codec API. A route declaration
is not native support: storage dtype, scales, ABI revision and native admission
must all agree. Bridge input storage and prepared FP16 arithmetic are distinct.
Do not introduce a second registry. QSA direct-native calls and the RoPE layout
imports remain outside this refactor.

A speculative provider implements `SpecFeature.prepare` and registers a factory
in `spec.features.FEATURES`; the proposer supplies the speculative method.
Persistent tensors stay in the builder's workspace. Keep feature fields and
model names inside `spec/`; the common package allows historical names only in
the `ROUTE_SPECS` literal. Prepared metadata continues checking the originating
builder identity. Capture uses persistent buffers and stable storage pointers.

## Validation and scope

Run the immutable CPU trace, source-calculation oracles and executor tests:

```bash
.venv/bin/python -m pytest -q --import-mode=importlib \
  tests/v1/attention/test_flash_v100_trace_golden.py \
  tests/v1/attention/test_flash_v100_impl_composition.py \
  tests/v1/attention/test_flash_v100_spec_metadata.py
.venv/bin/python -m tools.sm70.flash_v100_audit
.venv/bin/python tools/pre_commit/check_layering.py
```

The 813-case trace freezes native call ordering, write destinations, candidate
outcomes, invalidation and persistent storage identity. Its zero-valued native
stubs do not prove GPU precision or performance. The original 47-method and
14-metadata calculation fixtures are unchanged: the oracle expands current
helper bodies and validates their exact argument bindings before hashing.
`tools/sm70/route_parity.py` compares requested workload, routes and greedy token
IDs; `op_parity.py` compares real native outputs. Final recorded commands/results
are in `docs/design/architecture/flash_v100_refactor_progress.md`.

The user deferred DDTree algorithm work and its GPU gate. Its 321-line
`spec/verifier.py:tree_prefill` is the sole explicit length exception; the audit
reports both the whole-package maximum and the non-deferred maximum (200).
All other functions are capped at 200 lines, and the entry forward at 150.
This cleanup changes Python ownership and diagnostics, not CUDA arithmetic or
performance policy. It makes no new throughput claim or 35B speed qualification.

The [codec contract](../../kv_codecs.README.md),
[architecture overview](../../../../../docs/design/architecture/README.md) and
[Phase D report](../../../../../docs/design/architecture/sm70_phase_d.md) cover
current ownership and initialization compatibility. Historical A3 test counts
here are evidence at their recorded revision, not the current test count.

`KernelConfig.sm70_decode_strategy` still supports the previously introduced
`shared` default and the retained `legacy` setting. Their numerical/performance
differences and native revision checks are independent of this ownership cleanup;
no experimental route is deleted by A3.
