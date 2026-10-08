# Flash-V100 attention

`backend` owns registration, `impl` owns initialization and the forward entry
point with registered feature hooks. It binds methods from `decode`, `prefill`,
`verify` and `debug_compare`;
`state` owns their shared one-shot flags. `ops` owns
native loading, `kv_layout` owns cache views/gathers, and `metadata` owns common
host metadata construction. `spec/` owns speculative feature metadata.
`dense_prefill` owns
dense D256 operators and workspaces; `masks` owns reference visibility masks;
`debug` owns comparisons and tracing. Access mutable helpers/state through the
owning module. The old `flash_attn_v100` module forwards imports and patches.

## Routing and formats

`routing.ROUTE_SPECS` declares stage, codec set, head/GQA dimensions, page
alignment and chunk constraints. The 44 literal counters resolve their names
from this table. Concrete dynamic aliases are declared too; page/partition
and FP8 summary families are diagnostic observers, not extra implementations.
Counter names remain compatible, including historical format-bearing names.

`RouteShape` contains host-visible dimensions only. `RouteContext` carries
operator availability, policy and metadata hints; no device sequence-length
read is introduced. `select_route` preserves candidate priority. Uniform
decode, mixed resident rows and small-query XQA use one admission implementation
with their existing stage-specific hint/window rules. Native operator ABI,
stride, metadata and availability checks remain required; a codec declaration
alone does not prove an operator exists. More detailed grouped contracts are
owned by `ops/sm70_grouped.py`, whose codec contracts preserve FP16/E4M3
limits and expose rejection reasons.

Dense prefill declarations describe prepared FP16 K/V; bridge declarations
describe their quantized input storage. Supporting a codec through a bridge is
different from native arithmetic. BF16 is not declared as native XQA storage.

Fallback counters and logs are separate from `_route_counts`, preserving the
existing `VLLM_SM70_DEBUG=routing` route summary. Optional admission can return
None to try another native operator. Final selection names an explicit fallback;
unknown counter names and undeclared fallback targets fail instead of silently
inventing paths.

## CPU verification

`tests/v1/attention/test_flash_v100_routes.py` generates path × codec × shape
checks from the declarations and compares XQA admission against frozen original
predicates from A0. It covers context boundaries, metadata hint priority, graph
capture, batch enablement, missing operators, windows and partition overrides.
Existing mixed-row tests verify device-length use, grouping and scatter behavior
with CPU operator mocks. These checks do not measure GPU speed or precision.

The campaign proceeds without V100 verification at the user's request. Native
operator output, CUDA Graph replay and timings are unmeasured; do not infer GPU
qualification from the generated structural matrix or the CPU mocks.

## Method composition

The `FlashAttnV100Impl` class, its inheritance and method names stay stable.
Extracted functions are descriptors bound on that class; static methods retain
`staticmethod`. Their typed `self` refers to the core class. State reads/writes
use the `state` module, so legacy patches have one owner. The extracted
comparison super call uses the original class object, preserving its old
`__class__` cell semantics even if a public export is patched.

The method fixture records normalized parent calculation hashes for all 47
methods. Normalize docstring indentation, typed-self annotations, moved state
qualification and the explicit super receiver; other calculation nodes must
match. Mechanical feature hook calls are expanded back into their calculation
bodies, with argument order checked; the same 47 parent hashes still match.

## Speculative metadata ownership

The common metadata builder owns a `SpecMetadataState` through `spec_state`.
It injects immutable `MetadataInputs` and five `MetadataOps` callbacks for the
base builder and common metadata. The state owns persistent draft/small-query
buffers and speculative configuration. It neither imports nor receives the
common builder. The old private calls delegate to this owner for compatibility;
`METADATA_HOOKS` remains an external compatibility adapter, not the production
registration path. The per-request metadata packet also owns feature fields; neither common class
inherits a speculative mixin.

`spec.builder` preserves the extended build options; `spec.tree` owns tree
attachment/restoration; `spec.draft` owns drafting refresh; `spec.verify_metadata`
owns small-query attachment; `spec.smallq_metadata` owns grouped preparation.
The previous `smallq_metadata` path aliases the owner and retains patch identity.
Prepared grouped metadata validates the original common-builder identity from
immutable inputs. Persistent storage belongs to the same workspace across
replays. Fourteen calculation hashes remain identical to the frozen parent.

`SpecAttentionState` owns construction-time feature policy. Common assembly
injects the two native ABI limits and keyword-support probe, and receives the
configured prefill operator. Dispatch policies consume frozen common policy,
metadata or an explicit validation callback. Attention no longer inherits a
feature method mixin or invokes a single-provider hooks registry.

Legacy method names remain delegates at the common assembly boundary; verifier
calculations receive only VerificationConfig and VerificationOps. The ordinary
causality guard bypasses executor construction. Tests inject owned policy and
operators; the trace recorder observes the actual verifier predicate under its
original canonical event name. The proposer-side `SpecFeatureRegistry` selects per-builder implementations by
speculative method. The tree provider suppresses query expansion for tree
verification, the parallel provider consumes prepared grouped metadata, and the
linear provider expands query rows. All preserve explicit cross-method payloads
for existing callers. New features implement `SpecFeature.prepare` and register
a factory in `spec.features.FEATURES`; the common builder has no method names.
Registration does not alter config read timing or own persistent tensors.

## Shared decode strategy

`KernelConfig.sm70_decode_strategy` requests `shared` (default) or the retained
`legacy` strategy. The implementation captures the effective strategy once.
E4M3 D256/GQA6 single-row XQA requires native shared-strategy revision 1;
older artifacts select and count `decode_strategy_legacy_revision` explicitly.
Disabled or unavailable XQA does not produce an irrelevant strategy warning.
FP16/E5M2 retain their existing planning.

The shared E4M3 shape hint is the FP16 hint: page784 plans a p256 envelope,
other admitted aligned pages use the ordinary context planner (p256 below
32K, p1024 at/above 32K). Batch/small-query/explicit partition restrictions
remain in the existing native contracts. The standard native launch/reducer
already accepts `PARTIAL_T`; E4M3 continues using FP32 partials and uint8 cache
storage. Shared planning changes partition boundaries and disables the legacy
p64 envelope/automatic wave choice for these single-row calls. Numerical
outputs and performance can change. `--kernel-config
'{"sm70_decode_strategy":"legacy"}'` restores the retained policy.

This is a strategy change with CPU planning/ABI/workspace and host C++ syntax
evidence. GPU output error, route traces and timing have not been measured;
no speedup or numerical-equivalence claim is made. The older p256 measurement
alone does not describe the current p64/p256/wave source, and wave admission
also depends on the forwarded context bound. Retain those experimental paths.

## Per-request speculative metadata

Common metadata owns one `spec_state` packet. The Triton builder's result is
adopted as the existing FlashAttnV100Metadata subtype in place: object identity,
common fields and tensor references remain unchanged. Old field access forwards
to the packet; feature names are declared only by its Spec owner. Existing raw
fields migrate without tensor copies. Shallow copies get independent packets
while sharing tensors, and packets do not retain the enclosing metadata.
Older duck-typed external metadata remains accepted at the compatibility view.

## Prefill owner boundary

`prefill.PrefillExecutor` owns per-layer policy/geometry, explicit
`PrefillDriverOps` and the existing `V100Workspace`. Its calculations never
receive Impl. Common assembly constructs the owner and preserves legacy method
facades/instance overrides. The inner candidate executor receives native and
admission callbacks from this owner. Snapshot construction is per legacy call,
so later operator or policy changes affect subsequent calls without changing an
already created owner. Speculative callback names and capabilities belong to
`spec.attention`; no entire attention implementation is passed there.
