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
owned by the attention ops and will be consolidated in the grouped family.

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

## Speculative metadata hooks

The common metadata builder inherits the compatibility method surface from
`spec.hooks.SpecMetadataMethods` and invokes the frozen `METADATA_HOOKS`
registration at initialization, common attachment and capture preparation.
`spec.builder` owns feature selection and the legacy extended build signature;
`spec.tree` owns tree attachment/restoration; `spec.draft` owns persistent graph
buffers and drafting refresh; `spec.verify_metadata` owns small-query verifier
buffers and attachment. `spec.smallq_metadata` owns device/grouped preparation.
The previous `smallq_metadata` path is a module alias to that owner, preserving
patch identity. Internal imports use the owner path.

The generic `metadata.py` contains no family names. Its prefix-anchored and
decode shape/partition metadata remain local. Speculative methods preserve
the old names and arguments, including positional build options. Their super
calls continue after the feature mixin so the Triton builder executes once.
Fourteen extracted calculation bodies match frozen parent hashes; CPU tests
exercise tree restoration, hook ordering, capture length guards and persistent
buffer addresses across refreshes. Native metadata kernels remain unmeasured.

`spec.attention.ATTENTION_HOOKS` registers typed callbacks for scalar-tail
initialization, verifier ABI/policy, prefill wrapper policy, feature contract
validation, XQA exclusions, explicit fallback and capture route accounting.
`SpecAttentionMethods` preserves historical method/field names for external
callers. Common impl, metadata and backend entrypoints contain no family names;
the existing route names and compatibility exports remain intact. CPU forward
tests cover unsupported-layer rejection, allowed fallback and non-causal
capture, while checking the original route labels and base call count.
