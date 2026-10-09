# Flash-V100 attention

`backend` owns registration, `impl` owns initialization and the forward entry
point. It binds methods from `decode`, `prefill`, `verify` and `debug_compare`;
`state` owns their shared one-shot flags. `ops` owns
native loading, `kv_layout` owns cache views/gathers, and `metadata` plus
`smallq_metadata` own host/device metadata construction. `dense_prefill` owns
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
match. Feature metadata registration and removal of family names from forward
orchestration remain the next independent scope.
