# INT8-G64 KV codec integration contract

This is the interface handoff for the INT8-KV implementation, not an INT8
implementation or a claim of existing route support. The architecture stack
through A5 supplies shared codec admission, native reader traits and visible
fallbacks. `Int8G64Codec` does not exist yet.

Adding one Python codec and one CUDA traits specialization lets operators
reuse format-independent scheduling and attention math. In the current source
it does **not** automatically enable a runnable route: scale storage, writes,
cache accounting, native launch bindings and capability registration must be
implemented first. The work below belongs at those shared seams, rather than
in copied INT8 variants of decode, prefill and verify.

## Format and ownership

Use a distinct proposed canonical name, `int8_g64`. Do not alias it to the
existing `int8_per_token_head`: that mode has one FP32 scale per token/head,
whereas G64 needs one scale for each 64 channels of each token/head. Do not
alias it to FP8 just because both payloads occupy one byte.

The following separate-scale layout is a proposed initial contract for the
INT8 agent to finalize. It is not a currently accepted allocation or launch
ABI. An inline-scale alternative must declare its physical strides explicitly.

| Component | Proposed shape and meaning |
| --- | --- |
| K/V payloads | Two signed-int8 tensors `[blocks, page, kv_heads, head_dim]`. |
| K/V scales | Two FP16 tensors `[blocks, page, kv_heads, head_dim / 64]`. |
| Group | 64 consecutive channels of one token and one KV head. |
| Initial head dimensions | 64, 128, 256; reject other dimensions until padding semantics are declared. |
| Dequantization | Convert signed payload and group scale to FP32, multiply, then use the operator's declared staging/accumulation precision. |
| Addressing | Payload and scales use the same physical page IDs, token position and KV head. Strides are explicit; element offsets are not interchangeable with byte offsets. |

Specify rounding, saturation range, zero-group scale, finite-input policy and
scale precision in the writer/reference contract before implementation. Do not
reuse E4M3 bit reinterpretation or silently apply scalar `k_scale`/`v_scale`
instead of group scales. Retain global scalar fields only if their additional
meaning is explicitly defined; otherwise launch them as unity in that ABI.

With the proposed layout, K+V bytes per page before alignment are:

```text
2 * page * kv_heads * (head_dim + (head_dim / 64) * 2)
```

For D256 this is 528 bytes per token/head for K+V, versus 1024 for FP16.
These are storage calculations, not measured memory savings or speedup.
Scale buffers, page padding and scratch allocations must also be budgeted.
The current per-token-head accounting formula cannot be reused unchanged.

| Responsibility | Current owner / required seam |
| --- | --- |
| Canonical name and reference codec | `vllm/v1/attention/kv_codecs.py`; `KVCodec.dequantize(cache, scale: float, out_dtype)` currently has no group-scale argument. Extend the format contract centrally. |
| Config dtype registration | `vllm/config/cache.py::CacheDType`; currently no `int8_g64` spelling. |
| Quantization mode and page bytes | `vllm/v1/kv_cache_interface.py::KVQuantMode`, `AttentionSpec.page_size_bytes`; add G64 accounting and allocation semantics. Unknown names currently resolve to `NONE` here. |
| Reader | `flash-attention-v100/kernel/kv_codec_traits.cuh`; a G64 specialization needs scale context as well as signed payload reads. |
| Writer and cache binding | Cache writer/operator and backend storage binding; bind both scale planes and preserve logical page ownership. Shared storage/writer work in #1048 is a separate dependency, not already integrated wholesale into this stack. |
| Native registration | Shared format codes and dtype dispatch in the native launchers, then the packaged Python interface; advertise only actual implemented readers. |
| Route admission and telemetry | `flash_v100/routing.py::RouteSpec`, XQA admission and `ops/sm70_grouped.py::GroupedContract`; keep implementation scheduling shared. |

## CUDA reader contract

The current traits supply `storage_type`, `quantized`, `element_bytes`,
`vector_elements` and `vector_bytes`, plus scalar and eight-value readers.
FP16 and FP8 specialize storage while inheriting `KVReader`. That reader
intentionally rejects other format codes at compile time. A G64 specialization
must provide its own signed/group-scaled reader; extending the storage typedef
alone is insufficient.

For separate scales, introduce one shared reader context/ABI carrying scale
base pointers and strides. Pass it from the launch boundary to tile reads.
`load_half8(cache, physical_offset, column, lut)` currently has no separate
scale pointer; `half8_from_packed(raw, lut)` has no token/head/group location.
Those signatures cannot recover a separate G64 scale from eight raw bytes.
Keep the existing signatures as compatibility forwarding entry points for the
old formats when extending the reader interface.

For both regular and paired panel loads, apply the scale belonging to each
eight-value vector. A sixteen-value pair can cross a group boundary when its
starting channel is not aligned; declare and validate alignment or read both
scales correctly. The E4M3 conversion LUT is unrelated to group scales.
Scalar reads must select the same group as packed reads. Do not copy attention
loops to solve either case.

Scale placement and rounding are an operator contract. XQA currently stages
unscaled half values and applies the scalar K/V scale elsewhere. Per-group
scales cannot simply use that scalar placement. Define staged precision and
partial/reduce precision before advertising the G64 route. `quantized=True`
does not imply a valid numerical contract for every operator.

## Route reuse matrix

All entries below are **currently unavailable for `int8_g64`**. "Reusable"
describes existing shared machinery after the listed integration work; it is
not permission to add the codec to every support set preemptively.

| Family | Reusable machinery | Remaining G64 integration |
| --- | --- | --- |
| Scalar paged decode (`decode_scalar_paged`) | Page walk, masking, attention math, partition/reduce framework. | Group-aware reader context, launch dispatch/ABI, dtype capability and precision contract. |
| XQA q=1 decode (`decode_xqa_paged`) | Existing tile schedule, page specialization and partition planner. | Scale-aware scalar/vector/paired reads; admit the codec once at shared XQA storage/capability seams. `_xqa_kv_codec` currently allows only FP16/E4M3/E5M2. |
| Mixed decode rows and small-query XQA | The same XQA reader and row metadata machinery. | Qualify their separate batch, query-row, page and precision capabilities; q=1 support does not imply these ABIs. |
| Grouped FP32 verify | `grouped_fp32_reason` shape/metadata validation and shared family organization. | A codec provider/contract plus native grouped support and revision declaration. Current providers admit FP16/E4M3 only; adding an XQA reader does not create a grouped provider. |
| General paged prefill (`prefill_prefix_paged`) | Shared causal/window/page scheduling and native reader machinery. | Bind G64 scales, launch its reader, then declare native codec support centrally. Both native and Python admission currently restrict the supported formats. |
| Dense / Split-D / 75T FP16 prefill | Existing FP16 computation can consume an explicit dequantized workspace. | No direct G64 support follows from the traits. A new group-aware conversion fallback needs its own capability and counted declaration. The existing FP8 bridge admits E4M3/E5M2 only. |
| BF16, anchored-window and specialized speculative routes | Shared metadata where their existing contracts apply. | Explicit feature/layout/precision qualification; no inherited support from a dtype name or storage size. |

`RouteSpec.shape_reason()` and the declaration-generated coverage matrix are
format-independent and can check the new codec after registration. Actual
dispatch additionally requires operator capabilities and valid tensors. Add
support to the common family declaration once, preserving shape restrictions;
do not add an `if int8_g64` branch at each `_record_route` site. Unsupported
combinations must reject with a reason or use a separately declared fallback.
Preserve existing route labels for comparable telemetry.

## Minimal delivery and validation order

1. Freeze layout, writer/reference rounding and byte accounting. Validate page
   and scale addressing, zero groups, range endpoints and independent K/V scales
   on CPU. Declare tails/padding rather than guessing them.
2. Add the codec descriptor, group-aware reader context/traits, writer and
   shared launch binding. Preserve old-format behavior and include paths;
   build from shipped headers without external private kernel DSOs.
3. Register one family first (scalar decode or q=1 XQA), including truthful
   native capabilities and fallback rejection. Use declaration-driven tests
   for admitted and rejected codec/shape combinations.
4. Extend mixed rows, verifier and prefill only after their own launch contracts
   are implemented. Test non-contiguous pages, page boundaries, reordered page
   IDs, multi-KV-head scales and group-boundary vectors.
5. Keep CPU/static evidence distinct from GPU numerical error, route traces,
   CUDA graph replay and matched operator timings. GPU measurements are omitted
   in this architecture campaign at the user's request; no INT8 route or
   performance is currently qualified here.

The intended result is one G64 format implementation shared by operator
families, with capability declarations exposing exactly what is implemented.
It is not a format rename that silently opts into FP8 scale semantics.
