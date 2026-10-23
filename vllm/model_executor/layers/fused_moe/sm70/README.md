# SM70 MoE execution and ownership

Format methods adapt the existing FusedMoE load/apply lifecycle.
`Sm70MoEMethodBase` binds resolved policy and native availability; the existing
`quantization/sm70_moe_router.py` selects `Sm70MoeRoutePlan`. This package owns
shared stages, codecs, diagnostics and resource helpers, not a second selector.

```text
resolve policy / model contract -> prepare weights -> select plan
  -> prepare routes/input -> W13 -> activation -> W2 -> weighted reduction
```

## Stages and numerical contracts

`stages.py` and `single_token.py` share AWQ/FP8 execution. `fp4_stages.py`
shares NVFP4/MXFP4 sequencing. `Sm70MoEWeightCodec` and `Fp4MoECodec` bind
prepared layouts and native operations; they do not reinterpret strategy flags.
`STAGE_BINDINGS` and `FP4_STAGE_BINDINGS` describe covered stages, layouts and
arithmetic. Fused W13/activation or W2/reduce bindings consume those stages once;
unfused plans retain the original intermediate rounding.

Token rows, top-k IDs/weights, offsets and prepared weight pointers must satisfy
the selected plan. Outputs retain the existing token/hidden layout. Indexed,
compact, grouped, direct and split-K variants retain native shape limits and
accumulation/FP16 boundaries. Dynamic M dispatch stays at its current boundary;
initialization must not freeze the wrong prefill/decode branch.

## Policy, resources and compatibility

`KernelConfig.sm70_moe` owns per-format policy and explicit native projection.
Legacy aliases resolve at initialization; common stages consume modes without
reading environment settings. Parent/native conflicts retain validation.
Missing operators and unsupported contracts follow the existing selector's
fallback; binding declarations do not guarantee a loaded or qualified operator.

Layers retain weights and prepared state. `workspace.py` and `fp4_workspace.py`
reuse existing layer workspace/address-resolution mechanisms: borrowed buffers
must outlive capture/replay, capacity growth and AOT reload. Weight replacement
and release invalidate the corresponding prepared state. Diagnostics borrow the
engine's owner and retain historical labels.

Native packed dense/grouped descriptors share `csrc/sm70_turbomind/ops/packed_gemm.h`.
The native `gemm_runtime.cpp` owns workspace, tuning and prepared FP16 caches;
consumers borrow views with the existing engine/device/stream lifetime. Format
wrappers retain codec types, grouping and numerical epilogues.

GGUF keeps mixed formats and separate gate/up preparation. Skinny keeps modular
MoE integration. They share applicable contracts, stages and diagnostics, not an
interchangeable codec interface with all four formats. Clamping, interleaved
weights and different reductions remain explicit contracts. See the
[B integration record](../../../../../docs/design/architecture/sm70_phase_b.md).

Old imports and path names forward to canonical owners. New formats add codec
preparation, bindings and supported-plan tests; new paths add a declaration,
explanation and fallback to the existing selector. Neither copies a complete
`apply` nor calls back into an old format module from common stages.

## Minimal validation

From the repository root:

```bash
.venv/bin/python -m pytest -q tests/quantization/test_sm70_moe_mainflow.py \
  tests/quantization/test_sm70_fp4_mainflow.py \
  tests/quantization/test_sm70_moe_load_cache_release.py \
  tests/tools/test_sm70_path_explanations.py
.venv/bin/python tools/sm70_route_snapshot.py --category moe
```

These tests/snapshots cover stage composition, call ordering, resources and
static selection. GPU output, graph replay and timing require affected operator
tests. Keep observed traces separate from predicted operators; operator evidence
is not a model throughput measurement.
