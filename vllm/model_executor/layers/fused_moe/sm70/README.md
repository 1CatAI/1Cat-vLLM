# Shared SM70 MoE methods

`base.py::Sm70MoEMethodBase` owns the persistent/overflow buffer lifecycle.
FP8 is its first consumer. The format adapter supplies its legacy attribute
prefix, persistent capacity and empty weight/scale dtypes. Shapes, allocation
order, scratch policy and overflow reuse preserve the previous FP8 contract.

`quantization/utils/sm70_layer_workspaces.py::LayerWorkspaceView` gives the
base a format-independent view of the layer's buffers. The layer remains the
sole owner of those tensors. Existing `_fp8_buf_*` attributes, rebinding and
persistent addresses stay visible; no second buffer dictionary or global
registration is introduced. The existing opaque linear-workspace registry is
unchanged.

The legacy FP8 `_allocate_buffers` and `_get_buffers` methods forward to the
base. They retain their signatures; the original module's persistent-capacity
constant is still read when allocating. The shared `_apply_moe` pipeline now
owns permutation, stage selection, indexed/compact W13, W2 and weighted
reduction. `Sm70MoEWeightCodec` supplies `prepare_weights`, `gemm_w13` and
`gemm_w2`, together with transitional capability/policy/log bindings.
`quantization/fp8_sm70_weight_codec.py` binds the original FP8 native stages
and owns preparation. Bindings resolve through the legacy module so patches,
logger identity and helper exports retain their owner. The `layer` argument
lets subsequent codecs bind their own scale/zero descriptors.

Legacy single-token compact and comparison implementations remain callbacks
on the format adapter pending subsequent B1 scopes. AWQ/NVFP4/MXFP4,
GGUF and skinny MoE have not migrated yet. Their differing capacities,
temporary capture buffers and layouts must be preserved when they join.

CPU differential tests execute the frozen parent methods and the new base for
24 combinations of top-k, capacity boundary and scratch policy. They compare
tensor descriptors and every persistent-storage alias. All FP8 method bodies
match the parent AST after expanding mechanical wrappers and codec bindings.
Additional CPU tests compare all native-call argument descriptors, final
stub outputs and exact log messages across single-token indexed/compact,
batched/per-expert/dense and empty-input routes. Weight-preparation tests
compare parameters, prepared layouts and removal of original load tensors.
These use CPU native-op stubs, not numerical CUDA GEMM implementations. A CPU
`torch.compile(..., backend="eager", fullgraph=True)` check observes rebound
storage; retained opaque-workspace AOT reload checks also pass.

This is skeleton extraction, not a new native MoE kernel or speedup. GPU output,
CUDA graph replay and timings remain unmeasured at the user's request. The
full B1 shared stage/weight-codec skeleton remains in progress.
