# Layer weight offload for capacity-limited H3 execution

The original model's attention/MLP FP16 matrices alone occupy 40,076,574,720
bytes globally. A 32-GB V100 cannot hold that complete DiT, even before FP32
protected weights and activations. The explicit `--weight-offload layer`
deployment option stages individual DiT blocks and Qwen text/vision layers.
`H3Config(weight_offload="layer")` is the Python equivalent. The default
remains whole-component staging for the existing TP4 execution path.

The plan partitions the existing immutable host snapshot by actual storage
ownership. Private block storage moves to the GPU before its forward and is
released afterwards. Storage shared with other blocks or outer consumers stays
resident for the component context, preserving offsets, strides and aliases.
The first DiT block's normalization and AdaLN projection stay resident because
cache decision probes may call them outside the block's forward. Adapter
buffers follow their owning block. No weight precision or reduction changes.

Both normal and failed forwards release block storage. The context removes its
hooks on exit, rejects overlapping use and prevents a whole-component load
during layer staging. GPU copy streams/events are reused sequentially from the
original snapshot; block boundaries synchronize. Allocator retention stays
bounded. This mode does not support a fixed persistent FP16 weight-cache list.

All transfers inside sampling remain in the full denoise denominator.
`dit_layer_weight_staging` / `dit_layer_weight_offload` and the corresponding
encoder fields report subsets of the enclosing denoise/encode wall times;
they must not be summed again into request latency. This is a capacity option,
not an automatic or >80-TFLOP/s configuration.

## Development validation

Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8.93, V100 SXM2 32GB.
Evidence: `/data/minimax-h3/sm70-general-20260909/`.

- `layer-staging-cpu-v1.log`: 22 ownership, lifetime, failure cleanup,
  configuration and service checks pass, three GPU cases deselected.
- `layer-staging-gpu-v1.log`: FP16 and FP32 models with column-major weights,
  adapter buffers, cross-component aliases and 65-row tails reproduce
  whole-resident results bitwise across three load/forward/offload cycles.

Full TP1/TP2 H3 generation and peak-memory measurements remain pending. These
operator tests do not establish full-model quality or performance acceptance.
