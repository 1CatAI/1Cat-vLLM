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

`layer-staging-cpu-v2.log` additionally passes 35 checks, including explicit
resident consumers and both CLI modes, with three GPU cases deselected.

Source `aea5a0fc35` completes a TP1 original-weight LightX2V four-step request
on one V100, using column-major matrices and pageable host masters. The first
capacity check uses the smallest legal temporal extent (22 frames) on a
256x448 canvas; it is not a primary performance workload. The full request
passes basic media validation and peaks at 15,473,571,328 allocated GPU bytes.
Denoise takes 126.274887 seconds and the complete request 152.031041 seconds.
Recorded DiT weight loading takes 123.212966 seconds including the initial
3.769427-second resident setup outside denoise. This identifies pageable
layer transfers as the dominant cost, not a useful fast configuration.

The full contract, source/binary hashes, raw latents/RGB/PCM and stages are in
`/home/ymzx/h3-sm70-artifacts-20260909/runs/layer-offload-tp1-original-minimal/`.
Pinned-host full-output comparison, larger TP1 and full TP2 generation remain
pending. Neither the operator tests nor this basic media check establishes
independent official full-model quality or performance acceptance.
