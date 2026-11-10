# SM70 E4M3 attention capacity from the model window

## Purpose

Extend the existing grouped long-context and compact scalar-tail E4M3 decode routes to the configured model window. These providers already ship in the FA2 extension; this change extends their usable context range and validates it.

The existing long-route qualification is 262144 tokens. A block table describes the engine's allocated window even when the current request is short. Serving with a larger window, for example Yarn at `--max-model-len 1048576`, can exceed provider page-table or workspace checks during graph warmup. Existing grouped optimizations therefore cannot safely serve that configuration as qualified.

## Capacity and native compatibility

`sm70_grouped_long.py` resolves `model_config.max_model_len` and queries two independent native revisions: grouped dynamic context and scalar workspace. Revision 1 admits the served window within the native integer-addressing limits. Older binaries retain their known 256K capability. New Python code alone cannot expand an older native provider's contract.

The scalar provider must cover the grouped graph's entire context bound. A mixed build with an insufficient scalar ABI declines the scalar-tail route. External manifests retain their declared capabilities. Missing model configuration retains the conservative built-in qualification.

Page-table admission accepts at most `ceil(context / page_size)` columns. Only E4M3 calls with a qualified long-context graph descriptor and the dynamic native ABI can exceed the ordinary E4M3 table ceiling. FP16 negotiates its own served-window capacity with an independent native revision, using the same window validation and page rounding. Other codec qualifications retain their existing boundaries; FP16's shapes, pages and batch limits are unchanged.

## Workspace and native launch

The scalar provider allocates `ceil(max_model_len / 1024)` partitions before profiling and graph capture. Its cache key includes the resolved capacity and device. Serial layers and captured target graphs reuse the same FP32 numerator, maximum and sum buffers.

The native scalar entry validates shape, device/layout, integer bounds and workspace coverage of the page-rounded table. Partition indexing uses `grid.x`, removing the 65535-partition `grid.z` restriction. Reduction processes weights in fixed-size shared-memory tiles while preserving ascending FP32 accumulation order and stable softmax normalization.

The grouped kernel retains its 80 splits and arithmetic. Its E4M3 table check uses the structural integer limit; Python admission enforces the negotiated provider capacity. The existing operator names and execution family are reused.

## Graph behavior and scope

The existing graph manager consumes the family owner's context contract. Capture fixes workspace and launch grids for the served window. Replay uses device lengths without allocating or copying lengths to the host. No context buckets, scheduler policy or model-specific runner logic are added. Existing route policy and external manifests retain their meanings. No new configuration or environment variable is required.

## Validation

- CPU: model-window resolution, independent native revisions, mixed builds, page rounding, invalid bounds, workspace cache isolation and ordinary fallback.
- GPU: FP64 references over the same quantized KV, q1 through q8, multiple heads and request batches, page/partition boundaries, zero lengths and restored live graph inputs. Numeric cases extend through 2M tokens.
- A separate overprovided-workspace launch uses 65537 partitions and a short live sequence to exercise grid/reduction coverage without a long KV cache.
- Full engine: Qwen3.8-27B NVFP4, DFlash block 8, TP4 on four V100s, E4M3 KV, a 1M served window and 1024 generated tokens per measured request. Performance compares the ordinary path with the existing optimized routes enabled under the expanded context contract. The reported context range is 32K through 512K; kernel coverage beyond it is validated separately.

Production edits occupy two CUDA units and three existing Python family owners. The family README describes the boundary. The regression fixture updates source fingerprints while retaining the original legacy admission predicate bodies.
