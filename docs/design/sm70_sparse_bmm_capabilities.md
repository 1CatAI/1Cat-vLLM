# Sparse attention matmul selection on SM70

`KernelConfig.sm70_sparse` owns `indexer_decode_cublas`, `decode_bmm`, and
`prefill_bmm`. Each defaults to true; operators decide whether the actual
configuration is supported. These policies add no environment variables.
The existing indexer cuBLAS disable override remains compatible.

The indexer accepts FP16 queries with 128 values per head, a positive multiple
of eight heads, FP32 head weights, and block-major FP8 keys with FP32 scales.
Cache blocks may have padding. Native and flattened uniform request layouts
share one paged gather per request. The gather derives its live bound from the
row lengths on device, so replay masks unallocated graph-bucket tails. Query
row counts are not limited to a fixed speculative width; gather and score
storage must fit the existing indexer workspace budget.
Eager and piecewise calls use their live host key bound. Full-graph serving
retains the paged indexer: a captured cuBLAS key bucket otherwise performs the
same dense work when replayed for a much shorter context. The operator itself
still supports graph capture, including changed row lengths and masked tails.

Sparse decode accepts the packed 448-dimensional E4M3 component plus
64 BF16 RoPE values, including padded cache blocks. If paged QK-D is already
enabled, it retains low-workload shapes where its launch overhead is lower. The
selector uses the paged kernel's number of independent head/key tiles, including
its optimized head grouping, rather than a fixed head or query count. Prefill
uses dense FP16 keys with bounded query passes.
Neither route depends on a model name, tensor-parallel size, concurrency, or
speculative token count. Runtime layout rejection is logged with its reason;
startup reporting includes policy, metadata and hardware rejection.

QK accumulation and scores remain FP32 through the softmax boundary, matching
the existing attention precision. Probabilities retain the existing FP16 HMMA
boundary before PV. Unused slots, including stale indices past each length,
are masked before gathering and their key values are zeroed before PV.
