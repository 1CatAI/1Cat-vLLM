# Automatic direct I/O for large safetensors checkpoints

With the default loading strategy, checkpoints larger than 90% of available host RAM use direct I/O when the storage supports aligned reads and every participating tensor-parallel rank qualifies. Available RAM includes cgroup-v2 pressure and hard limits. Checkpoints that fit the budget retain the existing mmap or network-prefetch behavior. Startup logs explain why direct I/O was selected or rejected.

The loader reads decoder tensors in coalesced private buffers. Pipeline stages use their actual decoder ranges, tensor-parallel ranks share identical reads over their CPU group, and expert-parallel ranks keep their existing ownership filter. Embeddings, heads, visual tensors and explicitly retained PLE shards stay mapped. Unneeded target tensors are filtered before MTP and PLE loading touches them.

Explicit `--safetensors-load-strategy lazy`, `prefetch`, `eager`, and `direct` retain precedence. An explicitly requested direct strategy reports unsupported storage or ambiguous pipeline ranges as an error; automatic selection retains mapped loading in those cases. No new environment variable is required.

Direct I/O reduces page-cache pressure; it does not guarantee faster loading on a checkpoint that fits RAM. AMD-specific PLE mapping and full-model loading on untested hardware remain dependent on that platform's validation.
