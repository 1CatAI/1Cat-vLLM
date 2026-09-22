# SM70 prefill workspace tuning

This follow-up to the shared-weight/scale memory defaults reduces the persistent
score workspace of the FP32-accumulated Q8000/Q8192 prefill route. The same route
and policy apply to TP1, TP2, and TP4; no tensor-parallel gate is added.

The prefix score block can be selected before worker startup with
`VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS`. It must be a multiple of 8192 in
[8192, 131072]. The current build default remains 24576 until the memory,
quality, and speed gates establish a smaller default. The score buffer uses
`block_tokens * 8192 * 6 * sizeof(half)` bytes per device and is shared by the
Q8000 and Q8192 specializations. Graph addresses remain fixed for the worker
lifetime. Do not change this setting after workspace initialization.

Smaller prefix blocks change the online softmax merge order. FP32 accumulation,
causal masking, the tail implementation, and the attention route are preserved,
but numerical and end-to-end validation are required before changing defaults.

## Validation record

Integration base: `8d5d82334d0f0b32153fa2c66f9ce587f438ed7c`.
Baseline: 27B NVFP4, DFlash2/7, target E4M3 KV, draft FP16 KV, SM70 V100,
CUDA 12.8, Torch 2.10.0+cu128, 185 W, CUDA Graph enabled. No wheel build.

The preceding audit found a 2.25 GiB score buffer and approximately 2.66 GiB
of persistent warmup allocations per device. Expandable allocator segments
were rejected as a universal default: they helped TP1's estimated KV budget
but reduced TP2's budget and increased TP2 request-resident memory.

This scope is separate from prefill workspace growth and decode graph pointer
lifetime fixes in PRs 660/661. It changes the native Q8000/Q8192 score block.
GPU 0–3 are reserved for this task; the unrelated service on GPU 4–7 is not
modified. Task-owned source, build, compiler caches, and raw results are kept
under `/data/minimax-h3/task-cache/memory-workspaces-20260922`.

Validation is pending; no speed or quality acceptance is claimed yet.
