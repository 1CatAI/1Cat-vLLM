# SM70 prefill workspace tuning

This follow-up to the shared-weight/scale memory defaults reduces the persistent
score workspace of the FP32-accumulated Q8000/Q8192 prefill route. The same route
and policy apply to TP1, TP2, and TP4; no tensor-parallel gate is added.

The prefix score block can be selected before worker startup with
`VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS`. It must be a multiple of 8192 in
[8192, 131072]. The candidate default is 16384 (1.50 GiB per device); 24576 restores the
previous 2.25 GiB capacity. End-to-end promotion remains under validation. The score buffer uses
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

Initial matched Graph operator results: 16384 saves 768 MiB per device with
0.49–0.81% latency increase at 128K/256K across Q8000/Q8192. The 8192-token
option saves 1.50 GiB but increases latency by 2.00–2.26%, so it was not selected
as the default. All sampled outputs are finite; FP32 oracle relative L2 remains
0.00054–0.00184. The 24576 override is bitwise identical to the previous binary
on all five tested shapes. End-to-end validation is still pending.

The worker also preserves the complete allocator configuration while temporarily
changing `max_split_size_mb` for model loading. The previous partial settings
update reset user rounding and garbage-collection options and ignored the unified
`PYTORCH_ALLOC_CONF` alias. The post-capture KV capacity suggestion now includes
persistent warmup allocations, as the actual KV budget already does. This corrects
the suggestion without increasing KV allocation or removing the Graph reserve.
