# SM70 memory reuse defaults

Status: candidate under validation, based on main `949728e891`.

The 27B NVFP4 DFlash2 profile retained two 4-bit layouts, per-layer FP16 and
E4M3 scales, and separate Q8000/Q8192 score workspaces. The TP2 baseline loaded
19.539 GiB per GPU and left only 1.303 GiB for KV at memory utilization 0.85.
Sharing the weight layout alone reduced model loading to 13.620 GiB.

This change defaults shared QPN2/TurboMind codes and compact scales on for
already-compatible QPN2 projections. Explicit `VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT=0`
and `VLLM_SM70_NVFP4_QPN2_SHARED_SCALES=0` retain rollback. The existing QPN2
numerical contract, layout gates, and FP32 accumulation remain unchanged.
TurboMind fallback scales now use scratch retained per device, CUDA stream and
matrix size, matching its existing GEMM scratch lifetime. Each call restores
its layer's scales before use. Larger CUDA graph capture sizes are admitted.

The Q8000 and Q8192 native attention families share a maximum-sized score
buffer and completion event. A common host lock plus CUDA event orders reuse
across their internal and caller streams. The tail aliases the score buffer by
default; `PREFIX_TORCH_SERIAL_TAIL=0` restores concurrent tail allocation when
memory permits. Query, key and reduction buffers remain shape-specific. No
attention block size or reduction order was changed.

SM70 V2 no longer silently disables graph memory budgeting. Before KV
allocation, it reserves one measured steady-state activation peak for graph
pools. `VLLM_V2_CUDAGRAPH_MEM_MIB` explicitly overrides that estimate, including
zero; `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` disables graph budgeting.
This estimate is not a hard ceiling for the CUDA caching allocator. Validation
must compare reserved and actual capture memory, not just startup success.

Acceptance contract: V100 SXM2 32GB, 185W, CUDA12.8, Torch2.10.0+cu128,
Qwen3.8-27B QUASAR NVFP4, TP4, DFlash2/7, target E4M3 and draft FP16 KV,
context262144, chunk8192, max sequences32, memory0.85, CUDA Graph. Matched
standard vLLM bench C1/2/4/8/16/32, long retrieval32K/128K/exact262144 boundary,
and 32 MBPP questions use the same requests and full natural answers.
No wheel builds or enforce-eager service benchmarks.

Evidence pending: native build, exact scale/graph replay tests, shared-score
lifetime tests, per-card memory deltas, and matched endpoint acceptance.
A draft PR is not a passed promotion gate. Previous r32 tail-only operator
result saved396 MiB with identical output and +0.592% graph latency; it is not
an end-to-end claim. The separate full-FP32 75T and35B migration targets remain
open.
