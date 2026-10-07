# Flash-Next host-backed KV on SM70

## Scope and current status

Keep the IQ3_S target and FP16 MTP weights on GPU, the packed PLE tables in
pinned host memory, and move persistent attention KV to host memory. This
document specifies the next implementation; host-backed active decode is not
implemented or benchmarked yet. Small, bounded GPU buffers remain necessary
for attention, recurrent updates and CUDA graph replay.

The measured TP4 capacity route on four 16 GiB V100s uses 14.281 GiB of model
weights per rank, approximately 0.95 GiB of non-PyTorch runtime memory and a
0.40 GiB cache budget. It runs short C1 requests but cannot sustain the measured
C4 cohort. These measurements use eager execution and are not the FULL-graph
latency baseline. PLE storage is 26.82 GiB on the host.

## Community implementations reviewed

| Implementation | Relevant mechanism | Limit for this workload |
| --- | --- | --- |
| [vLLM OffloadingConnector](https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md) | Pinned CPU block pools, asynchronous DMA, cache keys, group-aware prefix reuse | Promotes reusable blocks back to GPU; does not make active attention consume host-resident history |
| [SGLang HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design) | Layer-wise load overlap, host page layouts and write-back scheduling | Required attention data is loaded into GPU before computation |
| [SGLang HiSparse](https://docs.sglang.io/docs/advanced_features/hisparse_guide) | Complete host KV plus bounded device slots; native sparse selection, hit detection, miss gathering and asynchronous backup | Model/backend admission and speculative decoding restrictions prevent direct Flash-Next MTP4 reuse |
| [LMCache](https://github.com/LMCache/LMCache) | External KV storage, retrieval and transfer integration | Storage integration alone does not supply a host-reading Flash-Next attention kernel |
| [KTransformers](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/long_context_introduction.md) | Chunked GPU prefill and CPU sparse attention at decode | Its post-hoc sparse selection is a different numerical contract; do not substitute it for QSA selection |
| [FlexLLMGen](https://github.com/FMInference/FlexLLMGen) | CPU attention and overlap of heterogeneous storage/computation | Throughput-oriented design; CPU round trips need separate small-batch MTP latency measurements |

The SGLang source audit used commit
`288915879266e9f21dccd3e1f0e22577d960c8a9`:

- `python/sglang/srt/arg_groups/hisparse_hook.py` rejects speculative decoding
  and restricts admission to supported sparse model families.
- `python/sglang/kernels/ops/kvcache/hisparse.py` has a separate speculative
  swap primitive limited to 2--4 steps. This is not end-to-end MTP support;
  Flash-Next target verification has five query positions.
- `python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh` and
  `hisparse_spec.cuh` implement device-side slot planning and host gathering.
- `test/registered/kernels/ops/attention/qsa/test_qsa_hicache.py` tests QSA
  host transfers and reordered physical pages. This validates a useful storage
  layout reference, not a V100 active-decode performance claim.

SGLang, vLLM, LMCache, KTransformers and FlexLLMGen repositories declare
Apache-2.0 licenses. Any source transplant must retain the source's individual
headers and notices and check transitive dependencies separately. No community
kernel source is copied by this design document.

## Separate the cache owners

Flash-Next is not a dense transformer with one interchangeable KV pool:

1. Twelve target QSA layers own attention K/V. At TP4, two model KV heads are
   replicated across ranks: each rank has one head with dimension 256.
2. Each QSA indexer also owns raw and compressed key caches. The indexer must
   choose the same token positions before attention can gather its working set.
3. Thirty-six target GDN layers own convolution and recurrent states. Active
   recurrent state and speculative rollback snapshots are updated each round;
   they are not immutable historical attention pages.
4. PLE has convolution/history state in addition to its host-resident embedding
   tables. Moving the tables does not move that state.
5. The MTP module has its own cache/state ownership and speculative positions.
   Sharing target weights must not alias target and draft cache slots.

Current `Qwen4ExpQSAFlashAttentionBackend.supports_kv_connector()` returns
false. The existing native offloader already handles hybrid prefix boundary
states, but that support does not replace an active host-KV QSA backend.

## Recommended first route

Use FP16 pinned host storage for persistent attention K/V and fixed-address
GPU working buffers. Retain the existing QSA selector and FP32 GDN arithmetic.
Do not add KV quantization or a new sparsity heuristic.

At each target layer:

1. Produce the original five-query top-k selection.
2. Resolve hits in the GPU buffer and deduplicate misses across the five
   queries. Gather the exact selected host K/V entries, preserving the
   selector's logical order, causal mask and per-query membership.
3. Run the existing sparse attention arithmetic on remapped device slots.
4. Back up newly committed K/V to host storage. Tentative draft/target writes
   remain versioned until the acceptance result commits their positions.

Checkpoint length, rejection rollback, slot eviction and reuse must be one
transaction. An in-flight copy may neither overwrite a slot still read by
attention nor resurrect a rejected token. Use fixed-size device workspaces and
explicit stream/event dependencies, with no CPU top-k result readback on the
captured decode path.

The GPU buffer must cover the union of selected tokens, not just one query's
2048-token budget. The current indexer output allows another three boundary
positions. Worst-case union capacity is five times a per-query selection,
multiplied by active requests; a tiled gather/attention path may be needed to
bound memory. Reusing one layer buffer across the serial QSA layers is a
candidate, subject to transfer and graph-lifetime validation.

Initially retain compressed index keys and active GDN/PLE state on GPU.
Then assess host backing for raw index keys and inactive/checkpoint states.
If the fixed active GDN/MTP state still prevents C4, test layer-staged state
updates without lowering precision. Moving attention KV alone does not prove
that C4 fits.

GPU direct attention over mapped host memory is an alternative to gathering.
Test it against gather-plus-device-attention: repeated reads by attention CTAs
can amplify link traffic. CPU exact attention is a separate fallback candidate
if these GPU routes fail the capacity/latency gate. A post-hoc CPU sparse
algorithm must not silently replace the model's original QSA selection.

## Capacity and transfer accounting

Target attention K/V alone consumes `12 * 2 * 1 * 256 * 2 = 12288` bytes per
context token per rank. At 32K tokens that is 384 MiB per rank per request,
excluding indexer caches, MTP, recurrent state, page padding and workspaces.
Host capacity must budget all of those owners as well as PLE and OS reserve.
Start with per-rank host pools; deduplicate replicated KV only after proving
the corresponding rank values and lifetimes identical.

For a miss-only selection of 2048 tokens, attention K/V traffic is about
2 MiB per QSA layer per query. Twelve layers and five disjoint query selections
could require roughly 120 MiB per rank per round before boundary positions,
indexer traffic and writes. Actual unions, hit rates and concurrent transfer
bandwidth determine latency. GPU-to-GPU NVLink topology does not establish
GPU-to-host bandwidth.

Measure mapped gather, DMA staging, D2H write-back and simultaneous four-rank
traffic on the intended machine before choosing the transfer route. Include
PLE traffic in the concurrent test. Compute `new host bytes / measured link
bandwidth` as a lower bound, not as an end-to-end speed prediction.

## Implementation and validation order

1. Remove unused EP GPU resources in pure TP; preserve TP collectives and
   legitimate EP/DP/PCP/EPLB behavior. Record actual bytes reclaimed.
2. Capture cache owner/page geometry and per-request fixed state costs, without
   uniform-pool padding hiding the physical requirements.
3. Test exact M5/M20 host gathering, slot translation, graph replay, copy
   overlap and four-rank contention with PLE.
4. Integrate one QSA layer and compare selected positions, attention output,
   MTP acceptance/rejection rollback and slot reuse with the resident route.
5. Integrate the complete model and measure 2K/8K/32K prefill, C1/C4 round
   latency, emitted tokens per round, acceptance intervals and actual host/GPU
   peaks. Compare the same model, precision and graph settings.

Precision reductions require a separate decision. Host placement itself
introduces no quantization error; any numerical changes from a different
attention reduction order must be measured and reported.
