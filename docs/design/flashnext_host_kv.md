# Flash-Next host-backed KV on SM70

## Scope and current status

Keep the IQ3_S target and FP16 MTP weights on GPU, the packed PLE tables in
pinned host memory, and move persistent attention KV to host memory. This
document specifies the next implementation; host-backed active decode is not
implemented or benchmarked yet. Small, bounded GPU buffers remain necessary
for attention, recurrent updates and CUDA graph replay.

PLE placement must be explicit for a weights-on-GPU, tables-on-host deployment.
The existing cascade policy fills device headroom before host memory: freeing
an EP communicator can consequently put PLE rows back into VRAM. A sufficient
`VLLM_QWEN4EXP_PLE_HOST_GIB` budget alone does not override that ordering when
cascade is enabled. For the current packed table, use the existing 7 GiB
per-rank host budget together with `kernel_config.ple_disk_cascade=false`.
Check the actual row placement and preserve host reserve before admitting KV
capacity; no additional environment variable is introduced.

The measured TP4 capacity route on four 16 GiB V100s uses 14.281 GiB of model
weights per rank, approximately 0.95 GiB of non-PyTorch runtime memory and a
0.40 GiB cache budget. It runs short C1 requests but cannot sustain the measured
C4 cohort. These measurements use eager execution and are not the FULL-graph
latency baseline. PLE storage is 26.82 GiB on the host.

## Community implementations reviewed

| Implementation | Relevant mechanism | Limit for this workload |
| --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata) | Native Flash-Next QSA and MTP; authoritative pinned host KV, persistent per-layer GPU hot pages, GPU-side miss resolution inside CUDA graphs | Published IQ3_S results use INT8 KV and a single GPU; FP16 TP4 and C4 need separate sizing and measurement |
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

### Strata: the closest active-decode reference

The source audit pins upstream commit
`82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`. Strata declares an MIT license;
preserve its notice and audit dependency notices before copying source.
Its implementation directly addresses this model's native sparse attention,
rather than only storing reusable prefixes outside GPU memory:

- [`kv_stream.hpp`](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/include/strata/kernels/kv_stream.hpp)
  defines authoritative host K/V and a logical-block-to-device-slot page table.
  Native pages contain four tokens. Resident pages persist across rounds.
- [`kv_stream.cu`](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/src/kernels/cuda/kv_stream.cu)
  deduplicates selected blocks across all queries of a resolve call, protects
  current-call hits, and uses CLOCK eviction for remaining slots. A GPU copy
  kernel reads only missing pages from mapped host memory into resident slots.
  Resolve and copy are capturable; CPU selection readback is unnecessary.
  Ordinary vector loads implement the copy, without a TMA dependency.
- [`verify.cpp`](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/src/core/verify.cpp)
  resolves the batched target selection before existing GPU QSA attention.
  Writes update authoritative host storage and any resident copy. The draft
  path uses a windowed ring; preserve our drafter's attention semantics rather
  than assuming that ring/window policy is interchangeable.
- [`layer.cpp`](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/src/core/layer.cpp)
  keeps pooled index keys in GPU memory. Active GDN state also stays on GPU.
  Host attention KV does not imply all historical index/state storage is host
  backed. FP16 streaming is supported alongside quantized KV formats.

The upstream parity executable covers streamed versus resident attention for
FP16 and quantized formats, random multi-query batches, eviction, partial pages
and ring restoration. These are useful test cases, not tests reproduced here.
Likewise, the [V100 fork](https://github.com/jmnargi/Strata-V100) demonstrates
a real SM70 integration, but its single-GPU Q2_0/spec8 results are not a
Flash-Next IQ3_S TP4/MTP4 baseline.

#### Published IQ3_S evidence

[Community report #469](https://github.com/Niko1221/Strata/pull/469) measured an
RTX 2080 Ti 11 GB on PCIe Gen3 with a Threadripper 3960X, INT8 KV streaming
and 32,768 resident cells. Engine version was 0.1.33. Fresh prompts have unique
prefixes; the first three rows are medians of three runs with 400 output tokens
each. Drafting uses MTP4 plus built-in suffix drafts, so its token throughput
does not share our acceptance benchmark's round denominator. The report did
not measure a matched resident-versus-streamed pair.

| Fresh prompt tokens | Decode tokens/s | Draft acceptance |
| --- | --- | --- |
| 4,096 | 39.6 | 65.6% |
| 32,768 | 38.0 | 63.7% |
| 128,000 | 34.1 | 55.7% |
| 250,000 (one run) | 31.8 | 53.4% |

KV block reads hitting VRAM were 99.6% at 4K and 94.3% at 250K; peak GPU
memory remained 10,525 MiB. This is evidence that active host backing can
retain a high hot-page hit rate even at long contexts. Context length, draft
acceptance and CPU expert work also vary, so the throughput difference cannot
be attributed exclusively to KV placement. Their INT8 traffic is not our
FP16 traffic, and their measured hit rate is not a forecast for TP4/C4.
The report's [detailed logs](https://github.com/Niko1221/Strata/blob/11aed66e5b00d882e81c483abbad39862c4e796d/bench/results/2026-10-02-community-rtx2080ti-11gb/README.md)
record about 825 MiB read from host KV over the fresh 250K request's 400-token
decode, lasting 12.571 seconds. That is approximately 66 MiB/s averaged over
decode, far below the machine's reported 13.1 GB/s host-to-device probe.
This average supports low transfer volume; it does not bound individual miss
bursts, page-resolution overhead or four-rank contention.

Strata is therefore the primary implementation reference for the first QSA
host-KV route. Start with FP16, persistent per-layer hot pools and graph-safe
GPU miss resolution. Measure actual selected-page unions and miss bytes before
choosing a smaller pool. At our geometry, 32,768 resident tokens in each of
the twelve target QSA layers cost 384 MiB per rank, excluding MTP, indexes,
state and metadata. This footprint must fit alongside resident weights.

The pinned implementation enforces at least 20,480 resident cells and requires
that one resolve call's selected-page union fit in its slots. Miss workspaces
are also sized from slots. Do not copy that sizing assumption into an M20
call or shrink the pool without enforcing capacity before writes; bound the
union, enlarge independent workspaces, or tile requests/queries. Current-call
page protection, overwrite coherence and rejection rollback must be tested
before model integration.

Strata's prefill staging differs from its sparse decode path. Optional
next-layer prefetch uses a separate stream with ready/release events and is
off by default. Its documentation labels older prototype prefill gains as
historical measurements, not a demonstrated speedup of the final upstream
port. Do not infer prefill performance from the decode hit rate.

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
That reuse discards each layer's cross-round hot set: it cannot simultaneously
claim the hit rate of twelve persistent layer caches. Compare both layouts
with the actual fixed-state memory budget and miss traffic.

Initially retain compressed index keys and active GDN/PLE state on GPU.
Then assess host backing for raw index keys and inactive/checkpoint states.
The complete host-storage goal also includes historical compressed index keys:
stage them into bounded layer buffers or scan them in chunks with exact score
and top-k merging. The initial resident-index variant is an intermediate step,
not a claim that all historical cache storage has moved off GPU.
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
indexer traffic and writes. This is an all-miss, disjoint-query upper bound,
not the expected transfer volume or a measured latency. Strata's published
hit rates show why using that bound as normal traffic is overly conservative.
Actual unions, page granularity, hit rates and concurrent transfer
bandwidth determine latency. GPU-to-GPU NVLink topology does not establish
GPU-to-host bandwidth.

Measure mapped gather, DMA staging, D2H write-back and simultaneous four-rank
traffic on the intended machine before choosing the transfer route. Include
PLE traffic in the concurrent test. Record selected-token unions and actual
buffer hit/miss rates at 2K, 8K and 32K; worst-case disjoint selections are not
an observed traffic measurement. Independent per-layer QSA selectors cannot
reuse another layer's selection as a prefetch prediction without validating
the original semantics. Compute `new host bytes / measured link
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
