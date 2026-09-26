# Automatic Prefix Caching

## Introduction

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

!!! note
    Technical details on how vLLM implements APC can be found [here](../design/prefix_caching.md).

## Enabling APC in vLLM

Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example:

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vLLM to process this long document *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vLLM to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vLLM to reuse the processing results of the chat history across all future rounds of conversation, allowing vLLM to serve future requests with much higher throughput and much lower latency.

## Limits

### Sparse checkpoints for aligned Mamba caches

`--prefix-cache-retention-interval` (the `CacheConfig.prefix_cache_retention_interval`
field) controls Mamba checkpoint retention. Its name, token unit, and value
semantics follow upstream vLLM:

| Value | Retained checkpoints |
| --- | --- |
| `None` | Every eligible state, preserving dense admission |
| `0` (default) | Necessary replay boundaries and detected shared-prefix junctions |
| Positive integer | Those boundaries plus periodic checkpoints at this token interval |

A positive interval must be a multiple of the resolved cache-hit alignment, not
necessarily a power of two. For 800-token alignment, `16000` selects a checkpoint
every 20 blocks. The former draft environment variables
`VLLM_MAMBA_SPARSE_CACHE_INTERVAL` and
`VLLM_MAMBA_SPARSE_CACHE_INTERVAL_BLOCKS` are replaced by this configuration field;
the old value `0` maps to `None`, not to the new default `0`.

This is a Mamba-focused adaptation of upstream
[#43447](https://github.com/vllm-project/vllm/pull/43447),
[#45845](https://github.com/vllm-project/vllm/pull/45845),
[#47782](https://github.com/vllm-project/vllm/pull/47782), and the replay-boundary
fixes [#53945](https://github.com/vllm-project/vllm/pull/53945) /
[#54713](https://github.com/vllm-project/vllm/pull/54713).
Other cache types retain their existing 1Cat admission policy. Sparse Mamba
admission applies in `align` mode when the manager block size matches the
coordinator alignment; other geometries keep dense admission.

Free uncached blocks are reused before cached blocks. The retention mask keeps
both model-level MTP/EAGLE replay positions when required, without removing the
existing speculative safety backoff. When a group has a longer cached prefix than
the combined hit, the request records that shared junction; scheduling
materializes its aligned state and admission keeps it for later sibling requests.
The first newly observed fork can still require replay. This does not predict
future forks or pin caches against eventual capacity eviction.

The parameter changes retention, not state computation, block size, or allocated
VRAM. Smaller positive intervals retain more recovery points; larger intervals
reduce checkpoint pressure but may require more replay. Cold-prefill contention
with active decoders remains possible.

The earlier custom-policy GPU experiments used Flash-Next AWQ on four V100 GPUs
with native FP8 MTP3. They do not constitute GPU acceptance of this revised
upstream-aligned implementation or of other models such as 27B variants.

### Workloads without reusable prefixes

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
