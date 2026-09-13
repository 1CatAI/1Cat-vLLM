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

In 1Cat, `VLLM_MAMBA_SPARSE_CACHE_INTERVAL` opts into sparse
checkpoint admission for aligned Mamba prefix caches. Its default is `0`, which
retains dense checkpoint admission. A positive value is measured in tokens and
must be a multiple of each Mamba manager's block size; invalid values fail at
initialization. For example, an interval of `16000` is valid for 800-token blocks.

Sparse admission applies only when the cache mode is `align` and the coordinator's
alignment equals the manager's block size. Other geometries use dense admission.
It retains periodic checkpoints and both prompt replay boundaries needed by
ordinary and speculative lookups. It preserves the speculative one-block backoff
and does not change state computation. Free uncached scratch blocks are recycled
before cached blocks whenever prefix caching is enabled, including when the
sparse interval is zero.

This reduces checkpoint churn that can evict another request's reusable prefix
during a long prefill. It does not pin conversation caches: finite capacity and
changed prefixes can still cause misses. A prefix that ends between retained
checkpoints can require additional replay. Cold-prefill contention with active
decoders also remains possible.

Real-model validation used Flash-Next AWQ with four V100 GPUs and MTP3. Shared
manager code does not imply that other models, including 27B variants, have
completed real-weight validation.

### Workloads without reusable prefixes

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
