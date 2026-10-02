# Qualified Flash-Next batch defaults

Flash-Next's small-batch FP16 path admits MTP verifier and draft projections
through the same local geometry, layout and alignment checks as ordinary decode.
The numerical policy remains unchanged: the MTP HC schedule retains its FP16
K512 partial boundaries, whereas ordinary concurrent HC keeps FP32 partials.
Other speculative methods remain outside the batch quality qualification in
`vllm/model_executor/models/config.py`. Microbatching and batch-invariant mode
retain their existing fallbacks. Native arithmetic and weight precision are
unchanged; online QPN8 remains opt-in.

The 14 controls tagged `Flash-Next qualified batch` in environment metadata
are enabled by default, retaining explicit `0` overrides. Their qualification
comes from [#703](https://github.com/1CatAI/1Cat-vLLM/pull/703) and
[#704](https://github.com/1CatAI/1Cat-vLLM/pull/704). These references contain
bounded model quality and performance evidence; they do not establish general
accuracy or a less-than-20-ms MTP round. The recorded all-acceleration MTP4
round is 21.411234 ms. Each metadata description explains the operation,
default rationale and use of the off switch.

## Memory trade-off

Startup and `/v1/sm70/acceleration` expose the default controls and a per-rank
estimate of additional packed weight copies for the reference TP4 layout.
The estimate covers target and draft buffers separately, excluding allocator
overhead, graphs, temporary workspace and KV cache. It is not measured free
memory and does not guarantee that a chosen context/concurrency fits.

For the 48-layer target plus one MTP layer, GDN uses 725.625 MiB, target HC
330 MiB, draft HC 6.875 MiB, router 122.5 MiB and shared expert 76.5625 MiB per
rank. Smaller budgets can disable the packed copies with these controls:

```bash
VLLM_SM70_QWEN38_BATCH_FASTPATH=0
VLLM_SM70_QWEN38_GDN_INPUT_BATCH=0
VLLM_SM70_MTP_HC_BATCH=0
VLLM_SM70_MTP_ROUTER_BATCH=0
VLLM_SM70_MTP_SHARED_BATCH=0
```

The independent GDN and aggregate batch controls retain their previous OR
semantics: both must be off to disable packed GDN. Runtime checks determine
whether loaded projections actually prepare these buffers. For other layouts
the report omits the estimate rather than guessing their memory consumption.

## Regression command

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m tools.sm70_flash_next_route_snapshot \
  --baseline-ref BASE_SHA --output routes.json \
  --expected-changes tests/models/qwen4_exp/data/flash_next_batch_default_changes.json
```

This runs historical and candidate loader/dispatch predicates on CPU tensor
doubles over the 324-configuration matrix. The expected changes identify only
Flash-Next rows; unrelated model rows retain their routes. It does not replace
paired model quality or target throughput measurements for admission/default
changes. M1/prefill and unsupported local geometries retain their fallbacks.

Small-batch dense output under the MTP reduced-reduction policy can differ
from the original cuBLAS result in FP16 bits. Admission therefore needs paired
model quality, rather than a claim of universal bit identity. No global
accumulation flags are changed for MTP.
