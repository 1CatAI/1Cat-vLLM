# Qualified Flash-Next batch defaults

Flash-Next's small-batch FP16 path admits MTP verifier and draft projections
through the same local geometry, layout and alignment checks as ordinary decode.
The numerical policy remains unchanged: the MTP HC schedule retains its FP16
K512 partial boundaries, whereas ordinary concurrent HC keeps FP32 partials.
Automatic GDN packing also stays within that boundary; an explicit legacy
GDN batch opt-in is retained for controlled tests of another proposer.
Other speculative methods remain outside the batch quality qualification in
`vllm/model_executor/models/config.py`. Microbatching and batch-invariant mode
retain their existing fallbacks. Native arithmetic and weight precision are
unchanged; online QPN8 remains opt-in.

The 14 controls tagged `Flash-Next qualified batch` in environment metadata
are enabled by default for qualified operation, retaining explicit `0`
overrides. Gated RMSNorm is resolved per engine in
`KernelConfig.sm70_rmsnorm_gated_exact`: Flash-Next ordinary decode and MTP
select it automatically; unrelated models retain their previous route.
The compatibility getter remains off. The loaded layer holds its own decision,
and the resolved value participates in the compilation hash. Their qualification
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

With the MTP reduced-reduction policy, M2/M4/M5 output projections retain
baseline cuBLAS routing: the native oracle showed different FP16 bits there.
M8 and the measured small router rows remain admitted. An explicit request for
FP16 accumulation also keeps its original dispatch. No global precision flags
are changed for MTP.

## Promotion validation remains open

The earlier candidate passed MBPP 12/12, 32K retrieval 3/3 and Chinese QA 6/6
with 17/21 exact token sequences. A matched 1-GiB-KV configuration improved
C1 throughput by 13.24%; its long C4 sample lacked four simultaneous decoders
and is excluded. A valid four-request profile at 32K capacity, 4K inputs and
1024 outputs, TP4/MTP4, FP16 KV, 1.5 GiB KV/rank and prefill budget 4096
initially regressed by 3.85%. The same KV budget with prefill budget 8192
exhausted memory. Packed weights added about 1.25 GiB/rank at load; manually
specified KV bytes do not automatically shrink to reserve peak workspace.

The small-row routing follow-up measured C4 at 220.410 versus 222.553 token/s
(0.96% lower), while its warmup was about 1% faster. These adjacent values do
not prove that the guard repaired the earlier performance regression: control
throughput also changed across runs. A single-switch dense-batch diagnostic is
queued. The current final artifact still requires paired quality and target
speed validation before promotion. PR #796 remains draft.
