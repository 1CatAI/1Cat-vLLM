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

## Recorded promotion validation

The final ordinary-wheel source pair is control `cf2a1285e40c09ae498ab8bee5443fb2a147199d`
and candidate `c2b819954378573b3742a8d0aa272c665a7472d0`. All 15 native library
hashes are identical. Hardware/software: four Tesla V100-SXM2-32GB on GPUs 0–3,
Torch 2.10.0+cu128, CUDA 12.8, Flash-Next NVFP4, TP4, FP16 activations/KV,
MTP4 with greedy draft, greedy target temperature 0/top-p 1/top-k -1,
32K capacity, concurrency limit 4, prefix cache, V2 runner, FULL_AND_PIECEWISE
graphs, prefill budget 4096, 1.5 GiB explicit KV/rank, memory utilization 0.95.
These settings are shared by both arms.

| Measurement | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| C1 pure decode, 31744 input / 256 output | 79.832 tok/s | 92.199 tok/s | +15.49% |
| C1 TTFT, reported separately | 6.721 s | 6.606 s | -1.70% |
| C4 pure decode, four 4096 / 1024 requests | 225.400 tok/s | 227.778 tok/s | +1.05% |

C1 excludes one warmup and uses the median of two measurements. C4 excludes
one warmup and uses three measurements; each has four simultaneous decoding
requests and zero preemption. C4 timing counts returned tokens only while all
four are decoding, with no new admission/prefill. The small C4 improvement is
within the observed run variation; the evidence supports no material slowdown,
not a strong C4 speedup claim. C4 uses 32K *capacity*, not four 32K input prompts.

All 21 paired quality requests have exactly matching token sequences. Both
arms score MBPP 12/12, 32K retrieval 3/3 and Chinese QA 6/6, with natural stopping.
This is a paired subset qualification, not a general accuracy or 35B speed claim.

A subsequent loader-only scope correction restores the historical fallback for
36 unvalidated Flash-Next/DFlash proposer matrix rows, retaining explicit legacy
opt-in. All 288 other matrix rows, including every measured ordinary/MTP row,
are unchanged against the tested artifact, and all 28 GEMV/GDN operator function
ASTs are identical. The correction and reporting changes introduce no new
measured-route or numerical change.

## Retained rejected evidence and capacity limits

The earlier pre-guard candidate passed all subset scores but had only 17/21
exact token pairs. Its matched 1-GiB-KV profile improved 8K C1 by 13.24%; its long
C4 sample lacked four simultaneous decoders and is excluded. A valid C4 profile
at the final 1.5-GiB KV/prefill-4096 settings initially regressed by 3.85%.
The small-row follow-up later measured 220.410 versus 222.553 tok/s (-0.96%),
with its warmup about 1% faster. A one-switch dense-batch-off diagnostic reached
221.970 tok/s, only 0.7% above the adjacent enabled candidate. These samples do
not alone prove that the guard repaired the initial regression; final repeated
matched measurements and paired output are the acceptance evidence.

With 1.5-GiB KV/rank and prefill budget 8192, the earlier candidate exhausted
memory during a 448-MiB PLE temporary allocation. Packed copies added about
1.25 GiB/rank at load. Manual KV byte budgets do not automatically shrink to
reserve temporary/graph peaks. Startup status reports this limitation and the
packed-copy off switches; the supported 4096 profile does not establish that
8192 fits. No automatic KV-budget or prefill routing change is made here.
