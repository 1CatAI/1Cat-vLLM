# SM70 DFlash2 NVFP4 Prefill Promotion

## Scope

Promote the existing bounded QPN2-packed NVFP4 prefill operator for the
quality-audited Qwen3.8-27B DFlash2 contract. The change does not alter kernel
arithmetic, the DFlash2 draft model, target verification, sampling, attention,
or prefix-cache semantics.

Integration base: `onecat/main` at
`62ad1e02693f4c857f3b7547cef1860ee54e8053`.

## Retained evidence

No target-only baseline is rerun in this campaign. Historical evidence is:

| Contract | 32K prompt tok/s | 64K prompt tok/s |
| --- | ---: | ---: |
| DFlash2 before the D256 closure | 3125.12 | 2588.53 |
| DFlash2 with the D256 sidecar | 3476.53 | 3103.02 |
| DFlash2 with D256 and QPN2-packed prefill | 3959.14 | 3450.29 |

The last row used the same NVFP4 target, official BF16 DFlash2 draft, TP4
V100, FP8 E5M2 target KV, FP16 draft KV, prefix cache, Mamba alignment, and
CUDA Graph decode contract. All retained requests were uncorrupted and kept
the same first-token hash. Raw artifacts are under
`/data/minimax-h3/task-cache/v100-dflash2-prefill-32k64k-20260827/`.

The often-quoted 5170.96 tok/s exact-8K and 2438.89 tok/s 256K results are
FP8 target-only contracts. They remain useful upper-bound references but are
not relabeled as NVFP4 DFlash2 measurements.

## Dispatch change

The QPN2 decode and bounded prefill implementations are already in main but
both environment gates default to off. They now resolve to on only when all of
the following hold:

- speculative method is `dflash` with seven draft tokens and the checkpoint
  exposes the DFlash2 selector contract (`selector_top_k=16`);
- PP1/TP4, `max_num_seqs=1`, and no DBO or microbatching;
- the existing exact Qwen3.8 dense NVFP4 layer-shape gate passes.

Explicit `VLLM_SM70_NVFP4_QPN2=0` and
`VLLM_SM70_NVFP4_QPN2_PREFILL=0` remain hard rollbacks. Other speculative
methods, target-only service, different TP, and concurrent service retain the
existing route.

## Promotion gates

- focused resolver, shape, zero-copy-layout, and dispatch tests: 5 passed;
- latest-main DFlash2 route hit plus cold 32K/64K throughput: passed;
- unchanged short decode path, acceptance behavior, and structured-output
  health before marking the Draft PR ready.

## Latest-main remote validation

PR commit `8aec0f5e4419f44c70232edd49b68058b9fa224a` was exercised on the
four-V100 deployment host with both `VLLM_SM70_NVFP4_QPN2` and
`VLLM_SM70_NVFP4_QPN2_PREFILL` unset. The runtime contract was the official
BF16 DFlash2 draft with seven speculative tokens, NVFP4 target, TP4, FP8 E5M2
target KV, FP16 draft KV, 256K maximum length, 4096 batched-token budget,
prefix cache plus Mamba alignment, probabilistic draft sampling, and CUDA
Graph decode.

The run hit all required routes: automatic QPN2 decode, automatic
QPN2-packed bounded prefill for `M>=1024`, the fixed-SHA D256 sidecar, direct
paged chunked prefill, the FP8 E5M2 prefill bridge, and the context-only
DFlash2 chunk path.

| Case | Mean request wall | Request-wall tok/s | Pure-prefill tok/s |
| --- | ---: | ---: | ---: |
| 32K | 8.0618 s | 4064.62 | 4069.25 |
| 64K | 18.3891 s | 3563.85 | 3566.94 |

Relative to the retained pre-closure DFlash2 measurements, request-wall
throughput improved by 30.1% at 32K and 37.7% at 64K. Relative to the retained
QPN2-packed q4096 run, the remote result was 5.5% and 5.2% faster,
respectively; this cross-host delta is recorded but is not used as a kernel
claim.

All six measured requests reported `is_corrupted=false`, returned token ID
`248046`, and retained output-token SHA256
`54363ddee68f4a5db81c9d37e5fb738d28f5b67dc7f725ad7333172b1ea157da`.
That hash is identical to the retained pre-closure, D256-only, and
QPN2-packed runs. The operators themselves were not changed by this PR; its
only source change is strict-contract default routing with explicit rollback.

The raw result and log are retained at
`/data/minimax-h3/task-cache/v100-dflash2-nvfp4-prefill-promotion-20260829/remote-four-v100/`.
Their SHA256 values are `e6d6033492cc3c5af37dc90219f07c33eb1559907743ecf7355c13c861606beb`
and `949531b1bf03a2921ed664a95f19e0af0adc7b842fd7183087798cff5984e654`,
respectively.

The M<=8 QPN2 decode route was already enabled in the merged DFlash2 quality
audit. That audit passed 24/24 structured API cases, 5/5 long alternating
prefix-state cases, three-seed MBPP/HumanEval/LiveCodeBench gates, and a
target-only versus DFlash2 PPL comparison of `5.4993116/5.4993622`. This PR
does not change that operator or the verifier/sampler path. Its newly promoted
large-M route reuses the previously bitwise-equal packed prefill operator, and
the latest-main long-prefill run preserved the retained output hash.

## 2026-08-29 opaque dispatch and practical defaults

A release audit on `onecat/main` at
`cda10f1220f78cb1fe9d62b0001912d0a0b59c95` found that the first integration
used a Python `M >= 1024` branch around the prefill operator. AOTInductor
captured that dynamic branch into the decode graph: the graph pool grew from
`0.13` to `0.28 GiB`, and a complete DFlash2 round regressed from about
`17.4` to `41.98 ms`. Sidecar loading alone did not fix the graph because the
Python shape branch remained visible.

The retained implementation exposes one opaque
`nvfp4_qpn2_prefill_dispatch_sm70_out` operator for all M. Its C++ dispatcher
keeps M below the threshold on the established QPN2/TurboMind path and sends
large M to the QPN2-packed prefill kernel. The prefill kernel creates and
releases its bounded FP16 dense workspace within the operator; no large
persistent workspace is captured by the decode graph. Production wheels link
the dispatcher into `_C`; a separately loaded fragment is supported only for
source-overlay validation against an already accepted native extension.

The route passed exact V100 screens for M8/M16, normal/gated output, and
pointer-zero ephemeral versus explicit workspace (`max_abs=0` throughout).
The focused CPU contract suite passes seven tests, including strict admission
and explicit rollback.

For the practical NVFP4 DFlash2 service contract, source configuration now
also defaults the already quality-audited verifier/GDN metadata paths and the
QPN8 dense-order rerank. Admission requires Qwen3.8-27B compressed-tensors
NVFP4, FP16 execution, TP4/PP1/B1, q7/top16 DFlash2, FP8 E5M2 target KV,
`max_num_batched_tokens=4096`, no DBO, and Flash-V100 on SM70. Every explicit
environment value wins. Candidate-order rerank remains off because its FP16
tie-cutoff changes candidate sets; dense-order is the quality release path.

The four-V100 source-default proof removed all QPN2/prefill booleans and all
DFlash2 fastpath booleans from the launch file. Logs showed automatic defaults,
opaque QPN2 prefill, QPN8 `dense_order=True`, and a `0.13 GiB` graph pool.
Three 512-token single-request runs measured complete rounds of
`17.4035/17.3569/17.3697 ms` (mean `17.3767 ms`). One cold 32K token-ID prefill
measured `4039.49 tok/s`; the following decode measured `17.3640 ms/round`, so
the large-M route did not poison or slow the decode graph. Three distinct
function-call schemas and one strict JSON-schema response also passed on the
same source-default service. Raw evidence is in
`/data/minimax-h3/task-cache/v100-dflash2-release-audit-20260829/remote-18ms-restart/source-default-validation.json`.
