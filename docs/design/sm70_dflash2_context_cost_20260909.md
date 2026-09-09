# DFlash2 context-length cost audit, 2026-09-09

## Scope and frozen contract

Measure growth in complete verification-round cost at 64K, 128K and near the
256K capacity boundary, with a same-prompt-template 1K anchor. This is a latency
diagnostic, separate from the continuing natural-output quality campaign.
No production kernel, arithmetic, weights or serving default changes here.

The harness branch starts at main `80545c010bbf6f5ed06458d992c189d75d0eff8f`,
which contains optimization PR #556. The serving checkout remains frozen at
`a7cc5ae305149d7a9ffdf42fb224dff34e5606aa`, with the same native libraries as
the approximately 16-ms candidate. This avoids changing the runtime while
measuring context-length scaling. Target QUASAR/draft revisions are
`d8e6fbfa3e3a78899b440222b827430045a05b44` /
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`.

Use physical GPUs 4–7, four V100-SXM2-32GB GPUs, TP4/B1/q8, CUDA 12.8,
Torch 2.10.0+cu128, Python 3.12.13, E4M3 target KV, FP32 logits/state and
FP16 draft transport. Preserve the fixed Gemma reduction, packed GDN,
BV2 value tile, combined split, QPN2 cap64/publication, grouped attention,
sparse selection, context/probe overlap and CUDA graphs. The server retains
262144 total context capacity, a 4096 prefill chunk budget and four maximum
request slots with only one live request.

The prompt uses the frozen long-context corpus's prefix, repeated filler and
coding-task suffix. Exact input lengths are 1024, 65536, 131072 and 261888
tokens. The last point reserves 256 tokens inside the 262144 capacity; it is
not a 262144-token input followed by out-of-capacity generation. Each diagnostic
request has at most 256 output tokens, honors EOS, and retains T1/k20/p.95/seed0
and the frozen xhigh template. Length stops receive no quality credit. Inputs
are never clipped, and all actual token counts are retained.

## Measurement and attribution

`benchmarks/profile_sm70_dflash2_context_cost.py` records a cold/warmup request
and three unprofiled repeats per input length. It separates TTFT, engine decode,
complete-round mean, emitted token throughput and draft acceptance, and retains
per-chunk token counts/times. Client stream intervals are transport evidence,
not instrumented GPU-round latency. No trace or tensor dump runs in this service.

`benchmarks/sm70_dflash2_context_trace.py` collects twelve q8 rounds in a separate
service after a warmup request and eight preceding q8 rounds. The gate rejects
prefill chunks, initial eight-token prefills, shorter tail queries and multiple
requests. Only capture boundaries synchronize. Every rank retains scheduled
width, computed positions and output positions; missing rank/window evidence
fails the client instead of producing a partial trace. The first and last
captured transitions are excluded from steady attribution.

The trace uses CUDA Graph node activity and NVTX with Nsight Systems 2025.3.1.
Attribute target graph, target head/sampling, state, draft and host work on the
same critical rank. Retain GPU event union and uncovered wall time separately.
Kernel service, phase envelopes and independent rank maxima must not be summed
as complete-round wall time. Static CTA/register/shared-memory evidence does
not establish achieved occupancy or HBM throughput.

## Validation and artifacts

CPU checks cover seven q8/prefill/tail/multiple-request combinations, four exact
prompt lengths and rejection beyond 262144 total tokens. Scoped pre-commit runs
on both benchmark modules. GPU measurements are pending; this document makes
no 64K/128K/256K latency or quality claim before their artifacts exist.

Artifacts, private launch wrappers, checkpoints and task-local compiler caches
are under `/home/ymzx/.cache/1cat-dflash2-context-cost-20260909`. The retained
frozen candidate and rear-four-GPU queue are under
`/data/minimax-h3/task-cache/v100-quasar-dflash2-15ms-20260908`.
The ongoing dataset campaign is checkpointed before this focused sweep and
resumed afterward, preserving completed same-startup pairs and retaining any
interrupted partial case separately.
