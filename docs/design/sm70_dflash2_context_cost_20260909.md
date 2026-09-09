# DFlash2 context-length cost audit, 2026-09-09

## Scope and frozen contract

Measure growth in complete verification-round cost at 64K and 128K, with a
same-prompt-template 1K anchor. The user stopped 256K measurements after observing
abnormally slow prefill; queued 256K jobs remain held. This is a latency
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

`benchmarks/profile_sm70_dflash2_context_cost.py` records an initial/warmup request
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
on the benchmark modules. The first GPU observations below exposed a missing
native prefill dependency and do not qualify the intended fast route.

Artifacts, private launch wrappers, checkpoints and task-local compiler caches
are under `/home/ymzx/.cache/1cat-dflash2-context-cost-20260909`. The retained
frozen candidate and rear-four-GPU queue are under
`/data/minimax-h3/task-cache/v100-quasar-dflash2-15ms-20260908`.
The dataset campaign is checkpointed, preserving completed same-startup pairs
and retaining any interrupted partial case separately. Its queued continuation
remains held during the prefill route investigation.

## Missing native prefill dependency

The frozen service selected `FLASH_ATTN_V100`, but its `lib-v4` dependency set
did not contain FA2 and its launch did not set `VLLM_SM70_FA2_D256_LIBRARY`.
The startup log explicitly warned that
`_vllm_fa2_C::sm70_d256_splitd_n32_dense_fwd` was absent and long prefill would
use a slower fallback. The E4M3 bridge is resolved from that same missing
library, so it was unavailable too. The later logs show direct paged E4M3
prefix prefill, without the D256/v37 bridge route.

The warning was missed before the long sweep. The partial unprofiled results
are retained as fallback observations, not expected Flash-V100 scaling:

| Input tokens | Complete round, ms | Pure decode, tokens/s | Accepted drafts/round | Emitted tokens/round | Initial TTFT, s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16.419 | 254.61 | 3.2295 | 4.1967 | 1.650 |
| 65536 | 34.972 | 142.97 | 4.0784 | 5.0196 | 81.468 |
| 131072 | 53.575 | 80.67 | 3.3390 | 4.3390 | 221.823 |

Round/decode values are medians of three repeats in one startup. All 12
completed requests reached the 256-output-token diagnostic cap; none receives
quality credit. Repeated token IDs and acceptance match within each length.
The initial 128K request could reuse the previous 64K prefix, so its TTFT is
not a cold-prefill measurement. Repeat TTFT also includes prefix-cache hits.
No 256K request completed and no graph-node trace was collected in this run.

PR #548 is already merged at `8d9c3518992059105d89939e8a46d75184505d8e`.
Its CMake-built FA2 library, SHA256
`ec00745c34b3d146b0200fb9454c1419322072b0ccf0d551d958cbe701e4e15b`,
contains Split-D dense/paged, v37 and the E4M3 bridge. Its eight v37 source
hashes match the frozen serving source. A private copy is frozen under this
audit's `native/fa2-ec00745c` directory; every other native dependency stays
fixed. This is a dependency-loading repair for the diagnostic service, not a
new kernel or a quality promotion. PR #548 disclosed a remaining model token
divergence; operator accuracy alone cannot close that model-quality gate.

The corrected launch explicitly selects this sidecar. The client checks native
availability and the loaded FA2 SHA on every rank before long requests, then
requires actual exact-bridge route hits. Snapshots occur between requests.
It resets the prefix cache before each new input length and requires the
request's computed-prefill-token counter to equal the full input length;
an HTTP reset response alone does not prove a cold request. Full API usage,
engine prefill time and computed-token metrics are retained. The first focused
repair run stops at 128K; 256K jobs must not resume automatically.

Missing FA2 explains the prefill fallback. It does not by itself attribute
the q8 decode slope: target verification has a separate grouped E4M3 FP32
dispatch and still needs a same-route graph-node trace after this repair.
