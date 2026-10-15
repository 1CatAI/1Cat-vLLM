# Flash-Next MTP4 batch scaling on four 16GB V100 GPUs

The optimization target is the MTP4 verifier, with the qualified C1 cost of
approximately 16 ms as the reference. One verification round processes five
target rows per request. Its cost includes target forward, target sampling and
state commit; draft generation is separate. HTTP time per emitted token and
prefill throughput cannot substitute for this measurement.

## Workload and evidence boundaries

Integration base: `onecat/main`, `6f3557c973b4602905037e086fd4acba9b0d867f`.
The owned branch starts from the qualified deployment composition
`699c864d3c3a841b8626b657a1c8f236457ab1b5` (decode #1204, prefill #1205 and
deployment #1206). Native libraries were rebuilt from that source in an owned
CMake directory. All 12 packaged libraries have recorded SHA256 values and
no dependencies under private build directories or another worktree.

Hardware: local GPUs 4–7, Tesla V100-SXM2-16GB, TP4, driver 580.178.04,
CUDA 12.8, Torch 2.10.0+cu128, Python 3.12.13. The model is SC117's
Qwen3.8-Flash-Next-GSQ-RCO-abliterated IQ3_S with the matching stock MTP
weights converted to Q8_0. Context capacity is 262144 total tokens, prefix
caching, aligned recurrent state, disk PLE and host QSA E4M3 history.
Experts retain the required 80% device / 20% host placement.
Sampling remains temperature 0.6, top-p 0.95, top-k 20, seeds starting at
20261014; thinking and natural EOS remain enabled. Draft length is four with
greedy draft sampling. The measured long prompt has 32774 tokens.

The earlier 16 ms evidence used the original GSQ-RCO model on 9683 and kept
compressed indexer history on device. The first local batch diagnostic uses
host compressed indexer history and a 512-token prefill chunk to fit the
larger state pool. Its absolute C1 cost must not be substituted for the
qualified 16 ms result. The original-layout C1 control is measured separately.

The owned-source original-layout C1 node capture confirms this distinction:
36 interior complete rounds have a 16.009 ms median and 16.234 ms mean, with a
16.243 ms TP4 envelope mean. Trace service on the critical rank is 3.295 ms
dense/shared projections, 2.882 ms expert projections, 2.322 ms HC/fused
communication and 1.985 ms QSA history/attention. The inactive-capture HTTP
control emits 143.1 tokens/s in its common decode interval with acceptance
length 2.838. Nsight injection is still present in that control; it is not a
fresh uninstrumented production result. Full per-round data and exclusions
remain in `c1-original-layout-node-breakdown.json`.

## Initial verifier diagnostic

Synchronized worker events retain all four TP ranks. Align the same step
ordinal, require `tokens=5*C` and `drafts=4*C`, and select the rank with the
largest verifier event sum. Stage values below come from that same rank.
Event profiling adds synchronization and is diagnostic evidence, not an
unprofiled production speed result.

| Actual concurrent requests | Complete rounds | Verifier mean ms | Target forward ms | Sampling ms | State commit ms | Draft ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1, repetition 0 | 112 | 27.251 | 26.829 | 0.405 | 0.017 | 4.707 |
| 1, repetition 1 | 112 | 27.247 | 26.824 | 0.405 | 0.017 | 4.706 |
| 2, repetition 0 | 93 | 69.715 | 69.262 | 0.433 | 0.020 | 7.230 |
| 2, repetition 1 | 97 | 69.580 | 69.127 | 0.433 | 0.019 | 7.232 |
| 4, first actual overlap | 32 | 102.197 | 101.650 | 0.524 | 0.024 | 10.583 |

The C4 row is from the first warmup batch's actual four-request overlap. Its
later cached batch never reached M20, and the HTTP common-window check failed.
That failed attempt is retained and cannot be reported as sustained C4.
C2 verifier inflation is about 2.56x; C4 is about 3.75x. Nearly all of it is in
target forward. GPU kernel attribution and paired candidate timings are needed
before selecting a performance default.

## Full-width MTP4 batch measurements

The compact-state/tiled-kernel candidate with grouped shared-key scoring
produced sustained full-width verifier rounds at all five concurrency levels.
The existing scheduler option `--mixed-prefill-min-tokens 512` was fixed for
this pass: its adaptive floor of 16 previously starved new prefills, so C6/C8
never overlapped. This is a lab configuration change, not a new scheduler
default. Only rounds with five target rows and four draft rows per request
are included below. All four ranks must be present.

| Concurrency | Complete rounds | Verifier mean ms | p50 ms | p90 ms | p99 ms | Forward ms | Sampling ms | State ms | Draft ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 202 | 27.422 | 27.268 | 28.019 | 29.934 | 26.997 | 0.408 | 0.017 | 4.798 |
| 2 | 191 | 52.964 | 52.745 | 53.764 | 56.661 | 52.510 | 0.435 | 0.019 | 6.760 |
| 4 | 206 | 93.346 | 92.797 | 95.453 | 101.870 | 92.796 | 0.527 | 0.023 | 10.753 |
| 6 | 182 | 138.721 | 138.250 | 140.552 | 150.466 | 138.115 | 0.580 | 0.026 | 15.069 |
| 8 | 168 | 185.889 | 185.226 | 190.645 | 196.910 | 184.995 | 0.864 | 0.030 | 19.122 |

With grouped scoring disabled, C2 was 60.521 ms and C4 was 102.246 ms under
the earlier tiling pass. The first control C2 was approximately 69.6 ms.
Those C4 controls have fewer overlap rounds and are not a paired sustained-C4
production result. Per-request verifier share above is 27.4/26.5/23.3/23.1/23.2
ms. It remains above the qualified 16 ms device-indexer C1 reference.

Node attribution localizes the remaining cost to compressed-indexer scoring,
rather than main selected-history attention. In the short retained trace,
C1/C2 indexer GPU service is 13.644/28.931 ms on the critical verifier rank;
the native shared-key reader accounts for 13.008/28.292 ms. Main selected
history attention is only 0.663/1.033 ms. These are traced service sums from
64 C1 and seven C2 rounds, not the event-profile absolute timings above.
The trace retains a separate wall residual and incomplete-round exclusions.

## Candidate changes and retained fallbacks

* QSA prefill staging now checks per-request query length, rather than treating
  every batch above 32 total rows as prefill. MTP4 C8 is M40 with query length
  five. The existing native reader handles its 32+8 fragments. True prefill
  retains staging; unsupported direct inputs retain the protected reader.
* `sm70_hcx_batch_max_m` defaults to 8. An explicit limit up to 40 uses existing
  native eight-row HCX tiles. The limit belongs to each module, not mutable
  shared runtime policy. Larger batches keep projection/reduction fallbacks.
* `sm70_gguf.small_m_dp4a_max_m` defaults to 20. An explicit limit up to 40
  uses existing native experts in fragments of at most 20 rows. Fragment size
  also respects the cache's resident slots and miss ring. Each fragment finishes
  before the next resolution can evict its weights. Batches above the configured
  limit retain TurboMind. The opt-in also affects prefill tails of 21–40 rows
  and therefore requires numerical and model-quality evidence.
* `qsa_host_kv_state_compact` defaults to false. The bounded host-KV allocator
  gives each recurrent group independent device banks and distinct scheduler ID
  ranges. The block tables and zeroer translate those IDs into local bank IDs;
  null ID zero remains zero. History IDs, prefix hashes and reference lifetimes
  remain global. PP or CP above one is explicitly rejected. This avoids charging
  every layer for every recurrent group's IDs. For C8, the tested layout saves
  463.46 MiB per rank, rather than the initially estimated 1.6 GiB.
* `sm70_qsa_shared_key_batch` defaults to false. It reuses the existing
  single-request native scorer for each request in a small batch, then keeps
  shared batched top-k/expansion. Admission requires FP16 H4/D128 on SM70,
  two to eight queries per request, valid CPU query starts and at most 64 rows.
  All requests are admitted before any scorer executes; other layouts retain
  the paged scorer. No new native provider or kernel is introduced.
* `sm70_qsa_indexer_single_rank` is an experimental opt-in, default false.
  It binds only for SM70 FP16 H4/KV1/D128, host compressed-indexer storage and
  TP4/PP1/DCP1/PCP1. With one to 64 rows, rank zero selects indices and the
  existing TP communicator broadcasts them. Every rank still writes its own
  raw/compressed keys. Larger batches retain independent selection. Device
  indexer history, including the qualified C1 layout, remains on its existing
  path. The 32K task gate and fresh ordinary-worker paired task/performance gates pass (see the updated evidence below).

These are explicit opt-ins until the paired end-to-end gates pass. No kernel
sidecars, copied extensions or preloads are required.

## Validation and negative results

The five QSA GPU cases cover M32, M33, M40, M64 and M65, including changed KV
pages, selection indices and lengths under graph replay. They pass against
the protected reference reader. The compact pool tests cover C1/C2/C4/C6/C8,
the adjacent insufficient capacity, unique per-group quotas, null IDs and
prefix-cache lifetime. CPU cache/dispatch tests pass (46 cases), and typed
policy/CPU block-table tests pass (12 cases).

Four-card GPU tests pass (23 cases): compact-state zeroing with guarded foreign
IDs and changed graph inputs; block-table request permutation and padding;
tiled GGUF experts against independently dequantized Q8 oracles at M20/M21/M30/M40;
and compiled HCX payload ownership at M5/M8/M9/M10/M20/M30/M40/M41, including
changed inputs and final mixing fallback. GGUF error bounds are rtol/atol 0.003;
HCX retains the existing 0.002 relative-norm allowance. These operator tests
do not establish full-model quality or performance.

Grouped scoring passes 14 GPU cases with device and mapped-host histories,
C2/C4/C6/C8, request reorder, missing pages, zero lengths and padded rows.
Changed-input graph replay uses rtol 0.003/atol 0.0002 for scorer reductions;
expanded selection matches the independently invoked native request scorer
exactly in the tested cases. The single-rank prototype passes a four-rank
graph test at M1/M5/M10/M20/M30/M40/M64/M65: only rank zero executes selection
through M64, while M65 falls back on all ranks. Changed requests/pages/lengths
match independent selection exactly. Typed policy/hash tests pass too.

An aligned main-history cache microbenchmark rejected increasing hot capacity
from 32768 to 49152: both are about 0.219 ms for the representative expanded
indices. An earlier unaligned synthetic selection produced about 10 ms, but
does not represent the indexer's four-token expansion and is not model evidence.

Combined synchronized worker events and Nsight node tracing failed twice
(one CUDA launch/Xid43, one CUDA-event recording segmentation fault during
mixed prefill). All five event-only measurement batches completed before the
second trace failure. Subsequent event and node passes are separate; partial
traces are used only for attribution. Those failures are not discarded or
reported as successful full-width trace coverage.

The first compact C8 start captured M5/M10/M20/M30/M40 but its hand-built
startup warmup allocated IDs outside the new state-group ranges. Warmup now
uses the existing BlockPool and respects each group's capacity; 17 warmup
tests pass. The next start exposed an active-M33/padded-M40 output gate shape
error in the existing host QSA wrapper. Active gates are now sliced with the
other active inputs; six CPU wrapper cases cover empty input, M3/M5, M7/M10
and M33/M40 padding. These failed starts remain separate from performance data.

Rejected unmodified C8 starts include 1024-token and 512-token chunks that ran
out of device memory. A 64-token chunk with M30 graphs instead entered
TurboMind prefill and attempted an event dependency on an uncaptured stream.
This is distinct from CUDA IPC handle failures; #1208 is not a prerequisite
for that reproducer. A later M5-only C8 start also ran out of memory. Raw failed
logs and contracts are preserved; none is a speed baseline.

## Reproduction and retained artifacts

Run the service with the qualified sampling/MTP/KV settings and an owned GPU
group. Use `VLLM_SM70_MTP_PROFILE=1` and
`VLLM_SM70_MTP_PROFILE_INTERVAL=1` only for the diagnostic phase pass.
The benchmark requires a JSON list of prompt cases with `name` and `prompt`.

```bash
python benchmarks/flashnext_prefill/measure_mtp_concurrency.py \
  --base-url http://127.0.0.1:18041 --cases /path/to/prompt-cases.json \
  --concurrency 1 2 4 6 8 --service-log /path/to/worker.log \
  --out /path/to/requests.json
python benchmarks/flashnext_prefill/analyze_mtp_verifier.py \
  /path/to/worker.log /path/to/requests.json /path/to/verifier.json
```

For node attribution, launch the same service under Nsight with
`--cuda-graph-trace=node:host-only --capture-range=cudaProfilerApi` and
`--profiler-config '{"profiler":"cuda"}'`. The optional diagnostic worker
`mtp_trace_worker.TraceWorker` annotates target and draft graph submissions
after graph capture. Capture only short warmed generation intervals. Report
GPU service sums, same-rank verifier wall and TP4 envelope separately; node
tracing is composition evidence, not an accepted absolute latency.

Local raw artifacts are retained at
`/home/ymzx/桌面/1cat-vllm/reports/flashnext-mtp-batch-20261014/`.
The local public API is stopped. The 9683 API is outside this experiment and
remains running. No shared environment packages or model files are modified.

## Single-rank host-indexer candidate

Only the single-rank selection option changes relative to the full-width
grouped-scoring event pass. All five measured request streams complete;
the collector performs no Nsight capture. Values remain synchronized worker
diagnostics, not accepted uninstrumented throughput.

| Concurrency | Complete rounds | Before ms | Candidate mean ms | p50 ms | p90 ms | p99 ms | Forward ms | Sampling ms | State ms | Draft ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 202 | 27.422 | 20.979 | 20.823 | 21.678 | 23.439 | 20.557 | 0.404 | 0.017 | 4.124 |
| 2 | 197 | 52.964 | 38.099 | 37.782 | 38.939 | 43.372 | 37.648 | 0.432 | 0.019 | 5.330 |
| 4 | 169 | 93.346 | 58.831 | 58.283 | 61.249 | 65.323 | 58.283 | 0.525 | 0.023 | 7.673 |
| 6 | 168 | 138.721 | 84.812 | 84.353 | 86.517 | 91.666 | 84.198 | 0.588 | 0.026 | 10.236 |
| 8 | 178 | 185.889 | 111.006 | 110.241 | 113.423 | 123.378 | 110.115 | 0.861 | 0.030 | 12.568 |

C2 improves a further 28.1%, C4 37.0%, C6 38.9% and C8 40.3%.
C8 is 6.94 times the approximately 16 ms device-indexer C1 reference;
its 13.876 ms per-request verifier share is not a 13.876 ms batch wall.
The candidate host-indexer C1 is 20.979 ms, so it does not meet the qualified
device-indexer C1 latency. Draft costs are separate. Eight submitted 32K retrieval/arithmetic requests, executable order-preserving
Python deduplication, automatic weather-tool calls and tool replies pass with
thinking, production sampling and natural EOS. The tool test initially required
`finish_reason=stop` for a successful tool call; that harness mistake was fixed
to admit the expected `tool_calls` completion, and tool/reply checks were rerun.
Fresh ordinary-worker paired task and emitted-token measurements completed; see the updated evidence below. Further batch latency optimization remains necessary.
The affected final CPU suite passes 73 cases. A stale warmup fixture originally
patched the removed runner platform/environment hooks; it now exercises the
existing bound warmup policy. No runtime compatibility workaround was added.

## Updated acceptance target and completed clean control

The requested costs refer to the **whole MTP4 verifier batch**, not its share
per request: C1 approximately 16 ms, C2 17–18 ms, C4 approximately 22 ms and
C8 approximately 28 ms. The previous 111 ms C8 result does not satisfy that
objective even when divided by eight. All further results retain five target
rows and four draft rows per request, and separate draft time.

Fresh ordinary workers with profiling and NVTX disabled completed paired
root-selection controls, using the same capacity-eight host-indexer setup.
Common decode intervals exclude prefill and generation boundaries:

| Concurrency | Retained independent selection tok/s | Root selection tok/s |
| ---: | ---: | ---: |
| 1 | 76.46 | 107.91 |
| 2 | 99.35 | 116.72 |
| 4 | 94.74 | 157.36 |
| 6 | 95.92 | 144.65 |
| 8 | 98.71 | 150.58 |

Both runs pass the same task gates: eight seeded 32K exact-field retrieval
answers with natural EOS, executable deduplication code, automatic weather
tool invocation and its response. These are emitted-token speed evidence,
not replacement measurements for the verifier cost. The local GPU-indexer
C1 control is 14.655 ms and C2 25.472 ms at capacity two. A capacity-four
GPU-indexer experiment failed near full VRAM and is not accepted speed evidence.

## Bounded shared-page indexer staging

`sm70_qsa_indexer_stage_keys` defaults to false. A qualified host FP16
H4/KV1/D128 indexer borrows one engine-owned device bank, bounded to 32768
compressed keys rounded to physical pages (about 8 MiB at page size 204).
Every selection resets its map, deduplicates live physical pages across up to
eight requests, and refreshes their keys. The host history stays authoritative;
missing pages and overflow retain the existing scorer's paged-reader semantics.
The native `run_staged` entry reuses its original arithmetic; `run` keeps its ABI.
PP/DCP/PCP above one, other query layouts and absent native entries retain their
qualified paths. The bank never exposes borrowed output views and does not
cross engine configuration transfers.

The short scorer test at 32K improves C8 shared-prefix scoring from 3.74 to
0.36 ms; independent histories improve approximately 1.9x. This is not a claim
that arbitrary independent contexts obtain the shared-prefix gain. Twenty-three
GPU tests pass, including changed key contents, page reorder, padding, missing
pages, graph replay, exact selection and a deliberately overflowing one-page bank.

The full model diagnostic uses an 8 MiB QSA score tile and the same sampler,
256K capacity, prefix cache, 80% GPU / 20% host experts and MTP4. A previous
32 MiB score-tile run exhausted VRAM on a 64 MiB prefill score allocation;
that failure is retained. The bounded setting completes all 32K shared-prefix
streams with no incomplete TP4 verifier rounds:

| Concurrency | Rounds | Whole verifier mean ms | p50 ms | p90 ms | p99 ms | Draft mean ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 202 | 17.887 | 17.733 | 18.616 | 19.950 | 3.881 |
| 2 | 193 | 29.413 | 29.124 | 30.535 | 32.590 | 4.610 |
| 4 | 191 | 36.992 | 36.497 | 38.333 | 41.594 | 5.935 |
| 6 | 181 | 52.411 | 51.824 | 54.096 | 60.300 | 7.513 |
| 8 | 178 | 66.309 | 65.201 | 69.296 | 80.457 | 8.807 |

These are diagnostic GPU events with critical-rank alignment, not clean
production throughput. They remain above the requested whole-batch costs.
The staging candidate's fresh clean paired throughput, long-output drift and
256K boundary gates remain pending; no default is promoted.

## Follow-up candidates and rejected directions

Large dense projections now have an opt-in `sm70_gguf.small_m_hmma_batch`.
It reuses the existing segment format and native decoder across token tiles;
qualified DMV Q4/IQ4 banks provide their equivalent full scale representation.
The six-array prepared-operand ABI is retained. The accelerated initialization
subset is K2560/N>=1536 or K1536/N2560 in the existing qualified formats and
layouts. Runtime acceleration covers M9..16 and M25..40; M1..8 and
M17..24 retain the previous projection routes. Compatible small segment
peers share the same batch launch. Larger batches retain canonical restoration. The native batch operator has separately tested M1..40 coverage. The old native entry retains
its M1..32 contract; the batch entry is explicit. Eighty-six dense native GPU
cases and 111 affected storage/runtime cases pass, including capture/replay
and the M40/M41 fallback boundary. The initial wider M9..40 model experiment
produced C1/C2/C4/C6/C8 costs 17.931/26.655/38.265/51.922/63.368 ms:
C2 improves, but C4 regresses. That wider runtime admission is rejected;
M9..16 retains its C2 opportunity and preserves the prior M20 route. The
narrowed policy still needs matched full-model evidence.

A single-grid HC experiment passed its small numerical check but did not
improve C8 (approximately 0.192 ms for both retained and candidate calls).
Its implementation is removed from the runtime and retained as a research
patch in the private artifacts. Existing eight-row HC tiles remain qualified.

A separate default-off `sm70_gguf.grouped_gate_up` candidate reuses the
original lattice weight decoder for up to four routes to one expert. The
accelerated shape is M9..20, source types18/21/22, N divisible by32 and K by256,
with the existing Q8_1 intermediate. Routing is sorted and admitted on the GPU
each replay; outputs return directly to their original route order. Sparse
routes retain the original single-route algorithm. Changed-input graph tests
exercise both live admission outcomes and exact Q8_1 bytes; full-model benefit
remains unqualified. The first merged grouped/sparse kernel regressed sparse
routes because they inherited its register and shared-memory footprint; it is
rejected in favor of a separately guarded retained kernel.

Rejected experiments are preserved in the task artifacts:

* Parallel mapped-host indexer scoring in request-major and interleaved grids
  slowed independent-history scoring approximately threefold. They are removed.
* The retained full-batch HC matrix path is slower at C1/C2 and improves C8
  only slightly; it is not promoted as a scaling fix.
* Expert lane-width and bank-layout changes improve the leading integer kernels
  by only about 10%; they do not establish the requested scaling.
* Upstream MoE MMQ at M40 takes 1.6–1.9 ms for gate/up in representative real
  banks, versus about 0.22 ms for the retained integer route; it is rejected.
* Sorting routed experts alone passes exact Q8 output checks but slows the
  complete scorer by approximately 2–4%; its private native entry is removed.
* The expert-cache entry already holds 248 route IDs. At top-k ten the C8
  native path uses two twenty-row fragments. The 136 count in startup logs is
  host weight slots, not request-entry capacity; it does not justify a ring fix.

Hardware counter attribution is not available to the local unprivileged user
(`RmProfilingAdminOnly=1`). Cache/decode hypotheses are therefore based on
paired exact-shape experiments, not claimed Nsight Compute counters.

## Combined narrowed dense and grouped-expert diagnostic

The narrowed dense policy (M9..16) plus grouped experts completes all five
32K shared-prefix cases under the same capacity-eight, score-tile 8 MiB,
chunk 512, host-history contract. Whole verifier includes target forward,
sampling and state commit; draft remains separate.

| Concurrency | Complete rounds | Whole verifier mean ms | p99 ms | Draft mean ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 202 | 17.768 | 20.321 | 3.871 |
| 2 | 202 | 26.779 | 31.678 | 4.601 |
| 4 | 203 | 36.705 | 41.995 | 5.915 |
| 6 | 203 | 51.769 | 59.463 | 7.472 |
| 8 | 190 | 64.895 | 80.106 | 8.808 |

There are no incomplete TP4 rounds. This combined experiment does not isolate
the contribution of grouping: the preceding wider dense policy also differs.
Relative to page staging alone, its C8 improvement is about 2.1%; this is not
the requested 28 ms C8 target. At C8 forward is 64.005 ms; sampling 0.860 ms and
state commit 0.030 ms cannot explain the inflation. A new matched node trace
will decide the next bottleneck rather than attributing it from micro gains.

The eight submitted 32K retrieval/arithmetic tasks, executable Python function,
automatic weather-tool call and its reply all pass with natural EOS. The
quality run still has worker diagnostic fences, so clean emitted-token
qualification and the 256K boundary remain pending. The first model run uses
the owned native build immediately before a symbol rename/formatting pass;
the final formatted source artifacts are subsequently rebuilt and installed
atomically. The final artifacts pass 54 affected GPU cases, including changed-route
graph replay, independent expert output oracles and four-rank indexer state
checks. Clean model speed and longer-context gates remain separate before
any promotion.

## Matched node attribution and mixed-segment fusion

Five decode-only node captures use the final rebuilt source and identical
C1/C2/C4/C6/C8 contracts. Their interior counts are150/113/87/60/49.
Node tracing changes absolute latency: traced whole verifier means are
22.105/29.934/39.397/57.967/71.533ms, separately from event measurements.

| TP0 GPU service ms | C1 | C2 | C4 | C6 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense/shared projections | 3.489 | 6.879 | 6.224 | 9.291 | 14.847 |
| Routed expert projections | 2.880 | 4.639 | 7.573 | 12.015 | 15.439 |
| HC and its fused exchange | 2.504 | 4.648 | 7.564 | 10.575 | 13.877 |
| Indexer and top-k | 3.556 | 3.876 | 4.434 | 5.109 | 5.955 |
| Expert-cache resolution/wait | 1.659 | 1.398 | 1.498 | 3.285 | 3.761 |

These are service sums, with overlapping streams; they are not an additive
wall decomposition. C8 peer broadcast includes waiting for TP0 indexer work.
The actual local GPU4–7 mesh has mixed NV1/NV2 links; the remote9683 NV2 mesh
is a separate hardware contract. CPU affinity matches NUMA node1.

The captured operand descriptors expose a dispatch problem: QKV banks mix a
large batch-enabled segment (family5) with existing small K/V segments
(family3). Requiring every segment to have a batch tag splits compatible
projections and restores the small peers before separate calls. The candidate
admits one native batch launch when all segments have the compatible layout
and at least one calibrated large segment enables batching. The old M20
fusion remains unchanged. M9..16 and M25..40 are qualified separately from
larger prefill fallback.

Real-weight QKV screens across four layers improve M10 by2.07–3.63x,
M30 by1.48–2.04x and M40 by1.98–2.81x. M20 arithmetic is unchanged.
Relative output errors remain below0.00077 versus the FP32 dequantized-weight
reference, under the predeclared0.003 norm bound. The affected111 GPU runtime
cases pass; three additional mixed-format cases exercise11 boundary/interior
M values, changed-input replay and exactly one native batch dispatch at the
accelerated sizes. The matching complete model curve is being collected;
these micro gains are not reported as whole-verifier gains.

Two original-down scheduling screens are rejected: cooperative two/four-lane
source reads usually regress despite exact outputs, and one-warp-per-route
regresses IQ4_NL while offering no representative M20 gain for Q2_0. Both
research patches and all15 real-weight cases are retained outside the runtime.

The mixed-segment model experiment completes all five 32K shared-prefix
cases with181/214/182/178/165 complete TP4 verifier rounds and no incomplete
rounds. Whole means are17.513/26.114/36.692/50.150/61.131ms; C8 p99 is68.697ms
and draft mean8.768ms. C8 improves5.8% relative to the preceding combined
64.895ms diagnostic. Retrieval, executable code and automatic tool call/reply
all pass. This remains above the requested18/22/28ms targets and uses worker
fences; ordinary-worker qualification remains separate. Source/native sampler
acceptance varies between runs, so HTTP gains alone are not attributed to the
projection change.

An isolated HC grid-barrier screen compares the existing cumulative arrival
counter with distributed arrival flags. Both take approximately1.62us per
barrier and pass ordering/wrap checks in that screen. No native protocol change
is justified; the private prototype is retained as a negative result.

## Closed verifier interval audit

The five captures use the pre-mixed-fusion grouped-twenty-row candidate
(the untraced C8 event mean was 64.895 ms). They were swept by GPU timestamp on each
selected critical rank. Each interval belongs to exactly one category, a
separate overlap bucket, or the idle residual. Categories below sum to verifier
wall; they are not summed kernel service. Node tracing perturbs absolute latency
and these numbers must not replace the untraced event measurements above.

| Concurrency | Complete interior rounds | Verifier wall ms | HC ms | Experts ms | Dense/shared ms | TP collectives ms | Overlap ms | Idle ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 150 | 22.105 | 2.390 | 2.879 | 2.489 | 3.661 | 0.999 | 2.150 |
| 2 | 113 | 29.934 | 4.592 | 2.411 | 3.577 | 3.988 | 5.689 | 2.590 |
| 4 | 87 | 39.397 | 7.399 | 7.560 | 5.488 | 4.823 | 0.167 | 3.035 |
| 6 | 60 | 57.967 | 10.565 | 11.416 | 5.086 | 5.821 | 4.334 | 4.364 |
| 8 | 49 | 71.533 | 13.702 | 12.595 | 7.753 | 6.724 | 9.138 | 4.338 |

The remaining wall belongs to selected-history attention, GDN, routing, cache
resolution, normalization, sampling/state commit and uncategorized kernels.
Raw `scaling-nodes-{1..5}-closed.json` files retain those categories and each
round; `scaling-nodes-closed-summary.json` retains their means. The closure
assertion tolerates at most 1e-7 ms. This audit includes target forward,
sampling and state commit, and excludes MTP draft calls. Expert and HC scaling
remain primary optimization targets. A collective can include a wait on the
indexer root, so its duration does not establish intrinsic link cost.

## Forty-row expert/cache experiment

The default fragment and mailbox capacity remain twenty rows. An explicit
`sm70_gguf.grouped_gate_up_max_m=40` permits one 400-route lease with a
512-integer mailbox. Reply capacity accounts for all request entries plus
spare restoration; it grows to the next power of two when required. Retained
quantization and ordinary gate/up/down kernels still run in twenty-row tiles.
Grouped original gate/up has a separate native forty-row entry point, keeping
the legacy operator and opaque schemas unchanged.

The matched diagnostic event curve with forty-row gate/up, without grouped
down, is 17.752 / 26.270 / 37.187 / 47.641 / 61.115 ms for C1/2/4/6/8,
with 202 / 200 / 190 / 196 / 174 complete rounds and no incomplete rounds.
C6 improves versus the mixed-fusion twenty-row control; C8 does not. This is
not evidence that a faster isolated gate/up microbenchmark speeds the service.
Eight concurrent 32K exact retrieval/arithmetic responses, executable Python,
automatic weather tool selection/reply and natural EOS checks all pass.

The next default-off `sm70_gguf.grouped_down` candidate groups original
IQ4_NL/Q2_0 down weights across at most four equal-expert routes. It keeps
FP16 projection boundaries, the retained four warp-owned route subtotals and
their addition order. Only M25..40 is accelerated, with the forty-row lease
policy and admitted Q8 intermediate/native expert shape. Smaller fragments
keep the retained down path. Unsupported shapes use that path or its existing
TurboMind continuation; a requested qualified route without its native build
rejects initialization.

Twenty-seven real-weight down screens and twenty-one threshold/interior
screens preserve the prior bytes, including the saved prior-binary references.
At M40, sort-inclusive speedups are 1.12..1.92x for IQ4_NL and 1.17..1.57x
for Q2_0. Sparse M20 IQ4_NL regresses and remains outside acceleration. Sparse
M25 IQ4_NL only improves 1.013x; this boundary result is not a broad speed claim.
The native, cache, state/graph and complete expert-chain checks pass 141 GPU
cases after correcting a test parameter that shadowed its weight tensor.
The combined service screen below establishes measured 32K performance and
task evidence; 256K quality qualification remains outstanding.

Additional rejected screens are retained in the report directory. A bounded
mailbox-plan scan changed isolated cache time by less than 1% and was removed.
A four-float HC partial-workspace pad appeared faster in a one-call graph,
but the 64-call graph screen removed launch skew and showed no C8 gain
(0.13680 ms retained versus 0.13803 ms padded). Its source was removed.

## Rechecked whole-batch HC path

The earlier whole-batch HC rejection did not establish a reliable native-path
comparison: its launcher did not explicitly enable the fusion policy, and a
single-call graph mixed GPU work with peer launch skew. Rechecking the native
operator directly, with observed CUDA dispatch and sixty-four calls per graph,
shows bit-identical outputs and critical-rank per-call times:

| Concurrency | Retained tiled HC ms | Whole-batch HC ms | Speedup |
| --- | --- | --- | --- |
| 1 | 0.021842 | 0.021777 | 1.003x (retained fallback) |
| 2 | 0.045594 | 0.043849 | 1.040x |
| 4 | 0.073743 | 0.065897 | 1.119x |
| 6 | 0.103555 | 0.092633 | 1.118x |
| 8 | 0.136066 | 0.121190 | 1.123x |

This supersedes the earlier HC rejection for the measured full-mesh TP4,
separate-projection, local-schedule path. The default-off typed policy
`sm70_hcx_batch_fusion` owns its larger peer buffers and local workspaces.
Producer-fused output projections, non-full-mesh execution, unsupported local
modes and M1..16 retain the original eight-row path. M9..16 was excluded after
the ordinary-worker acceptance regression described below. M>40 retains the existing
projection/collective continuation. Requested qualified fusion rejects a
missing native build. Original operator signatures remain unchanged.

The four-rank Inductor/graph payload test passes all M5/8/9/10/17/20/24/25/30/
32/33/39/40/41 cases, including input mutation, retained/fast alternation,
materialized/final HC consumers and output ownership checks. These results are
kernel/state evidence. The combined service performance and task gates are
being measured separately; 256K boundary qualification remains outstanding.

## Combined expert-down and HC service screen

With mixed dense fusion, forty-row expert leases, grouped down and whole-batch
HC enabled, the matched MTP4 diagnostic GPU-event verifier costs are:

| Concurrency | Complete rounds | Incomplete | Mean ms | P99 ms | Draft mean ms |
| --- | --- | --- | --- | --- | --- |
| 1 | 202 | 0 | 17.650 | 20.388 | 3.875 |
| 2 | 220 | 0 | 25.681 | 28.084 | 4.580 |
| 4 | 214 | 0 | 35.286 | 42.640 | 5.936 |
| 6 | 186 | 0 | 45.974 | 58.584 | 7.455 |
| 8 | 193 | 0 | 54.660 | 64.421 | 8.749 |

C8 improves from 61.131 to 54.660 ms (10.6%) versus the prior mixed-fusion
screen. C4 improves from 36.692 to 35.286 ms and C6 from 50.150 to 45.974 ms.
These are whole-batch costs, never costs divided by concurrency. The workload
keeps 256K capacity, 32K shared input, MTP4, C8 capacity, the same host QSA and
80% GPU expert storage, 512-token prefill chunks and natural-EOS sampling.
Profiling events/fences are enabled. The 16/18/22/28 ms C1/2/4/8 targets remain
unmet. Ordinary-worker throughput/acceptance and paired task checks are a
separate gate; no default promotion follows from this screen.

## Ordinary-worker acceptance audit

Fresh ordinary workers, without event fences, NVTX tracing or worker wrappers,
completed matched mixed-fusion control and combined expert-down/HC passes.
Both use the same source/native artifact, 384 output tokens, 32774 input
tokens, capacity eight, sampling and per-request seeds. The control retains
twenty-row gate/up and tiled HC; the combined pass enables forty-row gate/up,
grouped down and whole-batch HC. Each row is the measured common interval
with all submitted requests still decoding; rates are aggregate emitted tokens.

| Concurrency | Control tok/s | Combined tok/s | Control acceptance length | Combined acceptance length |
| --- | ---: | ---: | ---: | ---: |
| 1 | 110.15 | 125.34 | 2.477 | 2.824 |
| 2 | 169.49 | 133.84 | 2.630 | 2.121 |
| 4 | 213.31 | 235.15 | 2.339 | 2.473 |
| 6 | 251.40 | 277.44 | 2.458 | 2.481 |
| 8 | 259.09 | 305.02 | 2.259 | 2.451 |

The C2 emitted-token regression is approximately 21%, despite a slightly
faster diagnostic verifier. This candidate is not accepted as a general MTP
speed improvement. C1 acceptance also changes, so its rate improvement cannot
be attributed to verifier compute alone. Both passes complete the eight-way
32K retrieval/arithmetic, executable Python, automatic tool-call/reply and
natural-EOS gates. These bounded task checks do not establish unchanged MTP
acceptance or broad quality. The runtime HC acceleration range is consequently
narrowed to M17..40 while C2 acceptance is investigated. The native operator's
correctness range remains M9..40; admission and measured benefit are distinct.

The subsequent shared-expert experiment reuses the existing fused gate/up and
down operations in bounded twenty-row fragments. It is explicitly enabled by
`sm70_gguf.shared_expert_batch` during the experiment, admitting Q8_0 down projections at
M9..40, and preserves the canonical continuation for other projections and
larger batches. The existing two-integer internal down descriptor remains
valid; the explicit opt-in adds a third policy integer without changing the
registered operator schema. Real-weight screens across six layers show
1.34..2.18x isolated M40 speedups and paired relative-norm error at most
0.000622 against the predeclared 0.003 bound. M5/M8/M20 retain their prior
bytes. GPU graph tests cover M5/8/9/10/16/17/20/21/24/25/30/32/33/39/40/41,
mixed source formats, the independent FP32 oracle, changed inputs and native
execution/fallback. Eleven format combinations pass.

The first ordinary-worker shared-expert pass emits 110.24 / 151.25 / 244.01 /
253.33 / 292.45 tok/s at C1/2/4/6/8. C2's additional same-seed repetition emits
159.54 tok/s, with acceptance length changing from 2.444 to 2.579. Eight-way
retrieval/arithmetic, coding and tool/reply gates pass. C6/C8 do not exceed the
previous combined pass, and C2 remains below control. This is an unpromoted
experiment; an isolated shared-expert speedup is insufficient to accept it.
Its separate synchronized-verifier curve is 17.701 / 26.318 / 36.261 / 47.069 /
61.686 ms, with 181 / 192 / 181 / 181 / 180 complete TP4 rounds and no
incomplete rounds. The experiment does not improve whole-verifier cost and
was removed, including its policy and runtime code. Its patch and tests remain
in the raw artifact directory; operator micro speed does not justify retaining
another runtime option.

A subsequent ordinary run keeps shared-expert acceleration disabled and HC
acceleration limited to M17..40. C2 repetitions emit 142.16 / 150.06 / 155.10
tok/s with acceptance length 2.326 / 2.326 / 2.568. Its C1 cold prime and warm
repeat have acceptance length 3.031; the eight short math primes used by the
earlier paired run were omitted. Thus this isolated run does not establish
recovery or isolate a numerical cause. Per-request seed alone is insufficient
to treat these different runs as identical generation histories. The HC
admission restriction remains while the acceptance variance is unresolved.

The latest combined C8 node capture closes a 60.184 ms traced verifier wall
with 12.465 ms HC/fused communication, 9.069 ms expert projections, 6.599 ms
dense/shared projections, 6.453 ms TP collectives, 3.874 ms selected-history
attention, 2.423 ms GDN, 9.191 ms overlap, 3.399 ms idle and 6.711 ms in the
remaining categories. Categories are exclusive wall intervals; traced absolute
time is not the 54.660 ms diagnostic result. The observed native expert route
groups average approximately four rows per expert. Neither these counts nor
collective durations establish memory bandwidth or intrinsic NVLink cost;
Nsight Compute profiling counters remain unavailable to this user.

## HC scheduling and partial-write follow-up

The instrumented HC screen partitions each CTA's control-thread interval,
separately from the ordinary-worker verifier measurement. At C8 its mean
117.942 us consists of 24.627 us initial peer writes, 4.472 us weight prefetch,
26.112 us peer reads/combine, 14.968 us down partials, 5.750 us first grid
rendezvous, 18.591 us RMS/down reduction, 7.816 us LoRA peer writes,
0.466 us LoRA completion, 3.310 us second grid rendezvous, 5.949 us LoRA
shared-memory load, 5.164 us HC up/output writes and 0.717 us output completion.
These are instrumented per-CTA phase intervals, not hardware counters or a
sum of per-stage rank maxima.

The retained small-HC local schedule issues three independent peer loads
before inspecting their epochs. The first whole-batch kernel instead waited
for each peer in sequence. The candidate reuses the retained independent-load
schedule with the same tags, timeout and canonical addition order. Its down
projection transposes partials in shared memory before coalesced global writes.
The LoRA input tile is unused during this phase and owns that temporary view;
no new shared-memory allocation, host pool or function attribute is required.

The private research screen, using 64 calls per graph and twenty timed replays
on all four ranks, improves C8 HC from 0.121 to 0.104 ms. Outputs match both the
packaged native baseline and private retained kernel byte-for-byte; changed
graph inputs pass too. This screen only admits the idea to the ordinary native
build. Fresh normal-artifact graph/state and model measurements remain the
qualification gate; its private research library is not a service dependency
or a production speed result.

The normal CMake artifact subsequently passes 44 CPU policy/dispatch cases,
the four-rank compiled payload/state suite and the direct native changed-input
graph screen. C1/2/4/6/8 direct native costs are 0.022427 / 0.042894 / 0.059882 /
0.083102 / 0.107008 ms; C1 is a retained fallback. Corresponding retained
eight-row tiles are 0.022294 / 0.045446 / 0.075752 / 0.104005 / 0.136468 ms.
The native correctness range includes M10; runtime acceleration still begins
at M17. The new ordinary `_C` SHA256 is
`283b58ff6a99f358b74a4c2d579f6757b996cbe4ca79f908c721cbf5e03cda77`.
Whole-model qualification remains separate.

A subsequent two-shape expert-down screen compacts four-route group leaders
and schedules persistent work blocks, retaining the original integer dot and
unroute order. At M40, 128 expert IDs and 149 active groups, the sort/plan-inclusive
cost changes 0.102674 to 0.100932 ms for IQ4_NL and 0.077722 to 0.076186 ms for
Q2_0; outputs remain byte-exact. These 1.7%/2.0% gains fail the predeclared
8% screen threshold. The research source is archived and not integrated.

## Latest whole-model HC result and next indexer candidate

The normal coalesced HC artifact completes ordinary task checks and the
separate synchronized-verifier run. Whole-batch verifier means at C1/2/4/6/8
are 17.652 / 26.573 / 36.066 / 43.682 / 53.886 ms, with 187 / 202 / 222 /
189 / 199 complete TP4 rounds and no incomplete rounds. Draft means are
3.857 / 4.604 / 5.949 / 7.499 / 8.754 ms. These are whole batches, without
dividing by request count. C2/C4/C8 targets of 18/22/28 ms remain unmet.
Ordinary common-interval aggregate rates are 125.41 / 138.15 / 239.56 /
261.18 / 298.48 tok/s; additional C2/C8 repetitions give 141.43/327.25.
Retrieval/arithmetic, executable code, automatic tools and natural EOS pass.
The C2 regression remains unresolved, and this is not a general MTP speed win.

A two-hop HC peer reduction retains bytes but regresses the direct graph
screen: C4 0.061217 to 0.071124 ms and C8 0.105724 to 0.126098 ms. It remains
research-only and is rejected. Its extra rendezvous does not amortize the
smaller peer fanout.

The next default-off candidate is `sm70_qsa_indexer_persistent_keys`. It
reuses the existing FP16 shared-key scoring family, with one bounded per-layer
bank plus the existing shared refresh bank. Physical-page tags and int64
write versions identify reusable pages. Both fused and reference compressed
writers advance versions before selection or its MTP skip path. Versions
never rewind with speculative visibility. Every cache bind clears tags,
including same-tensor rebinding before recapture. Hash collisions use freshly
copied spill pages, then the existing authoritative paged reader for overflow.
The default staging path still refreshes every live page on every selection.

The accelerated subset is the existing SM70 FP16 H4/KV1/D128 host-indexer
path with TP4 root selection, CP1 and PP1. External KV transfer, KV offload
and speculation methods other than ordinary MTP have untracked writers and
are explicitly rejected when this candidate would be selected. Other layouts
retain the existing path. A mismatched cache binding retains protected
refresh; closed owners reject reuse. Normal native loading requires the
matching `run_staged_banked` operation; prior native schemas remain unchanged.
This changes neither attention selection arithmetic nor KV storage format.

At compressed page size204, 16384-key per-layer banks add approximately48MiB
on the selection root across twelve QSA target layers; the shared refresh
bank remains32768 keys. Startup, cold prefill and captured C8 generation must
fit the unchanged expert placement and memory contract. A private NUMA1
operator screen improves shared-prefix C8 from0.367 to0.197ms and independent
C8 from2.976 to2.309ms. These local figures do not establish whole-verifier
speed, actual256K performance, broad numerical coverage or production quality.
The one-bank variant regressed independent C8 by7.6% and was rejected before
runtime integration. Normal-artifact state and model gates remain separate.

The normal banked-indexer artifact subsequently completes the model gates.
Its whole-batch C1/2/4/6/8 verifier means are15.925 /24.678 /33.724 /41.812 /
53.130ms, with187 /181 /201 /213 /193 complete TP4 rounds and zero incomplete
rounds. Corresponding p99 values are18.019 /28.322 /46.876 /59.045 /66.672ms.
The single-stream mean returns to approximately16ms under this capacity-eight
contract, while the desired C2/C4/C8 means remain unmet. C8 improves only0.756ms
in this pass; its isolated indexer improvement is not a whole-batch speed claim.

Fresh ordinary workers emit122.06 /182.21 /251.03 /288.93 /322.03 aggregate
tok/s; extra C2/C8 repetitions give169.11 /308.37. Acceptance lengths change
and C8 overlaps the prior repetition range. All five retrieval/arithmetic,
executable-code and tool/reply gates pass, as do eight short natural-EOS checks.
Observed peak root GPU allocation is16078MiB, leaving67MiB available on the
16384MiB card after driver reservations; other ranks use16050..16052MiB.
This fits the measured32K workload with unchanged80% GPU expert placement,
but does not establish256K boundary memory or quality. No default is promoted.

The normal indexer SHA256 is
`ca41596598165c7ecb39c127f4b6c82ef5458dc5bdad16cf4abc4ca8e1237842`.
Fresh-process dependency inspection and all four actual worker maps confirm
ordinary source-owned extensions without a private research DSO or preload.
Actual worker JIT-monitor entries confirm version tracking and bank planning.
The first48 GPU and29 policy/cache tests, six writer-ordering cases and full
changed-source hooks pass. A further18 GPU cases include the actual204-token
compressed page size, stale-binding refresh fallback, same-address generation
reset and changed-input replay after cache rebinding and recapture. Hash
collision winners may change between launches; zero-copy reuse is required
for unchanged pages with stable distinct leases, not for conflicting hashes.

A follow-up dense screen tests smaller row chunks in the existing weight-major
family. The trace shows244 registers/thread,112..128 CTAs of128 threads for
large input projections, and80 CTAs of256 threads for the1536-to2560 output
projection. These are launch resources, not measured occupancy or stall
counters. Smaller chunks reduce registers to approximately168 but reload and
decode weights. Across six M20/M40 shape screens, the dominant Q6_K input
projection does not improve. One mixed IQ4_XS/Q8 case improves8.1%, while small
chunks regress the output projection. Outputs and changed-input graph replay
retain the normal artifact's bytes. The general candidate fails its8% benefit
gate and is rejected; no runtime policy or native change is added for it.

## HC independent-row parallelism

The prior HC launch uses256 threads, with several exchange, row reduction and
copy phases assigning independent rows to eight warps. The16-warp candidate
keeps the same80-CTA resident grid, FP32 reduction order, six down-projection
warps, five up-projection warps, shared-memory lifetime and peer epochs. Extra
warps handle independent row exchanges/reductions and coalesced copies; up
weights and eight-row gate tiles remain guarded to their original warps.
The compiler reports113 registers/thread and43040 static shared bytes for the
512-thread research kernel, without spilling. This is a launch-resource fact,
not a measured occupancy or memory-bandwidth claim.

Four-rank64-call graph screens preserve bytes with changed inputs. Critical
rank maxima at C4 improve0.064814 to0.056273ms, and at C8 improve0.107693 to
0.096882ms. A384-thread intermediate gives0.059313/0.097945ms. The512-thread
schedule is then folded into the existing native HC batch operation, retaining
its ABI, workspace and runtime M17..40 admission; M9..16 still uses the retained
runtime path while the acceptance-history issue remains under investigation.

The normal CMake artifact passes44 CPU policy/dispatch cases, the compiled
four-rank payload/state suite and direct changed-input graphs. Direct C1/2/4/6/8
costs are0.022916 /0.037545 /0.055384 /0.075440 /0.097872ms, versus retained
8-row tiles0.022454 /0.045633 /0.074343 /0.104196 /0.136121ms. C1 is the same
fallback; the direct M10 native result does not widen runtime admission.
The new normal `_C` SHA256 is
`a71013eeb238309d695d7f05977a20ea8681734af041b4671d08d45c16f5c700`.
These local wins require separate whole-model qualification; they do not
establish the requested28ms C8 verifier cost.
