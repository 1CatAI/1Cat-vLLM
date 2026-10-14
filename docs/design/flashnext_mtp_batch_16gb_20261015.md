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
  path. The 32K task gate passes; fresh ordinary-worker paired performance is pending.

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
Fresh ordinary-worker paired task and emitted-token measurements are pending.
The affected final CPU suite passes 73 cases. A stale warmup fixture originally
patched the removed runner platform/environment hooks; it now exercises the
existing bound warmup policy. No runtime compatibility workaround was added.
