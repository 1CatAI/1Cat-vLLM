# DFlash2 long-context verification curve

## Frozen scope

Integration base: `80545c010bbf6f5ed06458d992c189d75d0eff8f`. The task includes
the diagnostic-only changes from PR #586 at
`720457ff4a8f6361e8160bd1cc9210d327d9b8de`. Runtime numerical/performance
comparisons use a frozen source and native-library manifest, physical GPUs
4–7, TP4/B1/q8, QUASAR target revision
`d8e6fbfa3e3a78899b440222b827430045a05b44`, DFlash2 draft revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`, CUDA 12.8, Torch 2.10.0+cu128,
E4M3 target KV, FP32 logits/state and the existing compensated attention.
Original FlashQLA GDN prefill and the verified FA2 sidecar stay enabled.
Capacity remains 262144; measured inputs stop at 131072 tokens.

For adjacent lengths 32K/64K/128K, candidate complete-round growth must not
exceed frozen baseline cold-prefill unit-token cost growth. Every long-context
absolute round cost must improve, and 1K must not regress. The initial
diagnostic reference ratios are 1.115, 1.211 and 1.350 for 32→64, 64→128 and
32→128 respectively. Freeze the final ratios after three independent baseline
startups; never relax them by slowing prefill or the shorter decode point.

## Implementation order

1. Repeat the restored-path baseline, one verified cold request and five warm
   requests per length/startup. Retain source/library hashes and all-rank route
   evidence. Supplement the existing 64K q8 trace with a 128K trace.
2. Independently screen aligned E4M3 vector loads and QK live-range/unrolling
   changes, preserving the 80-split, N32, K16 compensation and FP32 partial
   contract. Require byte-exact output and full partial/max/sum buffers.
3. Develop overlapping loads only after these measurements. If necessary,
   evaluate 80/160/320 splits, grouped-head KV reuse and versioned workspaces.
   Any graph specialization belongs in the actual MRV2 graph manager.

Model gates retain fixed-prefix distributions, acceptance and natural-output
checks. Arithmetic variants require independent FP64 error measurements before
the final FP16 cast. Never remove precision compensation as an optimization.
No experimental route is enabled by importing its builder or benchmark.

## Progress and artifacts

Three baseline startups completed 72 requests, with identical token IDs and
acceptance across and within startups at every length. The baseline median
complete rounds at 1K/32K/64K/128K are 16.350/25.821/35.047/53.267 ms.
Cold-prefill throughput is 3525/4063/3640/3009 tokens/s. The frozen growth
limits are 1.1163476857, 1.2095236995 and 1.3502489828 for 32→64, 64→128 and
32→128. First-use prefill overhead affects the first 1K request; medians use
the three independent cold observations, and 1K is not the curve denominator.

The curve reporter verifies distinct worker startups, the full computed-token
count for cold requests and five measured requests per length. Its percentiles
describe request-average round costs, not individual GPU-round latency. It
refuses to overwrite a frozen curve and checks absolute costs as well as ratios.

The separate 128K trace measures 36.849 ms target grouped attention per
rank/round, versus about 7.317 ms of QPN2 projections, 0.878 ms of draft
attention and 0.678 ms GDN recurrence. Critical-rank interval is 54.871 ms and
GPU union is 52.899 ms. The NCU probe returns `ERR_NVGPUCTRPERM`; no hardware
counter claim is made. The separate 32K trace is complete: its critical-rank
interval/GPU union is 27.615/25.584 ms. Target attention accounts for almost
all additional GPU time at 128K; draft attention and GDN recurrence remain
approximately 0.88 and 0.68 ms respectively. These are profiler observations,
not the unprofiled acceptance costs.

Initial operator candidates preserve the 80-split/N32/K16 compensation
contract. With sixteen distinct layer KV allocations, paired GPU-graph
measurements give:

| Candidate | 32K attention, ms | 64K attention, ms | 128K attention, ms | Byte checks |
| --- | ---: | ---: | ---: | --- |
| Frozen three-group control | 9.232 | 17.985 | 35.453 | Reference |
| Guarded 16-byte KV loads | 6.012 | 11.540 | 22.744 | 50/50 |
| QK unroll1 | 7.529 | 14.379 | 28.264 | 50/50 |
| QK unroll4 | 9.319 | 17.990 | 35.437 | 50/50; no stable gain |

The combined screen independently compares unroll1, vector loads plus
unroll1, two padded three-head groups plus unroll1, and one six-head group
plus unroll1 with the frozen control. All 200 output/full-workspace/canary
cases match. At 128K the control costs 35.458 ms; the respective candidates
cost 28.258/18.474/17.383/12.766 ms. The six-head candidate's 32K/64K costs
are 3.497/6.592 ms. These remain operator measurements, not complete-round
gains or model admission. In particular, some unroll/grouping variants slow
the 1K operator and cannot replace the short path without further evidence.

The initial multi-candidate loader exposed a native-module alias: two DSOs
used the same module name, and CPython returned the first module for both.
The second candidate's initial results are withdrawn and the report marked
invalid. Source-derived native module names, actual loaded-file verification
and distinct-callable checks now prevent this failure. The table above uses
the corrected independent bindings. A CPU regression check rejects the old
aliased pair before any GPU work.

QK unroll1 reduces the compiled register count from 234 to 94 without spills;
unroll4 uses 140 and does not gain stable speed. These are compiler resource
observations, not achieved occupancy. Follow-ups evaluate unroll2 and constant
1648/3296-page addressing with the same arithmetic. Other page sizes and
8-byte-only strides retain their existing address/load implementations.

Combining six-head reuse, vector loads and unroll1 reduces the sixteen-layer
128K attention cost to 9.959 ms. A disjoint V panel lets otherwise idle QK
warps load and convert values while the six QK warps compute; it reduces that
cost further to 9.092 ms (32K/64K: 2.557/4.736 ms). Both candidates pass all
50 byte-exact output, FP32 partial/max/sum and canary cases. V prefetch uses
73728 bytes of dynamic shared memory with an explicit device opt-in limit
check. It preserves the existing CTA barriers and online-softmax warp barrier.
Page-specialized addressing and unroll2 together reach 9.218 ms without V
prefetch; these independent results determine which combinations to test.

The system sanitizer executable failed before testing because its injection
library was absent. Reruns use the previously validated, complete CUDA 12.8
sanitizer bundle. The vector-only path passes memcheck, racecheck and
synccheck with zero errors/warnings. Its first unprofiled startup gives
16.133/22.068/27.824/38.765 ms complete rounds at 1K/32K/64K/128K, with identical
token IDs, finish reasons and acceptance to the frozen control. All four
ranks capture the candidate in 16 actual target attention calls. This is an
independent screen, not three-startup curve acceptance; the growth gates
still fail. The combined V-prefetch model screen requires its own native
sanitizer gates before launch.

The actual model capture uses a 3296-token KV page with strides
`(1687552, 256, 256, 1)`. The initial sixteen-layer performance screen used
1648-token pages; both are correctness cases, and `--performance-page` allows
timing the exact captured page geometry. The loader also retains the
8-byte-only stride fallback. No context result is extrapolated to 256K.

Private launch/build manifests and raw results are retained in the task
artifact archive; generated libraries and private cache paths are excluded
from Git. No new production route or default has been enabled.

Prior rejected experiments remain recorded in the context-cost and long-verify
worklogs. Historical E5M2 and FP16-partial Pack-GQA timings are design references,
not quality/performance evidence for this E4M3 FP32 path.
