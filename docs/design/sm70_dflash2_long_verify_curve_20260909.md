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
Capacity remains 262144. The frozen first campaign stops at 131072 input
tokens; the subsequent target revision below explicitly adds the 256K tier.

The user revised the objective on September 10: continuously reduce absolute
complete-round cost and the incremental cost of longer contexts. Prefill
growth ratios are reference observations, not admission limits or a stopping
condition. Every long-context absolute round cost must improve, 1K must not
regress, and context increments must not increase. Report both additional
milliseconds and milliseconds per 1024 additional context tokens. Never
improve a ratio by slowing the shorter point, prefill or acceptance.

The subsequent September 10 target revision sets explicit complete-round
latency goals:

| Context tier | Complete-round target |
| --- | ---: |
| 32K | <= 17 ms |
| 64K | <= 18 ms |
| 128K | < 20 ms |
| 256K | < 22 ms |

1K must not regress and the <15 ms short-context goal remains. Output quality,
compensation and acceptance requirements are unchanged. The 256K target
supersedes the previous instruction to stop all measurements at 128K; it does
not qualify any untested kernel range. Keep the current 132096-token serving
gate until the longer operator and model checks pass. Existing frozen reports
remain immutable. Within 262144 service capacity, a boundary-window performance
probe constructs a separate 261888-token prompt and reserves 256 output tokens;
report that exact input length and the actual sampled context range. Do not
truncate an existing prompt, label this as a full 262144-token cold prefill, or
silently raise model capacity. Operator checks separately include 262144 and
the speculative q8 boundary headroom.

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
reference ratios are 1.1163476857, 1.2095236995 and 1.3502489828 for 32→64,
64→128 and 32→128. The original frozen report remains immutable even though
these ratios no longer gate admission. First-use prefill overhead affects the
first 1K request; medians use
the three independent cold observations, and 1K is not the curve denominator.

The curve reporter verifies distinct worker startups, the full computed-token
count for cold requests and five measured requests per length. Its percentiles
describe request-average round costs, not individual GPU-round latency. It
refuses to overwrite a frozen curve and checks absolute costs and context
increments. A candidate can now pass despite exceeding the prefill ratio;
slowing an anchor still fails the absolute-cost gate.

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
independent screen, not three-startup acceptance. It failed the superseded
prefill-ratio gate. Subsequent V-prefetch and qk2/page/V-prefetch candidates
each passed their own memory/race/synchronization checks before model screening.

The actual model capture uses a 3296-token KV page with strides
`(1687552, 256, 256, 1)`. The initial sixteen-layer performance screen used
1648-token pages; both are correctness cases, and `--performance-page` allows
timing the exact captured page geometry. The loader also retains the
8-byte-only stride fallback. No context result is extrapolated to 256K.

Private launch/build manifests and raw results are retained in the task
artifact archive; generated libraries and private cache paths are excluded
from Git. The new serving route remains explicitly opt-in and has not been
enabled by default or merged.

## September 10 implementation and rejected candidates

The qk2/page/V-prefetch operator takes 2.401/4.417/8.444 ms for sixteen
independent layer allocations at 32K/64K/128K with actual 3296-token pages.
The corresponding initial model startup gives 16.143/18.521/20.484/24.452 ms
at 1K/32K/64K/128K. Tokens and acceptance match the frozen baseline. This
still requires integrated-route quality and repeated-startup admission.

Next-K prefetch preserves byte-exact output and full FP32 workspace, but
both tested load/softmax warp partitions lose performance. Eight load warps
cost 8.815 ms at 128K versus the 8.439 ms paired control; four load warps
cost 9.394 versus 8.444 ms. The extra warp-role work outweighs the overlap.
Neither is selected for serving. The extended byte gate covers 132096 visible
tokens, providing bounded generation headroom after a 128K input.

Increasing to 160/320 splits also loses: 128K costs 9.002/9.897 ms versus
8.469 ms for 80. The independent FP64 screen records final-FP16 max error
0.001952 for all three at 128K; this does not replace a native pre-cast FP32
audit or model admission. Arithmetic variants remain rejected and disabled.

`VLLM_SM70_E4M3_LONG_ATTENTION_MANIFEST` enables the experimental loader and
an additional MRV2 B1/q8 graph. The loader verifies the actual DSO SHA and
module identity, accepts the 80-split six-head workspace, and retains native
input validation. Eligibility depends on q8/GQA6/D256 E4M3 tensors and
validated 1648/3296 page layouts, not target weight quantization or model name.
The descriptor carries a 132096-token upper bound; replay chooses it from
the existing CPU sequence-length upper bound. Larger bounds and other shapes
retain the full-context graph. No device-to-host length read is introduced.
Workspaces are fixed per operator source SHA, capacity, device and CUDA stream.
CPU tests cover the boundary, fallback, switching back, missing captures and
refusal to read a device hint. Native graph-switch/model gates remain pending.

The integrated serving source `239d71c7100b3bce5526268be2cafb4cff8ba8f2`
uses candidate source SHA
`8459d57c6b72993ba47f5c3fe3953bd8343c4174974a05f329984e1ef070f738` and DSO SHA
`dac8262f3d023ce35f1618bbd6fe0f569993f1e9e1f2a60e8291880f76a97339`.
The first unprofiled integrated startup gives 16.078/18.382/20.356/24.326 ms
at 1K/32K/64K/128K, with pure decode 293.7/210.2/237.9/177.7 tokens/s.
All 24 request token sequences, acceptance counts and finish reasons match
the frozen control. This is still a single-startup screen. Three independent
paired startups and the frozen seeds 0/1/2 natural-output campaign follow.
The selected DSO also passes expanded memcheck, racecheck and synccheck
coverage, including the actual 3296-token pages, with zero errors/warnings.

The first fixed-prefix diagnostic completes its 1K control/candidate/control
captures, then exhausts GPU memory during the 32K warmup. Snapshot buffers
grow after the initial memory profile; this instrumented failure is not a
performance result. The retry reserves additional diagnostic memory by using
GPU memory utilization 0.6; capacity stays 262144 and uninstrumented performance
runs keep 0.8. The partial captures and failed report remain in the archive.

For the completed 1K captures, all full-vocabulary logits, distributions,
top-p support, top-1 and EOS probabilities are exact in both A/B and A/A.
There are 320 raw intermediate mismatches in each pair. Every mismatch is
either a bijective physical-slot renaming or unused convolution storage:
prefill writes only `kernel_width - 1` history columns; the verifier reads the
window beginning at `num_accepted_tokens - 1`. With no initial prefill state,
the old convolution allocation is not read. All verifier output storage is
compared in full. The offline comparer retains raw differences and separately
reports their explanations; it rejects changed live history, invalid selectors,
padding-to-live changes and inconsistent or aliased slot mappings. It must not
use repeated-run TV as a numerical tolerance. EOS IDs come from the frozen
generation configuration, not another tokenizer's constants.

## Repeated integrated results and the expanded target

Three independent paired startups complete 144 requests (72 per arm). Each
startup includes one cold warmup and five measured requests per context/arm,
with reversed arm order in the second startup. All paired token sequences,
finish reasons and acceptance records match. The unprofiled request-median
results are:

| Context | Paired control, ms | Candidate, ms | Candidate pure decode, tokens/s | Accepted drafts/round | Emitted tokens/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 16.210 | 16.130 | 292.76 | 3.778 | 4.741 |
| 32K | 25.553 | 18.371 | 210.32 | 2.894 | 3.879 |
| 64K | 34.701 | 20.418 | 237.20 | 3.863 | 4.863 |
| 128K | 52.826 | 24.451 | 176.76 | 3.339 | 4.339 |

The candidate passes absolute-cost and incremental-cost checks against both
the original frozen curve and the new paired controls. The 32K-to-64K increment
is 2.047 ms, or 0.06397 ms per additional 1024 tokens; 64K-to-128K adds 4.033 ms,
or 0.06302 ms per 1024. These results have not reached the new 17/18/20 ms
targets. The complete natural-output campaign remains a separate admission.

The diagnostic retry completes all six fixed tapes: 1K, 32K, 64K, 128K, MBPP28
and MBPP3. Each arm has 96 all-rank target snapshots. All native logits,
full/sampled distributions, support sets, top-1 and EOS probabilities are
exact in control/candidate and repeated-control comparisons. The 1768/1752
raw state differences are fully explained by the validated storage layout;
no live-state or other unexplained differences remain. Diagnostic GPU memory
utilization 0.6 provides 451076 KV token slots, exceeding the unchanged
262144 service capacity. These dumps do not contribute performance samples.

The integrated 128K trace now attributes 8.615 ms to target attention and
7.387 ms to the three QPN2 projection categories, averaged across ranks. Draft
proposal GPU service is 3.919 ms. The same fixed rank 0 has a 26.419 ms round
interval and 24.423 ms GPU union. Across critical ranks, actual profiled round
p50/p90/p99 are 26.524/26.631/26.754 ms. Those are individual **profiled**
intervals and must not replace the unprofiled request-average distribution.

The first expanded operator screen passes 45 byte-exact output/full-FP32-
workspace/canary cases, including 262152-token physical-page and stride
boundaries. At 261888 tokens the sixteen-layer attention working set takes
70.344 ms for the frozen control and 16.335 ms for the selected one-stage
candidate. This is not a 256K model result. The restored serving control's
single cold/warm screen gives 2265.5 cold-prefill tokens/s and a 95.694 ms
warmed complete round; the bounded experimental graph is deliberately not
selected beyond its current admitted domain. Repeated 256K model acceptance
still needs a separately validated extended serving route.

A new private feasibility builder separates compensated QK production from
two disjoint PV column partitions. It retains 80 logical context partitions,
K16 compensation, N32 online updates and probability residual products. The
QK producer uses 80 registers and 48384 shared-memory bytes; the PV consumer
uses 102 registers and a 25248-byte shared layout, without spills. These are
compiler resource observations, not achieved occupancy. Extra score storage,
kernel boundaries and repeated softmax work may erase the benefit, so byte
checks and complete-working-set timing decide whether to continue. Prototype
score scratch is capture-owned; no serving route is installed by this builder.
Its first screen preserves output and complete partial/max/sum bytes in all
65 cases, including the expanded boundary (130 checks across the one-stage
and staged candidates). It is slower: 32K/64K/128K/261888-token attention costs
3.162/5.849/11.194/21.714 ms versus 2.401/4.417/8.451/16.341 ms for the paired
one-stage candidate. The staged source SHA is
`d8493061867f9d044ce7a70e2298306816d0f283aaa84984c69031b668aee825`; DSO SHA is
`02b2a3080c99ad1c837e98fda1222eeb25fefca71ca97e23b939122f60b32f2d`.
It is rejected for serving. A bounded operator trace separates producer and
consumer costs before any follow-up; extra parallelism alone is not a gain.
The selected one-stage DSO separately passes extended-boundary memcheck,
racecheck and synccheck with zero errors.

Prior rejected experiments remain recorded in the context-cost and long-verify
worklogs. Historical E5M2 and FP16-partial Pack-GQA timings are design references,
not quality/performance evidence for this E4M3 FP32 path.
