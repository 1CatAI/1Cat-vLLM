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

## Expanded-domain model checks and terminal scheduling

A private process-only extension raises the experimental graph bound to 262152
before capture. The public source remains bounded at 132096, and service
capacity remains 262144. The original selected DSO completes one startup with
one cold warmup and five measured requests per arm at 261888 input tokens plus
256 output tokens. Median complete cost is 95.419 ms for the control and
38.602 ms for the candidate. All token IDs, acceptance and finish reasons
match. Accepted drafts/round are 4.02 and emitted tokens/round are 5.12.
This is a capacity-bound performance probe, whose responses reach the output
limit; it is not a natural-EOS quality case or three-startup acceptance.

The 256K fixed-prefix control/candidate/control diagnostic also has exact
logits, TV, support and EOS probabilities. Its 272 raw differences per
comparison are explained storage differences, with zero unexplained changes.
The separate natural-output campaign retains exact paired 10107-token
HumanEval and 75828-token LiveCodeBench responses, both ending naturally.
All twelve structured pairs (JSON, schema, one tool, parallel tools; seeds
0/1/2) have exact tokens/acceptance, valid structure and natural termination.
The other sixteen code pairs are separate resumable jobs. The first of those
also completes an exact 51562-token LiveCodeBench pair with natural EOS; it
does not complete the rest of the campaign.

The expanded 256K middle-window trace attributes 16.517 ms of rank-average
GPU service to target attention and about 7.387 ms to QPN2 projections.
Its profiled critical interval p50/p90/p99 is 33.834/34.201/34.822 ms.
Attention agrees with the actual-page independent working set; these data do
not establish an extra allocation/TLB bottleneck.

A second trace observes actual B1 scheduling through the end of the request.
At computed position 262139, after 252 emitted tokens, the final four steps
have one scheduled token and zero proposals. They leave the FULL q8 graph and
run eager target forward. The scalar E4M3 FP32 partition attention kernel has
grid `(1,6,256)`, 256 threads, 40 registers and 12880 static shared bytes;
its sixteen target-layer calls cost 59.377 ms per captured q1 step/rank.
The draft phase still runs. This explains a substantial terminal penalty and
identifies a separate optimization scope. Full request accounting must retain
those steps even though the speculative-round counter does not count them.
Adjacent profiled scheduling intervals overlap GPU work; their service sums
must not be presented as a closed wall-clock decomposition.

## PV reuse and quality localization

The PV-reuse builder interchanges independent M fragments so a raw V fragment
and its residual-scaled copy are loaded/formed once per N16 panel. Each output
accumulator still consumes main0, residual0, main16 and residual16 before its
N32 online update. Source SHA is
`c70e6046c374d18f1c51ad126622e03ecfc34a46a656de225ae2376d384968c0`; DSO SHA is
`30c468456c6e5bfb8d97819a3cad1d6e598db20d0ca18e9ca2d681b12ef32961`.
It uses 112 registers without spills and the same 73728 shared-memory bytes.
All 130 paired operator checks pass, as do its own extended memcheck,
racecheck and synccheck. The actual-page sixteen-layer costs are:

| Input tokens | Prior selected attention, ms | PV reuse attention, ms |
| --- | ---: | ---: |
| 1024 | 0.580 | 0.552 |
| 32768 | 2.401 | 2.244 |
| 65536 | 4.428 | 4.111 |
| 131072 | 8.471 | 7.841 |
| 261888 | 16.384 | 15.150 |

Its first standalone model startup is **not admissible**. The 1K responses
match retained controls, but longer free generations differ, including
acceptance and a 64K finish reason. The 32K/64K/128K request medians of
18.280/19.993/23.727 ms therefore cannot establish a paired speedup; 256K is
39.875 ms and does not improve the previous screen. The experiment also uses
a fresh compiler-cache namespace and the expanded graph domain. Fifteen
shared native-library hashes still match the frozen run. Do not attribute
the trajectory difference to the new operator without an exact-input check.

The next 32K/128K fixed-prefix A/B/A captures have exact native logits and
zero TV, support changes and unexplained state differences. Comparing their
control against the earlier startup's matching fixed tapes is also exact.
These checks do not replace the failed free-generation gate. A diagnostic
native shadow and natural proposal/state captures are used to locate the
first difference; the new candidate remains disabled for promotion.

Paired QK products are another independent scheduling experiment: produce
two separate zero-initialized K16 products, then consume their compensation
updates in the original order. All 130 operator checks pass. The 128K working
set is 7.769 ms versus the paired PV-reuse parent's 7.818 ms; at 261888 it is
15.001 versus 15.095 ms. This small local benefit has no model admission.
Its source SHA is
`0f1d3703382b94ccacee3958eb9f89e0b4f13cdb25ca48269ef7e31006fa731f`; DSO SHA is
`14377f48ce548858c8d2af830e892cb68abe3abe1da29ee2d60713698117a30a`.

The staged QK/PV follow-up with preferred shared carveout 100 does not help:
128K is 11.208 ms versus 11.193 ms without the preference and 8.446 ms for
the earlier one-stage selection. The resource occupancy API permits two PV
blocks/SM with either preference; this is a resource bound, not achieved
occupancy. The producer/PV trace costs 3.580/8.232 ms, motivating reuse and
consumer scheduling work instead of claiming that extra parallelism suffices.

An independent warp-pipeline prototype uses eight producer and sixteen
consumer warps, two score/V panels and named ready/free barriers. Its initial
binary (source `416d076ee5962e1587b3c0c9854eb8a1837ba808bc1bcf82a804645216b143cf`)
fails the expanded byte-equality check despite zero reported synccheck errors.
It also spills under the 768-thread register limit. It has no performance or
serving acceptance. A serialized variant isolates overlap from other causes;
both preserve failed artifacts. The serialized variant still produces NaNs
on independent random inputs, so disabling overlap alone does not fix the
prototype. Constant-input, first-tile checks follow before any timing claim.
Named-barrier synchronization follows the
[PTX producer/consumer memory-ordering contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar),
including explicit participating-thread counts and unaligned instructions
for divergent producer/consumer warps.

The structured-client diagnostic also found a prompt-counting defect:
canonical JSON serialization reordered tool/schema keys before rendering,
whereas generation preserved their insertion order. Actual prompts differ
in token count (339 versus 338 for one tool and 452 versus 454 for parallel
tools). The private client now uses the chat renderer with generation's wire
order, retaining the original counts as evidence and requiring equality to
the generated request's reported prompt count. Sampling, prompts, natural
EOS and total context capacity are unchanged.

The natural-state diagnostic also needs to find auxiliary target states
through the context-probe sampling wrapper. Its earlier immediate-caller
lookup fails there before a scored observation. The revised lookup requires
both the same model-runner object and the same input-batch object, rejecting
unrelated frames. The affected CPU suite passes 19 tests with one GPU skip;
this change observes existing tensors and does not alter inference arithmetic.

An isolated q1 builder reproduces the traced scalar E4M3/FP32 partition and
1024-token merge using the frozen serving source. The extracted reference and
PV-unroll4 candidate pass all 24 output/full-workspace/canary checks against
the frozen production DSO, including stride padding, zero-length replay and
262144 visible tokens. The sixteen-layer 261888-token workset is
60.565/60.574/60.580 ms for production/extracted/unroll4 respectively; the hint
has no useful performance benefit. Source/DSO hashes are:

| Scalar q1 variant | Source SHA256 | DSO SHA256 |
| --- | --- | --- |
| Extracted reference | `903401903455e91fda49685b170b7bd932b1975d291f0c022b7a951ad99d0501` | `c521b0f96fa17bd8f474e5104ed2b35d1c82074d73af2e447172c90a05b0d584` |
| PV unroll4 | `e81f28ba2bf8215e7a6e037c2f599697e84cef582d1be0148c71f6f2c082a6f5` | `e9336ac7bd888bd3a9b43559423c4661de74b8a5d46d6ee1c7591afc845c06cc` |

Explicit four-value prefetch applies the original FP32 FMAs in token order
and passes all 24 checks, but regresses the sixteen-layer 261888-token workset
from 60.564 to 63.124 ms. It is rejected without a model trial. Its source/DSO
SHA256 are `9723b252d9c54ebd96d1aa99c4ea3544c8cf66696e58e5fafa218d161e0619a8` /
`e822ade9e58e14510be2c40818327c4a608e4978d9970e5871d7fc3b231bf69d`.
A separate six-head scalar prototype shares each decoded K/V value across
independent per-head FP32 chains, retaining the original 1024-token partitions
and merge. It requires its own full-workspace checks and performance screen.
The scalar builder installs no serving route and does not change q8 arithmetic.

## September 10 online-softmax correction and natural diagnostics

The first warp-pipeline failure is localized to code extraction: the builder
copied the TWO_PASS statistics-only row loop instead of the online softmax
branch. It never wrote P, the compensated probability residual, or row rescale
before PV read them. A constant-input first-tile probe confirms an incorrect
P/residual with correct V and max/sum, including with overlap disabled. The
builder now anchors the online branch and requires all three publications and
its warp memory barrier. Corrected source
`15b44fd2fefe692000d18a112cb3f278802c6e5bc7448a62c6a885dc02014385` has DSO SHA256
`2fdf18b3d0d71d84a9797a046554fe56e15a70182151b15206cbe06080b73717`.
This correction is not a quality or speed pass; the original failed artifacts
remain retained. The current 768-thread prototype still has register spills.

The no-forced-token 32K control/candidate/control audit has 30 observed steps
and all four ranks. After checking consistent physical-slot renaming and
source-defined unused convolution storage, all captured target/proposal
logical tensors agree. Each comparison retains 2352 raw storage differences.
The natural comparator now supports the same explicit convolution-width
explanation as the fixed-prefix comparator; tests reject changed live values
and inconsistent mappings. The focused suite passes 22 tests with one GPU skip.

This audit cannot explain away the PV candidate's free-generation failure.
Its 128-token output differs from the retained uninstrumented selected result
at token 97 (zero based), while the failed PV result first differs at token 59.
In addition, graph-captured persistent shadow counters contain non-count
values after control replays. They are invalid evidence. A revised diagnostic
allocates persistent counters and reference workspaces before graph capture,
matching initial workspace contents before each native comparison. A separate
uninstrumented same-startup route A/B is required to distinguish candidate
behavior from startup/layout or diagnostic effects. No claim of a model-wide
PV equality pass is made from these observations.

The original selected combination has completed 17 of 30 frozen natural-EOS
pairs at this checkpoint. This includes all 12 JSON/schema/tool/parallel-tool
pairs over seeds 0/1/2 and five seed-0 code pairs. Completed long-code outputs
include 75828, 51562, 83004 and 24318 tokens with exact output IDs, acceptance
and finish reason. These are paired non-regression observations, not new
benchmark scores. Remaining natural code pairs continue at complete-pair
boundaries alongside the optimization queue.

## Matched PV evidence and scalar q1 whole-round result

The subsequent uninstrumented same-startup A/B uses the original three-group
route as control and PV reuse as candidate. All 12 pairs at 32K/128K have
identical output IDs, acceptance and finish reason. Both arms reproduce the
previously observed historical token differences at positions 59/21,
respectively. The prompt corpus SHA, prompt IDs and sampling contract match
the retained runs. Thus the observed cross-startup drift is not specific to
PV reuse. It remains unresolved and is not converted into an allowed quality
tolerance. Complete-round medians are 25.584/18.401 ms at 32K and
53.061/23.817 ms at 128K for this single control/candidate startup.

Moving persistent shadow buffers outside the shared graph pool repairs the
counter corruption: the actual-input 32K diagnostic observes exactly 1856
native comparisons (29 q8 replays × 16 layers × four ranks), with zero changed
output, numerator or max/sum elements. Control replays leave the counters at
zero. All captured target/proposal logical tensors agree across the 30-step
control/candidate/control runs, retaining the explained raw storage differences.
The original selected natural-output campaign is now 19/30 complete; seed-1
HumanEval-10 adds an exact 16503-token natural-EOS pair.

The scalar six-head candidate passes memcheck, racecheck and synccheck with
zero errors and six byte/canary checks in each run. Its first uninstrumented
model A/B keeps the prior selected q8 kernel in both arms and uses the scalar
candidate only for eligible eager q1 calls. All four ranks observe 384 real
candidate calls over six boundary requests, using CPU context bounds
262140–262143; no GPU length is read on the CPU. All 12 request pairs have
identical output IDs, acceptance and finish reason:

| Input | q1 control complete round | q1 candidate complete round | Pure decode control/candidate |
| --- | ---: | ---: | ---: |
| 1K | 15.959 ms | 15.893 ms | 295.901 / 297.120 tokens/s |
| 261888 | 38.563 ms | 36.428 ms | 132.252 / 140.000 tokens/s |

At the boundary, accepted drafts/round and emitted tokens/round remain
4.02 and 5.12. This is one paired startup, not the required three. The complete
round includes the terminal q1 overhead. The scalar optimization reduces it
by 2.134 ms, or 5.53%; the 22 ms goal is still unmet. The initial service-client
attempt failed to unpack the collective RPC results envelope before any
benchmark request; the corrected client validates all four rank identities.
The explicit worker probe and operator harness are now included for review.
No default route is enabled.

The revised QK/PV warp pipeline is rejected on performance. At 128K/261888,
the PV-reuse baseline is 7.812/15.093 ms, versus 8.834/17.123 ms for eight
producers and 9.933/19.340 ms for six. Both retain the same 80-register limit
and spills. Four producer warps remove spills at 96 registers, while preserving
all 65 byte checks, but are slower again at 11.301/22.210 ms. Removing spills
alone does not provide a useful pipeline; none of these variants enters a
model trial.

A separate feasibility probe predecodes the exact E4M3 values into FP16,
without restoring precision lost by E4M3 encoding. It passes 65 byte checks but
only changes the 261888-token attention workset from 15.090 to 15.034 ms,
excluding population/invalidation and the additional mirror memory. It is
rejected: the gain does not justify a mirror cache. The recorded failed build
attempt caught use of the FP8 paired-loader option with FP16 data; the tested
probe uses the existing FP16 vector loader. No serving KV representation changes.

Available Triton caches share 189 compilation identities; 91 cubin hashes
differ. For those 91 entries, PTX agrees after excluding debug location/file
sections and assert-filename strings. This rules out a PTX arithmetic change
in those shared artifacts, not a change in actual dispatch, arguments or
machine-code behavior. All raw hashes and the excluded differences are retained.

Native manifests for this stage:

| Candidate | Source SHA256 | DSO SHA256 |
| --- | --- | --- |
| q1 six-head KV reuse | `e624c2f2c2eaa0d770b46aef7d2b4f84a710bc895bf0141b4670fd21ba498f69` | `8d6ede73f56b9edc270eab507d23d56f437567db362263c96d60b2a5ae05f98a` |
| four-producer QK/PV | `b005452e565468012b25a6f53c297ebc9a3482adde545ac041bfd170b264cde4` | `7dab6ff2d14e60a6f2c9803b9039bceeefc0dd879005f9333beef20d8ede8b99` |
| six-producer QK/PV | `a79561fc9c1a8a0a06590455e8e3d64807efb70a6aed412a50abdf229a1c12f7` | `1e9da088a750af9892b8c50a12187c652e3061dbd645091152cba7ed01a8b74b` |
| lossless decoded mirror probe | `eca466df3e1a64c945627e70447470250b83a1683cb6a2b32fd23724d06071e5` | `c31c0a3c136f453ea3f2b27f0fdd1275d3c29963ec3aa0ded889100651b0aeba` |

## Three-startup q1 result and the next operator screens

The six-head scalar q1 comparison has now completed three independent paired
service starts, each with a cold request and five measured requests per length
and arm. All 36 request pairs (72 requests) have identical output IDs,
acceptance and finish reason. The summary checks distinct server PIDs, frozen
source/library hashes, the prompt/sampling contract and 384 candidate calls on
each rank in each startup. Both arms retain the previously selected q8 kernel.

| Input | Control complete round | Shared-q1 complete round | Pure decode control/candidate |
| --- | ---: | ---: | ---: |
| 1K | 15.951 ms | 15.893 ms | 296.070 / 296.663 tokens/s |
| 261888 | 38.563 ms | 36.436 ms | 132.268 / 139.854 tokens/s |

Complete-round values are the median of the three startup medians; pure
decode values summarize the measured requests. Terminal q1 time is included.
Boundary accepted drafts/round and emitted tokens/round remain 4.02 and 5.12.
The second startup's 1K candidate is 0.184 ms slower; it remains in the
aggregate and makes no new scalar calls. The aggregate short-context median
does not regress. The request-average p50/p90/p99 distributions are retained
separately from actual GPU-step timings in
`scalar-q1-three-startup-summary.json`.

An independent actual-input boundary shadow also passes: 64 q1 comparisons
per rank, 256 in total, have byte-identical output, full partial numerator and
max/sum workspaces against the frozen production scalar operator. The paired
requests retain identical output and acceptance. Shadow timing is diagnostic
only. The original selected q8 natural-EOS campaign has completed 20/30 pairs,
including a new exact 62396-token seed-1 LiveCodeBench-21 output.

The next q1 candidate constructs a 256-entry FP32 shared-memory E4M3 lookup
table using the original decoder. A completed initialization barrier precedes
all reads. It reuses the same decoded values across six independent head
chains, retaining the original dimension/token order, partition size, FP32
state and merge. Thirty output/full-workspace byte checks pass. Its sixteen-layer
261888-token operator workset is 15.965 ms, versus 33.326 ms for scalar sharing
alone and 60.577 ms for the frozen production operator. This is an operator
screen; its own sanitizer and full-service gates are required. It does not
broaden the public worker probe's context eligibility or enable a default.

Revisiting staged QK/PV with PV-value reuse still fails the performance screen.
One D256 column uses 512 threads; two D128 columns use 256 threads each. Both
retain all 80 logical partitions and N32 updates and pass 130 byte checks in
total. Sixteen-layer 128K/261888 worksets are 9.276/17.899 ms for one column and
10.537/20.420 ms for two, versus 7.814/15.088 ms for the one-stage PV parent.
Neither staged variant advances to a model trial. The one-column diagnostic
trace attributes 3.931 ms to QK, 6.762 ms to PV and 0.229 ms to merge per
sixteen layers at 128K. These instrumented service sums cannot be used as
unprofiled latency or assumed overlap savings. The PV kernel uses 108
registers/thread and 49824 bytes of dynamic shared memory; these are static
resources, not achieved occupancy.

A separate explicit `profile` entrypoint records intra-CTA `clock64`
boundaries around K loading, QK with V loading, online softmax, and ordered
PV. It exists to distinguish phase dependencies before another scheduling
rewrite. Timestamp deltas include probe overhead and waits and do not report
hardware utilization. No serving route imports the probe.

| Candidate | Source SHA256 | DSO SHA256 |
| --- | --- | --- |
| q1 shared E4M3 lookup | `82e3e0486f549125b94f5b38555e03e792fea51e349d3d5f972cf269871f632f` | `576b7dc765dd6850a6760a7b4dd248e803570d5c5d193ee3af83fc0d68d89a7a` |
| staged PV reuse, one column | `9aed322a8a098e9bb51f7113a0774bab3ca66766051281c20f54bfd9443c3b20` | `f01d7ea03de216445a325a95e5213fdb9f18b0ae8ff1299acabfa3aac7ffafce` |
| staged PV reuse, two columns | `fe00ab5a5e39e177f22bc3ecfda24d155bc8ae25ca78784c61f01d2150aa006f` | `16e5d1271f11e3201c5db9f61120cb6aea51c72d9b502007ddcfaec07f4ddfba` |
| intra-CTA phase probe | `aac2d83f53b52e738e9b71903ddc4f077d4ecb00ab95bcd354860067226f5534` | `abdea58968713b5017bf2d6f5898735a9cb741016264fa79c5df458d3d147906` |

The lookup candidate's own memcheck, racecheck and synccheck each complete six
byte/canary checks with zero reported errors. Its first boundary shadow uses
261888 input tokens and 256 output tokens: the two outputs and acceptance
records agree, but this trajectory exercises no eligible scalar q1 calls.
The client rejects the missing coverage. This is retained as a route miss,
not a scalar quality pass. A separate 262136-input/eight-output boundary
diagnostic is used to leave no full q8 window; its timing is excluded.

A subsequent uninstrumented paired startup does exercise the lookup operator:
384 calls per rank over six boundary requests. All twelve request pairs are
exact. Complete-round medians at 1K are 16.028/15.929 ms for control/lookup,
and at 261888 are 38.822/36.796 ms. Boundary pure decode is
119.426/126.003 tokens/s, accepted drafts/round 3.563636 and emitted
tokens/round 4.654545 in both arms. These acceptance values differ from the
older three-startup scalar campaign, so its 36.436 ms result cannot rank the
two scalar implementations. The unchanged control first differs from that
older output at tokens 114/62 for 1K/261888. All fifteen checked shared native
libraries retain their hashes. This startup repeatability issue is retained;
it is not assigned to the lookup kernel, which is disabled in the control.
An explicit same-startup control/shared/lookup comparison follows before
claiming incremental lookup benefit.

The clock probe completes sixteen-layer working sets at 128K and 261888 with
byte-identical output and full FP32 workspaces. It records 65536/130944 N32
tiles respectively. Aggregated CTA cycles at 261888 divide into 14.86% K load,
34.19% QK with V load, 20.16% online softmax and 30.79% ordered PV. The 128K
fractions closely agree. These include waits and timestamp overhead, and are
not kernel wall-time fractions or achieved utilization. They motivate removing
common-case branches and auditing the QK dependency chain. The original
selected natural campaign is now 22/30 exact natural-EOS pairs; the new seed-1
LiveCodeBench-64/93 outputs contain 94767/47005 tokens respectively.

The explicit short boundary-generation shadow completes 48 actual scalar
comparisons per rank (192 total), with byte-identical output, partial numerator
and max/sum storage; both requests retain identical output and acceptance.
The timestamp probe also passes its own memcheck, racecheck and synccheck,
including full 262144 visibility, page crossing, graph replay and timestamp
canaries. These checks validate the diagnostic; they do not remove its timing
overhead or grant model admission to another operator.

## Fixed-q8 tile specialization and a separate arithmetic screen

The fixed-q8 prototype clones the existing partial kernel with a constant
query-row count, selected only when the actual query shape has eight rows.
Queries with two through seven rows retain the original kernel. A second
variant removes visibility checks only when the complete N32 tile precedes
the minimum of all eight GPU row lengths. Zero-length and rejected rows,
partial tiles and the causal tail retain the original masking. No CPU copy of
GPU lengths is introduced. All 80 logical partitions, K16 compensation,
probability residual products and N32 numerator/max/sum updates are unchanged.

The two variants pass 130 byte-equality checks in total, including full FP32
workspaces. Sixteen-layer worksets show that fixed-q8 specialization alone
regresses 128K/261888 from 7.816/15.092 to 7.932/15.325 ms and is rejected.
Adding the complete-visible-tile specialization reaches 7.703/14.872 ms,
approximately 1.4% faster than the PV-reuse parent. The smaller gain needs a
same-startup complete-round A/B against that parent, using separate MRV2
graphs; operator timing does not establish service benefit. Its native
sanitizer checks precede that service job.

QK remains a substantial phase. A separate arithmetic prototype sums the
unchanged FP32 K16 products with explicit FP64 additions, then rounds to FP32
before the original scale and softmax. It preserves N32 state updates and PV
compensation, but changes the QK summation arithmetic and has no admission.
The explicit nearest-even operation follows the
[CUDA double-precision intrinsic contract](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html).
This is a feasibility experiment, not a claim that FP64 improves either
reference error or latency for this workload.

The builder can additionally expose the actual FP32 combine accumulator before
FP16 conversion. The independent numerical harness first proves that each
diagnostic build retains the regular build's full partial/max/sum workspaces
and produces exactly the same final FP16 values after conversion. It then
compares both ordinary and pre-cast outputs against independent FP64 QK,
softmax and PV, reporting maximum absolute error, p99 absolute error and
relative L2 for two seeds at five lengths through 262144. An increase in any
registered error metric rejects the arithmetic candidate before performance
testing. Passing this operator screen still does not establish recursive or
model quality, logits/distribution equality, or acceptance non-inferiority.

The new options are private builder switches: `--specialize-full-q8`,
`--all-visible-tiles`, `--qk-fp64-sum`, and `--diagnostic-output-fp32`. None
changes the frozen serving source or enables a production default.

The arithmetic screen completes and rejects the FP64-sum candidate before
timing or any model trial. All ten pre-cast diagnostics agree with their
corresponding regular builds' workspaces and converted outputs. All final
FP16 reference-error metrics are nonexpanding, but four of ten cases expand
at least one pre-cast FP32 metric. At length 3297, seed 20260910, maximum
absolute error grows from `2.779396474e-6` to `3.216708786e-6` (about 15.73%);
p99 grows from `1.495418274e-6` to `1.531674442e-6`. Two cases also change
final FP16 output values. These are operator differences, not measured model
token flips. The original K16 compensation remains selected.

The exact visible-tile candidate completes 25 byte/canary checks under each
of memcheck, racecheck and synccheck, with zero reported errors or races.
The initial racecheck invocation accidentally included timing worksets and
was stopped; the corrected invocation uses `--correctness-only` and reruns
the complete required checks. Timing collected under a sanitizer is excluded
from all performance claims. The subsequent service A/B compares PV reuse
against visible-tile specialization at 1K/32K/64K/128K/261888, with scalar q1
unchanged in both arms.

| Candidate | Source SHA256 | DSO SHA256 |
| --- | --- | --- |
| fixed q8, rejected speed | `eff0d41b399cbdb343f3b5543cd4577af2571b033fa9a3904d2fd9ad96cd07d3` | `6bfb9f20bc062f91731519faa05254bb53c4342052c6b0b733431f3a1ac35134` |
| fixed q8 and complete visible tiles | `3b0c9688ce17e1870408ef81fb5cd9b63a677b7cfd7d4777b8df77dd0fc24132` | `7dc632c2ff110cc751bceeb5fb683ede6066e443dad673d06c9e05429e01d0a2` |
| FP64 sum of K16 products | `e477e605fb94b9003cf71308da371b9051d7c72eb841e4e6e7fb4dbd6a8484da` | `e86ee04a6c7e949f9d3ded8ca7613b208bd89f4d3b642804d57e3f51f37daa1d` |
| PV parent, pre-cast diagnostic | `f30d85ef59aa86ca99d48aeba96eaa4a63dca8fbe4aaa1c375295d42f6ed284e` | `fc08222e8d768691d052817d89aadb2c7e5a7aa0207397da04fe4d70739ed0af` |
| FP64 K16 sum, pre-cast diagnostic | `9ba8011dbdc11103fb4746e07e3b798953bfa17d54557f973cdcb7638084e353` | `23ae5cb0b011b797e9279c6f2377863e2b337d301423b0ca38f4b28f59254699` |

## Physical N64 and scalar page-map screens

The physical N64 prototype computes two independent QK tiles together but
retains two consecutive N32 softmax/PV updates, the 80 logical partitions,
K16 compensation and probability residual products. It admits only aligned
q8; other query shapes and unaligned strides use its PV-reuse parent. Three
V-loading schedules are screened independently. All fail the working-set
speed gate despite byte-identical output and full FP32 workspaces: the first
two complete 130 checks in total, and the softmax-overlap variant completes
65. None advances to a service or sanitizer campaign.

| Sixteen-layer working set | PV parent | Raw V prefetch | V after QK | V during softmax |
| --- | ---: | ---: | ---: | ---: |
| 131072 tokens | 7.811 ms | 9.757 ms | 7.922 ms | 8.977 ms |
| 261888 tokens | 15.084 ms | 18.927 ms | 15.330 ms | 17.378 ms |

The softmax-overlap run has its own paired parent at 7.815/15.095 ms; it is
not compared by subtracting measurements from the earlier launch. Raw V
prefetch uses 128 registers and spills; V-after-QK uses 116 without spills.
Holding only the first N32 V tile across QK and loading the second during
softmax still uses 128 registers with spills. The wider tile does not produce
a useful local gain on this workload.

The scalar q1 page-map prototype exploits the validated 3296-token page and
1024-token partition contract: each partition spans at most two physical
pages. It replaces per-token page/offset arrays with two page IDs and two
consecutive PV segments, preserving each head's original FP32 FMA order.
Together with and without the existing E4M3 lookup, the candidates complete
45 byte checks. The compact lookup variant improves the 128K sixteen-layer
workset from 8.011 to 7.554 ms but regresses 261888 from 15.965 to 17.436 ms;
neither compact variant is admitted. A separate unused 4096-byte dynamic
shared-memory reservation tests whether the changed resource limit explains
the long-context regression. Resource bounds are not achieved occupancy.

The original selected natural campaign reaches 24/30 exact natural-EOS
pairs, including all twelve structured/tool cases. The final seed-1
LiveCodeBench-131/162 outputs contain 49357/61290 tokens. Six seed-2 code
pairs remain. These comparisons do not establish new benchmark scores or
admit a later untested combination.

The same-startup scalar control/shared/lookup experiment completes all saved
requests with matching output and acceptance, but records no eligible scalar
q1 calls in any arm. Its postprocessing failed because of a missing `Path`
import; offline recovery validates the retained requests and preserves the
original failure. Neither its timings nor its zero-hit counter establishes
incremental scalar speed or scalar operator quality.

| Candidate | Source SHA256 | DSO SHA256 |
| --- | --- | --- |
| physical N64, raw V prefetch | `cf4ac891a5c3e7d38354ae5ec7af8d3e8800f23a6d56113828827bfd9b3e088f` | `1350da745048a53137347715c1d4155ce4770ddd5a87b4745ad186af639206be` |
| physical N64, V after QK | `cf41ce8b2d8d171c900a3d943dc9d8f0698b5493a00f46a816939058340ad60d` | `4f342b5cde1189e6b15bb78e87472f7081354f7e722b390e844151aee265a0c3` |
| physical N64, V during softmax | `1d44e0ae130d906877c9ed24308a6d079cbe5191d72db4a9a85d0737945cd5f2` | `9fecce8b8d1626a6ac3bd3de52b736219e3bcfdf03e9c78a0a731b342fa84bac` |
| scalar compact pages | `46980a019914f5a80861f9e7173aada8f292fe2c0bc2a17d1e1b0c6bddf5252d` | `00c602f201323071c5790443e08d8294febae2dbca53f44701e70bce177815b7` |
| scalar compact pages with lookup | `3cf205e3f3c2744b9a059663a6c5ff3db84cef53fd37ad5b663700154d7d2a97` | `eb54d60463e9811b1c116da25e7f56655925f9bfe0d61f691f48f06e2c5c537c` |
| scalar compact lookup, 4096-byte reservation | `5e68578c0252c7525496abab98aef5ed5f774528ac1dab6c5e1c587912cfa632` | `6d2b2b1ec5e0501de9abf49a81670cca2fb2ee700be66365f98db118a62fee21` |

The next exact q8 candidate keeps each warp's three online max/sum rows in
registers and publishes them at the final output barrier. PV still consumes
the original shared row scales, and every row retains its ordered N32
update. The builder switch `--register-softmax-state` requires the fixed-q8
specialization; it changes no serving default. The generated visible-tile
source without this switch retains its original SHA. Native byte checks,
working-set timing and, if faster, its own sanitizers precede any service
trial. New 32K/261888 complete-round traces of the current visible-tile path
are collected separately from uninstrumented admission results.

## Visible-tile repeated service results and next resource controls

The visible-tile/PV-parent comparison completes three independent paired
startups (PIDs 1094304, 1101605 and 1103057), 180 requests and 90 exact pairs.
All contexts retain prompt, tokens, finish reason, sampling and acceptance.
The scalar q1 operator remains the original implementation in both arms.
The values below are medians of the three startup medians after one cold
request and five measured requests per context/arm. No profiler or tensor
dump is enabled.

| Input tokens | PV-parent round | Visible-tile round | Visible pure decode | Accepted drafts/round | Emitted tokens/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16.008 ms | 15.982 ms | 295.464 tokens/s | 3.777778 | 4.740741 |
| 32768 | 18.222 ms | 18.185 ms | 212.458 tokens/s | 2.893939 | 3.878788 |
| 65536 | 20.084 ms | 20.040 ms | 196.015 tokens/s | 3.030769 | 3.938462 |
| 131072 | 23.786 ms | 23.687 ms | 165.712 tokens/s | 2.984615 | 3.938462 |
| 261888 | 37.547 ms | 37.298 ms | 124.306 tokens/s | 3.563636 | 4.654545 |

These are small gains; the third startup's 32K candidate is 0.005 ms slower
and remains included. The aggregate short-context median does not regress.
All four revised long-context targets and the short-context <15 ms target
remain unmet. Do not rank this trajectory against earlier scalar-q1 trials
with different accepted outputs.

| Visible-tile input | Request-average p50/p90/p99 | Cold TTFT median | Cold prefill median |
| --- | --- | ---: | ---: |
| 1024 | 15.982 / 16.061 / 16.149 ms | 0.273 s | 3989.826 tokens/s |
| 32768 | 18.185 / 18.235 / 18.243 ms | 8.288 s | 3977.766 tokens/s |
| 65536 | 20.014 / 20.058 / 20.065 ms | 18.417 s | 3573.957 tokens/s |
| 131072 | 23.674 / 23.706 / 23.711 ms | 44.248 s | 2971.068 tokens/s |
| 261888 | 37.298 / 37.494 / 37.530 ms | 117.995 s | 2223.944 tokens/s |

Request-average quantiles are not actual GPU-round quantiles. Cold requests
verify the full computed-token count; cached repeat prefill is not used for
the cold throughput. Complete-round increments are 1.855, 3.647 and 13.611 ms
for 32K→64K, 64K→128K and 128K→261888. Their marginal costs are 0.057968,
0.056977 and 0.106546 ms per additional 1024 tokens. The last interval spans
127.75 such units and includes terminal q1 work. See the retained
`full-q8-visible-three-startup-summary.json` and its six hashed arm reports.

The compact scalar lookup with the extra 4096-byte reservation passes all
45 paired operator checks. At 261888, sixteen-layer working-set cost is
15.026 ms versus 17.444 ms without the reservation and 15.967 ms for the
previous lookup. The 128K pair is 7.556/8.013 ms versus the previous lookup.
Changing this resource reservation removes the observed compact-layout
regression, but it does not measure achieved occupancy or isolate all cache
effects. Its own sanitizer campaign and an actual service comparison follow;
this local result is not a complete-round gain.

Register-held q8 max/sum passes 65 full-workspace byte checks but regresses
every workset. Its 128K/261888 costs are 7.787/15.079 ms versus the paired
visible-tile parent at 7.695/14.862 ms. The relevant build uses 125 registers,
a 24-byte stack frame and no reported spills. Reject it before sanitizer or
service trials. Source SHA is
`3fb342d6737c1fe81220b2f18d805c3f394d1e728cda4e41683a7aaa4a15a905`;
the DSO SHA is
`153c8521e3dc173de9806fd5e1aef40e9768be8735c89ab448b1d126ed7aea12`.

A separate `--qk-head-rows` prototype stages Q by head and uses one M8/N32
QK tile per head, storing scores back in the original token/head order.
K16 compensation and all softmax/PV updates stay in place. Matrix-shape
equivalence is explicitly not assumed: the
[PTX WMMA contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-mma)
does not specify accumulation order or rounding for FP16 operations. The
byte gate must pass before timing; any difference instead requires the
independent reference audit. The normal source remains unchanged without
this experiment's switch.

The M8/N32 candidate passes 65 byte checks but is slower at every measured
context. At 128K/261888 it takes 7.801/15.076 ms versus its paired parent at
7.699/14.876 ms. Rotating the next Q/K fragment load ahead of the original
current K16 correction also passes 65 byte checks but loses:
7.883/15.279 ms versus 7.696/14.893 ms. Neither advances. Their source/DSO
hashes are respectively:

| Candidate | Source SHA256 | DSO SHA256 |
| --- | --- | --- |
| M8/N32 QK by head | `b99968cac226fc4443b097f71a4dd0a5ad3882b739eee697ff4f6ca28e04c2a7` | `bafa8a3a5fccb2fc07f94f45a5100b2ee58a6b68d2346a7d3be63d348a2cf06d` |
| QK operand rotation | `cda4d682a262bf859b54885f3044ca8b490fb55100d8277943114c1f43543e35` | `08c18843465cb7ee8e44b69e6ad8842cc9adb051ef33079cb3cae9a46cf149a9` |

The compact lookup/reservation candidate's own three sanitizers each pass
six cases with zero errors or race hazards. Its first uninstrumented service
pair uses visible-tile q8 in both arms and hits 384 actual scalar calls per
rank. All twelve request pairs are exact. At 1K the round median changes
15.995→15.968 ms; at 261888 it changes 37.160→35.602 ms and pure decode
137.245→143.250 tokens/s. Accepted drafts/round remain 4.02 and emitted
tokens/round 5.12. This is one startup, not completed repeated-start admission.

An actual-q1 diagnostic compares original, shared, lookup and compact lookup
using saved live operands and isolated output/workspaces. The retained subset
contains eight unique K/V pointer pairs per rank, with 96 exact candidate
comparisons across four ranks. Its final client assertion incorrectly expected
sixteen unique pairs and the job exits 1. Preserve the original failed report
and the separate subset analysis; this is not evidence that all sixteen
attention layers were sampled. Median per-operand graph times across ranks
are approximately 3.69–3.85 ms original, 2.02–2.10 ms shared,
0.96–1.00 ms lookup and 0.90–0.94 ms compact lookup. These operator events
and sums over eight samples are not complete-round costs. Additional route
metadata is required to close the smaller observed complete-round gain.

## Latest whole-round attribution and small-Q coverage

The new visible-tile 32K trace contains 57 analyzed q8 intervals after edge
exclusion. Its critical-rank mean interval is 20.121 ms, GPU event union
18.499 ms and uncovered time 1.623 ms. A 39.244-ms outlier remains included.
QPN2 service is 7.370 ms, draft service 3.933 ms and target grouped attention
2.379 ms on those same critical ranks. These instrumented values do not
replace the 18.185-ms unprofiled endpoint result.

The 261888 trace observes 54 q8 steps, one q6 step and four q1 steps in the
whole diagnostic request. In the analyzed inner intervals, q8 target
attention takes 15.027 ms and draft 3.889 ms. The q6 interval uses eager
target execution and target attention takes 24.435 ms. The q1 intervals use
the original scalar implementation. A partial verifier therefore loses both
the q8 graph and the q8-specific scheduling wrapper. All such costs remain
in the complete-round denominator; do not remove them to claim a target.

Earlier query-shape fixtures exercised q2 and q5 in addition to q8; they
did not qualify every q2–q7 shape. The benchmark now exposes an explicit
`--tail-queries` suite, a `--performance-query-rows` selector, and a hashed
reference binding to the actual frozen `grouped_e4m3_fp32_paged_fwd` entry.
The reference requires precision revision 4 and retains DSO SHA
`a751fed902279b0de23537c4aad2dc4fee360146d7fce7ef0c4f255a77f48b02`.
Default benchmark queries and candidate entrypoints are unchanged.

All six q2–q7 shapes pass 60 byte checks against that production reference,
including page crossing, padded strides, zero/rejected rows, restored lengths
and graph replay. Sixteen-layer q6 worksets improve from 12.176 to 7.796 ms
at 128K and from 23.717 to 15.066 ms at 261888. Its own small-Q sanitizer
suite precedes any eager service-route extension. No new shape is enabled
merely by adding the benchmark selector.
