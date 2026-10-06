# Flash-Next GGUF MTP4 complete-round optimization

The performance objective is C1 at or below 12 ms per round in the existing
acceptance benchmark, with correct target output, unchanged acceptance and
no C4 regression. A kernel service-time saving is a hypothesis until the same
installed wheel's end-to-end A/B confirms it.

## Measurement contract

Use Flash-Next GSQ-RCO IQ3_S, FP16-loaded MTP4, TP4 on four SM70 V100s,
FP16 KV and FP32 recurrent state. Preserve the benchmark's eight natural
prompts, greedy sampling, natural EOS, 600-token limit, 8192-input/256-output
C1 probe and 128-input/600-output C4 probe. Report tokens per round together
with round latency. A result obtained after acceptance collapses is rejected.

Freeze source and installed core hashes, graph policy, topology, clocks and
configuration for each arm. Obtain the unprofiled baseline first; use a
separate short node trace to inspect order and overlap. Profiler overhead must
not be subtracted from or reported as accepted latency.

The per-round ledger must distinguish target verification, drafting,
sampling/state updates, host preparation, peer skew and GPU activity gaps.
Union intervals before reporting busy/idle time; summed concurrent kernel
service is not a wall-clock ledger. Leave unattributed time explicit.

## Starting evidence

The previously accepted main path measured 18.534 ms/round at C1 with
4.886 emitted tokens/round. This is a historical reference; a new normal-wheel
baseline and trace are required before choosing the next implementation.

The operator-integration branch reports 18.602 ms/round with its new switches
off and 18.394 with all switches on. Its HCX-off arm reports 19.780, but
the off arm's individual cohorts range from 18.540 to 20.583 ms/round.
Consequently the reported 1.4-ms HCX contribution requires a matched repeat.
Other retained experimental arms reduce tokens/round to approximately one;
these are numerical/dispatch failures, not eligible speed comparisons.

The branch is imported into an isolated source tree for normal-wheel tests.
All new routes retain explicit configuration switches for ablation. No route
is promoted based on the aggregate microbenchmark estimate.

## Normal-wheel HC load qualification

Source `4309a6e9e5ab787184d31f494711d801e25d0815` built core SHA256
`9f3f9264edcb80998e92ba71cfbe2f52ae077725a86e3771d0bb4df210532cf3`.
Same-process, same-wheel four-rank ABBA across eight real HC weight pairs:

| Batch | Original, µs/pair | Optimized, µs/pair | Saving |
| --- | ---: | ---: | ---: |
| M=5 | 21.428 | 20.820 | 2.84% |
| M=20 | 31.690 | 31.035 | 2.07% |

These are maximum rank median graph times. The M=5 delta estimates only
0.058 ms across 96 pairs; it does not substantiate an end-to-end speed claim.
The research DSO's larger saving must not be substituted for this result.

Output and injection match by raw FP16 bit pattern on all ranks at
M=1,2,4,5,8,10,20, including changed-input graph replay. Tag-wrap/batch-transition
stress against the replicated reference has block relative maximum error
2.13e-4 and zero injection error. Four CPU policy tests and installed dependency
checks pass. Clean-process ABI, core hash and loaded-library checks exclude
private kernel DSOs.

## Current complete-round baseline

The source-complete control wheel measures 18.531 and 18.516 ms/round in the
two unobserved C1 cohorts: mean **18.523 ms/round**, **4.886 tokens/round**.
C4 measures **43.190 ms/round**. The eight-prompt mean acceptance is
47.630%, with prompt-cluster bootstrap 95% CI [37.225%, 60.005%]. The pooled
acceptance is 44.305%; it uses a different denominator and must not replace
the prompt mean. The two short natural completions terminate at EOS.

Historical-wheel long outputs differ on six of eight prompts. Consequently
this run does not establish historical bit equality. New routes require a
same-wheel control, teacher-forcing comparisons and matched acceptance data.

### Recorded activity ledger

The node trace retains 37 common TP windows, aligned by target replay ordinal.
Every target replay contains 1,263 kernels on each rank. Trace window median is
20.918 ms, versus 18.523 ms unprofiled; profiler durations are composition
evidence and must not be used as accepted endpoint latency.

The following mean partition closes each common 21.383-ms window. Intervals
are unioned and assigned once, with target taking priority over draft, graph
work over outside-graph kernels, and kernels over copies. This is an activity
partition, not an attribution of causal critical-path savings.

| Rank | Target activity | Draft activity | Outside graphs | Copies only | No recorded activity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 13.567 | 3.838 | 0.946 | 0.187 | 2.845 |
| 1 | 14.309 | 4.227 | 0.683 | 0.131 | 2.034 |
| 2 | 14.312 | 4.219 | 0.673 | 0.132 | 2.048 |
| 3 | 14.311 | 4.233 | 0.667 | 0.132 | 2.040 |

Units are ms/window. Concurrent kernel overlap totals 1.24–1.30 ms/rank;
summed kernel durations therefore overstate wall time. Collective spinning
inside a recorded kernel remains busy in this table. One 39.002-ms window and
14.801-ms entry-skew outlier remain in the mean, rather than being silently
removed. Target GPU entry skew median is 0.241 ms and p90 is 0.335 ms.

The principal target kernel-service components, averaged over ranks, are
HC down/up 2.542 ms, dense projections 2.366 ms, shared gate/up 0.817 ms,
routed gate/up 1.793 ms, routed down/unroute 0.928 ms, QSA 1.619 ms, router
projection/top-k 1.128 ms and TP allreduce 1.187 ms. These numbers overlap and
must not be added to form the round ledger. The routed kernels carry
`turbomind::gemm::StridedPtr` arguments; that type name does not imply they are
generic TurboMind GEMMs. Shared down is included in the dense projection
component. Its exact critical-path contribution requires an ablation because
shared and routed experts run on different streams.

The CPU observer records 1.35–1.50 ms/rank of attention-metadata preparation.
The rank-0 output-serialization span includes approximately 9 ms waiting for
GPU completion; it is not 9 ms of CPU computation. Async scheduling and fused
GDN metadata for three cache groups already operate in the baseline. Nested
CPU/GPU envelopes must be inspected before attributing metadata time to an
unhidden bubble.

## Integrated-route qualification and next decision

The installed integrated wheel passes 50 SM70 operator tests and four-rank
repeated HCX/HCXO isolation comparisons. Synthetic HCX block/injection relative
L2 errors are 9.91e-5/7.37e-4; Q6 output-projection HCXO errors are
5.64e-4/3.63e-4. These tests qualify isolation correctness, not model quality
or endpoint speed.

Merged GDN alpha/beta weights contain two BF16 shards. The side-projection
preparation originally admitted only one F16 shard, so it never covered these
36 layers. It now preserves shard order and the existing FP16 dense contract,
declines overflow/nonfinite coefficients, and passes CPU and GPU comparisons.
Its model contribution remains unmeasured.

Measure HCX alone against the integrated wheel's switch-off control, including
GPU envelopes, target output, teacher-forcing error, acceptance and C4. Keep
expert MMA and hot-vocabulary changes off during that comparison. The next
ablation is selected from the observed critical path; single-graph drafting
and greedy verification are separate hypotheses, not presumed gains. HCX
promotion also requires resolving redundant HC weight packs and verifying its
FP16 normalization boundary against the agreed numerical gate.

### Matched integrated control and HCX ablation

With the same integrated extension and switches fixed before worker startup,
HCX alone gives the following completed endpoint measurements:

| Route | C1 ms/round | C1 tokens/round | C4 ms/round | Eight-prompt mean acceptance |
| --- | ---: | ---: | ---: | ---: |
| Integrated control | 18.298 | 4.886 | 43.264 | 44.755% |
| HCX only | 18.004 | 4.886 | 46.588 | 47.528% |

C1 averages the unobserved cohorts before and after GPU-event recording.
HCX saves 0.294 ms, while C4 regresses by 7.7%; this configuration is not
eligible for promotion. The control's long natural outputs also differ from
the frozen-main baseline. Its prompt-mean acceptance difference is -2.876
percentage points, paired bootstrap 95% CI [-5.788, -0.077]. C1 probe output
is identical, but that alone does not qualify long-output acceptance.

HCX versus the same-wheel control on 64 aligned teacher-forcing positions
has mean KL 0.000698, maximum KL 0.006981, 63/64 matching top-1 predictions,
maximum absolute logit error 0.7344 and maximum relative L2 error 0.07993.
The earlier control-versus-reference comparison has mean KL 0.000671 and
63/64 matching top-1 predictions. These are measured errors, not a claim of
bit equality. The staged draft M1/M5 MoE kernels pass four changed-input,
changed-route and graph-canary tests against their retained references.

Current-stream events give a lower-overhead round ledger: target replay is
14.375–14.381 ms across ranks, draft proposal is 3.314–3.330 ms, sampling
outside draft is 0.511–0.527 ms, and preparation not hidden behind prior GPU
work is approximately 0.095–0.099 ms. The observed round is 18.316 ms versus
18.298 ms without observers. HCX lowers target replay by approximately
0.27 ms and leaves the other envelopes essentially unchanged. CPU metadata
service and CPU graph-submission skew are not equivalent to an unhidden GPU
bubble; host preparation alone cannot recover the missing 6.3 ms.

The HCX integration now dispatches on actual M inside opaque operators:
M>8 retains the original projection and reduction chain rather than moving
it to the next HC boundary. Shared and routed expert outputs remain separate
through the small-M consumer and are summed by HCX in FP32. HC following an
already-reduced dense FFN retains its original path. Preparation admits all
required dense FP16 HC weights before changing any producers, and records
capability and fallback reasons. Twelve CPU dispatch, compilation and
precision tests pass. Model speed, C4 and acceptance of this repair remain
to be measured; it stays opt-in.

### Same-engine execution-policy ablation

A source-complete wheel captures both draft policies before requests. Five
arms run in one loaded engine, followed by a repeat switch-off control. Every
arm produces exactly the same eight natural output sequences and acceptance
counters; prompt-mean acceptance is 47.306%. The C1 probe also emits the same
4.886 tokens per round in every arm.

| Execution policy | C1 ms/round | C4 ms/round |
| --- | ---: | ---: |
| Control | 18.273 | 43.277 |
| Single-graph draft | 18.600 | 43.467 |
| Local argmax verification | 18.092 | 43.497 |
| Both | 19.543 | 43.225 |
| Repeat control | 18.283 | 44.344 |

Single-graph drafting is rejected for this workload. Local argmax verification
saves about 0.18 ms in C1; its C4 delta requires an unprofiled confirmation,
since the repeat C4 control itself drifts upward by 1.067 ms. The Nsight
library is attached throughout but collection is disabled during these
cohorts. Neither the approximately 0.6-ms difference from a separate
startup cohort with both switches enabled nor microbenchmarks is substituted
for the measured ablation. The acceptance variation between fresh engines
remains unresolved; within-engine bit equality does not explain it.

### Signed-nibble lattice experiment

The largest remaining expert gate/up service component is approximately
1.79 ms per target replay. A candidate decoder expands correlated IQ indices
and signs to signed scalar nibble codes at load time. IQ3_S needs 16 odd
integer levels, IQ3_XXS needs 16 signed levels and IQ2_S needs six. Each
32-weight record stores 16 code bytes, the original FP16 base scale and one
or two integer odd subscales. The dot keeps the existing integer scaling and
FP32 accumulation order, rather than expanding and rounding coefficients.

All three codecs reproduce official GGUF dequantization element by element.
The CUDA experiment reuses the existing activation encoding, gate/up,
SiLU/multiply and Q8 intermediate skeleton, replacing only the decoder.
It removes shared-memory correlated-codebook lookup, but reads more weight
bytes: 160 versus 98, 110 or 82 bytes per source 256-weight block. Its speed
is unqualified until a same-card real-shard M=5/M=20 comparison passes the
retained integer-dot and graph-canary checks. No model dispatch changes yet.

### Rejected scalar LUT and hidden-reference HCX paths

The signed-nibble experiment passes 36 GPU integer-dot, changed-input,
changed-route and graph-canary checks, with byte-identical outputs. Real TP4
expert shards use E=512, N=160, K=2560 and 47 unique experts at M=5. Cold-cache
ABBA records the following gate/up medians:

| Type | M=5 retained, us | M=5 scalar LUT, us | M=20 retained, us | M=20 scalar LUT, us |
| --- | ---: | ---: | ---: | ---: |
| IQ3_XXS | 44.032 | 48.128 | 119.808 | 139.264 |
| IQ3_S | 46.592 | 46.080 | 122.880 | 134.144 |
| IQ2_S | 40.960 | 49.152 | 106.496 | 136.192 |

The M=5 clock windows are respectively 1290, 1425 and 1530 MHz; mixed clock
windows at other points remain in the raw data. These are paired comparisons
within each point, not cross-type rankings. The approximately 0.5-us IQ3_S
difference is too small to qualify. The weighted service estimate regresses
by 0.228 ms across the three expert layer groups, before any model overhead.
The route is rejected. Numerical and timing data are retained in
`data/flashnext_iq_scalar_lut_20261007.json`.

The initial HCX large-batch repair using hidden producer Tensor references
fails the model gate: C1 emits 2.365 versus 4.886 tokens per round and target
output differs. Eight-prompt acceptance falls to 35.797%; 64 teacher-forcing
positions have mean KL 0.3635, maximum KL 4.4217, 90.625% matching top-1,
maximum absolute logit error 13.254 and relative L2 error 1.7374. Its physical
round latency is not a valid speed result. Short natural EOS checks pass,
which confirms that those smokes alone cannot qualify this change.

Producer temporaries must have explicit consumer edges for compilation and
graph memory planning. HCX now receives an owned two-plane Tensor payload;
views of its two contiguous planes reach the native consumer. Large batches
retain the original reduction, copy the result into the first plane, and
ignore the uninitialized second plane. Hidden Tensor registries are removed.
Fourteen CPU tests, including Inductor temporary reuse and mixed-M dispatch,
pass. Four-rank compiled graph replay passes at M=5 and M=20 with changed
inputs and live scratch allocations. At M=20 all three outputs match exactly;
the maximum M=5 block/injection relative L2 errors are 3.43e-4/7.74e-4.
The compiled test first exposed the original large-M injection view's
336-column row stride, which disagreed with the opaque operator's contiguous
fake output. The fallback now materializes contiguous outputs. This fixes a
compiler contract; the added C4 copy cost still needs an endpoint comparison.
Output-projection fusion can be screened independently
of the HC boundary through kernel configuration. Compiled Q6 fusion uses
180 registers/thread versus 92 without a producer projection; neither
variant spills. This is a screening observation, not a performance claim.

### Owned-payload control and next decoder screen

The owned-payload wheel's switch-off control completes all checks. Its
unobserved C1 cohorts are 19.188 and 18.275 ms/round (mean 18.731), both
emitting 4.886 tokens/round with identical C1 output sequences. The first
cohort's median is 18.310 ms; its slower tail is retained in the mean and
has not been attributed. The low-overhead observed cohort measures
18.305 ms, with target/draft/preparation envelopes 14.386/3.316/0.097 ms.
C4 measures 43.211 ms/round and eight-prompt mean acceptance is 47.812%.
Both short natural completions terminate at EOS.

On 64 aligned teacher positions against the prior integrated control,
mean KL is 0.000686, maximum KL 0.009153, and top-1 agreement is 63/64.
Maximum absolute logit error is 1.0645 and relative L2 error is 0.1271.
This records fresh-process numerical variation even without selecting HCX;
it does not establish its cause. The native HC boundary with a separate
output projection requires its own matched model gate.

The next expert-reader screen retains the original compressed bytes, base
scales and integer-dot arithmetic. It separates IQ2_S's two codebook words
into shared-memory planes and generates four-byte sign masks in registers
using PRMT. No format expansion or new precision reduction is introduced.
The existing native operator exposes an optional switch, disabled by default,
for same-wheel comparisons. Thirty-six changed-input/route graph and canary
cases, then real-shard cold-cache ABBA at M=1/5/20, are required before model
dispatch changes. Compilation alone is not speed or correctness evidence.

### Rejected owned-payload HC boundary

The matched owned-payload HCX cohort keeps the output projection separate.
It emits only 2.370 tokens/round instead of 4.886, and its C1 target output
differs from the control. C4 increases from 43.211 to 45.884 ms/round. Against
the same-wheel control at 64 aligned teacher positions, mean KL is 0.22484,
maximum KL is 2.45558, and top-1 agreement is 55/64. Maximum absolute logit
error is 13.2695 and relative L2 error is 2.0790. None of the eight natural
output sequences matches. The higher natural acceptance mean, 55.489%, is
not a correctness pass. This path remains disabled; its shorter physical
round is not an eligible performance result.

The owned payload and contiguous fallback repair compiler contracts, but
they do not explain or fix this full-model failure. A diagnostic-only graph
records actual M=5 inputs and outputs at every HC boundary, with explicit
owned buffers and TP frame checks. Each boundary is compared against the
isolated FP32-accumulating dense reference with FP16 intermediate boundaries.
Diagnostic graph copies are excluded from performance claims.

### Bank-aware decoder result

The original-record bank-aware decoder passes all 36 changed-input, route,
graph and canary checks with byte-identical outputs. Cold-cache, same-pointer
ABBA on real TP4 expert shards gives:

| Type | M=5 retained, us | M=5 bank-aware, us | M=20 retained, us | M=20 bank-aware, us |
| --- | ---: | ---: | ---: | ---: |
| IQ3_XXS | 45.056 | 43.008 | 119.808 | 112.640 |
| IQ3_S | 45.056 | 44.032 | 116.736 | 113.664 |
| IQ2_S | 40.960 | 39.936 | 106.496 | 101.376 |

These are single-kernel service measurements. Across 17 IQ3_XXS, ten IQ3_S
and twenty IQ2_S layers, the M=5 estimate is only 0.066 ms/round; it cannot
explain the 6.6-ms endpoint gap. The decoder stays disabled pending a model
ablation alongside a larger qualified change. Reported bandwidth counts
logical expert payload, not measured DRAM traffic.

### HC stage budget

A four-rank graph screen uses real checkpoint HC weights and the installed
native extension. Instrumented and uninstrumented outputs match byte for
byte. The maximum-rank median is 30.100 us without instrumentation; the
instrumented cohort is 28.861 us. This difference is not promoted as a gain.
Per-CTA local-clock stage medians identify 6.144 us in the combined LoRA
exchange, up-weight prefetch and second grid barrier, versus 3.072 us in the
first grid barrier. They do not isolate pure communication cost and cannot
be summed into a critical-path ledger. This budget motivates examining
readiness and synchronization after the model numerical failure is localized.

### Actual HC inputs localize a TP consistency failure

The diagnostic model records 94 M5 boundaries in the same frame on all four
ranks. Layer 0's MLP HC has block relative L2 error 2.72e-4. At layer 1's MLP
HC, after the PLE boundary, hidden inputs differ across ranks by up to 0.0858
while norm/down/up checkpoint weights are byte-identical. Ninety-two later
boundaries exceed 0.005 relative L2; early block errors are about 0.07–0.08.
The native sharded down/up design requires replicated HC inputs, which this
cohort violates. A local dense HC reference naturally differs by rank when
its input residual differs, whereas sharded HC assembles a common output
from those inconsistent inputs.

Four-rank eager and graph replay of the three saved real boundaries reproduce
the model's native outputs byte for byte. In a diagnostic isolation only,
replacing the residual input with the same rank-0 residual on all four ranks
reduces block errors to 2.17e-4–2.96e-4 and injection errors to
6.46e-5–2.21e-4. This is not a production broadcast fix or a speed result.
The first PLE stage attempt did not activate its helper because the forward
configuration context did not retain the diagnostic flag. The flag is now
fixed on the module, and reports save each completed step independently.
This failed attempt is not stage-localization evidence. Compiler payload ownership is not established
as the cause of this actual-input failure.

The actual control decode graph selects FP32 `all_reduce_sum2`. A hypothesis
based only on the registered environment variable's default FP16 local sum
is rejected: the selected SM70 profile enables the sum2 route at runtime.

### IQ3_S single-kernel counters

Nsight Compute samples one installed real TP4 IQ3_S gate/up launch at
M5/N160/K2560, with 47 unique routed experts. It observes 375 GB/s memory
throughput, 50% theoretical occupancy and 42.73% achieved occupancy.
There are 56 registers/thread and 512 threads/CTA, limiting residency to two
CTAs/SM. Schedulers have no eligible warp in 51.8% of cycles. These counters
justify examining instruction readiness and register-limited residency;
they do not establish a pure HBM or codebook-bank bottleneck. Profiling does
not fix clocks (reported SM frequency 1.17 GHz), so its 48.22-us duration is
not substituted for unprofiled ABBA or complete-round latency.

### Range-compiled partial materialization

Inspection of the retained HCX backbone graph shows the first MoE payload
being sliced to its first plane and passed directly to PLE's HC combine,
without `all_reduce_sum2`. The Python M branch was specialized while tracing
an already-reduced large batch. Small verification calls use local two-plane
payloads, so reusing that graph violates the replicated-residual contract.
The final HC mixer has the same Python M branch.

Move PLE combine and final-mixer materialization into opaque operations that
select the TP reduction at actual runtime M. Large batches consume only the
already-reduced first plane; small batches sum both planes across TP before
combining. The repaired source-complete wheel's four-rank compiled M20-first/M5-second
materialization tests pass. In the actual model, all twelve recorded PLE
stages are byte-identical across TP, with equal weight fingerprints. All 94
HC boundaries have maximum relative L2 below 0.000435, replacing the earlier
approximately 7% errors. The short diagnostic generation matches the control
prefix for 32 tokens. This cohort includes diagnostic copies and is not a
speed result. The same-wheel endpoint and numerical comparisons below exclude diagnostic
copies.

### Further IQ scheduling screens

A forced three-CTA register schedule spills registers and regresses IQ3_S M5
from 45.568 to 73.728 us in same-process graph ABBA. It is rejected.
A two-row-per-lane schedule gives no IQ3_S M5 improvement (49.152 us for both
arms). Both screens encounter two one-LSB Q8 output differences at M20,
so neither is described as byte-exact or admitted to model dispatch. Their
research builds also used fast-math unlike the packaged native build. The
negative screens do not qualify an equal-arithmetic implementation.

An additional installed-kernel NCU sample reports long-scoreboard stalls
42.73%, MIO throttle 15.84%, short-scoreboard stalls 5.66%, and math-pipe
throttle 3.82%. Shared-load bank-conflict count is 742,063. These metrics
support testing weight prefetch and shared-codebook scheduling; they do not
justify replacing endpoint measurements with a throughput estimate.

The initial aligned-record screen incorrectly treated the reader's per-row
alignment bytes as payload and failed before timing. Its conversion now
uses only payload bytes, then adds two leading alignment bytes per original
block. The initial row-ready HC screen differs in three FP16 hidden values
(maximum 0.00024414); its research compiler enabled fast-math while the
packaged HC build does not. Both next screens use the native math flags.
Neither is admitted to model dispatch on these failed attempts.

### Precise real-shard follow-up screens

With native math flags, aligned lossless records and original-record metadata
prefetch each pass nine byte-identical Q8 output checks (M1/M5/M20, three IQ
types). Alignment has no IQ3_S M5 gain (45.056/45.056 us) and regresses its
M20 point (116.736/121.856 us). Its IQ3_XXS and IQ2_S M5 savings are 2.048 us
each. Original-record prefetch saves about 1.024 us on IQ3_S M5 and 2.048 us
on IQ3_XXS M5. These service estimates are too small to justify another model
load or dispatch/storage change. They remain unselected.

The repaired wheel's new switch-off control records C1 cohorts of 18.317 and
18.321 ms/round and C4 45.846 ms/round. Eight-prompt mean acceptance is
45.864%. Relative to the previous switch-off control's identically conditioned
64 teacher positions, mean/max KL is 0.000721/0.006947 and top-1 agreement is
64/64. Long natural sequences differ. Both configurations resolve to page816
and use the same prompt tokens. The earlier C4 record is 43.211 ms/round;
this difference is retained for investigation, not discarded as a favorable
new denominator. The completed matched comparison is recorded below.

### Repaired HCX matched endpoint comparison

Both arms use the same source-complete wheel at `75470871a`, core SHA256
`1fb69a6549693048bf4c704cbef523fee1c72a9b85e2e3ca21ce9fd2327e2f1d`.
Only HCX is switched; output-projection fusion and diagnostics are off.
The workload is TP4 V100, CUDA 12.8, Torch 2.10, FP16 activations/KV,
FP32 SSM, FULL target/draft graphs, MTP4 and page816. C1 uses 8192 input and
256 output tokens; C4 uses 128 input and 600 output tokens per request.

| Measurement | Control | HCX |
| --- | ---: | ---: |
| C1 unobserved mean, ms/round | 18.3192 | 17.3988 |
| C1 emitted tokens/round | 4.8857 | 4.8857 |
| C4, ms/round | 45.8460 | 45.6969 |
| Eight-prompt mean acceptance | 45.8638% | 47.3311% |

All six C1 256-token output sequences are identical across cohorts and arms.
HCX saves 0.9204 ms/round (5.02%). The paired acceptance difference is
+1.4673 percentage points, with bootstrap 95% CI [-0.3854, +3.3670] points
(100,000 resamples, seed 20261007). This interval does not prove zero loss;
no decrease is observed. Natural outputs remain coherent, and the short
arithmetic and blue-sky completions terminate normally. Natural sequences
are not byte-identical. On 64 identically conditioned teacher positions,
mean/max KL is 0.0004822/0.0057093 and top-1 agreement is 63/64. Maximum
absolute logit difference is 0.7588; maximum logit relative L2 is 0.05972.

Low-overhead rank-0 median GPU envelopes locate the endpoint saving:

| Envelope, ms | Control | HCX |
| --- | ---: | ---: |
| Target replay | 14.3949 | 13.4769 |
| Four draft steps | 3.3147 | 3.3178 |
| Sampling outside draft | 0.5371 | 0.5274 |
| Execute before target | 0.1030 | 0.0952 |
| Target to next target | 18.3716 | 17.4336 |

These are medians of nested envelopes, not an additive closed ledger.
CPU target-entry skew increases to median 1.5371 ms while GPU before-target
work remains about 0.10 ms: the CPU timestamps alone do not establish a
redeemable GPU stall. A new graph-node trace will locate actual TP entry and
dependency waits. The 12-ms objective remains unmet by 5.4 ms. The historical
43.211-ms C4 result remains a separate unresolved control drift; the new
same-wheel comparison alone does not establish no regression against it.

### Additional precise screens rejected or deferred

The row-ready HC variant preserves every checked output byte, including two
changed-input graph checks, but increases the four-rank maximum median from
27.208 to 28.396 us. It is rejected. Publishing separate row readiness does
not redeem the synchronization budget in this topology.

A coalesced N32/K8 scalar-code layout increases expert storage to 20 bytes per
32 weights. IQ3_S M5 regresses from 45.056 to 52.224 us, despite higher logical
bandwidth; IQ3_XXS and IQ2_S also regress. Some Q8 packet scale-sum bytes differ
with identical decoded values, and the worst tested output relative L2 is
0.000273. This layout is rejected for latency.

Carry-free packed-byte sign restoration is exact on 45,056 exhaustive
codebook/sign combinations and all nine real-shard GPU points. Its M5 weighted
service estimate saves only approximately 0.067 ms/round. Together with the
approximately 0.063-ms metadata-prefetch estimate, it is deferred rather than
used to justify another model load. All three screens are private-DSO research
measurements; none changes packaged model dispatch or qualifies endpoint gain.

## Community designs and applicability

[SGLang's DeepSeek-V4.1 optimization account](https://staging.lmsys.org/blog/2026-09-28-deepseek-v41-optimization)
describes overlapping mixing-coefficient computation with Attention or MoE,
fusing adjacent output/input combinations and normalization, and having the
router produce the consumer's layout directly. These are dependency and
schedule changes, rather than a claim that fewer kernels necessarily shorten
the critical path. Flash-Next's HC down projection consumes the normalized
post-combine residual, so it cannot simply be launched alongside the preceding
block without changing that dependency.

[mKernel](https://arxiv.org/abs/2609.13585) publishes tile readiness to overlap
communication with computation, partitions compute/communication resources,
and accounts for pipeline fill/drain and synchronization. Its reported H200
multi-node results are not V100 latency estimates. Here the small-M critical
path and two-hop TP4 topology require measuring those fixed costs before
reserving SMs for communication or introducing a persistent kernel.

[DeepSeek TileKernels](https://github.com/deepseek-ai/TileKernels) includes mHC
kernels and an Ascend backend. Its documented NVIDIA requirements are
SM90/SM100 and CUDA 13.1; it is an algorithm reference for this SM70 workload,
not an installable V100 fast path. Flash-Next's gated HC also differs from
DeepSeek's Sinkhorn-normalized mHC, so a Sinkhorn-specific improvement does
not explain this model's measured HC cost.
