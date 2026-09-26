# SM70 quantized batch reuse, 2026-09-26

## Workload and scope

Baseline source: `fcf59f8e9ae50c186333e98e5cf6aae705f320de`.
Qwen3.8-27B-NVFP4 mixed NVFP4/channel-FP8 target, DFlash2 q7 probabilistic
sampling, TP4 on V100-SXM2-32GB, FP16 execution, E4M3 target KV and automatic
draft KV, Flash-V100, prefix caching, MRV2 full/piecewise CUDA graphs.
CUDA 12.8 / Torch 2.10.0+cu128 / Python 3.12.13. Max length 262144,
max sequences 16, max batched tokens 8192, GPU memory utilization 0.8.
The fixed performance workload is 2048 input / 256 output with temperature
0.7, top-p 0.8, top-k 20, seeds starting at 20260923. Performance requests
ignore EOS; natural-EOS quality is checked separately.

The updated C8 acceptance target is at least 20% lower total GEMM latency
(aiming for 30%) and at least 20% faster complete batch decode. Both rolling
and no-new-prefill decode are reported against the same saved main baseline;
neither the acceptance limit of two percentage points nor the C1, quality,
context-capacity and memory checks is relaxed. The FP8-only candidate below
is a localization experiment and is insufficient for this updated target.

The measured QPN implementation extends batch reuse to FP4 and FP8 at M9..32.
A full DFlash q8 verification step has M=8*C, so these are the C2/C4 paths;
partial steps also benefit. Target-model M<=8 retains its established kernels.
The current M33..64 candidate adds padded activation supply and in-place FP4
scale preparation. Earlier M64 candidates and their failed serving guards are
recorded below. The current candidate is also not qualified for promotion.
The follow-up also removes a duplicate logits projection when compact DFlash2
sampling requires the existing full-vocabulary fallback. Scheduler and
attention implementations are unchanged.

## Implementation and defaults

- M9..16: a common paired-projection kernel shares each packed activation
  fragment between two output tiles. Format-specific readers consume existing
  compressed FP4/FP8 weights and scales. No second persistent weight layout
  is needed. This eliminates the research C2 prototype's approximately
  1.87 GiB per-rank duplicate weights without adding mainline weight memory.
- M17..32: FP4 reuses each decoded weight over four eight-row tiles. FP4 and
  FP8 pack activations and use two physical reduction phases while retaining
  the original logical split order. FP8 gate/up processes all rows in one
  invocation, including M17 tails that previously needed M16 plus another call.
- The established FP4 intermediate FP16 activation rounding and FP8 FP32 SiLU
  contracts remain distinct. Small projections and unsupported geometry keep
  their existing dispatch. The activation pack is transient; the measured
  TP4 projections use at most 160 KiB at C2 and 320 KiB at C4 per invocation.

No new environment switch is introduced. Existing SM70 DFlash2 defaults
already enable compressed batch layouts and M64 warmup/tuning. Service
launches explicitly unset the four manual batch-layout/tuning overrides;
worker audits check the automatic settings of 1/64/64/64 and the normal
source-built extension. This does not enable previously rejected opt-in
experiments such as FP8 prescaled batch GEMM.

## Native real-weight microbenchmark

Seven representative TP-local projections are weighted by actual layer count,
including activation packing and gate/up activation. These are GEMM estimates,
not complete target forward or serving decode. Seven timing rounds per
projection are measured in eager and CUDA Graph execution.

| Full-q8 concurrency | Baseline GEMM ms | Final GEMM ms | Latency reduction |
| --- | ---: | ---: | ---: |
| C1 / M8 | 6.712 | 6.700 | 0.2% |
| C2 / M16 | 11.099 | 8.284 | 25.4% |
| C4 / M32 | 19.207 | 12.582 | 34.5% |

FP8-only C2/C4 GEMM changes from 5.313/8.118 to 4.043/5.611 ms,
reductions of 23.9%/30.9%. FP4-only changes from 5.786/11.089 to
4.241/6.971 ms, reductions of 26.7%/37.1%. C1 is unchanged within noise.
C2 has not reached the aspirational 30–50% GEMM latency reduction.

M64 still uses the original TurboMind implementation. Its microbenchmark
varies with legacy tuning; no M64 optimization gain is credited to this
QPN candidate. Earlier 19% M64 estimates belong to rejected candidates.

### FP8 M64 reevaluation

The follow-up candidate `fp8-fc-default` combines the stabilized context FC
with FP8-only M33..64 activation-supply tiles. FP4 M64 keeps its original
registry. The common batch policy pads activation shared-memory rows and
measures M32/M64 tiles using existing compressed weights and transforms.
Measurement first selects a legacy K32 reference, then admits new tiles only
with the same split-K boundaries, warp-K traversal and operation class.
Captured partial batches preserve that partition through a masked fallback.
Untuned shapes, M<=32, grouped MoE and AWQ retain established dispatch.

The normal extension is
`426476ba2ef2a6fe057c32359d56dd8ca052fb6d1f6731f9690f21eece298b87`.
All 141 focused GPU tests pass, including FP8 captured tails and changed-input
replay. Representative real-weight M8/16/32 results remain bitwise equal to the
baseline, and eager/Graph outputs agree at all four row counts. The M64 FP8
estimate falls 7.567 -> 6.883 ms (-9.04%). Combined M64 GEMM is 19.845 ms versus
21.289 ms; this includes independent legacy FP4 tuning variation and is not
all attributable to the new FP8 tiles. C2/C4 estimates remain 8.306/12.638 ms
(-25.16%/-34.20%). Uninstrumented serving guards are in progress; these are
screening results, not an accepted C8 service speedup.

### C8 follow-up: stable reduction and in-place scale folding

The FP8-only service screening completed: C1 rolling 243.61 tok/s (+0.19%),
C8 no-new-prefill 619.16 tok/s (-0.44%), and C8 acceptance -0.058 percentage
points. This does not meet the new C8 target. Adding padded FP4 batch tiles
without stabilizing the reference failed: GEMM 17.617 ms (-17.25%), C8 window
601.96 tok/s (-3.21%) and acceptance -2.852 points. Neither is qualified.

Fixed-partition real-weight probes established that original indexed/blocked
FP4 M16/M32/M64 tiles and the new activation-supply tiles produce identical
bits at the same split-K boundaries. However, the first-stage reference
autotuner itself changed the gate/up split count from the saved baseline's
7 to 13. The new candidate uses the established deterministic selector as
the FP4 reduction reference, then measures only compatible supply tiles.
This avoids making reduction semantics depend on startup timing noise.

For batch layouts with independent QPN2 compact scales, the candidate folds
the exact FP4 conversion factor into the existing TurboMind FP16 scale
allocation at load time. A finite/range check rejects unsafe scales without
mutation. Matching transforms consume those scales for larger decode and
prefill, while M<=32 retains QPN2 and its independent compact scales. This
adds no persistent weight or scale allocation. The ordinary registry/cache
cannot select a prescaled transform for an unscaled tensor.

The first normal-build implementation passed 17 scale/replay tests and 50
batch/tail tests. Actual TP-local weights measured M8/M16/M32/M64 GEMM at
6.788/8.414/12.811/17.105 ms. The M64 FP4 portion is 9.779 ms (-28.73%);
FP8 is 7.326 ms. Total M64 latency reduction is 19.65%, still below 20%.
These are microbenchmarks, not service speed results. Its normal extension
SHA256 is `d1cd034060c455d11c043de04630517c496d27f0660fb4b97cde8af48cd931e6`.

Service initialization exposed a dynamic-compile integration bug: the Python
large-M branch was retained when the compiled token range executed small M.
The repair moves that choice into the existing opaque C++ QPN2/TurboMind
dispatcher and adds a dynamic-row compiled regression. Serving qualification
is pending; the earlier raw failure is retained rather than counted as a run.

The repaired opaque-dispatch artifact
`fb606316ebfa32374a0411c3388188e458ca80c019c018f5f49a11d89b44f406`
passes 69 focused tests, including dynamic compiled M64-to-M8 execution.
The real-weight M64 estimate is 16.982 ms (-20.23%); M16/M32 are
8.395/12.769 ms (-24.36%/-33.52%). C1 rolling median is 243.01 tok/s
versus 243.15 (-0.06%), with acceptance -0.274 points. However, the completed
C8 full-48 three-run median is only 649.44 tok/s versus 621.91 (+4.43%),
and acceptance falls 50.51% -> 46.36% (-4.16 points). The candidate fails
both C8 service targets. Stable FP4 partitions alone have not established
the cause or repaired acceptance; full-step/route diagnostics continue.

The acceptance audit retains the failing candidate rather than attributing its
GEMM estimate to service speed. A fresh main restart on the same 48 prompts
produced 50.36% acceptance in one diagnostic control run. Replaying all four
ranks' reference GEMM plans in the candidate, including the draft context
reduction, recovered 50.26%. All 63 exported plans per rank matched the imported
reference. This is localization with a manual cache, not default-service
qualification. With ordered request admission, the reference produced
49.68495% acceptance. Replacing only its FP4 plans produced 49.72875%;
46 of 48 output token arrays were identical, and the worst request used
68 streamed verification rounds instead of the failing candidate's 182.
Replacing only the draft FP16 plans produced the same 49.68495% acceptance,
2766 draft rounds and all 48 identical output token arrays. Neither isolated
change reproduced the large regression. Imported plans remained unchanged;
previously unseen tail LM-head shapes added cache entries during execution.
The target FP8-only run produced 50.12987% acceptance and 32 identical
output arrays; the LM-head-only run had 49.68495% and all 48 identical arrays.
Combining all candidate plans produced 50.12212%, also failing to reproduce
the normal-service regression. These family results have a diagnostic
confounder: disabling tuning changed previously unseen captured-tail
dispatch from cache reuse to the default heuristic. They cannot isolate
production tail behavior. The corrected diagnostic keeps the production
tuning flags and stable context selector; exact imported plans already
short-circuit measurement. Production-policy reproductions are pending.

The route audit found changes outside the measured full M64 FP4 projection:
FP8 gate/up and LM-head references moved from K16 to K32, and the draft context
projection at M16 changed its split count. Smaller batches therefore need the
same numerical audit as the full C8 step. Client-window estimates also place
most of the acceptance loss after requests finish and the batch shrinks. These
estimates clip returned tokens at the output limit and are not substitutes for
the server's accepted/drafted counters.

Warmup now skips the unused small-row TurboMind calls for prescaled states and
uses the matching prescaled transform for larger rows. Previously it timed the
ordinary transform with shifted scales; the discarded warmup output was wrong.
The 21 warmup tests pass, including gated/plain format regressions. This repair
has not yet independently established default-service acceptance recovery.

The batch tuner also incorrectly restricted its ordinary FP8 reference to
K32 candidates before testing new supply kernels. The pending correction
retains the full legacy reference pool, including K16. A new full M64/N256/K16
candidate preserves that reduction family; it is feasible only for exact
M/N/K tiles. In particular, the 62080-wide vocabulary shard has an N128 tail
and cannot use the unmasked N256 iterator. Fixed-partition real-weight probes
showed a gate/up reduction from 126.8 to 120.1 us with identical bits.
The legacy K16 LM head was already faster than the forced K32 route in that
probe. These are diagnostic measurements; default-service results follow.

### Latest default-service acceptance retest

The installed normal extension is
`d08ae1443aa24f91bddfbc4c80ba2c6dd1122a4368d6b3db4a944cfe537fef68`.
Its build-tree hash differs because CMake removes the build RPATH on install.
The source retains the legacy FP8 reference pool, adds the compatible full
K16 candidate and corrects prescaled FP4 warmup. All 52 batch/tail/replay GPU
tests pass, including the wide vocabulary shard. Actual-weight Graph GEMM
estimates at M8/M16/M32/M64 are 6.781/8.404/12.795/17.026 ms. C8 GEMM is
20.02% lower than the fixed reference; FP4/FP8 contribute 10.126/6.901 ms.

The primary service uses automatic defaults, no imported LUT, no worker
extension and no profiler. The same 2K/256 workload, seeds, sampling and
original concurrent admission protocol were repeated three times:

| Metric | Main reference | Earlier failed candidate | Latest candidate |
| --- | ---: | ---: | ---: |
| C8 no-new-prefill decode, tok/s | 621.91 | 649.44 | 654.76 |
| C8 server acceptance | 50.51% | 46.36% | 51.06% |
| Verification rounds, all 48 requests | 2732 | 2920 | 2709 |
| Returned decode tokens per request round | 4.480 | 4.192 | 4.518 |

These are three-run medians. Latest individual acceptance results are
48.069%, 51.063% and 51.568%; the first run is 2.44 points below the reference
median, so this does not establish that every run avoids regression.
C1 rolling decode is 243.60 versus 243.15 tok/s, with acceptance 56.77%
versus 57.04%. Separate C8 rolling medians are 375.24 versus 365.98 tok/s
(+2.53%) and 55.67% versus 54.92% acceptance. The no-new-prefill speed gain
is only 5.28%. The C8 speed target still fails; no default promotion is claimed.

SSE events inside the identical all-eight-alive windows help separate the
aggregate acceptance change from full-batch speed. Across all three runs,
returned tokens per request round are 3.890/3.858/3.909 for reference/previous/
latest; the corresponding estimated batch-round intervals are
50.30/47.55/47.76 ms. These intervals include host and transport time and are
not GPU forward measurements. Most of the earlier aggregate acceptance loss
was outside the full-eight-alive window. The remaining whole-step gap cannot
be assigned to the aggregate acceptance number alone.

Evidence: `native-preserve-fp8-reference*`, `latest-acceptance-comparison.json`,
`latest-acceptance-rolling-comparison.json` and
`latest-acceptance-all-alive-decomposition.json`. An independent service
restart also completed: C8/48 acceptance is 50.829%, with 2722 request
verification rounds and 668.57 tok/s in the no-new-prefill window. C1
acceptance is again 56.766%. This check uses only an idle route-export worker
extension, with no imports, hooks or profiling; its single timing result is
kept separate from the primary uninstrumented three-run medians. All four
workers loaded the normal extension and automatic batch defaults. Actual
routes were exported after requests completed, and the owned service stopped.

### C1 to C8 latency growth: existing matched trace

The last matched complete-q8 stage trace uses the previous `fb606316` native
artifact, not the latest `d08ae144` acceptance-retest artifact. Rank-0 CUDA
event medians (C1/C4/C8 have 36/35/33 samples) locate the growth:

| Stage | C1 ms | C8 ms | Increase ms |
| --- | ---: | ---: | ---: |
| Target forward | 14.136 | 33.731 | 19.595 |
| Target sampling, including logits | 0.919 | 6.431 | 5.512 |
| Draft total | 4.359 | 7.660 | 3.301 |
| Complete GPU round | 20.013 | 47.946 | 27.933 |

Independent medians are not additive. Forward accounts for approximately
70% of whole-round growth, and sampling approximately 20%. These are batch
rounds (M8 versus M64), not separate serial per-request latencies. C8 Nsight
rank-0 target kernel means are FP4 GEMM 9.872 ms, FP8 GEMM 8.033 ms, GDN
4.494 ms, TP communication 4.663 ms, attention 3.281 ms and other 3.289 ms.
There is no paired C1 Nsight category capture in this run, so these C8
category costs do not establish each category's C1-to-C8 increase. The raw
source is `trace-stable-prescale/comparison-summary.json`. The latest
real-weight GEMM estimate independently grows 6.781 -> 17.026 ms; it must
not be subtracted from the earlier whole-service trace as if measured in
the same timing scope.

A further M64/K32 probe tested additional M-warp arrangements while preserving
K32 partition boundaries. Real-weight comparisons are bitwise within each
partition, but the new arrangements did not improve the dominant FP4 gate/up
and down costs together. They are not promoted. The smaller FP8-out-only
improvement is insufficient to establish a useful whole-step benefit.

### Matched service trace: how much GEMM saving reaches the forward

A new matched run compares the main reference `7449bff7` and the candidate
`d08ae144`, before the logits-reuse change. Both use the same complete-q8
observer and selected C8 CUDA-graph node trace. Rank-0 stage medians are:

| Stage | Reference C8 ms | Candidate C8 ms | Saved ms |
| --- | ---: | ---: | ---: |
| Target forward | 36.882 | 33.834 | 3.048 |
| Target sampling, including logits | 6.426 | 6.443 | -0.018 |
| Draft | 7.710 | 7.562 | 0.148 |
| Complete GPU round | 51.182 | 48.071 | 3.111 |

The C8 graph-node FP4/FP8 means change from 12.424/8.957 to 9.903/8.186 ms.
Their combined saving is 3.293 ms (15.40%), smaller than the independent
4.263-ms microbenchmark projection. TP communication increases by 0.174 ms;
GDN, attention and the remaining target kernels are essentially unchanged.
Most of the actual GEMM saving reaches target forward. The full service
trace does **not** demonstrate a 20% GEMM reduction. At C4, forward changes
30.276 -> 23.940 ms, consistent with the approximately 6.41-ms micro saving.

Importing each service's actual selected plans into a separate real-weight
M64 microbenchmark gives 21.018 -> 17.266 ms, a 3.752-ms saving. This
localizes part of the earlier overprojection to tactic selection. It does not
attribute the remaining difference between standalone and full-model timing
to a particular hardware bottleneck. Imported-plan tests and instrumented
traces are diagnostic; final speed gates still require ordinary serving.
Raw evidence: `trace-fulfillment-control`, `trace-fulfillment-latest`,
`fulfillment-paired-trace.json` and `micro-service-{control,latest}.*`.

### Sampling follow-up and acceptance investigation

All 33 observed complete C8 steps and all 34 C4 steps take the compact
top-k probe and then fall back to full-vocabulary sampling. The C8 probe
costs 1.314 ms within the 6.443-ms sample stage. The original dense LM-head
path computes the local projection twice in this case. The new optional
logits-processor method retains the first local result and defers the same
TP gather, vocabulary trimming, soft cap and scale until fallback is needed.
Fused compact kernels that do not materialize dense logits retain the old
fallback. No persistent weight copy or new user flag is introduced.

The 34 focused projection/cutoff tests pass. A diagnostic service additionally
checks all four ranks, 420 fallback calls per rank and row counts 8 through
64 against a fresh full projection: every result is bitwise identical.
This is direct logits-equivalence evidence, not a performance result.

The first ordinary logits-reuse service has stable C1 acceptance (56.766%)
and 245.03 tok/s, but its three-run C8 acceptance median is only 47.324%
versus the main reference's 50.512%. This candidate fails the two-point
guard. Independent startups exhibit different acceptance despite identical
source. M64 output-projection split 2 versus 3 is not a sufficient cause:
a failed restart also uses split 3. Replacing only the M48 QKV plan with
one from a good run does not restore acceptance (48.843% -> 48.305% in
the controlled diagnostic pair). Neither correlation is treated as proof.
The completed same-process comparison keeps tuned plans fixed and changes
only the sampling route between requests. With ordered admission over the
same 48 prompts, old/reuse/direct-dense/old decode is
667.90/681.65/683.18/669.87 tok/s. All 48 token arrays, all 2714 request
rounds and the 51.047479% server acceptance are identical in all four runs.
Reuse adds 1.91% relative to the two old-path timings' midpoint. The extra
0.22% from direct-dense is insufficient to justify another routing branch;
that experimental branch is not included in production source.

Normal concurrent admission in that same process then gives
663.56/666.52/654.93 tok/s and acceptance 48.580/48.546/47.908%. The first
normal-admission run differs in five requests; requests 2 and 9 account for
most extra rounds. This proves that restart/tactic variation alone is an
insufficient explanation: admission/batching also affects the generated
paths. These later results retain their own label and are not substituted
into the ordered comparison or hidden as warmup. Evidence:
`sampling-paired-comparison.json`, `sampling-admission-comparison.json`,
`barrier-full48-sampling-paired-*` and
`barrier-full48-logits-reuse-qualified` (the latter name is a test label,
not a claim that the 20% serving gate passes).

Code review also found that prescaled gate/up warmup was timing a fused
FP32 epilogue, while the actual QPN2 batch dispatcher writes FP16 gate/up
before the separate activation. Warmup now uses the full-width output and
the same unfused GEMM epilogue. It adds no persistent allocation. The 42
focused CPU tests pass; the 13 GPU cutoff tests are skipped in that CPU
run, rather than counted as GPU validation. This last warmup correction is
checked below in a fresh ordinary service with no worker extension or
route overrides. No promotion is claimed.

### Ordinary serving after logits reuse and matching warmup

The `fulfillment-default` run uses the same original concurrent admission,
dataset, seeds, sampling and automatic defaults as the saved main reference.
It has no profiler, worker extension, imported plans or sampling override.
Each row below reports three measured repetitions:

| Measurement | Reference tok/s | Candidate tok/s | Speed gain | Acceptance change |
| --- | ---: | ---: | ---: | ---: |
| C1/16 rolling decode | 243.15 | 245.19 | 0.84% | -0.27 points |
| C4/32 rolling decode | 297.30 | 330.77 | 11.26% | -0.59 points |
| C8/48 rolling decode | 365.98 | 388.00 | 6.02% | 0.00 points |
| C4 first-four-prompts no-prefill window | 477.65 | 590.49 | 23.62% | 0.00 points |
| C8 full-48-prompts no-prefill windows | 621.91 | 692.62 | 11.37% | -0.98 points |

The C4 and C8 window protocols cover different numbers of prompts; they are
each paired against their matching reference and should not be used to infer
a cross-concurrency scaling curve. The C8 individual speeds are
692.34/692.62/697.28 tok/s and server acceptance is 49.537/49.007/51.762%.
Every acceptance result is within two points of the saved reference median,
but three repetitions from one startup do not establish startup invariance.
The earlier failures remain recorded above.

Across the exact all-eight-alive intervals, the pooled SSE batch-round
estimate falls 50.301 -> 45.783 ms, saving 4.518 ms. Returned tokens per
request round change 3.890 -> 3.972. These are client observations over
q1..q8, with host/transport time, not complete-q8 GPU-forward timings.
The isolated same-process test establishes the logits-reuse contribution;
the full 11.37% combined gain is not attributed solely to that change or
solely to the warmup correction.

The natural-EOS quality pair has 14 correct natural completions and 15
natural stops out of 16 on both versions. Question 12 is wrong on both;
question 16 reaches the 4096-token limit on both. The strict 16-natural-stop
assertion therefore still fails, including on the reference. This is relative
parity, not an absolute quality pass. The 4K prefix check retains 3296 hit
tokens and the 32K C2 route smoke completes; these are route checks, not
long-context throughput baselines.

The final source-built normal extension SHA256 is
`72e750ee54acc7ca61aadfa8b3e1664974f79305c7b059025310d95f3963e412`.
The ordinary performance run loaded `d08ae144` before formatting-only native
source cleanup. All 4,048 GPU kernels have identical instruction hashes after
rebuild, with no additions/removals. The final artifact passes 105 focused
tests, including GPU batch/tail/replay and sampling-cutoff checks; the earlier
CPU-only 42-test run is separate. Resolved dependencies are standard CUDA,
driver and Torch libraries, with no task-sidecar dependency or preload.

Evidence: `fulfillment-service-comparison.json`, `fulfillment-final-build-manifest.json`,
`fulfillment-native-sass-comparison.json`, `fulfillment-final-gpu-tests.log`,
`fulfillment-default-quality16.json` and the matching endpoint/window JSON.
The C8 20% serving target still fails. PR #691 remains Draft; there is no
new PRO comparison or 35B-A3B AWQ/FP8 model-speed acceptance claim.

### Final source-built trace and remaining gap

`trace-fulfillment-final` captures the normal `72e750ee` extension with logits
reuse and matching warmup. Like the earlier control trace, it selects complete
q8 steps without new prefill. Rank-0 CUDA-event medians are:

| Concurrent requests | Forward reference/final ms | Sampling reference/final ms | Complete round reference/final ms |
| --- | ---: | ---: | ---: |
| C1 | 14.073 / 14.046 | 1.044 / 0.984 | 20.275 / 19.968 |
| C4 | 30.276 / 23.972 | 4.765 / 4.217 | 42.162 / 35.038 |
| C8 | 36.882 / 33.919 | 6.426 / 5.757 | 51.182 / 47.491 |

C8 saves 2.963 ms in forward and 0.669 ms in sampling; its whole q8 round
saves 3.691 ms. All 31 observed C8 cutoff fallbacks reuse the existing logits;
the remaining one of 32 steps succeeds on the compact path. C4 reuses logits
on all 35 observed steps. The probe subphase now includes the deferred gather
when reusing logits, so it must not be compared as a probe-only duration.

Six interior C8 graph-node samples per rank give rank-0 FP4 12.424 ->
10.050 ms and FP8 8.957 -> 8.554 ms: combined GEMM 21.382 -> 18.604 ms,
saving 2.777 ms (12.99%). Independent startup tuning changes the selected
plans; the earlier candidate trace's 18.089 ms is not the final trace's
measurement. The corresponding per-category slowest-rank sums are
21.593 -> 18.604 ms. Neither sum is a directly measured whole-graph latency.
The complete target graph's slowest-rank envelope is 37.847 -> 34.704 ms.
The actual GEMM reduction broadly reaches forward rather than disappearing
in another target phase, but the standalone 4.263-ms saving is not reproduced
in the full model. The C8 GEMM reduction gate therefore still fails.

GDN (4.481 ms), target attention (3.266 ms), target TP communication
(4.092 ms on rank 0; 5.213 ms per-step rank maximum), sampling (5.757 ms)
and draft (7.600 ms) remain material costs. Category kernel means and phase
medians have different timing scopes and must not be added as exact totals.
At unchanged emitted tokens, taking 20% off the control's 21.382-ms GEMM
would reduce its 51.182-ms round by only 8.36%, or improve round rate by
9.12%. A 20% faster round rate requires 42.652 ms, another 4.839 ms below
the final trace. This is a fixed-token diagnostic inference, not an endpoint
speed or acceptance prediction. The ordinary-service result remains the
separately measured +11.37% full-48 C8 window result above.

Full events, route exports, Nsight report/database and analysis are retained
under `trace-fulfillment-final`; `fulfillment-final-trace-comparison.json`
contains the control/final phase and kernel comparison. All owned benchmark
services are stopped after capture. Keep Draft PR #691 unmerged pending
the remaining C8, absolute-quality and cross-model acceptance work.

## Earlier service validation

Candidate `qpn-fc-default` completed the paired serving gates using the normal
extension and automatic configuration. It failed the C8 guard and is not
qualified for promotion. Three-run medians are:

| Concurrency | Rolling control/candidate tok/s | Rolling gain | No-new-prefill control/candidate tok/s | Window gain |
| --- | ---: | ---: | ---: | ---: |
| C1 | 243.15 / 240.48 | -1.10% | 344.88 / 343.49 | -0.41% |
| C2 | 266.85 / 277.67 | +4.05% | 369.04 / 478.69 | +29.71% |
| C4 | 297.30 / 321.13 | +8.02% | 477.65 / 577.46 | +20.90% |
| C8 | 365.98 / 363.81 | -0.59% | 621.91 / 618.40 | -0.56% |

C8 rolling acceptance fell 54.92% -> 52.72% (-2.20 percentage points), beyond
the two-point limit. Full-48 window acceptance fell 50.51% -> 49.67% (-0.84
points). C4 window acceptance was identical at 65.16%; its +20.90% speed gain
supports the expected whole-decode benefit of the -34.5% GEMM estimate. C2
window acceptance rose 63.31% -> 76.48%, so its +29.71% cannot all be credited
to faster GEMM. C1/C2/C4 windows repeat the first C prompts, whereas rolling
uses 16/24/32 prompts; this difference also prevents assigning the whole
window-versus-rolling gap to prefill alone.

Both quality runs had 14 correct natural completions and 15 natural stops out
of 16; this is relative parity, not 16/16 success. The candidate retained the
4K prefix-cache route (3296 hit tokens) and passed the 32K C2 route smoke.
The latter is not a long-context speed baseline. These results are recorded
in `paired-qpn-fc-default.json`; no failed gate has been waived.

Rolling decode capacity is C times emitted decode tokens divided by summed
per-request decode duration. It excludes each request's TTFT but includes
pauses caused by replacement prefill. The fixed workloads are C1/16,
C2/24, C4/32 and C8/48 requests, with three measured repetitions. ITL is the
distribution of per-request mean ITL, not individual SSE/token gaps.

The no-new-prefill measurement counts returned token IDs between the latest
first token and earliest last token of a concurrently admitted wave. It
includes q1..q8 and host/transport overhead; it is not a GPU-step recorder.
C8 covers all 48 fixed prompts in six waves per repetition. The initial
first-eight-prompt diagnostic is retained separately rather than substituted
for the complete C8 workload.

## Draft projection stability

The initial C1 acceptance regression was isolated with an eager native-artifact
A/B test. The 4,023 shared native GPU kernels had identical instruction streams.
Importing the reference TurboMind plans before candidate initialization restored
the two diagnostic requests exactly. Importing only FP8 plans did not restore
them (71 -> 121 verification rounds for the sensitive request); importing only
FP16 plans restored every returned token and the 71/77/71 round counts.
The sharded draft context projection was independently tuned on each rank,
including split-K 10 on two ranks and 12 on the others. Its different rounding
changes draft scores, acceptance and eventually the seeded generated path.
The recorded failed service candidates therefore do not isolate batch-kernel
arithmetic from this existing startup-tuning variability.

An experimental cuBLAS projection retained the TP4 column partition and
all-gather. It did not import a task LUT or introduce a user switch.
On the actual 1280-by-25600 local `fc.weight`, seven-round CUDA Graph medians
were 142.49/109.08/134.80/242.77 us for tuned TurboMind at M8/M16/M32/M64,
and 116.87/108.26/118.01/189.91 us for cuBLAS. Changed-input eager/Graph
checks passed. The ordinary heuristic was also slower. Omitting its packed
FP16 copy saves 62.5 MiB per rank; service model memory drops 9.57 -> 9.51 GiB.
Six existing projection-contract tests passed, but its three-run C1 service
median was 236.50 tok/s with 54.86% acceptance, versus 243.15 tok/s and
57.04% in the saved reference. The -2.74% speed and -2.18-point acceptance
changes failed the serving guards. The cuBLAS source change was withdrawn.
Disabling generic FP16 autotuning alone also failed (227.05 tok/s, 52.07%).
These are failed diagnostic candidates, not production defaults.

The remaining legacy split-10 tactic passed the first diagnostic C1 guard:
242.89 tok/s and 56.77% acceptance. It is now selected in source for the
existing SM70 TP4 sharded context-FC contract, M1..8/N1280/K25600. The
CTA8x256x64 two-stage kernel and split-10 reduction are fixed; an imported
or previously tuned incompatible plan cannot override them. A small host
cache retains the eight row-count plans. This adds no weight layout, device
workspace, environment switch or private LUT dependency. Other FP16 shapes,
M16+ projections, quantized GEMM tuning and sampling remain unchanged.
The runtime micro confirms the default selector; all eight changed-input
eager/Graph tests pass after seeding an incompatible split-16 cache. Full
service acceptance, performance and quality still decide promotion.

## Numerical and build validation

92 focused GPU tests cover the context projection and FP4/FP8 gate/up projections,
M9/15/16/17/24/31/32, changed-input CUDA Graph replay and exact reduction
order. Representative real-weight M8/M16/M32 results are bitwise equal to
the normal reference extension at three input amplitudes; eager and Graph
outputs also agree. The unchanged M64 path retains its existing tolerance.
A separate artifact audit compares 27 relevant C1 kernel instruction streams
and finds them identical to the baseline; this does not replace C1 service
latency and acceptance checks.

The completed `qpn-fc-default` normal extension SHA256 is
`75303f12d50f02ebfb1f4f7616e009125851a207041ecda4df9e3ede9ec39d00`.
The reference normal extension SHA256 is
`7449bff7e4cc50fd2c9e9d243b65199126c4063d3aacd7d17a034422173c1890`.
The candidate is built from the owned source tree and loaded as
`vllm/_C.abi3.so`, without a private sidecar or library override. Runtime
extensions and FlashQLA use the same source base. Build artifacts and raw
traces are not committed.

## Rejected candidates and limits

Retain these results so later work does not repeat the same experiments:

- Unrestricted FP4+FP8 M64 tiles reduced estimated GEMM by about 19.6%, but
  full-48 no-prefill acceptance fell 50.51% -> 47.24% (-3.27 points).
  The first-eight-prompt diagnostic also failed (-4.97 points).
- A two-stage tuner preserved the selected legacy split-K boundaries and
  passed 134 GPU checks, including captured tails. Full-48 acceptance still
  fell to 47.86% (-2.65 points), while its decode gain was only 0.25%.
  Cross-startup bitwise identity was not established: independent legacy
  tuning itself can select different reduction partitions.
- Restoring FP4 M64 while retaining the new FP8 M64 tiles passed the full-48
  window gate: 621.91 -> 628.43 tok/s, acceptance 50.51% -> 49.57%.
  It nevertheless failed the C1 rolling guard in two runs: approximately
  226.6 tok/s and 52.22% acceptance versus 243.15 and 57.04%. That run reused
  a compilation cache. The later FP16-only plan ablation above identifies
  draft projection tuning as a confounder; this candidate remains unqualified
  until its own ordinary-service gates are repeated.
- The initial broader candidate had the same 14/16 correct natural stops as
  the reference on its quality pair: one wrong answer and one request
  capped at 4096 tokens. Counting an answer embedded in unfinished reasoning
  would incorrectly report 15/16. The original four-question smoke was 4/4.
- Previous rejected paths include repeated M16/M32 slicing, per-step full
  FP16 weight expansion, prepared C2 duplicate weights, register caps alone,
  shared decoded-B staging, the tested Marlin replacement, and a slower
  no-copy fast-scale C2 variant.

No fresh PRO comparison or full 35B-A3B AWQ/FP8 model-speed claim is made.
The final patch does not change AWQ or grouped MoE operators. It does not
establish the broader objective of surpassing PRO by 5%.

## Reproduction and retained evidence

Use `benchmarks/benchmark_sm70_batch_gemm_reuse.py` with a single idle V100,
`--model MODEL/model.safetensors`, a normal `--extension`, an output path
and `--rows 8 16 32 64`. Run the baseline with `--write-oracles`, then the
candidate against the same `--oracles` directory, seed and row order.
For direct operator micros, set the FP8/NVFP4 dense tune maxima to 64;
the tested serving configuration sets these automatically.

Artifact set: `sm70-gemm-expand-20260926`. The final evidence uses
`qpn-only-eager*.json`, `qpn-only-kernel-tests.log`, `c1-sass-audit.json`,
`service-qpn-only.log`, request-level JSON records, native/build manifests,
controlled-admission token events, natural-EOS outputs and context checks.
Failed candidates retain their own artifact hashes, checkpoint directories
and `candidate`, `split-safe`, and `fp8-m64` result labels.
