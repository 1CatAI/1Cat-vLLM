# Flash-Next MTP4 complete-round latency

## Acceptance

The default path must average at most 15 ms per complete MTP4 round on TP4
V100-SXM2-32GB, using exactly 8192 input tokens and a 262144-token service
limit. The round includes four drafts, target verification, sampling, handoff
and preparation. Use endpoint steady decode elapsed time divided by the
completed speculative-round count. Never divide by emitted tokens or use
only target forward time. Record emitted tokens/s, tokens per round and
acceptance separately.

Speed admission uses a fresh process, no profiler and no supplied VLLM
performance variables. Record source SHA, native extension hashes, actual
routes, model identity, Torch/CUDA versions, topology and GPU power limits.
The current remote TP4 group is fully connected with NV2 and has a 300-W
power limit on every V100. Historical 185-W timings are not matched controls.
No wheel packaging is required during development.

PLE defaults to file-backed disk mmap. Only bounded results and flags use
mapped pinned memory. Reuse the mapped-result transport from #831 for MTP;
other speculative methods retain their existing transport qualification.
The resident PLE table must not become the default.

Target and draft share the numerical contract and tools in
[SM70 distribution acceptance](sm70_qwen38_distribution_acceptance.md).
`qwen38_distribution_probe.py` owns the limits and summary implementation;
`sm70_teacher_forcing_metrics.py` only aligns MTP captures with it. FP32
reassociation and lower-precision candidates may be evaluated with those
limits. Target and draft must pass separately. Also require matched task
accuracy, truncation/noncompletion checks and no material acceptance loss.
A route-hit or a forced reference output is not a quality-set result.
C4 short concurrency must not regress.

## Rebuild the baseline before optimization

The campaign has been rebased onto current main. #831 and #832 are merged;
PRs #859 and #885 were still open when checked on 2026-10-05. Record their actual
merge state rather than treating either candidate as a main dependency.
Include #914's batched PLE n-gram implementation. Freeze the final main SHA
before recording any new timings. The previous 22--24-ms results are historical
and cannot establish the new baseline.

First use `benchmark_sm70_mtp4_round.py` without diagnostic flags. It warms
up each fixture, records at least three repetitions, saves token sequences,
request timestamps, finish reasons and speculative counters. All inputs are
exactly 8192 tokens. `fixed8k` preserves the historical deterministic fixture;
its forced output past EOS qualifies timing only. The `natural8k/*` fixtures
use normal EOS and checkpoint sampling. Report their results separately.
A fixture selector reduces work while keeping the frozen prompt unchanged.

Then collect one node trace with `--node-trace --fixture <same fixture>` and
Nsight Systems `--trace=cuda,nvtx --sample=none
--cuda-graph-trace=node:host-only --cpuctxsw=none
--capture-range=cudaProfilerApi`.
Trace admission is always false. Annotations are installed after graph capture;
they preserve FULL graphs and distinguish target M5, draft step 0 at its real
width, draft steps 1--3, target head/sample, preparation and handoff.
Reset the prefix cache before both control and traced requests to keep prefill
conditions consistent. Report token tapes and round counts, and compare diagnostics against an
ordinary control in the same loaded engine. Do not assume cross-startup
identity or hide differences through a timing rescale.
Report the profiled endpoint round mean beside the unprofiled control. Do not
rescale kernel service sums to make them equal to unprofiled wall time.

## Choose work from the new trace

Build a table per target kernel/shape and TP rank, including calls per round,
active weight layout and padding, weight bytes actually read including repeated
passes, the weight-only lower bound at 750 GB/s, actual node service and
service minus lower bound. Include the target head even though it runs in the
sampling phase. Sort by that last column. Show rank service, critical rank,
GPU sums and wall envelopes separately.

The saved tensor inventory provides shapes, strides, storage identities and
packed-buffer sizes. It is **not** a DRAM-traffic measurement. Do not add all
resident weights or treat unique checkpoint size as per-round reads. Resolve
which packed tensors each selected kernel reads. Validate traffic on the
leading family with focused NCU counters; unknown read counts remain unknown.
For expert kernels account for selected experts and reuse across tokens.
The weight-only bound excludes activation traffic, communication and compute.

Recheck the historical dense, HC, small-kernel and draft bottlenecks. Current
source already contains TP-sharded M5 HC and some router/shared-expert batch
routes; the trace must prove their runtime selection. Prioritize missing
M2--M8 skinny-M projections only after the table establishes the gap. Prove
an exact-shape operator/single-layer improvement before whole-model runs.
Run end-to-end only when the accumulated predicted saving is substantial or
when preparing to merge.

Do not retest the rejected local HC split-K, resident MoE, directly reused
shortlist, single cooperative HC kernel, BV16 or four-warp candidates.
Their historical artifacts remain in the prior campaign branch and worklog.

## Strata reference

Read-only reference snapshot: `Niko1221/Strata` commit
`6f32ec070f23ced9f50e704d854d775da52591ab`, MIT license.
Use [draft-vocab documentation](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/DETAILS.md),
[the corpus/script builder](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/draft_vocab.py)
and [multi-token HC kernels](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/fused_gr.cu).
Its current CJK subset contains 106299 IDs; the English/code subset contains
40525. These sizes are references, not our vocabulary choice. Evaluate three candidates: the Strata English/code set, its CJK extension,
and a set ranked by actual target-model outputs. Corpus coverage is descriptive
and cannot admit or reject a draft set. Compare Chinese/English acceptance,
actual emitted tokens/round, complete-round latency and tokens/s with a fixed
MTP4 control; select by tokens/s. Retain the original token IDs. Preserve original token
IDs and use compact value/ID IPC instead of gathering full vocabulary logits.

The current Strata confidence cutoff defaults to zero in source; a 0.5 gate
is a proposed candidate here. Define whether confidence uses the full or
restricted vocabulary, validate its calibration and retain a fixed MTP4
control. Variable verify graphs need explicit window/round counts; changes
in draft count cannot satisfy the fixed-round target through denominator
changes. Report acceptance, proposals and emitted tokens per round together
with output speed.

HC reference ideas include token/stream norm blocks, multi-token reuse of
one weight load, double-buffered input tiles and row-group/stream parallelism.
SM70 uses ordinary copies rather than cp.async. Preserve attribution and the
MIT notice if reference code is copied or adapted. CPU expert execution and
single-card cache mechanisms are outside this TP4 scope.

## Current status

### Unprofiled main baseline, 2026-10-05

Integration base: `c60bbe1194403cb5c36404bc35bc2580d8ee72a0`.
Measured source: `e65592623899f1da9b1267c8cae44d5bcd03240e`.
Normal source-built native components; Python 3.12.14, Torch 2.10.0+cu128,
CUDA 12.8, driver 580.173.02. TP4 V100-SXM2-32GB, full NV2, 300 W per card.
FP16 computation/KV, FP32 SSM state, NVFP4 target experts, FP16 draft experts.
PLE disk mmap, fixed MTP4, max length 262144, input 8192, one request, 4-GiB
KV cache per rank. No profiler or user-supplied VLLM variables.

| Fixture | Complete round mean | Repetitions / rounds | Decode tokens/s | Draft acceptance | Tokens/round | Finish |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Historical fixed8k, greedy | 21.835 ms | 3 / 933 | 75.40 | 16.16% | 1.646 | length, deliberately ignore EOS |
| Chinese natural8k/0, T=1, p=.95, k=20 | 24.304 ms | 3 / 363 | 172.40 | 79.75% | 4.190 | EOS in all repeats |

Round means divide total endpoint decode elapsed time by total speculative
rounds. The fixed fixture samples are 21.820, 21.849 and 21.835 ms; the natural
fixture samples are 24.379, 24.274 and 24.260 ms. Outputs match across repeats
within each fixture. The Chinese arithmetic answer gives 240 km and 68.57 km/h;
this is an output-health check, not quality-set or numerical admission.
Speed results come from `benchmark_sm70_mtp4_round.py`; its JSON preserves
prompt hashes, complete token IDs, sampling, request timestamps and counters.

Both fixtures fail the 15-ms target. Node attribution, measured traffic,
quality/distribution and C4 admission remain pending. No structural-kernel
speed claim is made.

Two node captures measured 28.107 and 28.060 ms over 300 rounds. Both differ
from the original control's token tape and 311 rounds. Resetting the prefix
cache and disabling CPU context-switch tracing did not remove the difference.
The unprofiled CUDA-event run also produces the 300-round tape, including its
uninstrumented warmup: the difference is not exclusively caused by profiling.
AOT/cache state and restart variability remain hypotheses, not established
causes. The event harness now adds ordinary controls in the same loaded engine.
Do not rescale either node capture into the 21.835-ms control.

### Unprofiled phase diagnostic

Source `8d28bfc00f`, same hardware/max length/input/sampling, no profiler.
Three endpoint means are 23.844, 23.818 and 23.771 ms over 300 rounds each.
CUDA event target-to-next-target intervals on rank 2 average 23.784 ms after
excluding the edge rounds; their agreement with endpoints checks the interval
boundaries. Event instrumentation overhead has not yet been checked against
same-process ordinary controls, so these are diagnostic measurements.

| Same-rank GPU-stream wall interval | Mean |
| --- | ---: |
| Target M5 forward | 17.591 ms |
| Draft 0, M5 | 1.338 ms |
| Draft 1, M1 | 1.045 ms |
| Draft 2, M1 | 1.034 ms |
| Draft 3, M1 | 1.030 ms |
| Draft metadata/gaps beyond its four graphs | 0.114 ms |
| Target head/sample | 0.729 ms |
| Target execute outside forward, including preparation/waits | 0.863 ms |
| Sample/handoff outside head and draft | 0.032 ms |
| Remaining round boundary | 0.007 ms |

Use nested ranges only once in this wall table. Kernel service from Nsight is
reported separately. The 300-round synthetic case counts acceptance beyond the
output cutoff: raw acceptance length is 1.720, while actual steady emitted
output is 512/300 = 1.707 tokens/round. It is timing-only, not output quality.

### FP32-policy router/shared repair

Real 48-layer checkpoint weights, M5, FP16 inputs/weights/output, FP32
accumulation/reduction, cold streaming sequences larger than L2, graph timings:

| Operator chain | Vendor control | Native candidate | Saving |
| --- | ---: | ---: | ---: |
| Router | 0.616 ms | 0.386 ms | 0.230 ms |
| Shared up plus SiLU/multiply | 0.644 ms | 0.327 ms | 0.317 ms |

The worst FP64 comparison errors do not increase over the vendor across input
scales 0, .03, .1, 1 and 3. This is operator evidence, not model distribution or
quality admission. Shared up now stores FP32 K320 partials under the ordinary
worker precision policy and preserves the final FP16 projection/activation
boundaries. The legacy FP16-partial schedule remains available. An explicit
native FP32 capability allows old binaries to retain the vendor fallback.

The new integration base is `f415b92d9e`: it already contains the router repair,
PR #900 topology-qualified HC, and #924 GDN preprocessing. Reuse those main changes;
only the shared FP32 schedule adds new projection behavior here. The combined
0.547-ms estimate is relative to the earlier c60 control; do not count main's
router contribution twice. Full-round, shared distribution, quality/acceptance
and C4 gates on the final source remain pending. The 15-ms objective is unmet.

### Trace calibration and source-audited weight floor

`analyze_sm70_mtp4_phases.py` compares each CUDA-event request with its ordinary
control in the same loaded engine. It reports output/counter differences and
observer overhead; the initial diagnostic tolerance is 2% in either direction.
This is a trace reliability check, not a numerical gate or endpoint speed
admission. Missing controls cannot pass. Per-rank nested intervals are counted
once, with mean/p50/p90/p99 and an explicit boundary residual. GPU event origins
are independent across ranks. Never maximize each phase separately and add them.
The six focused observer/analysis CPU tests pass. GPU calibration is reported in the final-source section below.

The earlier reset-cache node capture at source `7778083c3f` provides these rank-0
native weight-read audits, sorted by service minus the weight-only floor.
They are historical profiled service, not new-main endpoint measurements.

| Native target family | Calls/round | Issued weight bytes/round | Weight floor at 750 GB/s | Profiled service | Difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| HC up with fused gather | 96 | 157,286,400 | 0.210 ms | 1.072 ms | 0.862 ms |
| HC down partials | 96 | 188,743,680 | 0.252 ms | 0.848 ms | 0.597 ms |
| Output projection | 48 | 377,487,360 | 0.503 ms | 0.913 ms | 0.410 ms |
| Packed GDN QKVZ and BA | 36 | 760,872,960 | 1.014 ms | 1.386 ms | 0.372 ms |

HC uses TP-local down `[96,10240]` and up `[2560,320]` packed weights; both
include padding/stream layout. GDN BA reads 32 padded output columns, not only
the 24 logical columns. M5 reads each listed pack once; unsupported wider
shapes must account for repeated row tiles separately. The native source loops
establish issued reads, not physical DRAM traffic. Activation/state traffic and
IPC waits are excluded from these floors. Their difference is an investigation
budget, not a promised optimization gain or a bandwidth-utilization counter.

The complete per-kernel artifact retains unknown traffic explicitly. Vendor
weight passes require counters; NVFP4 weight and FP16 scale reads depend on the
actual selected expert groups. Do not substitute resident tensor size or the
maximum 50 route groups for those unknown measurements. New-main node
attribution and NCU counters must supersede this historical ranking before
selecting the next implementation.

### Final-source default baseline and calibrated phases

Source `925e6ca6c350b786b76c398c9ea5742ef5176e03`, integration base
`f415b92d9e`, normal native `_C` SHA256
`474cdbe06936d726f6f596648b310de6665edc6f19b0b7475c59fd1be2c6a325`.
No wheel, private DSO, supplied VLLM variables or profiler. TP4 fully connected
V100-SXM2-32GB, 300 W, Torch 2.10/cu128, CUDA 12.8, FP16 computation/KV,
FP32 reduction/state, disk mmap, 262144 startup capacity, exactly 8192 input.

Fixed8k complete-round means are 23.189, 23.152 and 23.219 ms: weighted mean
23.186 ms over 900 rounds, 73.606 steady decode tokens/s, 18% draft acceptance,
and 1.707 actual steady emitted tokens/round. All three outputs/counters match.
This fails 15 ms. The historical 21.835-ms/311-round tape is a different output
trajectory and cannot be used to claim a paired speedup or regression.

The normal-EOS Chinese arithmetic smoke averages 25.699 ms over 288 rounds,
167.807 decode tokens/s, 83.594% acceptance and 4.313 actual emitted tokens/round.
All three outputs match and end normally; 240 km and 480/7 km/h are correct.
This is output health, not distribution or task-set admission. Fifteen real
SM70 kernel/graph tests pass without skips, including the FP32-policy M5/M10
shared dispatcher with changed replay inputs.

Three same-process ordinary/event pairs preserve token tapes and speculative
counters. Observer endpoint overheads are +0.029%, +0.116% and -0.124%, passing
the 2% reliability check. Their ordinary mean is 23.246 ms; event endpoint mean
is 23.248 ms. The same-rank interval closure below excludes edge transitions
and uses rank 3 for all three repeats, not independently maximized phases.

| Calibrated GPU-stream interval | Mean |
| --- | ---: |
| Target M5 forward | 16.913 ms |
| Draft 0, M5 | 1.344 ms |
| Draft 1, M1 | 1.090 ms |
| Draft 2, M1 | 1.074 ms |
| Draft 3, M1 | 1.065 ms |
| Draft outside its four graphs | 0.112 ms |
| Target head/sample | 0.718 ms |
| Target execute outside forward | 0.870 ms |
| Handoff outside head/draft | 0.031 ms |
| Round boundary residual | 0.005 ms |
| Complete interval | 23.221 ms |

The four draft graphs total 4.572 ms; their complete outer interval is
4.684 ms. The GPU interval differs from its matched event endpoint by 0.027 ms.
The separate default process mean is 23.186 ms; never substitute target forward
or divide either round value by output-token count.

A matching node capture on the same source preserves the 300-round tape but
measures 26.665 ms. Retain its 3.479-ms overhead explicitly; do not rescale its
kernel service into the calibrated wall table. Router/shared batch operators
are selected in its actual target M5 graph. Some short-kernel services are
perturbed substantially: prioritize a proposed change only after the exact
operator-chain microbenchmark confirms unprofiled benefit.

The first NCU filter matched no kernels because its default name basis is
`function`, not `demangled`; exit zero was not a valid counter capture. The
corrected captures contain six actual metrics each. Cold-weight router DRAM
reads are 2,659,136 bytes for vendor and 2,666,400 bytes for native, versus
2,621,440 source-audited weight bytes. They do not show extra complete weight
passes. Active-warps fractions are 6.25% and 1.56%; sampled SM throughput is
8.52% and 1.39%. These are single-kernel cold counter snapshots, not steady
model utilization. Their instrumented durations are excluded from speed
claims. Native and vendor steady graph timings remain a separate measurement.

The next screen reuses existing channel-QPN8 preparation and M2-M8 dispatch
for actual checkpoint GDN/output projections, comparing the current native
FP16 chains, not a slow vendor-only control. HC remains FP16. No defaults change
until shared full-vocabulary distribution, task quality, acceptance and C4
checks pass. Expert-plan snapshots record actual valid groups without per-round CPU
synchronization; the measured traffic qualification follows below.

### Traffic qualification and QPN8 screen

Source `5c95f6e627`, same native binary and model computation as the recorded
baseline. The plan-only diagnostic attaches no Nsight collector and preserves
the 300-round token tape. All four ranks record 48 layers and agree on valid
groups: after excluding edge rounds, the mean is 29.561 groups/layer, not 50.
Issued W13 codes plus FP16 scales total 726,491,919 bytes/round/card; W2 totals
363,245,960. Their 750-GB/s floors are 0.969 and 0.484 ms. The matching node
services are 1.693 and 0.840 ms. These are issued-weight floors versus profiled
service, not measured DRAM bandwidth. Do not claim 730 GB/s from the maximum
route count. Qualify steady kernels at the observed group density next.

The existing QPN8 native-chain M5 screen completes on real checkpoint weights,
FP16 inputs/output and FP32 accumulation, seven alternating graph trials:

| Projection chain | Current native FP16 | Existing channel-QPN8 | Saving |
| --- | ---: | ---: | ---: |
| GDN input, all 36 layers including BA and output splits | 1.182 ms | 1.153 ms | 0.029 ms |
| Output, all 48 layers | 0.674 ms | 0.422 ms | 0.252 ms |

The native GDN issued-weight rate is already approximately 644 GB/s. Halving
its QKVZ storage does not halve chain latency: the existing FP8 path retains
BA and split/staging work. Reject this full GDN conversion for insufficient
benefit; do not promote it or repeat parameter sweeps. Output projection stays
a candidate, with only 0.252 ms measured operator saving. Neither arm has model
numerical/quality admission; FP8 operator relative L2 errors reach about 2.7%
versus FP64, so operator timing cannot justify enabling it by default. No FP8
runtime defaults change and no model test is credited to these microbenchmarks.

### Production-chain qualification

Source `06aa730e6b`, unchanged normal native binary. These microbenchmarks
use real checkpoint weights and synthetic activations; they do not measure
model-round latency or admit a numerical change. Seven unprofiled CUDA-graph
trials, no dispatch overrides or candidate parameter sweeps:

- TP4 current three-kernel HC projection/transport chain over all 96 pairs:
  critical-rank median 2.273 ms. Its 346,030,080 issued weight bytes have a
  0.461-ms weight-only floor. Combine/norm and the final mixer are excluded.
  The difference includes both compute and transport, not just HBM reads.
- Current grouped NVFP4 layer with a 30-group M5 density fixture: plan/W13/
  SwiGLU 30.46–30.47 us; W2/ordered reduction 12.96–12.98 us; whole chain
  43.27–43.29 us. Independent expert-bank offsets agree. Weight plus scale
  floors are 20.48 and 10.24 us. This fixture does not preserve captured route
  multiplicities and cannot replace actual whole-model timing.

HC has the larger confirmed gap; isolate its local projection work from TP
transport and inspect counters before choosing a structural change. Avoid
retesting rejected cooperative HC or local split-K implementations.

### HC local counters and rejected prefetch

Unprofiled current local projection diagnostics measure 0.947 ms for 96 down
projections plus local packet preparation, and 0.851 ms for 96 up/mix projections
plus output clearing. They use extra local preparation kernels, so their sum
cannot be subtracted exactly from the 2.273-ms transport chain.

NCU captures of production local kernels (source `dbe942926f7973999cd50a663a03318b4c1ae5be`, after warmup,
cold replay cache, unmodified clocks) contain seven actual metric rows each.
Down/up read 2,072,480/1,676,288 DRAM bytes for weight packs of
1,966,080/1,638,400 bytes. There is no duplicate full weight pass. CTA counts
are 60/80 with one warp each; sampled active warps are 1.56%, global-memory
long-scoreboard stalls 59.28%/38.49%, and registers/thread 86/90. This identifies
limited latency hiding as an investigation target; NCU durations are excluded
from steady performance results.

A register-prefetch implementation keeps the following K16 weight/input tile
live during the current MMA. Source `55125727fe`, ordinary source-built native
binary, paired TP4 complete-chain graph trials: control 2.295 ms, prefetch
2.614 ms (+13.91%). Changed inputs at five scales preserve outputs on all four
ranks, but the speed regression rejects the implementation. Remove the unused
kernel/API candidate, retain its source patch and raw results as artifacts,
and do not launch a whole-model test or repeat parameter tuning for it.

### Draft-head QPN8 candidate

Existing native channel-QPN8 on the actual TP4 head shard (62,080 by 2,560):
M1 FP16 vendor head 0.453 ms versus QPN8 0.191 ms, saving 0.262 ms in seven
alternating graph trials. Issued code/scale storage is 159,048,960 bytes versus
317,849,600 FP16 bytes. Four invocations suggest an operator budget of 1.048 ms;
this is not an accepted model-round improvement. Synthetic hidden-state
scales up to 3 yield at most 0.262 logit error versus FP64, but no model gate
can be inferred from that test.

The owned candidate uses a draft-only head view after target/draft sharing.
The target's checkpoint parameter and quantization method remain unchanged.
The view reuses native QPN8 for M1..8, FP16 computation/FP32 accumulation and
the existing padding mask/global-ID/compact top1 transport. Wide batches use
the original head. Channel preparation shares the existing implementation and
bounds FP32 scratch by 4,096 weight rows. Raw full-vocabulary teacher-forcing
logits use the same candidate view as greedy proposals. No user environment
variable is added. The benchmark candidate remains unmerged pending target and
draft distribution, task quality/acceptance, C4 and complete-round validation.

A diagnostic reference worker restores main's vendor shared-up policy and
checkpoint draft head. Its requests are explicitly ineligible for default
speed admission. Use the frozen 16-prompt manifest (math, code, Chinese and
retrieval) with the same 8K prompt/256K capacity, and compare target/draft
separately through the shared distribution tool and limits.

### Structural campaign and numerical gates, 2026-10-05

Stop register-prefetch attempts. Priority is HC CTA-parallel reduction/finish,
fused router projection/softmax/top10/plan, fused shared-expert chain, segmented
QSA selection/attention, distributed GDN state updates, and expert plan/launch
fusion. All proposals first use whole-layer graph benchmarks. Keep default
endpoint timing at 8K input/256K capacity, no profiler and actual round counts.

The frozen 16-prompt reference/candidate comparison contains 2,656 target
positions and 2,176 draft positions. Target shared-FP32 comparison passes all
limits (zero KL/raw logit difference, 100% top1). Draft-head QPN8 gives mean KL
0.000466, p99 0.006037, max 0.020955 and top1 99.724%; raw maximum logit error
0.710938 is recorded only. Under the shared owner-revised contract this
draft distribution passes; restore automatic draft-only QPN8 preparation.
The target head and checkpoint parameter remain unchanged. Shared and output
projections retain independent diagnostic reference paths.

A full HC graph over 96 real checkpoint pairs includes combine/norm, down
projection/SiLU/gather and up/mix/gather. Unprofiled critical-rank medians:
production 2.713 ms, 8-warp CTA split 2.378 ms, 16-warp 2.450 ms. These are
operator-chain measurements, not endpoint speed. The 8-warp saving is only
0.335 ms and does not satisfy the HC-chain goal of 1 ms. Complete column-pair
ownership removes the separate down gather, but seven alternating full-chain
trials regress: control 2.709 ms, 8-warp finish 3.374 ms, 16-warp finish
2.977 ms. This implementation is rejected; do not integrate or tune it.

Output-projection QPN8 independently fails the same 16-prompt gate. Target:
mean KL 0.006661, p99 0.089210, max 0.552678, top1 98.117%, raw maximum
logit error 9.433594. Draft: mean KL 0.018349, p99 0.268329, max 6.915023,
top1 98.851%, maximum error 25.859375. No runtime default is enabled.

Compact top1 transport was default-off and accepted only one value/ID pair.
The structural candidate enables the existing IPC reducer automatically on
SM70 and extends its one-block protocol to 128 independent row pairs. C4
uses the same compact payload instead of reverting to NCCL. Topology and
graph-warmup guards retain the exact NCCL fallback. Ordinary hidden-state
all-reduce selection is unaffected. The four-decision TP4 graph benchmark
compares changed inputs, shard ties, width changes and repeated epochs against
NCCL. This is communication-only evidence; full-model gates remain required.

TP4 graph results pass all widths (1, 4, 5, 16, 128), changed values, ties and
repeated epochs. Four decisions at M1: NCCL 0.101535 ms versus compact IPC
0.036680 ms; M4: 0.103511 versus 0.036879 ms. The first benchmark attempt
did not finish communicator shutdown; destroy captured NCCL graphs before
the process group, retain its log, and credit only the completed retry.
This is not the model C4 acceptance gate or a complete-round speed result.

Source audit identifies three all-gathers per draft step: top1 value/ID,
`fc_embedding` output and `fc_hidden` output. The latter two carry hidden
tensors needed by the HC block, not logits. Compact top1 replaces only the
first. Removing the other two requires a distributed projection/norm/HC
dataflow; silently substituting token IDs is incorrect.

The published owned branch merges main `0e359c87d3` (including the newer GDN
projection tails, strided QKV, direct GDN output and compact target top-k).
The earlier 23.186-ms endpoint is a frozen historical control, not the current
main result. Rebuild normal native extensions before measuring the merged
endpoint; do not reuse the older extension with newer dispatch code.

The router structural prototype spreads checkpoint-FP16 projection across
80 CTAs, sharing each weight load across all rows. Writers publish their
results before a completion ticket; the last CTA runs normalized top10 and
the existing integer-only route-group plan. There are no spinning CTAs or
cooperative-launch assumptions. FP32 accumulation retains the FP16 router
logit boundary. The checkpoint graph compares projection/selection/plan over
48 layers; production dispatch remains unchanged pending measured gains and
shared model gates. It is not a register-prefetch experiment.

The 80-CTA SIMT/last-producer prototype passes real-weight operator checks
(all tested scales select the same top10 IDs, maximum router logit difference
0.0078125), but regresses the complete 48-layer router graph from 0.711178 ms
to 2.016435 ms. Reject it without model integration or parameter sweeps.
The unprofiled control must not be equated with the older 1.62-ms profiled
router family service sum. A further design must preserve tensor-core batch
projection and prove that its selection/plan finish does not serialize work.

The apparent 1.352-ms rebase regression was not reproduced under matched
conditions. The owner closed this investigation on 2026-10-05; no further
bisection, historical-runtime audit or regression experiments are scheduled.
The latest-main baseline is fixed at **23.216 ms per complete round**.

Matched fresh-process controls, normal extensions built from each source,
TP4/300 W/8K input/256K capacity/no profiler, three repeats:

| Source | Complete-round mean | Rounds per repeat | Acceptance |
| --- | ---: | ---: | ---: |
| Recompiled `e655926238` | 23.708757 ms | 300 | 18% |
| `925e6ca6c3` | 23.048813 ms | 300 | 18% |
| Latest-main candidate `6e36a5a83a`, main `0e359c87d3` | 23.215921 ms | 300 | 18% |

The matched old/new controls emit identical token tapes and the newer path
saves 0.659944 ms. The rebuilt historical source does not reproduce the saved
21.835-ms/311-round tape: its output matches the frozen newer tape instead.
The owner closed the historical discrepancy without further investigation.
These source controls remain historical evidence, not a new optimization task.

Latest-main default timing samples are 23.308418, 23.150912 and 23.188432 ms
(900 rounds total, 73.513 decode tokens/s, 1.706667 emitted tokens/round).
Natural-EOS Chinese arithmetic: 26.235914 ms/round, 83.594% acceptance,
164.375 tokens/s, 4.3125 emitted tokens/round; all three repeats terminate with
the correct 240-km distance and 480/7-km/h average speed. This is output health,
not task-set or C4 admission. The 15-ms objective remains unmet. Updated
same-process event calibration and one node-level capture are queued under
the shared GPU locks; preserve their overhead separately from endpoint speed.

## Current implementation sequence (owner revision, 2026-10-05)

Restore the admitted draft-only QPN8 head, compact top1 IPC, and the positive
8-warp TP4 HC split on the FP32 fused batched route. The measured operator
savings are approximately 1.048 ms across four draft heads and 0.335 ms across
96 HC pairs; these estimates are not an endpoint measurement. Keep output
projection QPN8 disabled. No new acceleration environment switches are added.

1. Build a 32768-token draft shortlist from corpus frequencies. Freeze eight
   bilingual prompts before collecting candidate results, each with exactly
   8192 input tokens and 600 output tokens. Compare full-QPN8 and shortlisted
   QPN8 heads: acceptance may fall by at most one percentage point, and draft
   head time must decrease. Report proposals, emitted tokens per round, whole
   round latency and tokens/s. Corpus coverage is descriptive, never a gate.
2. Fuse shared gate/up/SiLU/down into one kernel with deterministic tiled down
   reduction and no floating-point atomics; target <=0.3 ms for 48 layers.
3. Use one multi-warp CTA per token for router projection and top10, without
   a last-finishing-CTA protocol; target <=0.3 ms for 48 layers.
4. Rank draft single-CTA, grid-at-most-four and one-warp kernels by service,
   then combine dependent work. Deep HC, QSA and GDN changes are owned by
   the shared decode line; ensure their supported M5 routes are selected here.

Each new structure first needs a real-weight whole-layer graph microbenchmark.
Run full-model timing after accumulated predicted gains reach about 1 ms, or
before merging. Shared GPU locks govern every GPU job and source deployment.

### Structural operator screens, 2026-10-05

These are unprofiled, single-card CUDA graph measurements on an uncontested
V100-SXM2-32GB at 300 W. The TP0 tensors come from the same checkpoint as
the remote TP4 baseline. No full-model speed or quality is admitted here.
Seven alternating graph trials use real weights; shared/router chains rotate
all 48 layers to avoid a single hot-weight fixture.

| Operator chain | Control | Candidate | Decision |
| --- | ---: | ---: | --- |
| Four QPN8 draft heads, M1, largest shortlist shard | 0.833485 ms | 0.268585 ms | Reject 32K: acceptance falls by 14.07 percentage points |
| Four QPN8 draft heads, M5 | 0.840376 ms | 0.270705 ms | Original global IDs preserved |
| 48 shared gate/up/SiLU/down chains | 0.994468 ms | 0.690063 ms | 0.304405-ms operator gain; model gates pending |
| 48 routers, projection/top10/plan | 0.631368 ms | 3.190070 ms | Reject token-CTA prototype; no production dispatch |
| Four FP16 draft expert chains, M5 + three M1 | 0.436480 ms | 0.449876 ms | First fused version regresses; do not integrate |

The router candidate uses five 16-warp CTAs, 128 registers/thread, zero
local spills. More warps inside a CTA still leave at most five active SMs.
Its single-token tensor-core projection repeats the token across eight MMA
rows instead of sharing one weight load across the five real tokens. The
weight read instructions also repeat for each token. These structural limits
remain even without register spilling. Local hardware-counter collection
was denied by driver counter permissions; DRAM bytes/utilization are not
filled with estimates.

The shared candidate uses one residency-checked cooperative launch, 80
eight-warp CTAs, and fixed-order down reductions without floating atomics.
Maximum local-output differences versus the five-launch control are 0,
1.19e-7, 6.10e-5 and 9.77e-4 at input scales 0, .03, 1 and 3. This passes
a layout/finite-value screen, not the independent model distribution gate.
It remains above the 0.3-ms component goal.

The first draft fusion passes finite/zero-input/layout screens and shows
relative output L2 differences below 1.41e-4, but its down stage issues
strided scalar half loads. The next implementation pairs adjacent 128-bit
loads in padded shared tiles and reports up/down stage timing separately.
The up loop also avoids compiler-expanded register preloading. This is a
structural memory-access correction, not a parameter sweep.

### Restored defaults and rejected shortlist, 2026-10-05

Source `f61348a4f6`, normal extension `bf0f4cf15724`, same fully connected TP4
V100/300-W contract, 262144 startup capacity, 8192 input, no profiler and no
supplied VLLM variables. The three fixed-MTP4 round means are 21.739059,
21.593571 and 21.776461 ms. Across 963 rounds the mean is **21.703030 ms**,
1.512891 ms below the owner's frozen 23.215921-ms main baseline. The 15-ms
objective remains unmet. Synthetic acceptance falls from 18% to 15.031%;
this timing fixture runs past EOS, so matched natural outputs are required
before task-quality or acceptance conclusions. Numerical QPN8 admission
remains valid; maximum logit error is record-only.

The disjoint-corpus 32768-ID shortlist fails the frozen eight-prompt,
600-output-token acceptance test. Both arms use the same temperature, seeds,
8192-token prompts and compact value/ID IPC. Aggregate acceptance divides
accepted tokens by proposed tokens, with per-language results retained.

| Eight-prompt arm | Rounds | Acceptance | Complete round | Emitted tokens/round | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full QPN8 draft vocabulary | 1597 | 50.204% | 25.817724 ms | 3.000626 | 116.223 |
| 32768-ID frequency shortlist | 1963 | 36.131% | 25.489029 ms | 2.441161 | 95.773 |

The 14.073-percentage-point acceptance loss exceeds the one-point limit;
the shortlist remains benchmark-only. Chinese acceptance declines from
62.048% to 44.496%. A smaller head alone does not admit the candidate. Forced
600-token output is an acceptance/timing fixture, not a natural quality test.
The already-built 65536-ID candidate uses the same training-only ranking;
its operator proof must pass before a matched acceptance run.

The coalesced FP16 draft fusion also fails the operator screen:
0.436593 -> 0.535665 ms for M5 plus three M1 expert chains. Up/SiLU grows
from 0.319007 to 0.400374 ms; down/route sum grows from 0.114299 to
0.134492 ms. Reject it without model integration or parameter sweeps.
The next draft candidate reduces weight bytes through the existing
channel-QPN8 preparation and fragment decoder, with FP16 compute and FP32
accumulation; it has no default dispatch or model admission.

A calibrated main-source event diagnostic is complete (source `6e36a5a83a`).
Same-engine ordinary/event token tapes and round counters match in all three
pairs; overhead ranges from -0.025% to +0.949%, within the 2% limit.
Rank-3 target M5 averages about 16.780 ms and the complete draft interval
about 4.525 ms. These precede restored defaults and do not replace the
fixed main baseline. The separately profiled node capture measures
26.893 ms/round; its service sums are attribution only, never speed admission.
M5 has 1428 target nodes. Source-audited issued weight reads are retained
separately from actual DRAM counters, which have not been measured locally.

### Grouped QPN8 draft expert screen

Normal `_C` build `bf376946ad16`, real-checkpoint TP0 experts, one V100 at
300 W, no profiler, seven alternating whole-chain graphs. Four expert steps
(M5, M1, M1, M1) improve from **0.435937 to 0.163205 ms** (0.272732-ms
saving). A control running the same dequantized QPN8 weights through the
old FP16 kernels takes 0.435220 ms, isolating execution structure from
quantization alone.

Both kernels reuse the existing channel-QPN8 packing and fragment decoder.
Up pairs gate and value tiles before FP16 SiLU. Down uses ten route warps,
with a fixed-order FP32 sum after each weighted route's FP16 boundary.
There is no persistent grid or floating-point atomic. Launches decrease
from sixteen to eight across the four steps. Compute remains FP16 with
FP32 accumulators.

At scales 0, .03, 1 and 3, zero-input and finite-value checks pass. Relative
L2 difference from the independent dequantized-weight control stays below
0.000240; relative L2 versus original FP16 weights is roughly 4.3--4.9%.
These hidden-output differences are diagnostics, not logit distribution
admission. The benchmark-only worker prepares the proposer subtree alone,
retains original FP16 weights for unsupported widths/prefill, and preserves
shared-expert ordering and the outer TP reduction. Full-vocabulary target
and draft gates, matched acceptance, quality and C4 remain required.

### Follow-up qualification

The 65536-ID corpus shortlist also fails the same eight-prompt test:
acceptance 50.204% -> 47.318% (-2.886 percentage points), round mean
25.817724 -> 25.215015 ms, emitted tokens/round 3.000626 -> 2.888487,
and decode throughput 116.223 -> 114.554 tokens/s. Its critical-shard four
M1 heads improve from 0.850913 to 0.486953 ms; operator speed does not admit
the acceptance loss. Chinese acceptance falls from 62.048% to 51.014% even
though observed Chinese training-corpus coverage is 100%. Both corpus-only
shortlists remain disabled.

The matched checkpoint-head/zero-split HC control records 46.659% acceptance
on these prompts; restored defaults record 50.204% (+3.545 percentage points).
This is separate from the synthetic timing fixture's acceptance loss. The
original-path natural quality control passes all twelve automatic tasks
(four GSM8K, four HumanEval, four retrieval) and all sixteen outputs stop
normally, with no empty/invalid-character/repeated-line anomalies. Two Chinese
explanations pass manual review; two have minor precondition/comparison-count
omissions, recorded for matched candidate comparison. No candidate quality
or main promotion is inferred from the control alone.

The next shortlist uses target output IDs from 32 independent multilingual
training prompts, with natural EOS and a 600-token cap. Their short inputs
serve vocabulary collection, not speed admission. Before collecting results,
fix a 70% normalized model-output / 30% corpus-backfill mixture per language;
language weights and the frozen evaluation prompts remain unchanged. Count
emitted IDs directly, without detokenize/re-tokenize changes. Evaluation
records are rejected by the builder. Acceptance and head-time gates remain
unchanged.

Latest integrated main is `6ae4c0c1c9`, including #885's native disk-row reader
and #955's earlier model-input preparation. The ordinary extension passes
all fourteen CPU mapped-row tests; source/native deployment remains protected
by shared GPU leases. The frozen 23.216-ms baseline is not reopened.

Shared expert follow-up replaces the scalar down projection with packed FP16
MMA, preserving one cooperative launch and deterministic eight-way FP32
reduction. Padded activation rows prevent shared-memory bank conflicts. This
is an operator candidate, not an admitted default. A separate real-weight FC
chain screen moves the embedding residual addition ahead of the two TP
all-gathers, reducing them to one gather while preserving original FP16
arithmetic. It includes both norms and projections and remains benchmark-only.

Structural model probes now record actual opaque-op invocation counts and
widths on every TP rank; registration or buffer preparation alone cannot
qualify a candidate. Deep HC, QSA and GDN work in #961 explicitly screens
M1/M5, but serving dispatch and GPU acceptance are still pending.

### Structural model admission and C4 startup failure

The first cooperative shared chain fails matched full-model distribution:
target mean/p99/maximum KL 0.005431/0.084623/0.353846 and top1 98.6446%;
draft 0.008927/0.107390/5.343494 and top1 99.0809%. Operator error screens
cannot replace these model limits. The packed tensor-core down follow-up
changes the 48-layer graph from 0.993065 to 0.662876 ms, but has no admitted
model distribution and is not enabled in serving.

Restored defaults retain all twelve automatic quality passes and all sixteen
natural stops. Manual Chinese review finds the same two minor omissions as
the checkpoint-head control, with no new failures. This fixed sample is not
a large-sample accuracy claim. C4 remains unqualified.

The checkpoint-head C4 control fails before generation: rank zero needs a
304-MiB allocation while only 210.62 MiB is free. Inductor combo benchmarking
materializes a synthetic 62,080-by-2,560 embedding shard. Exception cleanup
then enters CPU graph-address registration while peers wait in CUDA
synchronization, hiding the original error. Preserve the capture exception
and register addresses only after successful capture. Move the unchanged
draft vocabulary lookup ahead of the compiled small-shape backbone, keeping
lookup and TP reduction inside the outer CUDA graph. This avoids putting the
full vocabulary table in combo benchmarking's arguments; it does not change
model precision, the 256K capacity, or the 4-GiB cache contract. Matched GPU
speed/quality and C4 checks are required for the new input boundary.

Eager teacher-forcing probes explicitly retain the serving decode semantic
context. For draft expert candidates, every forcing case on every TP rank
must additionally prove an opaque-op invocation. Graph capture route proof
alone cannot qualify an eager distribution measurement.

The first run with the new input boundary completes three unprofiled fixed
samples and the frozen eight-prompt acceptance cohort, then fails during eager
forcing on a 1,632-token prefill outside the decode compiler's [1, 10] range.
Its speed records remain valid, but neither numerical, quality nor vocabulary
training completion is inferred. Select the forcing semantic context using
the actual captured compiler limit: small decode uses it; large prefill keeps
its separate compiler. Preserve the existing speed records and retry only
missing diagnostics. Collect natural quality and independent vocabulary
training before forcing; each has its own completion flag, while overall
candidate admission still requires every requested stage.

### Parallel draft local top1

The remaining full-head selector scans 62,080 logits in one CTA. Split it into
512-column segments and merge the small set of maxima, emitting the existing
FP32 value/global-ID packet directly. Preserve first-index ties, NaN precedence
and vocabulary padding. Keep soft-cap/scaling, shortlists and older extensions
on their existing routes. Target projection precision and head are unchanged.

A source-built V100 real-weight four-head graph (including packet production,
excluding identical TP IPC) screens 0.854364 -> 0.776305 ms at M1;
M4 0.861030 -> 0.784558 ms; M5 0.862136 -> 0.786964 ms. Input scales zero,
0.03, one and three preserve the packets exactly. Finite values, ties, negative
infinity, NaNs and excluded padding pass at M1/M4/M5/M16/M128. The approximately
0.078-ms local gain is not a complete-model speed claim. Prepare it on the
owned default path, with model/C4 qualification required before merging.
