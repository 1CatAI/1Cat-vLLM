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
40525. These sizes are references, not our vocabulary choice. Measure
coverage on our own frozen corpus with an independent held-out split and
Chinese, Japanese, Korean, English and code statistics before selecting IDs.
Keep all required script/byte/special-token coverage. Preserve original token
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
The six focused observer/analysis CPU tests pass; GPU calibration remains pending.

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
checks pass. Expert-plan snapshots now record actual valid groups without
per-round CPU synchronization; measured traffic/floor ranking remains pending
those GPU data rather than assuming all 50 routes read distinct experts.

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
