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
