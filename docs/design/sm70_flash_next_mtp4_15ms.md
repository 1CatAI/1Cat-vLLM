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
--cuda-graph-trace=node:host-only --capture-range=cudaProfilerApi`.
Trace admission is always false. Annotations are installed after graph capture;
they preserve FULL graphs and distinguish target M5, draft step 0 at its real
width, draft steps 1--3, target head/sample, preparation and handoff.
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

Source/default repairs and diagnostic tooling are prepared. New main baseline,
node attribution, traffic table, quality/acceptance and C4 admission remain
pending. No new 15-ms or structural-kernel speed claim is made.
