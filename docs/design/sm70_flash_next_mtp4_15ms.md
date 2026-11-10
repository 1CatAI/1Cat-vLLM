# Flash-Next MTP4 default-path qualification

## Default behavior

Flash-Next MTP4 uses the complete 248,320-token vocabulary. On qualified TP4
SM70 hardware, the draft head uses channel-QPN8 storage with FP16 compute;
the shared target head and target output projections retain their original
parameters and methods. Prepare the draft pack after target-head sharing and
before KV allocation or CUDA graph capture. M1--M8 draft logits and compact
top1 use the same head view. Unsupported shapes/devices retain the checkpoint
head. Batch-invariant execution excludes these numerical fast paths.

The admitted path also enables eight-warp HC when its TP4 fused transport is
available, FP32 shared-expert partials when the worker disables reduced FP16
reduction, parallel local head selection, compact value/ID IPC, and one draft
FC feature gather. The FC change retains the original local FP16 projections
and residual rounding. Communicator topology/capture guards retain the normal
NCCL fallback. No additional performance environment variables are required.

PLE stays disk-mapped. Reuse the bounded mapped-result transport from #831
and the CPU row reader from #885 for MTP. Other speculative methods retain
their independent transport guard. Do not make the resident PLE table the
default. Failed CUDA graph capture must propagate its original error instead
of entering graph-buffer registration collectives during unwinding.

The ambiguous top-k/top-p reference fallback bounds sorting scratch through
row-wise sorting and two-row chunks. Preserve the original batched softmax
and cumulative-sum schedule, including a padded odd final chunk. Ordinary
sampling and captured reference paths keep their existing semantics.

## Measurement contract

The optimization target remains **15 ms per complete round**. Include draft,
target verification, target head/sampling, handoff and preparation. Divide
endpoint steady decode elapsed time by completed speculative rounds; do not
substitute target forward time or divide by emitted tokens. Report emitted
tokens/s, accepted/proposed drafts and tokens/round separately.

Use exactly 8192 input tokens, a 262144-token startup limit and fixed MTP4.
The qualified hardware is four V100-SXM2-32GB cards with full NV2 connectivity,
300-W limits, TP4, Python 3.12.14, Torch 2.10.0+cu128, CUDA 12.8 and driver
580.173.02. Compute/KV are FP16, recurrent state is FP32, target experts are
NVFP4 and draft experts remain FP16. PLE is disk mmap, one request uses 4 GiB
of KV per rank, and speed runs have no profiler or supplied VLLM performance
variables. No wheel build is needed during development.

Use the shared numerical document and tools in
[SM70 distribution acceptance](sm70_qwen38_distribution_acceptance.md).
Target and draft are evaluated separately on aligned teacher-forced prefixes:
mean KL <=0.001, p99 KL <=0.01, maximum KL <=0.05 and top1 agreement >=99%.
Nonfinite logits fail. Maximum raw/centered logit errors are recorded only.
`qwen38_distribution_probe.py` owns the limits; the MTP alignment tool does
not maintain a second threshold set. Also inspect task correctness, natural
completion, truncation/noncompletion and acceptance. Shortlist admission allows
at most one percentage point of acceptance loss on the same eight-prompt,
600-output-token cohort. C4 concurrency remains a separate qualification.

## Recorded model results

These are measurements of the recorded sources, not a new performance claim
for the subsequently synchronized main branch. Integration included main
`6ae4c0c1c9`; the final default speed source was `2b4f74d82f`. Later row-wise
sampler and metadata fixes were separately checked at `b7b2cb3540`.

| Quantity | Recorded result |
| --- | ---: |
| Frozen main complete-round baseline | 23.215921 ms |
| Default fixed8k arithmetic mean | 21.198709 ms |
| Repetitions / completed rounds | 3 / 963 |
| Per-repetition complete-round means | 21.237521 / 21.225780 / 21.132825 ms |
| Fixed-fixture acceptance / tokens per round | 15.031% / 1.5950 |
| Fixed-fixture decode speed | 75.241 tokens/s |
| Eight-prompt natural-cohort acceptance | 50.2035% (3,207 / 6,388) |
| Natural-cohort tokens per round / decode speed | 3.000626 / 117.2215 tokens/s |

The historical fixed fixture deliberately ignores EOS for timing. It is not
a quality or natural-acceptance result. Natural cohort round means are
25.5979 ms, a separate workload despite the same 8K input length.

Sixteen teacher-forcing tapes compare the final default with its mathematically
unchanged full-head control: target and draft KL are zero, top1 agreement is
100%, and raw logit differences are zero. Sixteen natural texts match the
reviewed default exactly; twelve automatic tasks pass and all sixteen outputs
stop naturally. Four Chinese cases require manual review; two partial answers
already exist in the reference and are not new regressions. Do not report a
16/16 quality score.

The initially admitted draft-only QPN8 head has mean KL 0.000465567, p99
0.006037387, maximum 0.020955119 and top1 agreement 99.724%. Raw maximum
logit difference 0.710938 is diagnostic. The FP32 shared branch and eight-warp
HC preserve their independently recorded numerical admission.

Matched C4 startup completes graph capture and sampler warmup, but the actual
8K prefill fails a 160-MiB activation allocation with approximately 63--69 MiB
free. **No C4 throughput/non-regression result is available.** Do not reduce
input length, KV budget or startup capacity to relabel this failure as a pass.
The 15-ms speed target is also not achieved. This integration carries the
qualified default changes and the explicit remaining validation limits.

## Reproduce and inspect

Run from the editable source tree after building its normal SM70 extension:

```bash
.venv/bin/python -m benchmarks.benchmark_sm70_mtp4_round \
  --model /path/to/flash-next --fixture fixed8k --repeats 3 \
  --out /path/to/rounds.json
```

The report records source, exact prompts/token tapes, sampling, resolved engine
settings, endpoint timestamps, completed rounds and acceptance counters.
Use `--fixture-manifest` for the frozen eight-prompt 600-token cohort.
Use `--quality-manifest` for separately scored tasks and
`--teacher-forcing-manifest` for fixed continuations. Forcing preserves the
serving decode context on small draft shapes and restores all observers even
when a dump fails. Forcing cannot qualify natural outputs or speed.

For diagnostics, use `--phase-events` or `--node-trace` on the same fixture.
The latter requires Nsight Systems CUDA graph node tracing. Install annotations
after graph capture, reset prefix cache identically, and pair diagnostics with
an ordinary control in the same loaded engine. Keep profiler overhead visible;
never rescale service sums into the unprofiled round mean. The analyzer splits
target M5, four draft steps, head/sample, preparation and handoff. Kernel tables
must distinguish service, critical-rank wall, actual issued bytes and a
weight-only floor at 750 GB/s. Resident checkpoint bytes are not DRAM traffic.

Thirty local head selection cases cover finite logits, ties, NaNs and padding.
Compact TP4 transport covers changing values, row widths and repeated graph
signaling epochs. Thirty-three sampler boundary cases preserve reference masks;
the forced twenty-row fallback peak drops from 229.403 to 50.354 MiB.
These operator checks are not complete-round or C4 endpoint results.

## Research archive and closed directions

Unadmitted kernels, shortlist controls and quantized-expert/target-head probes
are preserved at the immutable
[research snapshot](https://github.com/1CatAI/1Cat-vLLM/tree/fd65b839251de53aef94105b51aff082307a3602),
including the complete trace history and benchmark recipes. They are excluded
from this default-path integration, including their CUDA build registrations.
Queued research jobs use frozen bundles from that snapshot and do not mutate
the integrated checkout. New candidates require independent model gates before
promotion; operator timings below are not default speed.

| Candidate | Evidence / decision |
| --- | --- |
| Corpus 32K / 64K shortlist | Acceptance 36.1309% / 47.3177%; rejected |
| Target-output 32K / 64K shortlist | Acceptance 41.2438% / 44.0339%; rejected |
| CJK-extended 107,021-ID shortlist | Acceptance 45.5320%; rejected |
| Target output projections QPN8 | Distribution rejection; not enabled |
| Grouped QPN8 draft experts | Maximum KL 0.091476 and acceptance 46.2315%; rejected |
| Row-scaled INT8 draft experts | Maximum KL 0.062439 >0.05; rejected despite acceptance 49.3344% |
| Block32 INT8 draft experts | Four-step operator 0.436398 ->0.188733 ms; model gates pending |
| Single packed INT8 draft store | M4/M20 layer 1.127332/1.418424 ->0.247624/0.457482 ms; saves 562.5 MiB/rank; M2048 prefill slows 2.571028 ms; model/C4 gates pending |
| Target LM-head QPN8 | M5 projection 0.481403 ->0.192010 ms; model gates pending |
| Compact greedy MTP target verifier | Sampling-only five rows 0.029716 ->0.016824 ms; full-model admission pending |
| Cooperative shared-expert fusion | Operator gain but target/draft distribution failure; rejected |
| Token-CTA router | 48-layer chain 0.631368 ->3.190070 ms; rejected |
| Direct/cooperative row-major router readers | Slower despite exact output; rejected |
| Row-major INT8 draft reader | Four-step FP16/candidate 0.408668/0.463677 ms; rejected |

Keep full-vocabulary drafting; stop expanding or rebalancing the rejected
shortlists. Do not repeat the closed 1.35-ms rebase regression investigation,
HC local split-K, resident MoE, register prefetch, single cooperative HC,
BV16/four-warp GDN or the two rejected FP16 draft fusion variants. The ordinary
decode line owns further HC/QSA/GDN core work; verify M5 applicability when
that line qualifies a change.

Strata is a read-only reference, MIT-licensed snapshot
`6f32ec070f23ced9f50e704d854d775da52591ab`. Its draft vocabulary and HC designs
informed research choices; no CPU expert/cache scheme is enabled for TP4.
Preserve source/license attribution if adapting its code in future work.
