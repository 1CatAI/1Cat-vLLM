# SM70 concurrent long decode follow-up, 2026-09-27

Status: the final ordinary-service candidate improves C4/C8 pure decode by
15.44%/15.78% against the frozen main baseline, meeting this 32K campaign's
15% target. C1/C2 improve by 1.56%/6.90%. Every retained output token ID,
accepted/drafted counter and prefix-cache counter matches the baseline at
all four concurrency levels. Long retrieval and natural stops pass 8/8.

## Reproduction contract

The integration baseline is `onecat/main` at
`1e90d17f2c75e443b2a85a576ed68fa04c5f9dd6` (PR #697). Both service arms use
GPUs 4–7 of the same V100-SXM2-32GB host, TP4, CUDA 12.8, Torch 2.10.0+cu128,
Python 3.12, Qwen3.8-27B-NVFP4 with FP16 execution, E4M3 KV, Flash-V100 and
DFlash2 q7 with probabilistic draft sampling. Prefix caching is enabled;
maximum context is 262144, memory utilization 0.8, maximum sequences 16 and
the scheduled token budget 8192. The resulting attention page size is 3296.

The fixed synthetic performance fixture contains 32768 input tokens per
request and requests exactly 256 output tokens. Its SHA256 is
`348dd1b93ba266c1a9a18949b6c41205cb130d1c95ead57597e3d5d643b18ac9`.
Temperature/top-p/top-k are 0.7/0.8/20; request seeds are 20260923 plus the
request index. Fixed length and ignored EOS apply only to this speed fixture.
Natural-EOS retrieval is a separate quality check.

Each freshly started service clears inherited acceleration overrides. A shared
prefix is primed once. The engine is paused while requests are admitted in a
fixed order, then resumed with all C requests present. No replacement request
arrives during a wave. One warm wave is excluded and three repetitions are
retained at C1/C2/C4/C8. All raw waves, including cold/JIT-affected ones, remain
in the artifacts.

Pure decode counts actual returned token IDs between the latest first token
and earliest final token of the wave. It includes client/host overhead and
q1–q8 steps while all C requests are alive. It is not an output-throughput
number including TTFT, a rolling workload, or an event-timed full-q8 step.

## Confirmed initialization defect and repair

The coordinated TurboMind warmup skipped every QPN8 layer on the assumption
that it used only static QPN8 dispatch. Prepared secondary compressed layouts
actually switch to TurboMind above M32. Importing the rank-0 cache froze
tuning before those layouts were measured, leaving their captured batch
execution on the slower heuristic plan.

Warmup now includes the prepared secondary layout in its actual M33–64 range.
It uses the layout's dimensions, strides, activation fusion and scale format.
A separate deduplication set prevents batch-only layouts from suppressing an
ordinary layer's small-M warmup. The serving logs change from one FP8 shape,
nine calls and 14 coordinated cache records to six shapes, 19 calls and 24
records on every rank. No new persistent weight allocation is required.

Newly warmed layouts retain the previous deterministic split-K partition.
An internal dispatch policy restricts supply tuning to compatible partitions;
it does not redefine already tuned ordinary FP8 LM-head plans. This is a
capability/layout decision, with no model-name admission or extra user switch.
The warmup repair leaves the M<=32 QPN dispatch unchanged. A subsequent native
candidate optimizes full M32 FP4 tiles, as described below.

The existing sampler's exact tied-cutoff fallback now materializes row indices
once and reuses them for the logits, k/p parameters and final scatter. This
preserves vocabulary tie order and sampling semantics while removing repeated
boolean-index synchronization.

## Validation and first service result

The five representative real TP-local FP8 M64 projections retain bitwise
equality to fresh heuristic oracles at three changed-input amplitudes, in
both eager execution and CUDA Graph replay. Their layer-weighted graph cost
changes from 12.568 to 7.609 ms. This is a GEMM microbenchmark, not target
forward latency. Two experimental K16 tile registrations were not selected
by the service's plans and have been removed.

The sampler M64 micro changes from 3.194 to 3.051 ms with bitwise-equal output.
Always using a captured full-vocabulary reference costs 6.871 ms and is rejected.
An earlier external-script run imported another editable checkout; that failed
comparison is excluded. Reproduction scripts must set the owned source root
explicitly in `PYTHONPATH`.

Validation so far: 25 warmup CPU tests, 73 GPU batch/tail/replay/sampler tests,
and targeted Python lint/format checks pass. Three-wave ordinary-service
medians are:

| Concurrency | Main pure decode, tok/s | Candidate, tok/s | Change | Acceptance, both arms |
| --- | ---: | ---: | ---: | ---: |
| C1 | 159.133 | 159.907 | +0.49% | 33.395% |
| C2 | 236.175 | 239.275 | +1.31% | 31.305% |
| C4 | 377.689 | 390.587 | +3.42% | 37.093% |
| C8 | 612.982 | 667.487 | +8.89% | 51.029% |

Every compared output token ID and accepted/drafted counter is identical.
Both services pass 8/8 long-document retrieval cases with 8/8 natural stops.
The +15% thresholds are 434.343 tok/s for C4 and 704.928996 tok/s for C8.
Neither is met. The older main result from variable admission order is
retained but is not substituted for this matched ordered baseline.

The candidate normal extension SHA256 is
`f826092acf6c01afbc502471ecbbfa53936f1e536bfe9349bedac35d47dd4f0e`.
Flash-V100 is unchanged, SHA256
`50f844e678ad348427f21b51aa9650fb9253a915e0e3cbb18b13c4aaca3ab079`.
The services load normal package artifacts, without a task-only library or
profiling worker extension. No fresh PRO 6000 comparison, 2K rolling speed
acceptance, or 35B-A3B AWQ/FP8 speed acceptance is claimed by this result.

## Rejected screens and next bottleneck

A main-service diagnostic records median full-C8/q8 target forward 51.59 ms,
sampling 5.48 ms, draft 6.44 ms and complete GPU interval 64.56 ms. A target
graph contains 12.67 ms FP8 GEMM, 9.76 ms FP4 GEMM, 13.69 ms long attention,
1.22 ms attention merge, 5.42 ms TP work and 3.88 ms GDN. Profiling is used
for attribution only; its timings do not replace ordinary-service results.

- Smaller two-head attention CTAs preserve bits but regress C2/C4 attention
  from about 3.96/7.48 to 5.14/9.86 ms. Reject.
- Separate QK/PV production preserves bits but regresses C2/C4/C8 from
  3.95/7.47/14.67 to 4.63/8.95/17.85 ms. Reject.
- Global split-K QPN8 retains the original partition but is slower on the
  first three projections; gated output also fails bitwise equality. Reject.
- A direct all-reduce retaining the original two-stage rank order passes
  four-rank changed-input graph checks, but costs 43.93 rather than 30.64 us
  per 640-KiB collective. Changing the existing two-stage grid to 20 CTAs
  saves only about 0.59 us, or 0.08 ms across 129 target reductions. Neither
  warrants a service experiment for this target.

Raw evidence, failed builds/prototypes, source snapshots, service logs, token
events and trace files are retained under the artifact key
`sm70-long-decode15-20260927`. The local `CURRENT.md` records exact paths and
process ownership. Build products, weights and raw traces are excluded from Git.

## Native supply follow-up

The second ordinary-service candidate adds three changes without additional
persistent weight storage or user environment switches:

- Full-q8 attention prefetches the next compressed K panel into registers
  while the current P @ V executes. Shared-memory publication, FP32 partial
  state, probability compensation and split boundaries remain unchanged.
- Attention merge reuses each split's computed weight across the 256 output
  channels and launches 256 threads. Denominator and numerator reduction
  order remain unchanged.
- A complete M32 FP4 tile removes row masking and prefetches the next weight
  code and scale. A register bound preserves capacity for two resident CTAs;
  partial-row batches retain their established path.

The general SM70 sampler also tries a bounded top-k shortlist for large
vocabularies. It restores the reference's vocabulary tie order and retains
full-reference fallback for truncated ties and nearby cumulative-probability
boundaries. This does not change request RNG streams or the sampling settings.

The rebuilt native attention operator matches the previous native artifact
bit-for-bit for numerator, max/sum, output and changed-input graph replay.
For 16 distinct 32K layers, C4 costs 7.549 -> 7.130 ms and C8 costs
14.786 -> 13.878 ms. Separate C1/C2 micro timing contains warm-clock
transients; no isolated speed claim is based on those samples. Ninety GPU
tests pass, covering long attention through 262K, M9–32 FP4 exact reduction,
changed-input graph lifetime and tied-cutoff sampling.

Using the same ordinary-service protocol and frozen baseline:

| Concurrency | Main pure decode, tok/s | Native supply candidate, tok/s | Change |
| --- | ---: | ---: | ---: |
| C1 | 159.133 | 161.571 | +1.53% |
| C2 | 236.175 | 243.559 | +3.13% |
| C4 | 377.689 | 404.612 | +7.13% |
| C8 | 612.982 | 694.945 | +13.37% |

All 256 output IDs per request and accepted/drafted counters match every
corresponding retained baseline repetition. Long retrieval and natural stops
both pass 8/8. The C4 and C8 +15% targets still fail. These measurements use
normal package artifacts, without a profiler or diagnostic worker extension.
The service is stopped after the run.

Native artifact SHA256:

- Core: `fe367ba99191ff7f838bd007b0c40b39cde4d01c866ec7e2c1bc4227f3c0d614`.
- FA2: `7962e11a7af0c0f88107459df7292c9f8c4a552c7be58efc7b1a0e32ae3ff2f8`.

Hardware counters do not support calling the attention path HBM-saturated:
the profiled C4 kernel reaches 17.6% DRAM throughput and about 25% occupancy.
The original M32 FP4 gate/up reaches 32.6% DRAM throughput and 21.5%
occupancy; long scoreboard stalls account for 45.8% of its issue cycles.
These counters localize stalls and do not substitute for service performance.

Further rejected screens retain raw evidence: a peer-output push all-reduce
is bitwise exact but slower than the established two-stage reduction;
reusing M48 TurboMind plans at M32 slows the tested ungated projections;
GDN warp/tile changes are slower and change reduction results. Omitting
intermediate GDN state writes is diagnostic only and violates verification
state semantics, so it is never enabled for serving. Replicated attention
conversion tables and arithmetic conversion are exact but slower than the
current table at both C4 and C8, and are also excluded.

## Full-M32 FP8 and wider sampling support

The next native candidate applies the same two-phase weight prefetch principle
to complete M32 QPN8 batches. It preserves scale decoding and the existing
accumulator partition/order. Packed activation preparation also serves the
short-K split-12 route. M1–16 and incomplete M32 batches retain their original
implementations. Five real TP-local projections at three changed-input
amplitudes are bitwise identical in eager and graph execution; their weighted
GEMM time decreases from 5.644 to 4.755 ms. Forty-two native FP8 tests pass.

Concurrent large-vocabulary sparse rejection retains up to 64 candidates and
all ties at the actual top-k cutoff. It falls back for truncated ties,
nonfinite values, probability underflow and ambiguous cumulative boundaries.
The original single-request route is preserved. A shared compact sort orders
both sparse and general sampler shortlists by score and vocabulary ID in one
kernel, retaining original value bits. It reproduces the reference radix
ordering, including signed zeros and NaNs. No RNG stream or sampling parameter
changes. The sampling checks pass 111 tests, and the unchanged general sparse
rejection interface passes 48 further tests.

Ordinary-service results, using the same three retained waves:

| Concurrency | Main pure decode, tok/s | Candidate, tok/s | Change |
| --- | ---: | ---: | ---: |
| C1 | 159.133 | 163.147 | +2.52% |
| C2 | 236.175 | 252.300 | +6.83% |
| C4 | 377.689 | 435.870 | +15.40% |
| C8 | 612.982 | 703.119 | +14.70% |

All retained output IDs, accepted/drafted counters and prefix-cache counters
match the frozen baseline. Quality and natural stops pass 8/8. C8 remains
below its 704.928996 tok/s threshold; the result is not rounded into a pass.
Core SHA256 is
`6e4d51ab447a8ca9662450134c74b1cfd1822bc8271f96053b407078dd220a48`;
FA2 is unchanged from the native supply candidate. The uninstrumented service
is stopped after evaluation. Raw results: `wide-native-service-comparison.json`.

## GPU boundary guard and medium-message reduction

The wider sparse sampler now computes its conservative admission guard on the
GPU and copies only row decisions to the CPU. It consumes the actual GPU
request-slot map and sampling parameters. The CDF uncertainty margin is doubled
to cover reduction rounding relative to the retained CPU oracle. C8 guard wall
time falls from 0.138 to 0.040 ms in a microbenchmark; 54 request-fallback and
adversarial boundary tests pass. The first native test attempt caught a Triton
constexpr argument omission; it was corrected before the successful full rerun.

FP16 TP4 reductions on fully connected SM70 devices use 256 threads and 20 CTAs
for 512–768 KiB messages. The two-stage partition, rank accumulation order and
visibility barriers are unchanged. Explicit dispatch overrides retain their
previous behavior. Four-rank changed-input graph tests remain bitwise identical.
The representative 640-KiB operation decreases from 30.629 to 29.688 us;
512/768-KiB cases also improve. The 1-MiB case is slightly slower and excluded.
Wider 32-byte packets are slower and rejected. Thirty dispatch tests pass.

The next uninstrumented service, still with no acceleration overrides, records:

| Concurrency | Main pure decode, tok/s | Candidate, tok/s | Change | Acceptance, main → candidate |
| --- | ---: | ---: | ---: | ---: |
| C1 | 159.133 | 163.037 | +2.45% | 33.395% → 33.395% |
| C2 | 236.175 | 234.684 | -0.63% | 31.305% → 22.104% |
| C4 | 377.689 | 437.412 | +15.81% | 37.093% → 37.093% |
| C8 | 612.982 | 717.599 | +17.07% | 51.029% → 51.029% |

C1/C4/C8 retain every output ID, and long retrieval/natural stops pass 8/8.
C2 fails the acceptance guard. Its output matches six earlier runs, including
old-sampler controls, exactly: request 0 is unchanged; request 1 first diverges
from the frozen baseline at output index 151. A subsequent same-process audit
compares the original CPU guard and new GPU guard: four C2 waves return the
same output IDs and 311/1407 accepted/drafted tokens. Across 312 audited steps
and 4992 rows, masks and GPU/CPU sampling parameters agree exactly. This rules
out the new guard as the cause of this observed divergence. The subsequent
single-plan isolation below identifies the older instability.

Core SHA256 is
`5d9f324b3317347269ab4d20661dad30d4726c682df3d0ca37e9cd32f90ff49e`;
FA2 is unchanged. Model allocation is 9.57 GiB and the graph pool is 0.99 GiB.
The ordinary candidate's pre-capture KV budget is 11.45 GiB versus 11.00 GiB
in the frozen control; this difference has not been isolated. Both use the
same 0.8 memory setting and have ample capacity for this 32K workload. Do not
claim identical total KV capacity or C8 at maximum context from this run.

## C2 draft projection stability

Single-record cache replacements in the same diagnostic process isolate the
cause. Changing M8 or M16 vocabulary-projection plans does not change any of
the 256 output IDs. Changing only the M16 FP16 draft context projection
(`16 x 1280 x 25600` per TP rank) toggles the two previously observed paths:

| Context projection plan | Accepted / drafted tokens | Acceptance | Token path |
| --- | ---: | ---: | --- |
| CTA 16x128x32, split-K 12 | 355 / 1134 | 31.305% | Frozen baseline, every ID |
| CTA 8x256x64, split-K 8 | 311 / 1407 | 22.104% | Earlier divergent run, every ID |
| Restore CTA 16x128x32, split-K 12 | 355 / 1134 | 31.305% | Frozen baseline restored |

Each arm retains four waves. The model graphs, sampling parameters, request
order and other plans remain in the same process. Even the established M8
split-10 tree produces the divergent path when extended literally to M16;
single-request arithmetic is not a substitute for the concurrent reference.
These diagnostic timings are excluded from performance qualification.

The existing stable-context selector now covers its full default FP16 tuning
range, M1–16. It preserves the existing M1–8 tree and pins the qualified
M9–16 partition before any autotuned/imported cache lookup. Larger batches
retain their established deterministic heuristic dispatch. This prevents
startup timer noise on TP ranks from selecting a numerically different draft
projection; it adds no user switch or persistent weight copy and is independent
of the target model's weight quantization.

Twelve native GPU cases pass, including M9/12/15/16 tails, incompatible cached
plans, eager output and changed-input CUDA Graph replay. Raw causal evidence
is in `plan-audit-comparison.json` under the task artifact key. The diagnostic
service has been stopped.

## Final ordinary-service acceptance

A fresh normal service uses the fixed projection, stock acceleration defaults,
no worker extension, no imported task LUT and no profiler. One warm wave is
excluded and three retained waves give:

| Concurrency | Frozen main, tok/s | Final, tok/s | Change | Acceptance, both arms |
| --- | ---: | ---: | ---: | ---: |
| C1 | 159.133 | 161.621 | +1.56% | 33.395% |
| C2 | 236.175 | 252.465 | +6.90% | 31.305% |
| C4 | 377.689 | 436.004 | +15.44% | 37.093% |
| C8 | 612.982 | 709.714 | +15.78% | 51.029% |

All complete 256-token arrays and all accepted/drafted/prefix counters match
their corresponding frozen-baseline repetitions. Retrieval and natural stops
both pass 8/8. The C2 divergence is resolved by the default native selector;
no user tuning flags are needed. C2 itself does not meet a 15% speedup target.
Results are in `stable-fc-g47-comparison.json`; all four raw waves remain saved.
The benchmark service is stopped and the public API remains off.

The measured core artifact SHA256 is
`f47d29ea250cdb1ffbda0644fb54b3e1b2dc9cef670e276820f8827f162850d1`;
FA2 remains
`7962e11a7af0c0f88107459df7292c9f8c4a552c7be58efc7b1a0e32ae3ff2f8`.
The final formatting-only core rebuild is
`49b92da93e596ef8e3c2ec4b07907c5e2663c131413ce7f3dafdc8e4f68bb7b4`.
Both artifacts have byte-identical SASS dumps across all 4070 GPU functions
(dump SHA256 `a6a25ccee13c52a7651303d029fc8a954a249c3c9a7f126c38427b17ddec595f`).
Fourteen final native stability checks pass, covering the context projection
and FP4/FP8 captured batch partitions. These are normal package artifacts,
without RPATH/RUNPATH entries or a private library dependency.

This closes the C4/C8 32K no-new-prefill target. It does not claim a 15%
improvement at every concurrency, a new 2K rolling result, a fresh PRO 6000
comparison, or new 35B-A3B AWQ/FP8 service-speed acceptance. The reported
8-case retrieval gate is also distinct from the older 2K rolling campaign's
quality/acceptance limitations.

## Public-data merge review

The merge review adds 25 warmup checks and 226 GPU/operator/sampler/graph
checks on the final package artifacts. All pass. The non-16-byte-aligned KV
stride regression now covers full q8 as well as a five-row tail. No new
native binary is required for this test expansion.

Fresh ordinary main and candidate services use the same 56 public-data
examples, prompts, seeds and sampling: temperature 0.7, top-p 0.8, top-k 20,
presence penalty 0, repetition penalty 1, thinking disabled, natural EOS.
GSM8K and HumanEval each exercise C1/C2/C4/C8; the three long suites use C8.
Untruncated long inputs span 4,133 to 31,930 tokens. These are fixed review
subsets, not full benchmark leaderboard scores.

| Dataset | Examples | Main score | Candidate score |
| --- | ---: | ---: | ---: |
| GSM8K numeric answer | 16 | 15/16 | 16/16 |
| HumanEval test assertions | 16 | 16/16 | 16/16 |
| HotpotQA answer F1 | 8 | 53.750% | 53.750% |
| MultiFieldQA Chinese answer F1 | 8 | 65.530% | 65.530% |
| NarrativeQA answer F1 | 8 | 24.643% | 24.643% |

All requests stop naturally; 54/56 complete token arrays match exactly.
Every code and long-context output matches. However, the GSM8K C2 cohort
fails the two-percentage-point acceptance guard: 70.417% to 53.940%.
One request first differs at output index 155, after 34 identical emitted
chunks. Its 400-token wrong answer becomes a 1,626-token correct answer.
Its paired request remains token- and chunk-identical. The aggregate
accepted/drafted ratio changes from 5,628/7,308 to 6,537/9,233. This is an
unresolved trajectory/acceptance finding, not a demonstrated accuracy loss;
the better answer does not by itself clear the merge gate.

A supplemental 16-example check uses the model card's non-thinking
presence penalty 1.5, exercising the dense sampling fallback. GSM8K and
HumanEval are both 4/4 on both services, and eight NarrativeQA F1 scores
match. All requests stop naturally, 15/16 token arrays match, and acceptance
changes from 1,439/1,862 to 1,445/1,855 (+0.615 percentage points). This
supplement passes but does not replace the failed primary C2 gate.

Raw results and exact fixture hashes are recorded under artifact key
`sm70-long-decode15-20260927`, in `quality-706-comparison.json` and
`quality-706-comparison-official.json`. The primary fixture SHA256 is
`21b4cfa0de925a55cbb739d42b1344d20fe6370ed45154f387d34f21e9f4ca54`;
the supplemental fixture SHA256 is
`0e4dd476651529988cada66e45526239c0fbe2f47b4c2ffde6e1e6a48ff05e16`.
The initial review kept PR #706 in draft while this finding was investigated.
After reviewing these results on 2026-09-27, the project owner explicitly
approved integration with the C2 acceptance finding retained as follow-up.
This approval does not reclassify the failed acceptance check as a pass or
claim that its cause has been resolved.
