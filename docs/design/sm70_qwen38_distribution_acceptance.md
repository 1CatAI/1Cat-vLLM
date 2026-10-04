# Flash-Next decode: distribution acceptance and reduction segments

## Precision and acceptance contract

The owner-approved decode contract permits FP32 reassociation, split-K,
cross-operator fusion, persistent scheduling and communication fusion.
Activations and dense weights remain FP16; dot products and recurrent state
accumulate in FP32. Existing NVFP4 expert storage is unchanged. Exact top-k
and every selected expert are retained. Bit identity and identical greedy
continuations are diagnostics, not admission conditions. Incorrect PLE rows,
missing synchronization, corrupted transfers and invalid state updates remain
correctness bugs regardless of distribution scores.

Dense 8-bit experiments are a separate arm. Their speed is never credited to
FP16 fusion results, and default admission requires a separate owner decision.

## Initial distribution thresholds

Use natural-log KL, temperature 1, the complete valid vocabulary, and logits
before sampling processors, top-k, top-p or temperature scaling. For each
fixed teacher-forcing prefix compute `KL(P_default || P_candidate)` in FP64
from stable log-softmax. Report reverse KL as a diagnostic. Never compare
free-running continuations with different prefixes. Ignore padded vocabulary
entries using the tokenizer/model vocabulary contract, not probability cutoff.

Initial admission limits (all must pass, globally and in each language/task
stratum):

| Metric | Initial limit |
| --- | ---: |
| Mean forward KL, nats | <= 0.001 |
| p99 forward KL, nats | <= 0.01 |
| Maximum forward KL, nats | <= 0.05 |
| Top-1 agreement | >= 99% |
| Maximum absolute raw-logit error | <= 0.5 |
| Nonfinite logits | zero |

Also report median/p95/p99 logit error, additive-offset-centered maximum
error, top-1 margin and disagreement counts. A common logit offset has no
probability effect; raw and centered errors must both be visible. The raw
maximum limit is a conservative investigation gate, not a mathematical claim
that this alone bounds KL. A gate failure requires investigation and an explicit
contract revision with evidence; it must not silently trigger relaxed limits.

The mean-KL limit is 22x and 58x below the cross-implementation examples
(0.022 and 0.058) supplied by the owner. Those examples are contextual reference
values; their prompts, vocabulary and KL aggregation have not been independently
matched to this protocol. Pinsker gives mean total variation <= sqrt(0.001/2),
about 2.24%, but this bound is loose and does not guarantee task quality. The
independent top-1, tail and task gates therefore remain necessary. These are
initial engineering limits, not empirically calibrated guarantees. Before first
admission, repeat the default arm three times on identical prefixes to establish
measurement/replay noise. Noise exceeding a limit blocks interpretation rather
than automatically widening that limit.

Freeze prompt token IDs, tokenizer revision, reference continuation IDs,
probe positions and hashes before evaluating a candidate. Start with the
existing 36-case quality manifest (12 MBPP, 12 GSM8K, eight Chinese QA and four
needle contexts), add four English prose prompts, and probe at least 16
teacher-forced decode positions per case. Subsequent gates retain one or two
258K windows rather than repeating the entire long-context set. Preserve the
previously collected windows and their hashes. Save per-prefix summaries; compute whole-vocab
metrics one row at a time so evaluation does not retain all logits in RAM.
Run the distribution gate at C1 only. Concurrent distribution collection is
stopped; previously collected data remains historical evidence.
Task continuations are frozen from the recorded default FP16 quality arm;
the four additional English prose continuations are authored and identified
separately. Both capture arms use the same frozen IDs. Capture
logits without changing model computation or scheduler state. No timing result
from the diagnostic logit capture arm is an accepted performance result.

## Quality and performance gates

Run the fixed GSM8K, Chinese QA, needle and MBPP cases with the existing seeded
sampling recipe. Each stratum must score at least its matched default baseline.
Inspect repetition, invalid text, premature termination and unfinished thinking;
a single output that runs to the token cap or repeats is an anomaly to triage,
not an admission veto. Repeat the affected prompt with three different seeds
in both the candidate and baseline before attributing a regression to the
candidate. Record all six outputs and baseline cap failures separately. Keep counts,
full outputs and reproducible checking code. This small suite is an admission
screen, not a claim about all model capabilities.

The quality runner records `health_passed` and per-case `health_failures`
separately from task scores, and exits unsuccessfully on an unhealthy output.
Its automatic screen flags a missing natural EOS, an empty final answer, a
replacement character, or three occurrences of the same final-answer line
longer than 24 characters. Repeated-line flags require inspection; a clean
screen does not replace manual review for other repetition or invalid text.
The fixed-length timing requests are excluded from this natural-EOS screen.

Use ordinary installed source-complete wheels, the same model revision, GPUs,
TP, graph, KV/state dtype, prompts, sequence lengths, disk placement and
sampling contract in paired arms. Report decode separately from TTFT/prefill.
Use at least five steady-state C1 samples per arm; interleave matched samples
where feasible and report median, range and ratios, not a best run. C1 speed
and its single-step bottleneck determine performance admission. For scheduling
changes that preserve numerical results bit for bit, an operator microbenchmark
and one matched C1 timing comparison suffice; skip distribution, quality and
acceptance-rate tests. Changes to accumulation order or numerical boundaries
require teacher-forcing distribution checks and C1 timing, plus one quality
suite before merge. Do not run dedicated
C2–C16 throughput, distribution or budget campaigns. Kernels must remain correct
at arbitrary supported batch widths, including M=5. Use shape capability and
measurement for admission, never a hardcoded batch-width fallback rule.

The first endpoint target is at most 7.5 ms/token, followed by 6 ms/token,
with TP4, 262144 startup capacity,
8192 input tokens, disk-mapped ngrams, FP16 dense/activations and FP32
accumulation/state. Initial component targets are 95 us per GDN layer,
150 us per QSA layer and 0.45 ms for LM head plus sampling. These are planning
budgets, not measured results.

## Iteration cost and expected benefit

Before implementing a candidate, record calls per token from the existing
trace, expected saving per call and `calls × saving × 0.9` as its projected
endpoint benefit. The 0.9 correction applies to profiler-derived timing;
retain unprofiled measurements as such. Bundle changes projected below
0.1 ms/token into a larger segment change. After endpoint measurement,
compare predicted and observed savings; investigate relative deviations over
15% and correct the estimate before proceeding.

Use three measurement tiers. Develop operator candidates in independent Torch
JIT extensions with ccache. Rotate real layer weights to keep L2 cold, capture
CUDA graphs and report addressed weight bytes, measured DRAM bytes when
available, GB/s, grid and numerical error. Then capture a real layer or segment
to measure dependency gaps and actual fusion savings. Run the endpoint only
when accumulated projected saving reaches 0.5 ms/token or a PR is ready to
merge. Compile changed extensions incrementally during development; build a
normal source-complete wheel for merge, not for every candidate.

Keep the C1 endpoint entry in `benchmark_sm70_qwen38_quality.py`; use
`--timing-only` for scheduling changes. It checks disk space, Python and headers,
the GPU lock, idle GPUs, request metrics and spawn protection before loading.
Every completion writes a compact summary, including failed launches. Preserve
raw evidence and clean only owned obsolete caches/build products. While GPUs
are occupied, develop the next eligible kernel on CPU rather than repeatedly
polling long jobs. Batch independent code/log reads.

## Current budget and required updates

The historic 13.05005 ms C1 control and the research transport below are from
different arms; their difference is not an accepted paired speedup. The weight
traffic floor of about 3.06 ms assumes 2.45 GB/token/card at 800 GB/s. It is a
weight-only estimate, excludes activations/state/transport and does not predict
runtime. Recalculate from actual selected layouts and measured device bandwidth.

| Arm | C1 ms/token | Graph kernels/rank | Communication-related kernels | PLE wait | Gap to weight floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recorded disk FP16 default | 13.05005 | 1349 | about 291 | 2.15–2.20 ms unprofiled | about 9.99 ms |
| HC down scheduling, merged #812 | 12.89271 | 1349 | unchanged | not isolated in this arm | about 9.83 ms |
| Mapped result, fresh-cache research audit | 11.28–12.01 | not accepted yet | unchanged | formal measurement pending | not a paired result |
| Normal CUDA transport, six samples | 13.117008 | pending | pending | pending | about 10.06 ms |
| Normal mapped result, six samples | 11.082492 | pending | pending | pending | about 8.02 ms |
| Sample-triggered PLE prefetch | pending | pending | pending | pending | pending |
| First reduction-segment fusion | pending | pending | pending | pending | pending |

Recorded HC gain is about 1.2% with overlapping sample ranges. Retain the old
concurrent measurements without extending them. Each admitted change updates
C1 endpoint time, actual per-layer graph
counts/times, cross-card synchronization boundaries, PLE residual wait and
traffic floor. Communication kernel count is not the number of global syncs.

### Mapped-transport Step 0 refresh

The fresh installed-artifact run uses `eac8c525d` model source and standard native
targets matching its base, Torch 2.10/cu128, CUDA 12.8, four V100-SXM2-32GBs
with full NVLink connectivity, FP16 KV, FP32 state, disk mmap, no MTP, 94%
memory utilization, 262144 startup capacity and 8192 input / 513 output tokens.
Five controls with NSYS collection disabled measured 11.2201, 11.0289,
11.1497, 11.1635 and 11.1918 ms/token, median 11.1635. They are controls under
the profiler launcher, not separate fully unprofiled endpoint qualification.
Subsequent unrelated main native changes are not covered by this artifact.

The one graph-node trace has eight complete graphs per rank. Retain all raw
data; use graphs 2 through 6 on all four ranks for the budget. The first graph
has staggered profiler-start peer waits; the first two and the last are excluded
explicitly. Every selected graph has 1349 kernels. Source-verified HC markers
partition all nodes into embedding, 48 layers and the final HC mixer.

| Trace view | Measured result |
| --- | ---: |
| Graph kernels per rank/step | 1349 |
| Total kernels including work outside the model graph | median 1396 |
| Communication-related kernels per complete step | median 290 |
| Regular GDN layer span, excluding the PLE-bearing layer | mean 184.34 us |
| Regular GDN layer kernel activity union | mean 160.54 us |
| Regular GDN layer intervals without kernels | mean 23.86 us |
| QSA layer span | mean 303.34 us |
| QSA layer kernel activity union | mean 270.86 us |
| QSA layer intervals without kernels | mean 32.40 us |
| GDN graph kernel counts | 24–27 |
| QSA graph kernel counts | 33 |
| PLE combine-to-dequant bracket | median 988.41 us |
| PLE bracket before H2D starts | median 873.75 us |
| PLE H2D copy itself | median 4.69 us |
| PLE bracket after H2D ends | median 112.00 us |

The PLE bracket contains the host-result wait and graph/copy scheduling; it is
not a pure flag-wait measurement. It does not support parking sampled prefetch
under the 0.2 ms rule. A separate event measurement is still needed before
attributing the whole bracket to CPU lookup. The approximately 12.60 ms traced
step interval is also not the 11.16 ms control TPOT. Do not subtract their
components as if they were one unprofiled wall-clock budget.

The trace has 99 cuBLAS GEMVs, about 0.948 ms summed service: 48 shared gate/up
projections (0.504 ms), 48 shared down projections (0.393 ms), one PLE value
projection (0.024 ms), and two final HC projections (0.028 ms). These match the
loaded ordinary linear methods. Shared and routed expert work overlaps, so
eliminating that service cannot be claimed as the same endpoint saving.

Independent microbenchmarks measured a 4.09–4.13 us NVLink flag round trip
using system release/acquire publication across pairs 0–1, 0–2 and 0–3, with
generation checks across five graph replays. This includes the memory protocol;
it is not bare NVLink wire latency. The dependent-kernel graph boundary median
was 1.024 us for 1-CTA and 80-CTA producer/consumer pairs on all four GPUs.
The observed global-timer quantum is 1.024 us, limiting per-sample precision;
CUDA-event averages independently check the flag round trip.

Representative cold-cache NCU counters on actual layer-0 TP0 expert weights:

| Native expert kernel | DRAM read MB | DRAM write MB | NCU us | Read floor at 750 GB/s, us | Remaining NCU us |
| --- | ---: | ---: | ---: | ---: | ---: |
| W13 + SiLU | 5.137184 | 0.019488 | 15.264 | 6.8496 | 8.4144 |
| W2 + weighted reduce | 2.577024 | 0.000128 | 12.256 | 3.4360 | 8.8200 |

These actual isolated counters establish the expert floor; they do not fill
unmeasured dense/state/communication traffic or represent in-model warm-cache
DRAM bytes. Full per-layer read floors, under-bandwidth time and pure cross-GPU
flag waits remain explicitly unmeasured. Their absence must not be filled with
the planning estimate of 44 MB per layer.

Step 0 uses one new graph-node trace with mapped result transport. For each
layer report measured DRAM-read bytes divided by 750 GB/s, the remaining kernel
duration, gaps on the dependency path, and cross-GPU flag waits. Nsight Systems
does not measure DRAM bytes or separate spinning from useful work: populate those
fields only from Nsight Compute or explicit instrumented measurements. Mark
logical-weight estimates separately and avoid adding overlapping service sums
as if they closed endpoint time. Also measure a cross-GPU flag round trip and
a dependent-kernel boundary inside a CUDA graph. Attribute the old approximately
97 cuBLAS GEMV launches before replacing them.

## Implementation order

Rank interventions only by measured kernel service minus the weight-read floor
at 750 GB/s. Operator A/B rotates all 48 real checkpoint layers or flushes L2
between operations. Hot-L2 results do not admit or reject a kernel; a full-model
A/B is preferred. Keep in_proj, out_proj and LM head outside this campaign's
optimization scope because their current paths approach their weight floors.

1. Finish independent C1 qualification and merge #831 and #859. Shared expert
   and router come first: fold shared gate into router row 513, use at least
   80 router CTAs and parallel top-k, fuse gate/up with SiLU, and use one down
   kernel. Each fusion has its own PR and is merged after its C1 gate rather
   than accumulating research drafts.
2. Decompose PLE publish, lookup, writeback and flag times. Then use a bounded
   pinned hot-row cache accessed through GPU UVA, with CPU-worker misses.
   Keep the full ngram table on disk with mmap.
3. Expand QSA sparse attention to at least 300 CTAs, use multi-CTA top-k and
   fuse merge.
4. Fuse GDN conv, recurrence and gated norm per head. Do not use cooperative
   launch or grid-wide synchronization for this segment.
5. Improve routed-expert W13/W2 tile parallelism and NVFP4 lookup decode;
   target 550 GB/s. Recheck historical negative operators with cold L2.
6. Revisit HC/all-reduce segments after those items; prefetch weights before
   waiting on generation flags. Communication choices admit by topology and
   capability. Dense 8-bit remains a separately evaluated precision decision.

Do not narrow `_is_sm70_qwen38_decode_compile_contract` by TP or exact shape.
Use the existing kernel configuration and startup capability report. New scattered
runtime environment switches are not part of this design.

Mapped transport admission uses the current stream-memory-operation capability.
The deprecated v1 attribute reports zero on the CUDA 12 driver even when the
current API works; it must not disable an otherwise supported path. See the
[NVIDIA stream memory operation contract](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-driver-api/group__CUDA__MEMOP.html).
The delayed producer GPU oracle remains required after capability admission.

## Integration validation log

The standard installed wheel passed the delayed CPU-producer GPU oracle on all
four V100s: 40 changing-width graph replays per dtype, uint8 and FP16, exact
results and acknowledgements. The corrected oracle supplies an explicit capture
stream on each device; its initial default-stream reuse had produced empty
graphs on secondary devices. The corrected rerun passed both dtypes in 12.17 s.
No research DSO or runtime source overlay was used. A first baseline startup failed before profiling because the
whole-table GPU placeholder lacks `_cascade` after its guarded constructor.
The independent fix was merged in PR #818; it does not change model arithmetic.
The paired transport measurements below include this startup fix.

The capture tool includes the first 16 and last 16 reference tokens, so code
answers and completed reasoning are represented rather than only thinking
preambles. Long-context probes use one long request plus short English filler
requests at each concurrent width, avoiding 16 copies of a 258K context.
Greedy parity in the no-MTP concurrency timing driver is diagnostic only;
its completed timing report explicitly leaves quality unaccepted until both
distribution and task gates pass. The MTP timing contract is unchanged.

The current-main baseline then reached compilation but rejected 256K KV
capacity at 90% memory utilization: 3.24 GiB required versus 2.65 GiB available.
Both new paired arms use 94% utilization; historical 90% timings remain
separately labeled. Long-context acceptance is not shortened to bypass the
capacity check. The failed startup log is retained, and no timing was extracted.

Standard-package C1 smoke (same installed source, TP4/no-MTP/FP16 KV, disk mmap,
94% memory budget, 8192 input and 65 output tokens): CUDA transport measured
13.49959/12.73089/12.79927 ms; mapped transport 11.90525/11.41047/11.36301 ms.
The median difference is 10.85% lower TPOT. These three short samples are
preliminary, not the five-sample performance gate. Subsequent task/distribution
results are recorded below; concurrency acceptance is no longer required. All repeats within each arm matched
greedily, supporting diagnosis only. The worker capability RPC now includes the
per-layer transport decision and bounded registered-memory size.

Custom logits processors select the older model runner. Distribution capture
therefore uses an explicit diagnostic worker extension around the current MRv2
sampler. A one-logprob request materializes full-vocabulary logits in place of
the greedy TP-local top-1-only path. Model core/runner remain current; capture
and forced sampling are excluded from speed evidence. Every captured prefix
checks actual GPU input token, position and request mapping; active width is
recorded so partial cohorts cannot masquerade as C4/C8/C16 decode probes.

The normal installed-package quality pair completed all 36/36 tasks in each
arm, all 36 natural stops, zero replacement characters and maximum repeated
long-line count one. Full continuations matched for all 36 pairs (diagnostic).
Six 8192/513-token timing samples: CUDA median 13.117008 ms, range
12.903743–13.175692; mapped median 11.082492 ms, range 10.952298–11.197272.
Observed TPOT decrease is 15.51%, equivalent throughput increase 18.36%.
These are separate-arm samples. C1 teacher-forced distributions have since
passed; fresh installed-runtime integration, graph budget attribution and a
short C4 smoke remain before merge. Dense 8-bit remains disabled.

Five complete engine-interval samples per arm and width have now completed
with atomic cohorts, 8192 input / 513 output tokens and the same installed
source. The first sample of each new width includes initial cache/JIT effects;
it is retained rather than silently dropped. The final four samples agree on
the direction. These concurrent samples are retained as historical data;
there will be no further concurrent timing qualification.

| Width | CUDA median step ms | Mapped median step ms | CUDA aggregate tok/s | Mapped aggregate tok/s |
| ---: | ---: | ---: | ---: | ---: |
| C1 | 13.029 | 11.077 | 76.75 | 90.27 |
| C2 | 15.727 | 14.243 | 127.17 | 140.42 |
| C4 | 17.107 | 15.366 | 233.82 | 260.31 |
| C8 | 21.386 | 19.459 | 374.07 | 411.12 |
| C16 | 31.716 | 31.533 | 504.47 | 507.40 |

C16 is essentially unchanged within noise; no meaningful throughput gain is
claimed there. C2/C4/C8 improve by about 10–11% aggregate throughput. Kernel,
communication and PLE wait attribution still require the new graph trace.
These data remain pinned to the original paired source, not subsequent main
changes. The PR is now rebased onto main and a new normal wheel was built from
matching standard native sources, including the new GGUF target; fresh runtime
validation of that integration artifact remains pending.

The full C1 comparison is now complete: 3648 positions per arm across three
repeats, including Chinese, English, GSM8K, MBPP and needle retrieval. Forward
and reverse KL, raw and centered logit errors are zero; top-1 agreement is 100%
globally and in every stratum. Default-repeat noise is also zero. The data is
pinned to the original paired installed source. C1 subsets were validated and
preserved from the stopped wider collection; the subset retains two 258K
windows per repeat. Concurrent collection and all remaining concurrent work
were stopped when the admission contract changed.

The first teacher-forcing smoke captured 64 English C1 positions through the
current model runner in both installed-package arms. Mean/p99/max KL and
maximum logit error were zero; top-1 agreement was 100%. This checks the capture
plumbing and small-result transport, not the complete multilingual/long-context
distribution gate. Dedicated concurrent timing/capture is no longer required.

The cooperative GDN research segment retained FP16 inputs and FP32 state,
passed the FP64 small-shape oracle, and reduced C1 conv/update/norm/projection
from 26.21 to 24.07 microseconds per layer. C16 regressed from 61.59 to 131.73
microseconds. There were no register spills. Retaining the native batched
projection reduced the C16 result to 66.41 versus 62.06 microseconds in a new
paired screen, but remained slower at every measured width. Both prototypes
remain outside default dispatch. Their negative result is a scheduling and
synchronization issue, not a bit-identity gate failure. The screen covers only
part of a reduction segment and is not an end-to-end decode claim.

## Revised Step 0 priority audit and current qualification

The selected graph retains 1349 nodes/rank. Source and loaded tensor layouts
now attribute prepared parameter bytes to the major kernel families. These
bytes count weights actually addressed by the operator, including the FP16
NVFP4 block scales, rather than the whole checkpoint tensor or padding. They
are not NCU DRAM bytes. Weight floors exclude KV, recurrence state, activations
and transfers; a zero-weight operation still has memory traffic. The detailed
67-family table also records grid and block sizes. Rank-masked embedding gather
aliases still need input-to-node attribution, and full-model DRAM counters
remain pending. Do not relabel this audit as a complete DRAM budget.

| Priority component | Trace service ms/step | Weight floor at 750 GB/s, ms | Service minus weight floor, ms |
|---|---:|---:|---:|
| Shared expert + router + shared gate | 2.3928 | 0.3254 | 2.0674 |
| HC regular layers | 2.1276 | 0.4247 | 1.7029 |
| QSA attention chain, excluding projections | 1.2558 | 0.000025 | 1.2558 |
| PLE combine-to-dequant bracket | 0.9884 | not a weight-read operation | 0.9884 bracket |
| Routed experts | 1.1960 | 0.4915 | 0.7045 |
| GDN core, excluding projections | 0.5500 | 0.0010 | 0.5490 |

The routed-expert floor is higher than the earlier 0.42-ms estimate because
runtime block scales are FP16: selected W13 packed weights plus scales read
5.12 MB/layer, W2 plus scales 2.56 MB/layer, across 48 layers. Isolated cold NCU
reads total 7.714208 MB/layer, implying 0.4937 ms across 48 layers. This supports
the prepared-layout accounting but does not measure whole-model DRAM traffic.

Attribution totals 2.407 GB/card/token inside the model graph, plus 317.850 MB
for the local FP16 LM head, or 2.725 GB/card/token. Its weight-only floor is
3.633 ms at 750 GB/s. This revises the earlier 2.45-GB estimate without treating
those parameter bytes as actual HBM counters. Rank-masked embedding and any
traffic omitted by the weight-only definition remain separately identified.

The five single-CTA families consume 1.518 ms summed service: router top-k
0.3840, shared gate 0.2507, activation 0.2029, HC down gather 0.4843, and QSA
top-k 0.1975 ms. Router GEMV alone averages 9.91 us in this trace; the router
GEMV plus top-k chain averages 17.92 us. Distinguish the chain from a standalone
hot-L2 GEMV timing. Service sums can overlap and are not endpoint savings.

Current qualification uses a full current-main native source build at
6bd9925ff8, packaged with Python head 6bbffc5bf6. All 16 native hashes match
the full source build. The matched normal-wheel C1 arms measured:

| Transport | Median ms/token, six 8192-input/513-output samples | MBPP | GSM8K | Chinese | Needle |
|---|---:|---:|---:|---:|---:|
| CUDA copy | 13.07986 | 12/12 | 12/12 | 8/8 | 4/4 |
| Mapped result | 10.98609 | 12/12 | 12/12 | 8/8 | 4/4 |

Both arms' 36 task outputs reach natural EOS and have nonempty final answers,
zero replacement characters and no token-cap failures. Current-artifact C1
teacher-forcing passes at 768 positions across 16 fixed windows and three
repeats: zero forward/reverse KL, zero raw/centered logit error, 100% top-1
agreement, and zero default-repeat noise. This includes two 258K windows per
repeat. The short C4 smoke passed with four natural-stop, nonempty healthy answers,
including English, Chinese, code and arithmetic. Both C1 gates and the
pre-merge C4 execution gate are complete. The 16.0% TPOT reduction is transport-specific and does not include
a dense 8-bit arm or establish the 7.5-ms target.
