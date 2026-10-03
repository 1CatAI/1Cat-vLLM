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
teacher-forced decode positions per case. Include needle contexts at 8192,
32768, 131072 and 258048 tokens. Save per-prefix summaries; compute whole-vocab
metrics one row at a time so evaluation does not retain all logits in RAM.
Run C1 and C4/C8/C16 using the same prefixes and exact active batch ordering.
Task continuations are frozen from the recorded default FP16 quality arm;
the four additional English prose continuations are authored and identified
separately. Both capture arms use the same frozen IDs. Capture
logits without changing model computation or scheduler state. No timing result
from the diagnostic logit capture arm is an accepted performance result.

## Quality and performance gates

Run the fixed GSM8K, Chinese QA, needle and MBPP cases with the existing seeded
sampling recipe. Each stratum must score at least its matched default baseline.
Inspect repetition, invalid text, premature termination and unfinished thinking;
new candidate outputs that run to the token cap fail admission even if the
answer appears earlier. Record baseline cap failures separately. Keep counts,
full outputs and reproducible checking code. This small suite is an admission
screen, not a claim about all model capabilities.

Use ordinary installed source-complete wheels, the same model revision, GPUs,
TP, graph, KV/state dtype, prompts, sequence lengths, disk placement and
sampling contract in paired arms. Report decode separately from TTFT/prefill.
At least five interleaved steady-state samples per arm and concurrency point;
report median, range and paired ratios, not a best run. A C1 gain cannot excuse
a repeatable C4/C8/C16 regression above 2%. Report end-to-end throughput and
per-step latency because batch throughput is not reciprocal single-token TPOT.

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

Recorded HC gain is about 1.2% with overlapping sample ranges. C4/C8/C16 budgets
are pending matched measurements; no extrapolated values are substituted.
Each admitted change must update C1/C4/C8/C16 rows, actual per-layer graph
counts/times, cross-card synchronization boundaries, PLE residual wait and
traffic floor. Communication kernel count is not the number of global syncs.

## Implementation order

1. Complete mapped small-result transport using consumer graph H2D and host
   release/acquire flags. Keep exact row/byte/timing tests. The prior mismatch
   came from functionalization cloning the output before the wait; the fixed
   fresh-cache audit checked all 65 prefill/decode copies per request, zero bad
   bytes, and three greedy continuations identical to the baseline. Equality
   is supporting diagnosis, not the new quality contract.
2. Publish sampled token IDs and a sequence/version doorbell directly into
   mapped staging memory. CPU lookup starts after sampling; only the layer-2
   consumer waits. Handle request identity, accepted-token count, ngram history,
   prefill, batch reordering and cancellation explicitly. Keep the ngram table
   on disk with mmap; registered memory is bounded result/control staging only.
3. Prototype GDN or MoE from one TP reduction boundary to the next. HC combine,
   grouped norm and down/inject belong to the previous reduction tail; up and
   gate-mix belong to the next projection head. Target three to four kernels
   and at most two cross-card synchronization boundaries per decoder layer.
   Measure before extending the primitive to more layers. Kernel primitives
   admit by dtype/layout/hardware capability, never TP4 or model size alone.
4. Fuse push reduction with projection tails and reduce+RMSNorm with consumers;
   admit partial-NVLink topology paths by measured topology/capability and fix
   C16 scheduling before default selection. QSA follows the same boundary rule.
5. Investigate dense-8-bit scale/outlier/sensitive-tensor causes independently.

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
results and acknowledgements. No research DSO or runtime source overlay was
used. A first current-main baseline startup failed before profiling because the
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
preliminary, not the five-sample paired performance gate; full task/distribution
and concurrency acceptance remain pending. All repeats within each arm matched
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
These are separate-arm samples; interleaved and concurrent timings, teacher-
forced distributions and graph budget attribution are still required before
claiming complete admission. Dense 8-bit remains disabled.

Five complete engine-interval samples per arm and width have now completed
with atomic cohorts, 8192 input / 513 output tokens and the same installed
source. The first sample of each new width includes initial cache/JIT effects;
it is retained rather than silently dropped. The final four samples agree on
the direction, but five fully warmed and interleaved samples are still pending.

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

The first teacher-forcing smoke captured 64 English C1 positions through the
current model runner in both installed-package arms. Mean/p99/max KL and
maximum logit error were zero; top-1 agreement was 100%. This checks the capture
plumbing and small-result transport, not the complete multilingual/long-context
distribution gate. Full captures and C1/C2/C4/C8/C16 timing are in progress.

The cooperative GDN research segment retained FP16 inputs and FP32 state,
passed the FP64 small-shape oracle, and reduced C1 conv/update/norm/projection
from 26.21 to 24.07 microseconds per layer. C16 regressed from 61.59 to 131.73
microseconds. There were no register spills. Retaining the native batched
projection reduced the C16 result to 66.41 versus 62.06 microseconds in a new
paired screen, but remained slower at every measured width. Both prototypes
remain outside default dispatch. Their negative result is a scheduling and
synchronization issue, not a bit-identity gate failure. The screen covers only
part of a reduction segment and is not an end-to-end decode claim.
