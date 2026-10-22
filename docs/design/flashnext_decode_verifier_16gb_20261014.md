# Flash-Next verifier diagnosis on four 16GB V100 GPUs

The objective is target verification below 16 ms at approximately 32K input
tokens, while retaining emitted-token speed and output quality. Verification
includes target forward, target sampling and state update. Draft proposal is
reported separately. The final opt-in encoded target hot cache measures an
all-round mean of 15.482–15.494 ms at 32K, including conservative bounds for
two missing graph starts. This meets the mean objective; p90 and p99 remain
17.340 and 31.689 ms. Ordinary decode and near-256K completion also pass the
measured cases described below.

## Fixed workload

The reference source is `6f3557c973b4602905037e086fd4acba9b0d867f`.
Hardware is four V100-SXM2-16GB GPUs with a full NV2 mesh, driver 570.211.01,
CUDA 12.8, Torch 2.10.0+cu128 and Python 3.12.3. The model is Flash-Next
GSQ-RCO IQ3_S with Q8_0 MTP4, greedy draft sampling, TP4 and one request.
Target sampling is temperature 0.6, top-p 0.95, top-k 20, seed 20261014;
thinking and EOS remain enabled.

Context capacity remains 262144 **total input and output tokens**, chunk size
8192, prefix caching enabled, and FULL target decode graphs at M5. Expert storage retains
the configured 80% GPU / 20% host placement. Target QSA history uses host
E4M3; the reference has an 8192-token device hot cache and the configuration
candidate has 16384 tokens. Draft history uses host FP16. Device hot values
remain FP16 for both history types; the larger cache adds about 104 MiB per
rank including draft storage.
PLE remains on disk with its native lane and 512-MiB CPU row cache.
Measured prompts contain 574 and 32774 tokens, with 192 output tokens.
These measurements do not qualify 256K speed or concurrent requests.

## Reference measurements

Ordinary streaming HTTP excludes TTFT from decode. Metrics count actual
draft-verification steps, so the round estimator divides by draft count,
without subtracting a prefill step. Retain raw observations and first-run
variation; the HTTP estimate is not an independent verifier timer.

| Input tokens | Mean ms/output token | Estimated ms/round | Tokens/round |
| --- | ---: | ---: | ---: |
| 574 | 9.110 | 18.710 | 2.054 |
| 32774 | 8.721 | 21.256 | 2.450 |

CUDA-only Nsight graph-node capture observes 1120 target kernels on every
rank. Short/long target counts are 94/76, versus independent committed
verification counts of 93/75; retain the extra pipeline-end replay. Draft
graphs contain 70/71 kernels and 207/210 kernels. Requests are separated by
their prefill boundary, and TP replays are aligned by ordinal.

The following means close the same GPU0 target-start to next-target-start
window at 32K. Node tracing perturbs latency; this is composition evidence.

| Phase | Mean ms |
| --- | ---: |
| Target graph activity envelope | 22.124 |
| Target end to first draft graph | 0.862 |
| Dependent draft graph chain | 3.895 |
| Draft end to next target graph | 1.173 |
| Complete round | 28.053 |

Expert copies contribute 2.437 ms of target kernel service, expert publish /
CPU-plan wait 1.206 ms, and QSA host gather 1.990 ms. Dense MMA contributes
2.272 ms and 94 complete HC boundaries contribute 2.151 ms. Service can
overlap; these values must not be added into an endpoint wall table.

Explicit synchronized step timing retains every TP worker. At 32K, the two
reference requests have mean per-round maximum verifier CPU walls of
20.354 / 17.701 ms, p90 25.496 / 20.319 ms, and p99 40.704 / 25.687 ms.
Target sampling is approximately 0.42 ms and state update 0.02 ms.
Event collection, fences and interval-one logging perturb endpoint latency;
ordinary HTTP remains the accepted emitted-token speed measurement.

## Candidate and operator evidence

The existing expert copy kernel changes from 16 operation programs with
8 word splits, BLOCK1024/W8, to 4 operation programs with 32 word splits
and 3 parallel projections, BLOCK256/W4. This increases parallel work for
small miss plans without serializing all three projections in each CTA. Whole-word
storage, swap ordering and masked tails retain their exact byte contract.

The paired TP4 screen uses real per-expert byte geometry from layers 0, 1,
17 and 47, with production 410 device / 136 host slot counts. All six
policies pass eager and changed-input graph equality over complete host and
device banks, including guards. The initial 4-by-64 word split improves
common one/two-miss cases roughly 6–20%, with a 6.4% regression in one
eight-miss case. The projection-parallel screen additionally improves paired
miss plans by 0.5–17.5% versus that first candidate, while some swap cases
regress. End-to-end admission is required. Word lengths 255/256/257 and
operation counts around 4 and 16, through 128, also pass four-rank byte checks.

For QSA direct resolution (STAGE0), values that cannot be installed are
read later by the authoritative native attention reader. The candidate
omits the unused host read/decode and mixed-tile cached-value read, and
skips the entire value-materialization block when no page is installed. It
preserves tags, ownership, counters, resolved indices and consumed values.
Staging modes 1/2 retain their original value-read masks; staging writes
complete before the same release of cache metadata.

Correctness support retains the existing valid page4/D256 geometry and
FP16/E4M3 storage. The changed subset is direct resolution with failed
installations or mixed hit/miss tiles. Shape admission and existing native
and staged fallbacks are unchanged. No numerical tolerance is relaxed.
The first host KV suite passes 104 cases on both reference and candidate,
including protected-capacity fallback and changed-input graph replay. The
second candidate passes 108 host KV cases and 7 expert cache cases, including
probe-pointer signed/unsigned wrap, graph replay and decode/prefill interleaving.
The step-profiler suite passes 13 cases on GPU0.

The first candidate's 32K second request improves maximum verifier CPU-wall
mean from 17.701 to 16.704 ms, p90 from 20.319 to 18.388 ms, and p99 from
25.687 to 19.113 ms. Its first request is 20.598 ms with an 87.139-ms maximum,
versus the reference 20.354-ms mean; retain this first-request variation.
The second candidate's first/second 32K means are 20.054 / 16.864 ms; its
second-request p90/p99 are 18.373 / 25.294 ms. This does not establish a
further gain or satisfy 16 ms. Ordinary candidate HTTP qualification remains
pending while optimization continues.

Existing tests had compared the full 32-row workspace against a smaller
active request and passed capacity-sized metadata to attention. The
original kernel reproduces the failure. Tests now slice the active rows;
the original and candidate pass the same corrected oracle.

## Reproduce and retained negative paths

Run the installed source benchmark on an idle, locked TP4 GPU set:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node 4 \
  benchmarks/kernels/benchmark_sm70_gguf_cache_copy.py --output copy.json
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node 4 \
  benchmarks/kernels/benchmark_sm70_gguf_cache_copy.py \
  --boundary-only --samples 1 --repeats 2 --output boundary.json
```

Use the prepared environment's Python and its normal native artifacts.
Raw Nsight report, SQLite, HTTP responses, per-rank logs and copy JSON are
retained in the task's external `flashnext-decode-verifier-9683-20261014`
report directory, outside Git.

Optional NVTX scopes failed because the Python package is absent; CUDA-only
node tracing succeeds. A concurrent prefill compiler exhausted host headroom
and memguard terminated an early capture; the user chose compiler priority.
The serving CLI does not expose `--observability-config`; the existing
captured MTP profiling compatibility flags enable the diagnostic instead.

Initial direct-resolve screens retained one KV epoch across replays. They
measure a protected-cache regime: selected pages remain pinned and later
misses fall back instead of installing. They do not represent normal decode
churn. The corrected fixture advances the epoch before every resolve,
matching the protection transition of a decode KV write.

With advancing epochs, the adversarial FP8 M5 churn screen measures W4/W8
at 693 / 660 us, while shared/overflow selections regress about 10% at W8.
Eight-token tiles improve that churn case only about 2%; four-token tiles
regress both shared selections and churn. Retain the existing W4, 16-token
tile and per-page CLOCK atomics. Register counts are 127 for W4/T16, 76 for
W8/T16 and 72 for W4/T8, with no spills. This is a synthetic operator screen;
register reduction alone does not qualify model speed. No new tiling or warp
startup switches are added.

An exact-word HC gather experiment encoded Half storage words as integers in
FP32 and used the existing peer reduction. Every Half bit pattern and changed
graph passed exact equality, but the complete M5 boundary regressed from
69.824 to 79.296 us; the model's second 32K verifier mean was 17.472 ms. The
helper was reverted. Collective substitution alone is not a speedup.

The existing 32768-token hot-cache setting also failed admission: the first
real request needed a 144-MiB allocation with only 75.94 MiB free. It retained
the expert ratio and context capacity but cannot run on this 16GB deployment.
The 16384-token setting runs successfully and measures first/second 32K
verifier means of 18.435 / 16.059 ms, with second-request p90/p99 of
17.209 / 18.848 ms. It is a separate configuration screen and still misses
16 ms. Arithmetic, Python evaluation, two 32K retrieval tasks and a 1809-token
CPU explanation pass their task/EOS gates; the explanation has no six-word
phrase repeated more than twice. Ordinary throughput qualification is pending.

The two target-graph NCCL gathers occur after the early PLE boundary. PLE
materializes the pending MoE residual before adding its result, then calls
standalone HC `mix()`. Retaining the six-dimensional HCX fusion pack at that
boundary forces unpacking and two gathers, because standalone LL projection
consumes the existing five-dimensional pack. A candidate leaves partial
residual ownership intact and prepares the LL bank at PLE boundaries instead
of the unused HCX bank. This changes one target boundary on the measured
model and allocates no duplicate bank. HCX schedules at other boundaries,
partial residual ownership, normalization and request state were unchanged.

The actual layer-1 checkpoint screen on four ranks measures complete M5
boundaries at 76.672 / 19.140 us for reference / LL storage. M1/M8/M20 are
also faster; M21 and M256 retain fallback with exact outputs. Native arithmetic
uses the existing FP16 HC error gate, rtol 0.003 and atol 0.003, fixed before
the screen. Both variants replay changed inputs exactly against their own
eager results, with finite outputs. CPU dispatch/lifetime tests pass 21 cases;
the corrected fallback suite passes 17 GPU cases across two rank orders,
both storage layouts, M5/M20/M21 and M512. The model screen retains HOT8192
to separate this source change from cache tuning.

That model screen rejects the PLE storage change. First/second 32K verifier
means are 18.077 / 17.048 ms; the second-request p90/p99 are 19.262 / 23.022 ms.
Committed draft steps rise to 91 for 192 output tokens, versus 75 in the
preceding HOT8192 candidate. Four integer tasks pass, but the CPU explanation
remains in reasoning at the fixed 2048-token limit. This does not establish a
quality regression by itself, but gives no qualification for promotion.
The runtime and dispatch-test changes are reverted to the preceding candidate.
Retain the corrected fallback oracle and its passed GPU coverage. The large
operator speedup and node-trace NCCL waits did not predict lower live verifier
cost; do not promote this route based on the microbenchmark alone.

The old fallback oracle omitted the physical rank and simulated peers by
scaling an already mixed local result. That does not model each peer's
different input columns and nonlinear gates; it fails the unchanged HCX
fallback too. The corrected oracle computes remote projections and columns,
includes the actual local packet, and retains exact assertions and changed
input graph checks.

An asynchronous final verifier replay can finish after the HTTP stream.
The original before-counter read occasionally assigned that replay to the
next request. Retained API POST boundaries align all four workers and keep
every final replay with its preceding request; aligned counts differ from
independent committed draft metrics by at most one. Raw and aligned ledgers
are both preserved. New event probes wait for all-rank counters to remain
stable for 200 ms between requests. This does not change measured decode
timestamps or ordinary production requests.

## Ordinary whole-graph measurements and completeness

CUDA graph-level tracing records graph envelopes without per-round CUDA
Events or fences. The verifier window runs from the earliest target graph
GPU start to the latest `_postprocess_mamba_align_kernel` end across TP4.
Source inspection establishes `_post_update_kernel`, then
`_scatter_num_accepted_kernel`, snapshot copy and Mamba alignment as target
sampling/state commit. Draft preparation follows this boundary. CPU graph
launch-to-draft time includes previously queued work and is not an isolated
verifier measurement.

The matched ordinary HTTP pair retains identical prompts, seeds, sampling,
MTP, native artifacts and warmup sequence. Reference HOT8192 versus source
candidate HOT16384 measures 574-token mean TPOT at 8.957 / 7.714 ms, and
32774-token mean TPOT at 8.331 / 8.519 ms. The latter is about 2.3% slower;
there is no unconditional emitted-token speed claim. Tokens per committed
round differ, so output TPOT and verifier latency must remain separate.

Commit-aligned four-request traces measure first/second 32K verifier means
of 16.967 / 16.169 ms for the reference and 15.816 / 17.788 ms for the
candidate. The candidate's 86.557-ms tail remains included. Every committed
round is retained; one additional asynchronous pipeline-end graph is
recorded separately per request.

An eight-request steady HOT16384 capture has 584 complete state commits on
every rank: each request contains one prefill commit, 71 committed verifier
rounds and one pipeline-end replay. Five graph activity records are missing
across the ranks, affecting four of the 568 committed rounds. Direct target
graph ordinal alignment is invalid for this report and its old global
statistics are withdrawn. The three commit kernels, target graphs and
draft graphs all use the same CUDA context and stream on each rank.
Associating each graph with its following commit aligns complete request
and round ledgers. For a missing start, the preceding same-stream commit
end bounds the target start from below; this conservatively includes draft
and preparation time instead of imputing or dropping the missing window.
All remaining graph records map uniquely to a commit and all independent
committed-step counts reconcile.

The resulting eight-request all-round verifier mean lies between 16.014
and 16.043 ms; upper-bound p50/p90/p99/max are 15.265 / 18.984 / 24.134 /
31.897 ms. This remains above the 16-ms objective. Nsight's incomplete-event
warning and four conservatively bounded rounds are retained. A fresh single
capture range is required for a fully exact distribution. Interactive rearm
returned `StopRequest` errors and did not reliably honor a new output/range;
do not mix its report with the later HTTP requests.

HOT20480 passes the five screened task/EOS cases but measures synchronized
32K verifier means of 17.864 / 16.590 ms, slower than HOT16384 and with less
memory headroom. W2/T16 at HOT16384 uses 202 registers without spills but
regresses shared, overflow and advancing-epoch churn selections; retain W4.
No corresponding runtime or default tuning change is promoted.

The ordinary pair passes arithmetic, Python evaluation and both 32K needle
retrieval tasks. Its open-ended CPU explanation reaches the 2048-token cap
in both arms; a candidate 4096-token follow-up also reaches its cap. These
runs do not qualify EOS or long-output drift. The earlier HOT16384 screen's
1809-token natural EOS is retained separately. A 261959-input-token capacity
request reaches the 300-second client timeout without an OOM; it does not
establish successful near-256K completion. Keep both gates explicit before
promoting the deployment configuration.

An empty-plan expert-ID remap screen tests counts 0/1/5/10/50/128 and plan
sizes 0/1/2/4/20/256 with W4 and W1, with and without an empty-metadata branch.
All 36 cases per policy retain exact table/output bytes in eager execution
and changed-input graphs. The measured M5 empty-plan kernel changes from
1.832 to 1.992 us with the W4 branch, so no runtime change is retained.

The 564 exactly recorded steady rounds show mean maximum individual-rank
verifier duration of 15.656 ms, versus 16.017 ms for the complete TP4 window.
GPU target-start skew has mean/p90/p99 of 1.081 / 3.543 / 6.824 ms. CPU target
enqueue skew averages 8.579 ms; GPU queue-ahead averages 11.246 ms and includes
prior work. Enqueue skew correlates with GPU-start skew (0.607), motivating
an owned-process CPU affinity screen rather than treating queued time as
additional verifier computation.

## Encoded hot-cache model screen

The opt-in target setting `qsa_host_kv_hot_dtype="fp8_e4m3"` retains the
encoded bytes and original per-vector FP32 scales in device hot storage.
Both the existing native QSA family and its protected Triton reader decode
to the same FP16 operands used by the decoded cache. The authoritative host
history, page ownership, request masks, tentative writes, release ordering
and staging format remain unchanged. Default hot storage remains FP16.
Encoded hot storage requires E4M3 history; incompatible history is rejected.
An unavailable `run_hot_fp8` native artifact uses the qualified protected
reader, and unaligned native hot storage is explicitly rejected.

The optional `qsa_host_kv_draft_hot_tokens` independently bounds the draft
cache; its default None preserves the existing shared capacity policy.
The measured prototype doubles target hot capacity from 16384 to 32768
while retaining the 16384-token FP16 draft cache. Target storage changes
from 1024 to 520 bytes per cached token. Across twelve target QSA layers,
this configuration adds about 3 MiB per rank including scales, while leaving
the expert ratio, host histories, disk PLE and total context capacity intact.

GPU qualification passes 151 cases: the original 130 host/native tests,
18 new native/protected-reader cases over M1/M5/M20/M21/M32, selection
widths 1/15/16/17/65/2051/4096, hot capacities 48/128/32768, request masks,
causal bounds and changed-input graphs, plus unavailable-native fallback
and two alignment-rejection cases. Live hot bytes and scales match the
host source exactly. Native attention uses its existing FP16 comparison
gate, rtol 0.003 / atol 0.0002, unchanged before the candidate runs.
The first prototype's eight fallback failures are retained: its generic
reader still interpreted encoded bytes as FP16, and three fixtures used
invalid staging capacities. The corrected reader and fixtures pass.

The native module is built with the configured target's exact NVCC/link
arguments, writing only task-owned outputs and replacing the deployed
source module atomically. Shared venv packages are untouched. Original
native SHA256 is `ed1d3a854e673cec3c646076bbc7d3b3010d5840fa4c17867d1fc445a2e41eac`;
prototype SHA256 is `76fa88cc1fe323f019c0ad6942756dde84e5396bf46ae07b00cd27dc99115fd2`.
Rebuilding the final formatted source produces SHA256
`bf0dd515b90a2132f95ae408ab4e23bec4e0e19b81b1199f2d74b9d644b7a54b`.
All 24 affected encoded-cache, unavailable-native, alignment-rejection and
legacy E4M3-code tests pass again against that final artifact.
Rebuilding the final formatted source gives SHA256
`bf0dd515b90a2132f95ae408ab4e23bec4e0e19b81b1199f2d74b9d644b7a54b`.
Its 24 targeted GPU regressions pass, covering encoded/protected-reader
graphs, missing-native fallback, alignment rejection and all E4M3 codes;
the preceding 151-case suite remains the broader qualification evidence.

The TP4 M5 screen includes resolve plus native attention at 32774 input
tokens, width 2051, advancing KV epochs and alternating paired measurements.
FP16 HOT16384 / encoded HOT32768 shared-selection medians are
51.456/52.992, 43.264/45.808, 48.896/53.760 and 47.104/49.920 us across
ranks. Sixteen changing selections measure 719.776/595.888,
731.136/592.207, 713.152/598.064 and 716.224/590.608 us. Churn improves
17–20%, with shared selections 3–10% slower. This passes the predeclared
operator screen of at least 5% churn improvement and at most 10% shared
regression. Model admission uses the separate measurements below; the
repository's default hot storage remains FP16.

The encoded configuration's eight-request model capture retains all 640
committed verification rounds. State commits are complete on TP4; 638
windows are exact and two missing graph starts use conservative bounds.
The all-round mean lies between 15.482 and 15.494 ms, below the 16-ms mean
objective even at the conservative endpoint. Upper-bound p50/p90/p99/max
are 14.687 / 17.340 / 31.689 / 35.538 ms. Request means are 15.209, 14.780,
15.894, 16.804, 15.303, 15.072, 16.119 and 14.772 ms: this is an average
objective, not an every-round or p99 guarantee. The preceding FP16 HOT16384
capture under the same affinity policy has mean bounds of 16.079–16.091 ms;
the runs have different warmup histories, so this is consecutive-screen
comparison rather than an unconditional throughput gain.

Arithmetic, Python evaluation and both 32K retrieval tasks pass with natural
EOS. The paired concise 32K CPU task passes both seeds 20261014/20261015:
the encoded run ends naturally at 1088/682 output tokens versus 1333/1348 in
the preceding FP16-hot run. Both final answers correctly explain renaming,
RAW readiness, the ROB's ordered retirement and precise exceptions, with
concrete dependency examples. Reasoning and final output have no six-word
phrase repeated more than twice. The open-ended 2048-token CPU task still
hits its cap and remains an inconclusive long-answer stress observation.

Eight ordinary streaming requests, with capture inactive and the same 32774
input / 192 output tokens, measure mean TPOT 7.109 ms, or 140.7 output
tokens/s. Mean decode time is 1.358 s, excluding 1.309-s prefix-warm TTFT.
The HTTP round estimator is 18.859 ms with 2.681 emitted tokens per round;
it includes draft and scheduling and is distinct from GPU verification.
All requests emit 192 tokens. This is a candidate observation, not a paired
throughput gain over the earlier runs, whose acceptance counts differ.

The same capacity probe, with its timeout raised from 300 to 900 seconds
and unchanged 128-output limit, completes with 261959 input and 71 output
tokens (262030 total), natural EOS and the correct `SAFFRON45793` marker.
TTFT is 320.034 s and total time is 320.435 s. This establishes completion
near the 262144-total-token boundary for this request, not 256K speed or
concurrency. No OOM or memguard termination occurs. GPU memory reaches
16143 MiB per rank afterward, leaving roughly 17 MiB free: capacity passes
with very little device headroom. The earlier 300-second timeout remains
retained as an incomplete observation rather than a successful capacity run.

The final ordinary server runs without Nsight injection or profiler CLI
configuration, with target encoded HOT32768 and draft FP16 HOT16384. After
three warmup requests, task-owned worker main and helper threads are placed
on separate physical CPU cores using the recorded affinity policy. One
32774-input / 192-output smoke measures TPOT 7.836 ms, or 127.6 tokens/s,
and estimated whole-round cost 18.252 ms. Its acceptance is 0.335, versus
the earlier eight-request screen's different acceptance history. This is
a single deployment smoke, not a new mean-verifier distribution.
The key-gated public proxy returns 401 without authentication and 200 for
the authenticated model list and generation. Arithmetic generation returns
1517 with natural EOS. Memory protection remains active. The standard
deployment launcher retains its GPU locks and now records the selected
target/draft cache settings; its original script is backed up outside Git.
