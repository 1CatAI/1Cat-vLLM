# Flash-Next verifier diagnosis on four 16GB V100 GPUs

The objective is target verification below 16 ms at approximately 32K input
tokens, while retaining emitted-token speed and output quality. Verification
includes target forward, target sampling and state update. Draft proposal is
reported separately. Model qualification of the candidate is in progress;
the operator results below do not establish the 16-ms objective.

## Fixed workload

The reference source is `6f3557c973b4602905037e086fd4acba9b0d867f`.
Hardware is four V100-SXM2-16GB GPUs with a full NV2 mesh, driver 570.211.01,
CUDA 12.8, Torch 2.10.0+cu128 and Python 3.12.3. The model is Flash-Next
GSQ-RCO IQ3_S with Q8_0 MTP4, greedy draft sampling, TP4 and one request.
Target sampling is temperature 0.6, top-p 0.95, top-k 20, seed 20261014;
thinking and EOS remain enabled.

Context capacity remains 262144 **input tokens**, chunk size 8192, prefix
caching enabled, and FULL target decode graphs at M5. Expert storage retains
the configured 80% GPU / 20% host placement. Target QSA history uses host
E4M3 with an 8192-token device hot cache; draft history uses host FP16.
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
20.289 / 17.701 ms, p90 25.496 / 20.319 ms, and p99 40.704 / 25.687 ms.
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
versus the reference 20.289-ms mean; retain this first-request variation.
The second candidate model measurement is pending.

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

An eight-warp direct-resolve screen regresses FP8 M5 churn by approximately
30% and other M5 cases by approximately 12%; retain W4. A scalar reservation
of four CLOCK probe ranges improves FP8 M5 only 0–2%, with noisy FP16
regressions; retain the existing per-page atomics. These rejected choices do
not change the final runtime or introduce startup switches.
