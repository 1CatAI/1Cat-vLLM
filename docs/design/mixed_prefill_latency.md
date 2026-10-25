# Mixed prefill latency control

When resident requests are decoding, an incoming long prompt can occupy the GPU
for a large prefill chunk. Its decode/verification rows may use fast kernels
and still wait seconds for the rest of the step to finish.

The scheduler now reserves tokens for eligible resident decoders and limits
the aggregate prefill work sharing their step. The default step latency
target is 250 ms. A conservative initial budget is adjusted using completed
GPU measurements, so the policy follows the actual model, device, parallelism,
and context instead of an exact model-shape contract. This is a soft target,
not a deadline guarantee. No resident decode means the normal large prefill
budget is retained. Pure decode retains its existing FULL CUDA graph route.

`--mixed-prefill-step-latency-ms` changes the target; zero disables the policy
for comparison. There is no extra environment variable. GPU timing creates
events only for controlled mixed steps, checks their readiness without waiting,
and returns plain completed timing data to the scheduler for PP=1.
Indivisible multimodal chunks retain their existing scheduling behavior.

## Pipeline parallel feedback

For PP>1 the engine measures submission-to-result-consumption latency on its
monotonic host clock. Each `SchedulerOutput` owns its start timestamp, including
when structured-output sampling is deferred. This is an observed end-to-end
latency including queued work, communication, sampling and engine consumption
delay, not GPU compute time. No new device synchronization or result wait is
introduced. Workers skip local CUDA timing for these batches: the last stage's
events are not a complete pipeline measurement.

PP also limits prefill queued behind resident decoders that are temporarily
ineligible because their feedback is still in flight. These controlled batches
can have zero local decode rows. Execution eligibility, MTP placeholders and
recurrent-state feedback ordering are unchanged. Cancellation removes the
resident from protection through the normal scheduler request lifecycle.

PP starts with 32 aggregate prefill tokens per controlled batch, subject to the
configured minimum and maximum. This avoids an initially large chunk before
pipeline measurements are available. A stalled PP batch can reduce the budget
even when another scheduling limit makes the chunk smaller than the budget;
short fast boundary chunks still cannot increase it. Small budgets can grow by
at least one 16-token quantum when measured cost permits, so a temporary stall
cannot permanently trap the controller at its minimum.

The target remains soft: queued batches may already have been submitted, and
long-context decode alone can exceed it. Smaller prefill chunks favor resident
decode continuity at the cost of incoming prompt TTFT. Pure prefill without
resident decoders retains its normal budget, and pure decode retains the same
CUDA graph route. CPU regressions cover PP7/MTP cadence, queued and deferred
feedback, cancellation and budget adaptation. The GPU performance measurements
below predate PP support and do not establish its throughput or latency gains.

## Small chunks and recurrent state

Mamba align mode allows chunks shorter than a state block. The worker continues
updating the current running state in place. At a retained boundary the
scheduler ends the chunk, and the next chunk copies that state into a fresh
running block. Prefix lookup admits completed checkpoint states only.
Regression tests exercise allocation, worker state movement, boundary snapshot
immutability, and prefix hits with 560/1024 and 512/8192 token chunks/blocks.
In a hybrid Mamba/SWA draft cache, the retained sliding window must match the
replay checkpoint admitted by Mamba, including the speculative lookbehind.
Expired replay-window blocks keep normal cache priority instead of being
immediately reused. They remain evictable and do not enlarge the real-held
sliding-window reservation.

## Prefill kernels

The mixed attention/GDN routes incorporate the work in #841. Mixed-row plans
copy pinned host metadata asynchronously rather than creating device tensors
from pageable host lists. The NVFP4 small-prefill dequantization candidate uses
shared memory to transpose the original FP16 values into aligned vector stores;
it keeps the existing scale arithmetic and cuBLAS GEMM. Small prefills use
FP32 accumulation without reduced-precision partial reductions. The dense
prefill threshold is 256 rows; smaller batches retain TurboMind. Large chunks
and the small-row decode kernels retain their previous dispatch.

A first fused WMMA dequantization/GEMM candidate was rejected: on the screened
512–8192-row shapes it was approximately four times slower than the existing
dequantization plus cuBLAS path. It is not included in the implementation.

## Validation

Use `benchmarks/benchmark_mixed_prefill_latency.py` with four fixed 32K token-ID
prompts. Two resident requests reach steady decode before two additional
prompts arrive. The client reports resident token throughput during the
incoming prefill window, the longest gap between resident stream updates,
and each new request's TTFT. It counts returned token IDs, not stream chunks.
Resident EOS is disabled to sustain the load; output quality must be checked
separately with natural completion behavior. There is no prefill/decode barrier
in this mixed-load measurement.

The measured contract used Qwen3.8-27B QUASAR NVFP4, DFlash2 with seven
speculative tokens, TP4 on four V100 SXM2 32 GB GPUs connected as two NVLink
pairs, CUDA 12.8, Torch 2.10.0, Python 3.12, FP16 compute, E4M3 KV,
FLASH_ATTN_V100, a 262144-token maximum length, an 8192-token batch budget,
16 sequence slots, KV/state blocks of 2048/8192, align prefix caching,
0.8 memory utilization, and thinking disabled. The candidate was installed
from a complete wheel in an independent virtual environment and launched
outside the checkout with a cleared environment. Strict acceleration
validation passed without manual acceleration overrides.

Two measured warm runs followed one cold run. Medians of the two warm runs:

| Online mixed-load metric | Control | Adaptive 250 ms |
| --- | ---: | ---: |
| Resident output during incoming prefill (tokens/s) | 6.88 | 30.34 |
| Longest resident update gap within prefill (s) | 2.873 | 0.310 |
| First incoming prompt TTFT (s) | 14.287 | 17.994 |
| Second incoming prompt TTFT (s) | 20.631 | 35.491 |

This favors decode continuity: resident throughput increased about 4.41 times
and the longest gap fell about 89%, while the incoming TTFTs increased about
26% and 72%. The initial cold candidate run measured 32.14 tokens/s, a 0.330 s
gap, and TTFTs of 17.68/34.75 s. These are online measurements with overlapping
prefill; they are not pure-decode throughput claims.

For the separate pure-decode check, each cohort used fixed 32768-token IDs,
a 256-token output cap, temperature/top-p/top-k of 0.7/0.8/20, and fixed seeds.
One cohort warmed the route; two cohorts were measured. A 1024-token long
prefill threshold admitted the cohort before the common all-requests-alive
output window. No new requests arrived inside that window. EOS was disabled
only for this sustained performance measurement.

| Pure decode (tokens/s) | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| C1 | 184.78 | 185.31 | +0.29% |
| C4 | 477.39 | 484.88 | +1.57% |
| C8 | 617.61 | 609.41 | -1.33% |

The control source was b2335687 plus #841. The candidate incorporated main
1d1d1c9d as well as this work. These results validate the combined candidate;
they do not isolate the contribution of each kernel or intervening main fix.
Eight deterministic natural-EOS answer checks passed in both arms. Long
needle checks also used natural completion. The final SWA replay-window fix
passed four natural-EOS 32K needle checks (two cold, two repeated), with a
nonzero prefix-hit counter. Incoming TTFTs in this quality workload changed
from 18.04/35.86 s to 5.21/10.20 s on the repeated prompts. This is evidence of
correct cache reuse, not a comparison of uncached prefill kernel throughput.

The first dequantization/GEMM fusion candidate was rejected. Communication
and computation overlap is not implemented in this change. Broad scheduler
tests that import Llava were unavailable in the test environment because of
an unrelated Transformers Pixtral import mismatch; focused cache, state,
scheduling, kernel and packaging tests provide the change's validation.

## Context-only draft work and PP observations

An incomplete prompt chunk still populates the Eagle/MTP draft model's KV and
recurrent state. With a single data-parallel replica, it now stops after that
prefill pass instead of performing the remaining speculative decode steps:
these proposals cannot be consumed until the prompt finishes. The feedback
keeps its normal fixed-width shape and contains zero-filled unused proposals.
Final prompt chunks, mixed batches, DP collectives, profiling and warmup keep
their existing draft path. No device-to-host check or synchronization is added.

PP timing also observes standalone prefill batches. A zero control budget marks
these observations so they cannot train the mixed-prefill latency controller.
The engine emits one batch-latency sample per ten seconds, including token
counts and the mixed budget, without logging prompts or request identifiers.
The measured latency includes queued work and PP communication; it is not a
per-stage GPU kernel measurement. Pure decode does not create timing events.

Larger standalone prefill capacity can be configured through
`max_num_batched_tokens` while the separate mixed-prefill budget continues to
reserve resident decode capacity. The capacity must still be checked against
the deployment's peak memory. CPU tests cover these execution and scheduling
contracts; they do not establish a GPU throughput improvement.
