# SM70 DFlash2 batch latency, 2026-09-26

Status: implementation and isolated kernel validation; clean endpoint control
started after the other GPU task finished. No new end-to-end decode gain is claimed.

## Contract and ownership

- Integration: `onecat/main`, base
  `e889919e2192fa36b25c922366526fa3fe62edbc`.
- Branch: `codex/v100-decode-round-20260926-111504`.
- Retained artifacts: `/data/minimax-h3/task-cache/sm70-decode-round-20260926`.
- Target: Qwen3.8-27B-NVFP4, TP4 V100-SXM2-32GB, FP16 execution,
  DFlash2 q7 (eight verifier rows/request), target E4M3 KV, draft automatic KV,
  Flash-V100, prefix caching, max context 262144, memory utilization 0.8.
- Dataset: the existing 2048-input/256-output shared-prefix set, SHA256
  `1cea3c5dbbd22fde40ad08db21fae1065b74f994009882a37e89c16b82b3580e`.
- Sampling: temperature 0.7, top-p 0.8, top-k 20, min-p 0, repetition penalty
  1, seed 20260923. Fixed output length is for speed only; natural EOS quality
  remains a separate gate.
- CUDA 12.8, Torch 2.10.0+cu128, Python 3.12. Private caches and port 18879.
  Other sessions' services must finish before reserving four GPUs.

## Implemented candidates

### Batch context and metadata graphs

The previous pre-sampling context projection and graph metadata refresh only
admitted B1/q8. Capture exact context shapes already supported by the full draft
query graphs. Replay the matching graph for uniform no-prefill verifier batches;
ragged, prefill and uncaptured shapes retain their existing path.

Projection uses original positions before the acceptance decision. Only the
later store consumes accepted slot mappings. Capture initializes mappings to
`PAD_SLOT_ID`, and metadata replay copies all captured rows, including padding,
to prevent stale request metadata when a batch shrinks. Dispatch depends on
query shape and backend capabilities, not a model name or weight quantization.

### Request-local exact sampler fallback

Previously a tied/ambiguous cutoff in one request made the entire batch use
full-vocabulary sampling. Keep the original cutoff guard and full sampler for
each affected request. Other requests retain compact sampling. Packing preserves
request-slot IDs, positions, local draft steps and seeds. Full-logit gathering
and the all-ambiguous fallback remain unchanged. Penalties, grammar, logprobs,
and synthetic rejection are still outside the compact path's contract.

This removes unnecessary dense sampling work; it does not claim to remove a
second LM-head projection (the old fallback already reused its logits).

### Grouped attention partition-weight reuse

Compute each partition's exponential once in the denominator loop and reuse it
across output dimensions. Keep the existing max/sum order, FP32 arithmetic,
output accumulation order, thread count and synchronization. This common combine
kernel also serves the legacy FP16-partial and precise FP32-partial routes;
the optimization is not conditioned on a model or KV quantization name.

## Validation to date

| Gate | Result |
| --- | --- |
| Draft/sampler CPU tests | 32 passed, 19 skipped (CUDA unavailable in CPU run) |
| Existing DFlash2/alignment/structured-output/compact-aux CPU regressions | 192 passed, 21 skipped |
| Mixed request fallback GPU tests | 7 passed; C2/C4/C8, ragged rows, heterogeneous sampling, changed seeds |
| Deferred context writes under graph replay | 5 passed; B1/B2/B3/B4/B8, 16 replays each |
| Complete grouped-attention GPU suites | 91 passed on the normal rebuilt extension |
| Uninstrumented TP4 C1/C2/C4/C8 and acceptance | Pending GPU availability |
| Natural-completion quality | Pending |

The sampler GPU tests compare valid output token IDs and accepted lengths with
the unchanged full-vocabulary sampler. The context tests compare cache contents
with eager projection/store and verify rejected slots are untouched. These are
operator tests, not model-level acceptance evidence.

## Research-only microbenchmarks and rejected candidates

Research DSOs live only in the retained artifact directory. They are never
loaded by endpoint benchmarks or required by the installed package.

### GEMM scale lifetime

Actual TP-local weights, M64, service split-K/swizzle choices, paired eager and
CUDA Graph runs: grouped-scale register reuse reduced FP8 M64 registers from
146 to 130. Moving the next-stage fetch after current MMA reduced this to 129.
All compared eager/replay outputs were bitwise identical for three activation
amplitudes. However, the paired seven-shape layer-weighted GEMM sum only changed
from 16.7295 to 16.4402 ms (1.7%), and some layers regressed. This is not admitted
to production dispatch. In particular register reduction alone did not cross
the two-CTA residency threshold for the M64 tile.

The native reference measured 17.0636 ms; do not mix that with the paired
prototype baseline to inflate the compiler/lifetime change's benefit. The
research FP8 output tile also uses a different scheduler group axis; it is not
an exact replacement for that service tactic.

A follow-up occupancy candidate used launch bounds after shortening the scale
lifetimes. The M64/N128/K64 variants reached 128 registers, two resident CTAs and
zero compiler-reported spill stores/loads. They did not become faster: FP8 input
62.833 -> 62.981 us, QKV 60.365 -> 60.831 us, FP4 down 54.953 -> 54.717 us, all
bitwise equal to the original. The complete seven-shape paired sum regressed
17.3572 -> 17.6370 ms. This candidate is also excluded. Higher theoretical
occupancy alone is not sufficient evidence of higher useful throughput.

### Attention combine

The first 256-thread candidate normalized shared weights in a separate phase.
It improved C8 but regressed C1 by 8–14%; it was rejected.

The retained 512-thread candidate reuses unnormalized exponential weights with
no extra barrier. Synthetic partition inputs, 7 paired timing repetitions,
six changed-input graph replays/case, all bitwise identical:

| Batch | 2K old/new (us) | 32K old/new (us) |
| --- | --- | --- |
| 1 | 6.768 / 6.214 | 12.877 / 11.293 |
| 2 | 6.662 / 6.282 | 24.630 / 19.469 |
| 4 | 10.784 / 8.806 | 31.450 / 27.389 |
| 8 | 26.899 / 23.354 | 60.186 / 51.318 |

The tested contexts also include 128, 8192 and 262136. This is a combine-kernel
improvement, not whole attention or emitted-token speed. At 16 attention layers,
the C8 2K microbenchmark predicts only about 0.057 ms saved per verifier step.

## Remaining acceptance

Run matched clean control/candidate services, with no profiling hooks or private
DSO overrides. Separate rolling-request decode from all-C-alive windows without
new prefill, and include every verifier width. Compare acceptance and natural
output quality before admitting any change as a default. Re-run exact-source
trace attribution only after an uninstrumented endpoint benefit is established.
Retain the 4K cache-hit, 32K routing, 262K boundary and affected 35B-A3B AWQ/FP8
regression obligations. Saved PRO data are iteration context, not a fresh win.
