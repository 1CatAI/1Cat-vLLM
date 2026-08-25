# SM70 DFlash2 + ngram Hybrid

## Purpose

Add an optional prompt-ngram assistant to the MRV2 DFlash2 path. An ngram hit
must skip the DFlash2 query/selector while preserving DFlash2 context-KV state,
and a miss must fall back to the existing DFlash2 implementation unchanged.
The target verifier and the probabilistic rejection-sampling contract remain
authoritative, so the optimization cannot change the target distribution.

This work is based on `onecat/main` at
`d62ef5cb20b48de93a91562e777ac48985f44b76` and is isolated on
`agent/v100-dflash2-ngram-hybrid-20260825-050018`.

## Upstream audit

- vLLM prompt lookup: PRs #12193, #22437, #24986, and #29184 provide the CPU
  KMP and GPU-vectorized implementations. MRV2 still has no combined ngram +
  model-drafter route.
- SGLang: PRs #17260, #21243, and #22737 show that overlap scheduling requires
  complete request-token state and explicit accepted-token indexing. A stale
  host output list is not a valid lookup source.
- llama.cpp: its comma-separated speculative configuration gives draftless
  ngram proposers priority, falls back to DFlash, and still calls `process()` on
  every proposer so model-drafter state remains synchronized. This is the
  closest reference architecture for the first implementation here.
- Arctic Suffix Decoding and SAM-Decoding are useful follow-ups, but their
  confidence policy, external dependency, and longer trees are intentionally
  outside this first block-8 implementation.

## Initial contract

- The feature is opt-in and only valid for `method=dflash` with a DFlash2
  checkpoint. Standalone `ngram`, Eagle, MTP, DFlash1, and `dflash_ddtree`
  routing is unchanged.
- Lookup reads the authoritative MRV2 request-token state and supports normal
  synchronous and overlap scheduling without rebuilding history in Python.
- Structured-output requests bypass the ngram assistant until grammar-aware
  proposal masking is proved correct. Tool/reasoning parsers without a grammar
  are unaffected.
- A hit returns at most the configured DFlash2 draft width (seven tokens in the
  official block-8 setup). The normal DFlash2 proposer handles misses.
- Context K/V materialization always runs. Only the DFlash2 query, candidate
  projection, selector, and selector walk may be skipped.
- Greedy and probabilistic modes are supported. In probabilistic mode the
  ngram proposal is represented as a one-hot draft distribution inside the
  existing sparse target-rejection interface, preserving target sampling.
- Prefix-cache hits may rebuild missing DFlash draft K/V as before; this change
  must not corrupt either target or draft cache state.

## Test plan

1. Unit tests: configuration/routing isolation; KMP hit, miss, truncation, and
   overlap cases; per-request mixed-source state; one-hot sparse draft support;
   grammar bypass; prefix/state transitions.
2. CPU microbenchmarks: lookup at 1K, 32K, 128K, and 256K contexts.
3. V100 correctness: eager before CUDA graph; batch 1/2/4; hit/miss/mixed;
   compare proposed tokens, accepted trajectory, target output, and draft K/V.
4. V100 performance: report lookup, context-KV, query/selector, target verify,
   full round, emitted tokens per round, and pure-decode tokens/s for baseline
   DFlash2 and DFlash2+ngram at short and long contexts.
5. Quality: compare target-only, DFlash2, and hybrid on coding/general/tool
   datasets under the same sampling configuration. Report scores, completion
   counts, wall time, hit rate, conditional acceptance length, and failures.

## Promotion gates

- No statistically meaningful quality regression against target-only or the
  existing DFlash2 route.
- DFlash2-miss acceptance trajectory remains unchanged in deterministic tests.
- Ngram hits actually skip query/selector work in a trace.
- Hybrid pure-decode throughput is non-regressing at every measured context;
  otherwise the feature remains opt-in while the losing shape is investigated.

## Status

- 2026-08-25: Draft PR #287 implements opt-in MRV2 host lookup over the
  authoritative UVA request-token state, full-hit query/selector bypass,
  all-hit batch application, and one-hot dense/sparse rejection caches.
  Structured-output batches bypass the assistant, and intermediate chunked
  prefill materializes draft context K/V before returning without lookup.
- CPU split-history KMP microbenchmark (median/P95): 1K `2.66/2.74 us`, 32K
  `50.05/53.22 us`, 128K `190.72/201.03 us`, and 256K `369.57/393.17 us`.
  Focused validation reports `93 passed, 9 skipped` for the CPU DFlash2/ngram
  set, `10 passed` for MRV2 routing, and `24 passed` for the V100 ngram/AOT
  fullgraph set.
- PR #288 first restored the missing production optimization closure to main.
  On that restored source, no-assist practical coding measured `18.660 ms` per
  complete round, `135.70 tok/s`, and acceptance length `2.532`. Repeated runs
  of the historical MBPP item 28 stabilized at `18.484--18.616 ms`,
  `224.05--225.66 tok/s`, and acceptance length `4.184`. Older 27--31 ms
  branch measurements predate this closure and are invalid baselines.
- With ngram `[5,5]` enabled under the identical TP4 practical contract, the
  coding request measured `18.980 ms`, `131.14 tok/s`, and acceptance length
  `2.494`; only about 2.2% of eligible rounds were full hits. The matched MBPP
  item measured `18.700 ms`, `217.18 tok/s`, and acceptance length `4.071`.
  Thus low-hit short requests currently pay roughly `0.1--0.3 ms` per round
  and do not pass the default-enable performance gate.
- The 32-case MBPP natural-stop run (16K output cap, fixed per-item seeds) gave
  the pure-DFlash control `65,934` output tokens in `389.542 s`, `19,767`
  verification rounds, `19.707 ms` wall time per round, and acceptance length
  `3.336`. Hybrid produced `53,710` tokens in `327.613 s`, `16,700` rounds,
  `19.618 ms` per round, and acceptance length `3.216`. Cumulative full-hit
  rate reached about 10.1% and lookup averaged `0.016--0.018 ms`; skipped
  queries slightly reduced round cost, but the lower sampled acceptance path
  left raw aggregate output throughput at `163.94` versus `169.26 tok/s`.
- EvalPlus on the mapped 31 MBPP cases reports pure DFlash Base `30/31` and
  Plus `28/31`; hybrid reports Base `31/31` and Plus `28/31`. This is no score
  regression, but it does not override the failed throughput gate. The
  probabilistic one-hot proposal preserves the target distribution; it is not
  expected to reproduce the same sample-by-sample random trajectory.
- A fresh hybrid 32K chunked-prefill request measured `10.688 s` cold and
  `1.416 s` on an identical-prefix hit, versus the restored no-assist evidence
  of `10.58--10.65 s` and `1.405 s`. The context-only skip log was present and
  no cache corruption occurred.
- Promotion decision: keep `ngram_assist` opt-in. The merge audit removes the
  mixed-hit hazard: an n-gram draft is now applied only when every active
  request has a full-width hit and the complete DFlash2 query/selector can be
  skipped. Mixed-hit batches retain the unchanged DFlash2 proposals because
  overriding a row after paying the full query cost cannot improve latency and
  can alter sampled acceptance. A future default-on policy still needs matched
  evidence that full-query skips repay lookup/probe overhead; blindly reducing
  the ngram length is not justified by the current data.
- After the paired run was stopped, fresh graph captures on GPUs 4--7 began
  failing inside the draft paged-attention capture with
  `cudaErrorStreamCaptureInvalidated`. A clean pre-ngram main worktree failed
  at the same point, so this is not an ngram source regression. The NVLink
  topology requires resetting all eight GPUs, which was deliberately not done
  while the user-facing API remained live on GPUs 0--3. Do not count failed
  startups as performance samples.
