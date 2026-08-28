# SM70 DFlash2 drafter-free lookup chains

## Scope and provenance

This change adds the second-stage drafter-free chain on top of the existing
lookup-augmented DFlash2 (LABD) path. It is adapted from the Apache-2.0
implementations in
[`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090)
revision `69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c` and
[`Dmtrii-tesla/dflash2-ngram-vllm`](https://github.com/Dmtrii-tesla/dflash2-ngram-vllm)
revision `8fb3a6044158fd74c096b3cb1b274edbe4b490c6`.

The feature is opt-in (`VLLM_DFLASH2_CHAIN=1`) and requires the existing LABD
contract: DFlash2 emits its checkpoint-native seven-token block while lookup
may fill a target verifier wider than the checkpoint. The default gate is
greedy, B1, DP1, text-only traffic. Sampled requests retain the neural drafter;
upstream measured a regression when a point-mass context proposal displaced a
distribution-aware neural proposal at nonzero temperature.

## Runtime design

- The chain controller runs once per scheduler step outside CUDA Graph
  capture. Entry uses the previous normal step's raw suffix-match length, read
  asynchronously through pinned memory. It does not use the valid continuation
  count, which is clamped to the verifier width and cannot prove a long match.
- An active chain still stages target hidden states, prepares slot mappings,
  and materializes draft context KV. It skips only the DFlash2 query/selector
  graph. This keeps the draft cache warm and makes the first post-chain neural
  proposal valid.
- A full verifier block comes from request history. Every row is rewritten to
  a point mass on its proposed token. Missing continuation positions are
  cleared to token zero rather than retaining stale proposals.
- The first rejected token exits the chain. One intervening neural step is
  required before re-entry so stale pinned evidence cannot immediately reopen
  it.
- While active, the scheduler is pinned to the full target-verification graph.
  The original q15 contract uses q16; the current q31 contract uses q32 while
  the checkpoint still emits only seven neural drafts. Normal traffic keeps
  the existing adaptive q8/full-width policy.

This preserves target verification and does not introduce a direct-output or
unverified path. It also deliberately does not replace the SM70 compact top20
sampler, sparse rejection sampler, Flash-V100 grouped verifier, prefix cache,
or Mamba-align policy.

### Native q32 target verifier

The q31 LABD contract originally fell through to the generic XQA verifier. The
SM70 grouped verifier now has an exact q32 specialization for the production
`Hq=6`, `Hkv=1`, `D=256`, E5M2 paged-KV shape:

- three two-head CTA groups retain 64 query rows per CTA;
- 27 context splits launch 81 CTAs, keeping one wave available across V100's
  80 SMs;
- the stream-local Graph workspace is `27 x 32 x 6 x 256` FP16 outputs plus
  FP32 LSE, and the dynamic shared-memory contract stays within V100's 96-KiB
  opt-in limit;
- every proposed row is still evaluated by target attention and combined with
  online-softmax LSE. There is no direct-output or skipped-verification path.

Admission is limited to a seven-token DFlash2 checkpoint with target draft
widths 7, 15, or 31. Other widths and cache layouts retain their old backend.
At the q32 production page size 3,776, the isolated kernel is `2.71x` faster
than the row-wise baseline at 1K context and `3.90x` at 32K. Maximum absolute
error against the existing FP32 oracle is `1.53e-5`.

## Validation and measured boundary

The targeted DFlash2 suite passes `128` tests. It covers controller
entry/hold/reject/cooldown, stale-buffer clearing, full-width point masses,
environment defaults, routing, and lookup CUDA behavior. The full-vocabulary
draftless lookup microbenchmark on V100 reports:

After the TP-lifetime repair, the two focused DFlash2 files rerun as
`109 passed, 15 skipped`; eight controller/lifetime tests pass when selected
alone. The q32 kernel and backend-policy suites pass 29 GPU numerical/Graph
cases and eight policy cases.

| Context | LABD graph | Drafter-free eager proposal |
| ---: | ---: | ---: |
| 1K | 0.0099 ms | 0.1069 ms |
| 32K | 0.1245 ms | 0.1322 ms |
| 128K | 0.4651 ms | 0.4714 ms |

The paired end-to-end contract uses the production NVFP4 target, official
BF16 DFlash2 draft, TP4 V100, FP8 E5M2 target KV, FP16 draft KV, Flash-V100,
CUDA Graphs, prefix caching, Mamba align, 4,096 scheduler tokens, q15 adaptive
LABD, and greedy target sampling. Both arms use separate compilation caches.

| Request | Chain off | Chain on | Interpretable result |
| --- | ---: | ---: | --- |
| ordinary text | 139.63 tok/s | 156.60 tok/s | not attributable: output diverges at token 123 and the chain never engages |
| short repeated-context copy, q16 | 359.76 tok/s | 363.69 tok/s | +1.09%; all 512 output token IDs are identical |
| frozen syv 25K document, q32 | 337.107 tok/s | **359.770 tok/s** | **+6.72%**; all 512 output tokens and text SHA are identical |

The copy request takes 36 speculative rounds in both arms, with mean
acceptance length `14.222`. The chain engages on about 25 rounds. The observed
decode-time reduction is about `14.9 ms`, or `0.59 ms` per engaged round. This
is the actual remaining neural-query cost on the TP4 path; the target-hidden
projection and draft context-KV maintenance intentionally remain.

The result also explains why the upstream headline must not be generalized.
Its `381 tok/s` number is a 25K-context verbatim-document task at `14.97`
tokens per round, not ordinary chat. That implies about `39.3 ms` per q16
round. The local copy arm is about `39.5 ms` per round before chaining, so its
q16 target-verification service is already in the same range. The shorter
local prompt needs more q8/transition rounds and reaches `14.22`, which is the
larger gap to the headline. The frozen upstream-style 25K rerun closes that
uncertainty. It keeps the same 512-token output SHA256
`7d69c86ce0b10ca95d40ce33cbce8d798d9a215a8fbf2b4b9b4dc849a6dda55c`,
the same 28 rounds, 497 accepted draft tokens, and mean acceptance length
`18.75`. Pure decode moves from `1.51584` to `1.42035 s`. This reproduces the
upstream chain's roughly seven-percent workload-specific gain on TP4 V100.
Cold-request wall time remains prefill dominated (`9.336 -> 9.391 s`) and is
not credited as an end-to-end win. Repeating the identical cached prefix gives
`1.114 s` TTFT, `359.37 tok/s` decode, and `2.535 s` total wall time.

The first TP4 implementation also exposed a correctness race after a chain
ended. Request-state slots were used as controller identities even though the
scheduler reuses them across requests, and TP ranks independently queried D2H
event readiness before deciding whether to skip the neural draft graph. That
could make only a subset of ranks enter the drafter-free collective sequence
and eventually feed an invalid token ID to the selector. The retained fix:

- keys every pending lookup/rejection verdict by stable request ID rather than
  a reusable slot index;
- fences a pending entry/rejection event before a host branch so every TP rank
  selects the same collective sequence;
- tags rejection feedback with the proposal that produced it, preventing the
  normal proposal that admitted a chain from immediately terminating its first
  drafter-free step.

No token-ID clamp or selector-bound weakening is used. The cleaned route runs
the complete 18-turn iterative coding stress (`8,325` generated tokens) with
no assert or HTTP 500; mean acceptance length is `7.513`. This is a lifetime
and stability gate, not a scored coding-quality result. Chain remains opt-in
and greedy-only; production sampling, tools, and structured outputs keep the
neural drafter and the previously accepted quality path.

Artifacts are rooted at
`/data/minimax-h3/task-cache/v100-dflash2-labd-chain-20260828/`:

- `chain-overhead-sm70.json` is the lookup/proposal microbenchmark;
- `control-v2-greedy-q15-o512.json` is the chain-off end-to-end arm;
- `candidate-v1-greedy-q15-o512.json` is the chain-on arm.
- `results/q31-grouped-syv-labd-copy25k-o512.json` is the frozen q32 chain-off
  25K document control;
- `results/q31-chain-exact-q32-qpn2-clean-syv-copy25k-o512.json` is the matched
  cold chain-on result;
- `results/q31-chain-exact-q32-qpn2-clean-syv-copy25k-prefixhit-o512.json` is
  the identical-prefix rerun;
- `results/q31-chain-exact-q32-qpn2-clean-iterative18-o512.json` is the cleaned
  cross-request lifetime stress.

## Next gates

1. Extend the frozen 25K document control from verbatim reproduction to the
   upstream six-task LABD matrix. Report q8/q32 rounds, chain engagement,
   tokens per round, and pure decode separately; ordinary prose may not
   regress.
2. Capture a graph-node/NVTX trace and split the remaining draft stage into
   target-hidden projection, input/slot preparation, context-KV projection and
   insertion, and query/selector graph. A larger chain win requires moving
   cache maintenance off the critical path; merely skipping the already-small
   TP4 query graph cannot supply it.
3. Evaluate an event-ordered maintenance stream only if the trace shows useful
   overlap with target verification. The draft cache must be complete before
   re-entering the neural path, and output tokens plus post-chain acceptance
   must match the warm-cache control.
4. Run B1/B2/B4 distinct-prefix and shared-prefix concurrency ladders. Report
   per-stream decode, aggregate decode, milliseconds per forward, resident
   requests, KV occupancy, and preemptions. Drafter-free chaining remains B1;
   batching and hybrid recurrent-state capacity are separate optimization
   problems.
