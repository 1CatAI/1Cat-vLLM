# SM70 Qwen3.8 NVFP4 DFlash2 17 ms Campaign

> **Status:** planning record only. No source optimization or 17 ms result is
> claimed by this document. The historical baseline below is retained to make
> the next implementation reproducible.

## Scope and frozen baseline

This campaign targets the complete batch-one DFlash2 speculative round measured
with a Qwen3.8-27B NVFP4 workload on four V100 GPUs. Its historical baseline is
`onecat/main@34403018d917054dd7765d5e820ad29c8d342348`; the planning record was
integrated on `main@cadcf1d899b6d7511f815e7ee939b1e4676aff19`. It does not
credit route-hit logs as performance evidence.

The retained benchmark artifacts identify a mixed NVFP4/channel-FP8 target
with a BF16 LM head and a DFlash2 draft at revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. Private filesystem locations are
intentionally not recorded. The frozen workload uses TP4 on an owned four-V100
group, probabilistic DFlash2 with seven draft tokens, selector K=16, target
top-k=20, FP8 E5M2 target KV, FP16 draft KV, Flash-V100 for target and draft
attention, and FULL target and draft CUDA Graphs. Model/checkpoint labels are
benchmark evidence only and must not activate an optimized runtime route.

Two runtime contracts are deliberately kept separate:

1. The localization contract uses a 32K maximum model length, 512 maximum
   batched tokens, and disables prefix caching, Mamba alignment, and parsers.
   It is used only for short unprofiled repeats, graph-node traces, and
   exact-shape microbenchmarks.
2. The promotion contract uses a 256K maximum model length, 4096 maximum
   batched tokens, prefix caching, Mamba alignment, and the Qwen tool and
   reasoning parsers. A speedup is not accepted until it reproduces under this
   practical configuration.

The current accepted practical baseline on PR #288 is `18.465--18.537 ms` per
complete round for 512-token coding requests and `18.587--18.603 ms` for
1,024-token coding requests. The high-acceptance MBPP item 28 baseline is
`18.567 ms`, acceptance length `4.686`, and `251.60 token/s`, with a natural
EOS and EvalPlus base/plus `1/1`. A fresh current-source 16-prompt GSM8K run
measures request-mean acceptance `4.45732`, pooled acceptance `4.07740`, and
`19.3688 ms` per round with diagnostic counters enabled. Diagnostic timing is
not a production baseline.

## Acceptance gates

A short-context candidate is accepted only when all of the following hold:

- the no-diagnostic promotion contract measures a mean and median complete
  speculative round at or below `17.0 ms` across at least three independent
  steady-state requests; the p90 must be at or below `17.5 ms`;
- the same fixed prompt, seed, sampling parameters, output cap, CUDA Graph
  shapes, and GPU set are used for the baseline/candidate pair;
- paired request-mean acceptance does not fall by more than `0.05`, the
  per-position counters remain healthy, and no stale-buffer, graph-replay, or
  draft-KV mismatch appears;
- the existing GSM8K, MATH-500, HumanEval, corrected MBPP, and WikiText PPL
  gates do not regress versus target-only and the accepted DFlash2 baseline;
- every retained performance optimization has an exact engine-contract
  admission predicate, a rollback switch, focused numerical coverage, and an
  unprofiled endpoint win. Once these gates pass, the optimized path is the
  default; rollback remains for diagnosis rather than permanent opt-in.

Long-context work is evaluated at 32K, 128K, and 256K. The candidate must not
make complete-round latency or pure-decode throughput worse by more than 1%
at any length. Numerical probes require finite outputs and a dtype-appropriate
absolute/relative error envelope; official-sampling quality and acceptance
must remain non-regressing. Bitwise output or greedy-token identity is not a
promotion requirement when a changed but valid reduction order explains the
difference. The 17 ms target applies to the short round; long-context results
are reported as an explicit decay curve because the verifier attention cost
necessarily grows with resident context.

## Trace-first implementation sequence

1. Reproduce the current no-diagnostic NVFP4 baseline in this worktree and
   record source SHA, extension hashes, route logs, GPU clocks, and per-request
   round statistics.
2. Capture the smallest generation-only Nsight Systems trace that contains
   steady CUDA Graph replays. Split every round into draft, draft-to-target,
   target verification, and target-to-draft phases, then expand the target
   phase into exact kernel categories and launch counts.
3. Microbenchmark only the largest current-source bucket at its production
   M/N/K, TP, dtype, and graph shapes. A microbenchmark result is directional
   evidence, never an endpoint claim.
4. Implement the smallest exact change that removes the measured bottleneck.
   Rejected DDTree index reuse, skipped verification, generic auxiliary-stream
   overlap, variable K, and approximate reranking are not retried: they either
   changed output/acceptance or increased the complete round.
5. Promote a default only after focused CPU/GPU numerical tests, graph replay,
   three clean short repeats, the practical 256K configuration, the quality
   suite, and the 32K/128K/256K decay sweep pass.

The first trace should determine whether the remaining roughly `1.6 ms` is in
NVFP4 QPN2/QPN8 projections, draft non-causal KV work, target verification,
or scheduler/metadata boundaries. No saving is assigned to a hypothesis
before that trace and its exact-shape microbenchmark exist.

## Draft PR record

Purpose: reduce the complete Qwen3.8-27B NVFP4 DFlash2 round to 17 ms on SM70
without changing the sampling contract, acceptance, or task quality.

Test plan: unprofiled paired endpoint repeats; Nsight graph-node phase
breakdown; exact-shape microbenchmarks; focused numerical and CUDA Graph replay
tests; GSM8K/MATH-500/HumanEval/MBPP/PPL quality gates; 32K/128K/256K decay
sweep.

Test result: pending current-source baseline and trace.
