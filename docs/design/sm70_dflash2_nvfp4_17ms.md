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

## Acceptance and runtime-binary audit

The apparent NVFP4 acceptance regression was reproduced on the frozen 64-row
GSM8K order, but it is not caused by NVFP4. With the same current source and
the same official stochastic contract, FP8 measured request-mean/pooled
acceptance `4.49887/3.89772` and 60/64 accuracy; NVFP4 measured
`4.53642/3.79374` and 59/64. The request mean is slightly higher for NVFP4.
Its lower pooled value comes from a longer generated trajectory and is not a
valid quantization comparison by itself. The corresponding request-mean
completion tokens per verification round are `4.49761` and `4.53214`; this
closely related metric was previously mislabeled as request-mean acceptance.
All future records report both fields explicitly.

Both current paths are nevertheless below the repaired August 22 FP8 baseline
of `5.38880/4.56444` (`5.38884` request-mean completion tokens per round).
This is a real regression rather than only output-length mix: current FP8 is
lower on 61 of the 64 matched prompts, with paired
request-mean delta `-0.89123` and bootstrap 95% interval
`[-1.02152,-0.75597]`. Restricting each run to outputs of at most 512 tokens
still moves the mean from `5.60920` to `4.63777`.

Single-variable runs exclude the target quantized operators and compact
sampler as the main cause. Disabling NVFP4 QPN2 on the fixed 16 rows measured
`4.37420` request mean instead of recovering the current `4.57520` control.
Disabling FP8 target QPN8 on all 64 rows measured `4.49095/3.80785`, and
disabling sparse target rejection measured `4.51820/4.02970`; none approaches
the repaired baseline. All completed 64-row variants retained 60/64 FP8
accuracy.

The causal fault is binary provenance. The low-acceptance runs loaded
`_C_stable_libtorch` SHA256 `fe986a...9ba8c`, even though the checked-out source
contains the batched-weight RMSNorm repair. That binary predates the per-layer
weight implementation: its focused GPU suite fails 10 of 13 cases, silently
uses weight row zero for the other draft layers, and mismatches 81--96% of
elements on multi-layer fixtures. The repaired binary SHA256
`7689e5...d449` passes all 13 cases bitwise. The low result also numerically
matches the historical pre-repair `4.5395/3.9725` baseline.

The source now verifies the loaded stable extension once with deterministic
nonzero per-layer inputs. A conforming binary continues to use the single
grouped kernel after warmup. A stale binary emits an explicit warning and uses
the bitwise per-layer fallback, so a wheel/source or worktree/symlink mismatch
cannot silently lower acceptance again. The new fallback tests pass for both
runtime behaviors; the full CPU DFlash2 suite reports 70 passed and 12 skipped.

The repaired-binary FP8 closure uses the same frozen 64-row GSM8K contract and
the dense selector baseline. It records request-mean acceptance `5.37699`,
pooled acceptance `4.62217`, and request-mean completion tokens per round
`5.37814`, with 61/64 correct and 62/64 natural stops. Against the repaired
August 22 baseline, paired request-mean delta is `-0.01180` with bootstrap 95%
interval `[-0.08421,0.06282]`; every position-wise pooled acceptance value is
higher. The artifact records stable-extension SHA256 `7689e5...d449`. This
closes the large acceptance regression: the stale-binary run was lower by
`0.87812` request-mean tokens on the same rows, with bootstrap 95% interval
`[0.75099,1.00344]` after repair. Its concurrently collected throughput is not
a standalone performance baseline because NVFP4 compilation occupied the
other TP4 group.

The matching repaired-binary NVFP4 run records request-mean/pooled acceptance
`5.39851/4.57051`, request-mean completion tokens per round `5.40178`, and
59/64 correct with 62/64 natural stops. Its request mean is `+0.02152` versus
the repaired current-source FP8 run; the paired 95% interval
`[-0.10021,0.14795]` does not distinguish the two quantizations. The lower
NVFP4 pooled value is explained by its longer output trajectory (35,366 versus
32,658 tokens), not lower per-request acceptance. Relative to the stale NVFP4
run, the repair adds `0.86209` request-mean tokens with 95% interval
`[0.74547,0.98007]` and improves 62 of 64 matched rows.

The experimental draft-MLP QPN8 route is rejected independently. On the same
fixed 16 requests it changed request mean `4.57520 -> 4.51123` and pooled
acceptance `4.39123 -> 4.09891`; the request delta `-0.06397` exceeds the
`-0.05` gate. Its code and benchmark hook were removed rather than retained as
a nominal 17 ms result.

The corrected 16K coding-quality pair uses the same NVFP4 target, MBPP-32
prompts, request seed, graph mode, E5M2 KV, and natural-EOS sampling on both
sides. The dataset tests score no-DFlash `31/32` and DFlash2 `32/32`.
EvalPlus maps 31 of those rows: base is `30/31` versus `31/31`, and plus is
`27/31` versus `27/31`. At the plus level, each route alone passes one
different row; exact paired McNemar `p=1.0`. Both routes naturally stop on 31
rows and reach the 16K cap on one row. This provides no evidence of a DFlash2
quality regression under the repaired runtime.

The first score incorrectly reported one DFlash result as invalid syntax even
though the complete executable answer followed `</think>`. The extractor had
searched illustrative reasoning-era code fences before checking an unfenced
final answer. The scorer now gives the final answer precedence, its regression
fixture parses the affected `comb_sort` result, and both routes were rescored
with the same corrected extractor. The corrected DFlash results are dataset
tests `32/32`, EvalPlus base `31/31`, and EvalPlus plus `27/31`.

The opt-in selector-QPN8 candidate also passes the fixed 64-row acceptance
gate. Relative to the repaired dense selector, request-mean acceptance changes
`5.39851 -> 5.43951`, pooled acceptance `4.57051 -> 4.58679`, and GSM8K
`59/64 -> 60/64`. The paired request delta is `+0.04101` with bootstrap 95%
interval `[-0.02309,0.10669]`; 23 rows improve, 15 fall, and 26 tie. It
therefore has no acceptance-loss signal.

Its first clean directional speed signal is positive but not promotable: the
dense run on physical GPUs 4--7 measures request-mean/median complete round
`19.083/19.046 ms`, while QPN8 on GPUs 0--3 measures `18.077/18.029 ms`.
Aggregate output is `218.58 -> 229.41 token/s` and mean steady decode is
`282.68 -> 300.12 token/s`. Because these are different physical groups, they
are not a valid paired speed result. QPN8 also lowers available KV capacity
from 529,616 to 459,817 tokens through its prepared rerank layout. An unrelated
eight-GPU job occupied both groups before the same-GPU repeat; the candidate
remains explicit opt-in and is not a default or a credited 17 ms result.

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

Test result: the runtime-provenance acceptance repair and paired MBPP quality
gate pass. Selector-QPN8 passes the acceptance gate and has a cross-group
directional speed signal, but remains opt-in pending a same-GPU endpoint pair.
The 17 ms endpoint and long-context performance gates remain pending; no
rejected or unpaired speed candidate is credited.
