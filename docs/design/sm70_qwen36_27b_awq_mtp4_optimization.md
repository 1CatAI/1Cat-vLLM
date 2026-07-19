# SM70 Qwen3.6-27B-AWQ Native MTP4 Optimization Ledger

Date: 2026-07-10

This document is the decision ledger for quality-safe native MTP4 decode on
Qwen3.6-27B-AWQ, TP2, V100/SM70. It is separate from the NVFP4/TurboMind
operator ledger because its bottleneck is speculative verification economics,
not the Qwen3.5-NVFP4 decode GEMM route.

The architecture-specific target-verifier audit and newer-GPU fast-path matrix
are maintained separately in
`docs/design/qwen36_27b_awq_mtp_target_verifier_fastpaths.md`.

## Scope And Quality Contract

Accepted route:

- Qwen3.6-27B-AWQ with native MTP4, TP2 on two V100 GPUs.
- CUDA graphs enabled; eager mode does not count.
- Target full-attention uses Flash-V100 and Qwen GDN uses the quality-safe
  full-GDN/FlashQLA-SM70 route.
- Official sampling: `temperature=1.0`, `top_p=0.95`, `top_k=20`,
  `seed=20260620`.
- Long output must stop naturally and pass the established quality checks.

Do not enable `VLLM_SM70_QWEN_GDN_003_SPEC_ALLOW_DEEP_MTP=1` to claim speed.
That split-GDN diagnostic route can create self-consistent deep-MTP acceptance
and repeated output. It is not quality-safe for MTP3/MTP4.

## Accepted End-To-End Evidence

Artifacts:

- `bench_results/mtp_live_audit_20260710/mtp4_256k_awqdyn_live_macos6000_seed20260620.json`
- `bench_results/mtp_live_audit_20260710/nomtp_control/nomtp_256k_hot_macos6000_seed20260620.json`
- `bench_results/api_server_27b_awq_20260701/qwen36_27b_awq_mtp4_256k_profile_fix_port8001.log`
- `bench_results/mtp_static_vocab_20260710/static131k_256k_macos12000_seed20260620.json`

| route | measured output rate | output-token latency | result |
|---|---:|---:|---|
| no MTP, hot control | `54.462 tok/s` | `18.361 ms` | baseline |
| native MTP4, quality-safe | `80.111 tok/s` | `12.483 ms` | `1.47x` faster |
| native MTP4, static `131K` draft vocabulary | `97.698 tok/s` | `10.236 ms` | `1.22x` over full-vocab MTP4; env-gated |
| native MTP4, dynamic `98K + 2x512`, prefill `topk=2048` | `100.623 tok/s` | `9.938 ms` | first cold natural-stop gate passed; now the direct supported MTP default |

The MTP4 long run emitted `5,886` tokens and stopped naturally. Its accepted
tokens per speculative round, including the bonus token, were:

```text
A = 4.01637 output tokens / round
```

Therefore one observed MTP round costs `50.135 ms`. This is the correct unit
for verifier and draft costs; divide by `A`, not by four, to obtain output-token
cost.

The static-`131K` candidate emitted `6,063` tokens and stopped naturally under
the same official sampling parameters. It passed the established local
repetition/corruption checks and measured `A=4.08625`. Relative to the
full-vocabulary MTP4 evidence, output-token latency fell by `18.0%` and
throughput rose by `21.9%`. This is the first end-to-end candidate pass, not a
default-enable decision; broader prompts still need to prove shortlist
coverage and quality stability.

The dynamic `98K` candidate emitted `6,757` tokens and also stopped naturally.
Its four FP16 draft heads plus sampling measure `2.841 ms` p50 per round; a
one-time target-logits `topk=2048` bootstrap brings cold acceptance to
`A=4.01784`, a `1.6743%` loss from static `131K` and inside the agreed two-
percent limit. Its request rate is `2.99%` above static `131K`. This closes the
first cold single-prompt gate, but does not replace the required broader
quality suite. The later explicit default decision accepts the documented
per-prompt residual risk rather than claiming that the broad gate passed.

## Historical Full-Vocabulary MTP Cost Breakdown

This section records the pre-dynamic-vocabulary `80.111 tok/s` baseline. It is
not the current default-route breakdown; use the calibrated 2026-07-10 section
below for current numbers.

The event measurements below are rank-0 CUDA event spans from clean steady
`M=5` verifier rounds in the profile artifact above. They identify GPU critical
path work. They are not a sum over both TP ranks.

| bucket | ms / MTP round | ms / final output token | share of observed round | what it contains |
|---|---:|---:|---:|---|
| target verifier forward | `22.594` | `5.625` | `45.1%` | five-token Qwen target forward, including AWQ dense layers, GDN/full attention, TP work, and graph work inside the model forward |
| target logits | `2.040` | `0.508` | `4.1%` | target LM-head plus full-vocab TP logit gather; not communication alone |
| target rejection sampling | `1.400` | `0.349` | `2.8%` | top-k/top-p constraints, softmax, rejection/recovery, and bonus sampling |
| MTP draft GPU total | `12.135` | `3.021` | `24.2%` | four MTP heads, draft LM-heads, and draft sampling |
| bookkeeping GPU | `0.082` | `0.020` | `0.2%` | small post-sampling GPU bookkeeping |
| uninstrumented round remainder | `11.884` | `2.959` | `23.7%` | GDN state postprocess GPU work, staging, scheduler/queue handoff, and graph-boundary work not enclosed by the current event spans |
| **observed round wall** | **`50.135`** | **`12.483`** | **`100.0%`** | `1 / 80.111 tok/s * A` |

The `0.536 ms` state-update CPU value in the same profile is host submission
time and is non-additive with queued GPU work. `draft_wall_cpu ~=34 ms` is also
not a draft cost: the profiling reporter synchronizes the final CUDA event
before reading timings, so that wall value includes already queued target and
draft GPU work. Neither number may be added to the table.

The `11.884 ms` remainder is real observed round wall, but it must not be
labeled as plain CPU overhead. Add CUDA events around GDN state postprocess and
capture an Nsight Systems graph-node trace before assigning it to a kernel
family.

## What Is Inside The Two Largest MTP-Specific Buckets

### Target Verifier Forward

Independent `M=5` AWQ verifier microbench evidence:

| measured dense subset | ms / verifier round |
|---|---:|
| 63 MLP gate/up projections | `6.838` |
| 63 MLP down projections | `5.554` |
| 47 GDN `in_proj_qkv` projections | `2.912` |
| 47 GDN `in_proj_z` projections | `2.539` |
| 47 GDN `out_proj` projections | `2.473` |
| 16 full-attention output projections | `0.819` |
| **283 measured AWQ GEMMs** | **`21.134`** |

The `out_proj` measurement is in
`bench_results/mtp_cost_breakdown_20260710/awq_m5_linear_attn_out_proj_gpu0.json`.
The 283-GEMM sum is a standalone per-rank CUDA-event estimate, not an
arithmetical decomposition of the captured `22.594 ms` graph event. It does
show that AWQ dense execution is overwhelmingly dominant. MLP AWQ projections
alone account for `12.391 ms`, or `54.8%` of target verifier forward. Attention
backend changes can affect only a minority of this bucket.

### MTP Draft GPU Total

The native proposer profile divides `12.135 ms` as follows:

| draft sub-bucket | ms / MTP round | share of draft GPU |
|---|---:|---:|
| four MTP layer forwards | `2.709` | `22.3%` |
| four full-vocab LM-head plus probabilistic sample paths | `9.136` | `75.3%` |
| proposer boundary, tensor stack/copy, and outer runner boundary | `0.290` | `2.4%` |

Under the official stochastic sampling contract, the proposer calls
`compute_logits` and `compute_probs_and_sample_next_token` for every draft
position. This is why the draft cost is mainly full-vocabulary sampling rather
than Triton attention. The greedy local-argmax fast path does not apply.

The stable inner-proposer profile is `11.965 ms` total: the four forward and
sample spans sum to `11.845 ms`, leaving about `0.120 ms` for index/copy/stack
work. The outer `GPUModelRunner` `draft_total` event is about `12.135 ms`; its
additional roughly `0.170 ms` is call-window boundary work. These numbers are
from independently averaged event windows, so use `0.290 ms` only as an
accounting bucket, not as one kernel to optimize.

Per-position stable events:

| draft position | MTP forward | LM-head + full-vocab sample | measured sum |
|---|---:|---:|---:|
| 0, first pass | `0.787 ms` | `2.284 ms` | `3.071 ms` |
| 1 | `0.650 ms` | `2.285 ms` | `2.935 ms` |
| 2 | `0.635 ms` | `2.283 ms` | `2.918 ms` |
| 3 | `0.637 ms` | `2.284 ms` | `2.921 ms` |
| **total** | **`2.709 ms`** | **`9.136 ms`** | **`11.845 ms`** |

The `sample` event performs a local FP16 LM-head GEMM on each TP rank,
all-gathers full logits, casts/scales logits, applies draft top-k, materializes
full-vocabulary FP32 probabilities, generates exponential random values,
selects the token, broadcasts the selected id across TP, and retains dense
draft probabilities for rejection sampling. It does not all-gather the dense
probabilities a second time.

Model-free microbench landmarks for one draft row:

| isolated operation | measurement |
|---|---:|
| stock FP16 local LM-head shape, `1x5120 @ 5120x124160` | `1.931 ms` mean |
| TP2 full-logit all-gather, one row | `0.093 ms` p50 |
| dense top-k proposal after full logits exist | `0.464 ms` p50 |

Artifacts:

- `bench_results/mtp_cost_breakdown_20260710/lm_head_tp2_local_m1_n124160_k5120_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/tp2_compact_draft_logits_m1_v100.json`
- `bench_results/mtp_cost_breakdown_20260710/draft_sample_m1_vocab248320_topk20_gpu0.json`

These three isolated windows must not be added to reproduce `2.284 ms`: the
live path runs under its captured graph and uses different allocation/launch
boundaries. They are component-ranking evidence. Exact additive splitting
requires two additional CUDA events around live `_compute_logits_for_step` and
`_sample_from_logits`.

## FlashInfer Assessment

The hypothesis "verification is high because current Triton/FlashAttention has
no FlashInfer MTP optimization" is incomplete and is not the primary cause on
V100:

- Upstream FlashInfer supports SM75 and newer. V100 is SM70, so it is outside
  the supported architecture range.
- FlashInfer's GDN MTP operator explicitly requires SM90. This project already
  selects FlashQLA-SM70 or Triton for the V100 GDN route rather than that
  unsupported operator.
- FlashInfer's speculative/XQA attention support could affect standard
  attention query processing, but it cannot remove the dominant AWQ MLP GEMMs,
  GDN forward work, four draft LM-heads, or full-vocabulary sampling.

FlashInfer's compact speculative-sampling design remains an algorithmic
reference. Its shipped kernels are not a drop-in V100 solution.

References:

- <https://github.com/flashinfer-ai/flashinfer>
- <https://docs.flashinfer.ai/api/attention.html>
- <https://docs.flashinfer.ai/generated/flashinfer.gdn_decode.gated_delta_rule_mtp.html>

## Compact TP Logit Transport: Real V100 TP2 Go/No-Go

The official sampler has `top_k=20`, so a natural idea is to exchange only
per-rank top-20 value/id candidates instead of gathering all `248,320` logits.
The project already has a generic compact top-k helper. Before routing it into
MTP, a real two-V100 NCCL microbenchmark was run on GPU0/1:

```text
CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$PWD \
  /home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  benchmarks/benchmark_sm70_tp2_compact_verifier_logits.py \
  --warmup 50 --iters 500 \
  --json-out bench_results/mtp_cost_breakdown_20260710/tp2_compact_verifier_logits_v100.json
```

| path, rows=`5`, TP2 | critical-path p50 | mean | correctness |
|---|---:|---:|---|
| full-vocab all-gather then global top-k | `0.290 ms` | `0.310 ms` | reference |
| local top-20 then two small all-gathers then merge | `0.479 ms` | `0.498 ms` | exact top-20 values and ids |

Transport shrinks from `1,241,600` to `1,200` bytes per rank, but latency does
not follow byte count at this scale. Component p50 values are:

| isolated component | critical-path p50 |
|---|---:|
| full FP16 logits all-gather | `0.121 ms` |
| global top-k over full gathered logits | `0.231 ms` |
| local top-k over `124,160` logits | `0.172 ms` |
| compact value all-gather | `0.090 ms` |
| compact index all-gather | `0.097 ms` |
| compact candidate merge top-k | `0.178 ms` |

The isolated rows are diagnostic and should not be summed: synchronization and
allocation/launch behavior differ from the compound path. The decision is
nevertheless clear for the current generic implementation: it is about `65%`
slower at p50 even before adding exact top-p normalization and rejection
probability handling.

**Decision: do not implement the generic two-collective compact-top-k path as
an MTP speed optimization on V100.** Reconsider only with a fused one-kernel or
one-collective representation that measures at least `0.4 ms` faster than the
current full path in this real TP2 microbench, then validate exact probability
semantics and long-output quality.

## Draft LM-Head Route Study

The four draft LM-head and sample calls are the first MTP-specific target. The
following estimator deliberately credits only an isolated LM-head projection
delta and leaves every other measured live cost unchanged:

```text
estimated draft ms/round = 12.135 - 4 * (1.930383 - candidate_projection_ms)
```

It is a conservative component model, not a measured end-to-end result. It
assumes unchanged acceptance and does not credit possible gains in TP
communication, softmax, or sampling. A route is not accepted until its live
CUDA-event draft span, accepted tokens per round, and long-output quality are
measured together.

### P0: Target-Generated Static Vocabulary

FR-Spec proves that a draft model can use a frequency-ranked subset while the
target verifier retains the full vocabulary. VocabTrim's stronger result is
more applicable here: rank tokens from responses generated by the target model
on representative workloads, rather than from a generic external corpus.

The V100 microbenchmark keeps `M=1`, `K=5120`, FP16, and changes only the local
TP vocabulary width:

| global subset, approximate | local TP width | isolated projection | estimated draft ms/round |
|---:|---:|---:|---:|
| `248,320`, current full vocab | `124,160` | `1.930 ms` | `12.135 ms` |
| `131,072` | `65,536` | `1.061 ms` | `8.657 ms` |
| `98,304` | `49,152` | `0.778 ms` | `7.524 ms` |
| `65,536` | `32,768` | `0.529 ms` | `6.528 ms` |
| `32,768` | `16,384` | `0.296 ms` | `5.596 ms` |

Artifacts:

- `bench_results/mtp_cost_breakdown_20260710/fr_spec_lmhead_m1_localvocab124160_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/fr_spec_lmhead_m1_localvocab65536_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/fr_spec_lmhead_m1_localvocab49152_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/fr_spec_lmhead_m1_localvocab32768_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/fr_spec_lmhead_m1_localvocab16384_gpu0.json`

The first implementation candidate is global `131,072`. Its projection-only
estimate already clears the `draft < 10 ms` phase-one target without changing
the MTP backbone or training new weights. It also leaves much more coverage
margin than the `32K` subsets used in several published experiments.

Implementation constraints:

1. Build the ranking from target-generated assistant responses that represent
   conversation, coding, reasoning, Chinese, and long-form workloads. Record
   token coverage and target top-20 probability-mass coverage separately.
2. Keep the target LM-head unchanged. Select one global shortlist, then assign
   its rows evenly to draft TP ranks. Do not select independent per-rank top-N
   lists: the measured coverage loss is unacceptable. Populate the balanced
   draft-only packed FP16 views at startup from the original target shards.
3. All-gather the reduced logits once, then map shortlist positions back to
   global token IDs. Do not use the slower two-collective compact path.
4. In rejection sampling, represent draft probability `q` as zero outside the
   shortlist. The target distribution `p` remains full-vocabulary. Never
   renormalize `p` onto the shortlist for the accept/recover calculation.
5. Preserve the current dense route as fallback for penalties, grammar,
   allowed-token constraints, logprobs, or any sampling feature whose sparse
   semantics are not explicitly implemented and tested.

Correct speculative rejection with full `p` and the actual subset-supported
`q` preserves the target output distribution. A smaller subset can still lower
acceptance, so distributional exactness does not by itself prove speed.

For the `131,072` estimate, the new round cost would be about `46.657 ms` if all
other costs remain fixed. The no-regression break-even is therefore:

```text
A_break_even = 46.657 / 12.483 = 3.738 accepted output tokens/round
```

The break-even value is diagnostic only. The acceptance target is the matched
512-token static-`131K` reference at `A_ref=4.327731`. The maximum allowed
relative loss is `2%`, so the hard gate is:

```text
A >= 0.98 * A_ref = 4.241176
```

Keep reporting both the distance from `A_ref` and the pass/fail result. A
candidate between break-even and `4.241176` is rejected. The full quality suite
and a natural-stop long output remain separate required gates.

#### Offline Coverage And TP Ownership Gate

`benchmarks/analyze_sm70_draft_vocab_coverage.py` was added to prevent a
misleading static-vocabulary implementation. It deduplicates existing target
outputs, builds deterministic rankings, evaluates held-out token coverage, and
reports the physical vocabulary ownership on each TP rank.

The expanded local corpus contains `209` distinct Qwen3.6-27B-AWQ outputs,
`46,119` output tokens, and `5,219` distinct token IDs. A secondary ranking for
zero-frequency output tokens uses `5,089,461` tokens from `680` representative
LongBench prompt rows. The held-out sample is the accepted `5,885`-token
natural-stop MacOS generation. This is a preliminary route gate, not a
production vocabulary corpus.

| `131,072` shortlist construction | physical rows rank0/rank1 | held-out token coverage | runtime decision |
|---|---:|---:|---|
| global top-N, original ownership | `111,496 / 19,576` | `99.524%` | coverage is promising, but max local width `111,496` misses the speed target |
| independent local top-65,536 | `65,536 / 65,536` | `94.206%` | reject; local quotas discard too many globally useful rank0 tokens |
| global top-N, reassigned ownership | `65,536 / 65,536` | same `99.524%` | selected P0 design; requires one-time row redistribution |

The selected route keeps global ranking quality and makes the measured
`N=65,536` local projection (`1.061 ms`) applicable. It needs an extra packed
draft head of about `640 MiB` per rank. With the measured original ownership,
rank0 must send about `45,960` selected FP16 rows to rank1 at startup; this is
roughly `471 MB`. The full target shards remain resident and unchanged.

A real two-V100 variable-split NCCL microbenchmark validated that startup
operation:

| startup-only operation | critical-path p50 |
|---|---:|
| gather selected rows from original local shard | `2.371 ms` |
| redistribute `470.6 MB` with variable-split all-to-all | `12.371 ms` |
| gather plus redistribution | `14.099 ms` |

The resulting packed head is exactly `671,088,640` bytes per rank. The
`14.1 ms` cost occurs once at model initialization and is not part of decode
TPOT.

For the reduced global logits (`131,072`, one row), real TP2 V100 timings are:

| operation | critical-path p50 |
|---|---:|
| one full reduced-logit all-gather | `0.115 ms` |
| all-gather plus global top-20 | `0.273 ms` |
| local top-20 plus two small all-gathers | `0.435 ms` |

The collective/top-k path has a strong launch-latency floor and does not shrink
in proportion to bytes. This does not invalidate P0 because the LM-head itself
saves about `0.869 ms` per draft position, but it again rules out the generic
two-collective route.

Artifacts:

- `bench_results/mtp_cost_breakdown_20260710/static_vocab_coverage_holdout_v1.json`
- `bench_results/mtp_cost_breakdown_20260710/static_vocab_coverage_all27b_v2.json`
- `bench_results/mtp_cost_breakdown_20260710/static_vocab_coverage_all27b_longbenchbg_v3.json`
- `bench_results/mtp_cost_breakdown_20260710/static_vocab_tp2_rankings_all27b_longbenchbg_v3.pt`
- `bench_results/mtp_cost_breakdown_20260710/static_vocab_tp2_logits_m1_vocab131072_v100.json`
- `bench_results/mtp_cost_breakdown_20260710/static_vocab_tp2_row_redistribution_131072_v100.json`

Token occurrence coverage is only a proxy for acceptance. Before live model
integration, capture real MTP hidden-state logits and measure shortlist target
mass, draft top-20 recall, KL on common support, and predicted rejection
acceptance under the official sampler.

Upstream does not currently provide the required stochastic TP path. The
original FR-Spec implementation copies reduced LM-head rows on one GPU and has
no TP/NCCL ownership handling. Current SGLang can load a hot-token map, but its
EAGLE worker explicitly rejects reduced draft vocabularies when rejection
sampling is enabled and leaves the dense-probability remap as a FIXME. Do not
port that route as proof of exact `top_k=20, top_p=0.95` behavior.

#### Exact Rejection Representation Gate

Under the accepted sampling policy, draft `q` has at most 20 nonzero entries
per position because the validated Qwen MTP proposal applies top-k 20 and does
not apply draft top-p. Target `p` also has at most 20 nonzero entries after
official top-k/top-p. Therefore exact rejection needs only draft
`(global_token_id, probability)` pairs and target top-20 pairs; positive
recovery mass can exist only on target support.

The model-free V100 proof compares a reduced `131,072`-logit proposal and
sparse-q rejection against dense `248,320`-entry q and p tensors:

| check | maximum absolute difference |
|---|---:|
| target probability at top IDs | `1.49e-8` |
| `p` at sampled draft ID | `1.49e-8` |
| `q` at sampled draft ID | `0` |
| recovery residual at target IDs | `1.49e-8` |
| recovery residual total mass | `7.45e-8` |

All checks pass the `1e-6` probability tolerance. The remaining differences
come from running softmax over 20 values versus a dense vector containing
`-inf`; the mathematical sampling distribution is the same.

The unfused timing result is equally important:

| operation | p50 |
|---|---:|
| reduced draft proposal, retain sparse q | `0.322 ms` |
| reduced draft proposal, scatter q to dense full vocab | `0.374 ms` |
| dense-q rejection core, four positions | `0.332 ms` |
| PyTorch sparse-q rejection core, four positions | `0.599 ms` |

**Phase-one decision:** retain a dense full-vocabulary q buffer and scatter
only the 20 reduced-proposal probabilities into it. This reuses the existing
quality-passing rejection sampler and still receives the dominant LM-head
saving. Do not enable the multi-kernel sparse-q rejection prototype. Revisit
sparse q only as one fused kernel with a go/no-go threshold below `0.25 ms` for
four positions.

Artifacts:

- `bench_results/mtp_cost_breakdown_20260710/static_vocab_sparse_q_rejection_v100.json`
- `benchmarks/benchmark_sm70_sparse_q_rejection.py`
- `benchmarks/benchmark_sm70_tp2_vocab_row_redistribution.py`

#### Live P0 Prototype Result

An env-gated implementation now loads the deterministic ranking artifact,
selects the global top `131,072` rows, redistributes them once at startup into
balanced `65,536`-row TP shards, and runs an independent FP16 draft LM-head.
The target LM-head and verifier are unchanged. Probabilistic draft sampling
maps sampled IDs back to the full vocabulary and expands the reduced q tensor
to the existing full-vocabulary rejection-sampler contract.

The first real Qwen3.6-27B-AWQ TP2 route hit used the accepted 256K,
Flash-V100 target, Triton drafter, MTP4, prefix-cache align, compile/FULL CUDA
graph configuration. Both ranks reported the same ranking fingerprint and
`local_rows=65536`. CUDA graph capture completed without eager fallback. With
multimodal profiling disabled, model memory was `12.93 GiB` per rank and the
runtime retained `15.27 GiB` of KV-cache memory, enough for `444,873` tokens.

The 512-token official-sampling profile produced these clean 32-round
intervals:

| metric | original accepted baseline | static `131K`, interval 1 | static `131K`, interval 2 |
|---|---:|---:|---:|
| outer `draft_total` | `12.135 ms` | `8.412 ms` | `8.507 ms` |
| four MTP forwards | `2.709 ms` | about `2.6-2.8 ms` | about `2.6-2.8 ms` |
| four LM-head + sample spans | `9.136 ms` | about `5.36 ms` | about `5.36 ms` |
| mean accepted output tokens/round | `4.016` | \- | `4.328` over the full probe |

The mean clean `draft_total` is `8.460 ms/round`: `3.675 ms` or `30.3%`
below the original `12.135 ms`. The dominant LM-head/sample bucket falls by
about `41.3%`. This clears the phase-one `draft < 10 ms` component target.
The run emitted all `512` requested tokens, used the official
`temperature=1.0`, `top_p=0.95`, `top_k=20`, `seed=20260620` policy, and
reported draft acceptance `83.19%` with mean acceptance length `4.328`.

The 512-token profiler result alone is not end-to-end evidence because it
synchronizes event completion every round, uses the offline `LLM` harness,
and ends at a length limit. The subsequent unprofiled chat-API run used the
accepted macOS prompt and official sampler. A `6,000`-token cap was too short:
the output remained clean but ended with `finish_reason=length`. Repeating the
same request with a `12,000` cap stopped naturally at `6,063` tokens, passed
all established quality checks, and measured `97.698 tok/s` (`10.236 ms/token`)
with `A=4.086`.

This result clears the phase-one component, acceptance, natural-stop, and
single-prompt end-to-end gates. Keep the feature env-gated until a broader
prompt/domain suite confirms that the preliminary ranking corpus is adequate.

Implementation and artifacts:

- `vllm/v1/spec_decode/static_draft_vocab.py`
- `tests/v1/spec_decode/test_static_draft_vocab.py`
- `bench_results/mtp_static_vocab_20260710/route_hit_27b_awq_mtp4_static131k.json`
- `bench_results/mtp_static_vocab_20260710/route_hit_27b_awq_mtp4_static131k.log`
- `bench_results/mtp_static_vocab_20260710/profile_27b_awq_mtp4_static131k_macos512.json`
- `bench_results/mtp_static_vocab_20260710/profile_27b_awq_mtp4_static131k_macos512.log`
- `bench_results/mtp_static_vocab_20260710/static131k_256k_macos6000_seed20260620.json`
- `bench_results/mtp_static_vocab_20260710/static131k_256k_macos12000_seed20260620.json`
- `bench_results/mtp_static_vocab_20260710/api_static131k_port8001.log`

Reproduction controls:

```text
VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING=<absolute path to ranking .pt>
VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE=131072
```

Both variables are required and the feature remains off by default. The
ranking artifact must report the same TP size and full model vocabulary size
as the runtime. The API-server path can obtain MTP4 from
`VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1`. The offline
`benchmark_sm70_model_tokens.py` `LLM(...)` path does **not** apply that
server-only automatic MTP choice; it must also receive an explicit
`speculative_config` engine argument. The discarded first route-hit attempt is
preserved as
`bench_results/mtp_static_vocab_20260710/invalid_direct_llm_no_mtp.log` so a
future run does not mistake `speculative_config=None` for a static-vocabulary
failure.

#### FP16 Head Fusion Bound For A Three-Millisecond Bucket

The next target is specifically the four draft `LM-head + sample` spans, not
the whole MTP round:

```text
5.36 ms / four positions -> less than 3.00 ms / four positions
```

Draft-head W4 quantization is not part of this route. The retained design keeps
FP16 weights and evaluates whether the GEMV, local top-20 reduction, TP merge,
softmax, and random sample can be fused.

The existing SM70 TurboMind FP16 top-1 epilogue is an optimistic proxy for a
future top-20 epilogue. Real V100 timings are:

| global shortlist | local rows/rank | `torch.mm` only | fused FP16 head/top-1 proxy |
|---:|---:|---:|---:|
| `131,072` | `65,536` | `1.062 ms` | `0.694 ms` |
| `98,304` | `49,152` | `0.778 ms` | `0.520 ms` |
| `65,536` | `32,768` | `0.528 ms` | `0.356 ms` |

For `131K`, one packed FP32 `(value, token_id)` top-20 TP all-gather costs
`0.098 ms` p50. Merging 40 TP candidates, applying softmax, and sampling costs
`0.169 ms` p50 when q remains sparse (`0.211 ms` when expanded to full q).
Token IDs are below `2^24`, so the FP32 packed representation preserves them
exactly; the benchmark matched the full-logit global top-20 values and IDs.

The optimistic, sparse-q lower bound is therefore:

```text
(0.694 fused top-1 proxy + 0.098 packed TP gather + 0.169 finalize) * 4
= 3.844 ms
```

This already exceeds three milliseconds, and a real top-20 epilogue cannot be
cheaper than the top-1 proxy. Therefore the current `131K` FP16 head cannot
reliably meet the target through kernel fusion alone. The same bound is about
`3.15 ms` at global `98K` and `2.49 ms` at global `65K`.

The `65K` static ranking has only `92.76%` held-out token occurrence coverage,
versus `99.52%` at `131K`; simply truncating to `65K` risks acceptance loss.
The viable no-quantization route is a roughly `65K` FP16 base head augmented
with a GPU-resident context-dependent tail vocabulary, followed by a fused
FP16 top-20 epilogue, one packed TP exchange, and sparse-q rejection. This is
an exact target-distribution route when rejection uses the actual sparse q,
but acceptance and long-output quality remain hard gates.

Artifacts:

- `bench_results/mtp_cost_breakdown_20260710/fp16_lm_head_top1_m1_n65536_v100.json`
- `bench_results/mtp_cost_breakdown_20260710/fp16_lm_head_top1_m1_n49152_v100.json`
- `bench_results/mtp_cost_breakdown_20260710/fp16_lm_head_top1_m1_n32768_v100.json`
- `bench_results/mtp_cost_breakdown_20260710/fp16_static131k_tp2_compact_top20_packed_v100.json`
- `benchmarks/benchmark_sm70_tp2_compact_verifier_logits.py`

### P1A: Draft-Only TurboMind W4 LM-Head

A synthetic TurboMind W4A16 M1 benchmark was added specifically for the draft
LM-head shape. This is the TurboMind route, not Marlin.

| local TP width | TurboMind W4 projection p50 | FP16 projection | speed ratio |
|---:|---:|---:|---:|
| `124,160` | `0.368 ms` | `1.930 ms` | `5.25x` |
| `65,536` | `0.204 ms` | `1.061 ms` | `5.21x` |
| `49,152` | `0.162 ms` | `0.778 ms` | `4.81x` |
| `32,768` | `0.102 ms` | `0.529 ms` | `5.16x` |

The full-vocabulary W4 projection has `0.376 ms` mean and gives a mean-based
draft estimate of `5.918 ms/round` (`5.884 ms` if p50 is substituted).
Prepared full local weights use `317,849,600` bytes plus `19,865,600` scale
bytes.

Artifact:

- `bench_results/mtp_cost_breakdown_20260710/draft_lm_head_awq_turbomind_m1_k5120_gpu0.json`

This result proves only the hardware opportunity: the benchmark uses synthetic
weights. A usable path needs a draft-only quantized copy of the real LM-head,
real MTP hidden-state logits, and measurements of KL divergence, top-20
overlap, acceptance, and long-output quality. Quantizing `q` need not change
the final target distribution when rejection uses that exact `q`, but it can
reduce acceptance. Do not replace or quantize the target verifier LM-head.

### P1B: NanoSpec/MicroSpec Dynamic Vocabulary

The released repository calls the method NanoSpec; the latest paper title uses
MicroSpec. It constructs a context-dependent active vocabulary from prompt
tokens, recent tokens, previous draft candidates, and target verification
top-k tokens. Rows are gathered into a contiguous buffer asynchronously while
the next draft backbone step runs.

The V100 row-gather and reduced-head microbenchmark used a real full-shape
synthetic tensor `[124160, 5120]`:

| active local rows | row gather p50 | packed FP16 projection p50 |
|---:|---:|---:|
| `512` | `0.044 ms` | `0.056 ms` |
| `1,024` | `0.047 ms` | `0.054 ms` |
| `1,536` | `0.055 ms` | `0.061 ms` |
| `2,048` | `0.063 ms` | `0.065 ms` |
| `4,096` | `0.093 ms` | `0.077 ms` |
| `8,192` | `0.179 ms` | `0.141 ms` |
| `16,384` | `0.352 ms` | `0.290 ms` |

At about `3K` global active tokens, each rank gathers `15.7 MB` at about
`284 GB/s`. The `0.055 ms` gather is much shorter than the measured
`0.635-0.787 ms` MTP backbone step, so a separate copy stream has enough
overlap window. Crediting only the reduced projection gives an estimate near
`4.661 ms/round`, before any unhidden map-management cost.

Additional TP2 and sampler findings for global active vocabulary `3,072`:

| operation | critical-path p50 |
|---|---:|
| one full small-logit all-gather | `0.104 ms` |
| all-gather plus global top-20 | `0.185 ms` |
| generic compact two-collective route | `0.349 ms` |
| current generic dense proposal over only `3,072` logits | `0.390 ms` |

The first implementation should therefore all-gather the small logits once.
The sampler's roughly `0.39 ms` launch/multi-kernel floor does not shrink with
vocabulary size, so a fused sparse top-k/top-p proposal kernel is required to
realize the rest of the gain.

Artifacts:

- `bench_results/mtp_cost_breakdown_20260710/dynamic_vocab_lm_head_m1_local124160_k5120_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/dynamic_vocab_tp2_logits_m1_vocab3072_v100.json`
- `bench_results/mtp_cost_breakdown_20260710/dynamic_vocab_draft_sample_m1_vocab3072_topk20_gpu0.json`

Do not copy the released Python control path directly. Source inspection found
`.tolist()`, Python `set` updates, and CPU-side active-token maintenance in the
generation path, while one eviction path is commented out. Those operations
would add synchronization and graph breaks. The local design must keep the
active-ID set, deduplication, recency state, and row map on GPU. The published
evaluation also does not establish our exact `top_p=0.95, top_k=20` rejection
contract, so stochastic correctness remains a hard gate.

#### Dynamic Base-Tail Experiments, 2026-07-10

The matched acceptance reference was rerun with complete top-20 alignment
dumps. It reproduced `A_ref=4.327731` over `119` rounds and `476` draft
positions. Unlike the earlier top-10 diagnostic, every dumped draft q row sums
to approximately one under the official `top_k=20` sampler. Candidate updates
exclude target `-inf` padding and zero-probability draft entries.

The acceptance target and hard floor are therefore:

| gate | value |
|---|---:|
| matched target | `4.327731` |
| maximum relative loss | `2.00%` |
| hard floor | `4.241176` |

Baseline-trajectory replay looked promising: a `65K` base plus previous finite
target top-20 and accepted-output LRU covered `99.435%` of complete q mass and
predicted `A=4.21995` versus a same-trajectory estimate of `4.21812`. The live
tests showed that this counterfactual is insufficient because a restricted q
changes the sampled trajectory; future cold misses are then no longer measured
by the original trajectory.

| live host-managed quality prototype | measured A | loss from reference | gate |
|---|---:|---:|---|
| static `65K` floor | `3.68345` | `14.89%` | fail |
| dynamic `65K + 512` global LRU | `3.84962` | `11.05%` | fail |
| dynamic `65K + 4096` global LRU | `4.19672` | `3.03%` | fail |
| `65K + 4096`, full-head discovery every 64 rounds | `4.14516` | `4.22%` | fail; do not repeat |
| dynamic `98K + 512` global LRU | `4.32773` | `0.00%` | **pass** |
| output-leaking oracle `65K + 512` | `4.29412` | `0.78%` | numerical pass only; invalid production evidence |

The host-managed path remains an env-gated acceptance prototype, not speed
evidence. It keeps CUDA graphs enabled but uses CPU LRU maintenance and host
synchronization. Production integration must replace that control path with
fixed GPU buffers and graph-safe updates.

The fused TP2 microbenchmark now includes all previously missing costs:
non-contiguous reduced-row to full-token mapping, per-rank tail row refresh,
four sequential FP16 top-20 proposals, packed TP exchange, probabilistic
sample/broadcast, and one expansion of four sparse q rows to the current
`[4,248320]` rejection contract. The official target sampler remains
`temperature=1.0, top_p=0.95, top_k=20`. The matched draft-q baseline applies
top-k 20 but not top-p (`draft top_p=1.0`); standard rejection sampling
preserves the target distribution.

| configuration | full graph p50 | p90 | p99 | result |
|---|---:|---:|---:|---|
| `65K base + 512 rows/rank` | `2.031 ms` | `2.044 ms` | `2.071 ms` | speed pass, acceptance fail |
| `98K base + 512 rows/rank`, 20-value RNG | `2.805 ms` | `2.825 ms` | `2.840 ms` | numeric sampler pass, full-model acceptance fail |
| `98K base + 512 rows/rank`, baseline-width `99,328` RNG | `2.841 ms` | `2.852 ms` | `2.869 ms` | **speed and short acceptance pass** |

For `98K + 512/rank`, mapped IDs, top-20 values, sparse IDs, sampled token, and
dense-q row sums all match the reference path. A controlled `top_p=0.95` case
keeps `19` of `20` candidates; fused and PyTorch sparse IDs match, maximum
probability error is `7.45e-9`, and the sampled token matches. However, drawing
only 20 exponentials changes Philox consumption relative to the matched
reduced-vocabulary sampler. The first fused full-model run therefore diverged
after 23 output tokens and fell to `A=4.079365` (`5.74%` loss). This path is a
recorded dead end even though its probability distribution is valid.

The accepted kernel transports `(logit, full_token_id, reduced_row)` for each
candidate and draws the same `99,328` exponentials per draft step as the
baseline, while still avoiding full LM-head materialization and full softmax.
The `[1,99328]` baseline and `[99328]` fused RNG buffers match exactly for both
the current and next generator fills. The graph timing includes tail refresh,
four proposals, baseline-width RNG, and dense-q expansion; all 100 measured
iterations remain below `2.897 ms`.

The matched full-model host prototype also returns exactly the same `512`
output token IDs as static `131K` (SHA-256
`056e07c7f6f4dd14b83ae080dd5f92fa0202c6df203b3c67c227a332cf7df2e3`) and
exactly reproduces all acceptance counters: `[111,110,94,81]`, `396/476`
accepted drafts, and `A=4.327731`. Its `5.856 s` generation wall time is not
accepted speed evidence because host-managed LRU synchronization remains in
that prototype.

The RNG-aligned fused runtime reproduces the same 512 token IDs and text, the
same hash, and every acceptance counter. It records `A=4.327731`, `5.400 s`
generation wall time, `9.906 ms` request TPOT, and `100.947 tok/s` steady
decode. These wall figures are directional only: tail discovery and LRU/map
maintenance are still host-managed, so the production speed gate remains
blocked on a fixed-buffer GPU LRU implementation.

#### GPU-resident LRU status

The fixed-buffer GPU route was initially implemented behind
`VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_GPU_LRU=1` and is now selected by the
default MTP resolver. It is hard-guarded to FP16,
TP2, MTP4, one scheduled sequence, no periodic full-head refresh, the fused
proposal path, a `98,304`-row base, and `512` physical tail rows per rank. It
keeps the accepted `99,328`-value Philox consumption and does not enable eager
execution.

The first serial CUDA implementation was rejected: LRU update alone measured
`1.512 ms` p50 and a full refresh measured `1.480 ms` p50. The accepted
block-parallel kernel uses one 256-thread block, fixed shared-memory buffers,
parallel lookup/shift, and stable local compaction. The TP2 microbenchmark
checks 32 sequential updates against the host reference, including LRU order,
local IDs, source rows, and refreshed FP16 weights.

| GPU LRU component | p50 | p99 |
|---|---:|---:|
| LRU update plus local compaction | `0.140 ms` | `0.162 ms` |
| local tail-weight refresh | `0.038 ms` | `0.063 ms` |
| tail-ID all-gather plus map update | `0.155 ms` | `0.214 ms` |
| complete GPU LRU refresh | `0.203 ms` | `0.318 ms` |

On the 512-token short gate, GPU LRU exactly reproduces the accepted host-LRU
token IDs, text, and acceptance counters: `A=4.327731`, accepted positions
`[111,110,94,81]`, and `396/476` accepted drafts. Request TPOT falls from
`9.906 ms` to `8.395 ms`, while steady decode rises from `100.947 tok/s` to
`119.117 tok/s`. This is a valid short speed result, not yet the long-output
acceptance result.

The first 256K natural-stop chat run produced 6,130 tokens, stopped normally,
and passed the output-quality gate at `99.522 tok/s`, but its cold-state
acceptance was only `A=3.956746`. Relative to the matched static-131K baseline
`A=4.086253`, this is a `3.17%` loss and fails the required two-percent gate.
A second request on the same loaded service passed at `A=4.068040` (a `0.45%`
loss), `101.946 tok/s`, and normal quality. This isolates the remaining issue
to cold-state vocabulary coverage rather than steady-state LRU capacity.

The cold state was already seeded from the ranking; it was not empty. For the
current ranking, however, all 512 IDs in `ranking[98304:98816]` belong to TP
rank 1. Rank 0 therefore computes 512 reserved tail logits against inactive
rows. The next microbenchmark candidate is a shard-local 512-entry LRU on each
rank. It can activate up to 1,024 global tail IDs while preserving the existing
`[1,512]` local GEMM, 1,024-ID all-gather, and `99,328` RNG width. This route
must pass exact host-reference LRU tests and a fresh-process natural-stop gate
before it replaces the global-512 policy.

That candidate passed the implementation and microbenchmark gates. Both ranks
load the same ranking but select the first 512 non-base IDs owned by their
local full-vocabulary shard. Runtime updates process the same observed and
target-candidate stream on both ranks and admit only locally owned IDs. The two
LRU arrays are intentionally different; their gathered token map contains
1,024 unique, active, non-base IDs. Across 32 sequential updates, both
shard-local GPU LRUs, stable order, source rows, and refreshed weights exactly
match separate host references. Seventeen targeted spec-decode tests pass.

| shard-local GPU LRU component | p50 | p99 |
|---|---:|---:|
| LRU update plus local compaction | `0.137 ms` | `0.160 ms` |
| local tail-weight refresh | `0.040 ms` | `0.054 ms` |
| complete GPU LRU refresh | `0.206 ms` | `0.257 ms` |

The fresh-process 256K natural-stop gate still failed the acceptance criterion.
It stopped normally after 7,254 tokens and passed output quality at
`98.183 tok/s`, but measured only `A=3.982976`. This is an improvement over
global-512 cold `A=3.956746`, yet it is `2.53%` below static-131K
`A=4.086253`, outside the two-percent limit (`A >= 4.004528`). The route is
therefore not an accepted default.

Simply inserting prompt IDs is not a credible next experiment for this case.
The complete 72-token chat prompt has only two unique IDs outside the 98,304
base; one is already in the shard-local seed and the other is a chat-control
special token. A future cold-start bootstrap must use target-distribution
information already produced during prefill, or a separately validated
prompt-conditioned retrieval mechanism. It must not add a per-token full-head
calculation.

A priority-order LRU experiment was also implemented, microbenchmarked, and
rejected. It processed each target top-20 row from low to high probability,
then applied real accepted/recovery output IDs last. The implementation matched
its host oracle over 32 TP2 rounds and kept full refresh at `0.231 ms` p50 and
`0.265 ms` p99. The fresh natural-stop result nevertheless regressed to
`A=3.861660` and `94.842 tok/s` while output quality still passed. The likely
reason is that high-probability tokens are already concentrated in the 98,304
base; the tail is valuable precisely for lower-ranked target candidates. This
ordering was reverted. Do not repeat candidates-first, high-probability-most-
recent LRU ordering for this configuration.

The next isolated candidate reuses the full target logits already computed at
the final prefill position. A one-time top-k plus the existing GPU refresh costs
about `0.479/0.693/1.073/1.877 ms` p50 for `k=512/1024/2048/4096`
respectively on one V100. It adds no LM-head computation and occurs before the
first drafter proposal, outside steady decode and model graph replay. The
bootstrap runs once per request on both TP ranks and then leaves the existing
per-token GPU LRU path unchanged.

The fresh-process natural-stop sweep selected `topk=2048`:

| cold bootstrap policy | mean acceptance `A` | loss from static `A_ref=4.086253` | 2% gate margin | output | request rate | result |
|---|---:|---:|---:|---:|---:|---|
| none, shard-local tail | `3.982976` | `2.5274%` | `-0.021552` | `7,254` tokens | `98.183 tok/s` | fail |
| target logits `topk=4096` | `4.000639` | `2.0952%` | `-0.003889` | `6,262` tokens | `100.705 tok/s` | fail |
| target logits `topk=2048` | **`4.017836`** | **`1.6743%`** | **`+0.013308`** | `6,757` tokens | `100.623 tok/s` | **pass** |

All three requests used the official `temperature=1.0`, `top_p=0.95`,
`top_k=20`, and `seed=20260620` sampling configuration, stopped naturally,
and passed the local repetition/corruption gate. Both TP ranks logged the
one-shot bootstrap hit. `topk=4096` is a real narrow failure, not a rounded
pass. Its larger candidate set can replace more of each rank's 512-entry
static seed; the result is consistent with over-replacement, although the
single-prompt sweep does not by itself prove that mechanism.

This supersedes the old top-1 proxy that placed `98K` above three
milliseconds. No draft-head quantization is used. The accepted experimental
layout is `98,304` base rows plus a shard-local 512-entry LRU on each TP rank,
with a one-time `topk=2048` target-logits bootstrap. It satisfies the agreed
single-prompt cold-start gate. This historical env gate was later superseded
by the explicit 2026-07-10 default-route decision documented below; the
remaining English-prompt miss is still recorded as residual risk.

The first broader matched suite used six fixed prompts, natural EOS, and the
same official `temperature=1.0`, `top_p=0.95`, `top_k=20`, and
`seed=20260620` policy for dynamic `98K` and static `131K`. All twelve outputs
stopped naturally and passed the local repetition/corruption gate.

| prompt | static `131K` A | dynamic `98K` A | dynamic A loss | static rate | dynamic rate | decision |
|---|---:|---:|---:|---:|---:|---|
| Python pygame snake | `4.077597` | `4.019802` | `1.417%` | `86.339 tok/s` | `105.105 tok/s` | pass |
| Chinese realistic story | `2.220339` | `2.292208` | `-3.237%` | `57.057 tok/s` | `62.748 tok/s` | pass; task is intrinsically low-A |
| task-management SaaS design | `3.344203` | `3.355217` | `-0.329%` | `83.023 tok/s` | `88.128 tok/s` | pass |
| FastAPI task service | `3.988095` | `4.030366` | `-1.060%` | `97.392 tok/s` | `101.522 tok/s` | pass |
| Chinese V100 evaluation report | `2.960069` | `2.939671` | `0.689%` | `73.155 tok/s` | `76.019 tok/s` | pass |
| English backend implementation | `3.788684` | `3.676643` | **`2.957%`** | `91.005 tok/s` | `91.881 tok/s` | **fail** |
| **aggregate** | **`3.503842`** | **`3.526681`** | **`-0.652%`** | **`84.029 tok/s`** | **`90.415 tok/s`** | aggregate pass, per-prompt fail |

Negative loss means the dynamic run measured higher acceptance. The result
answers the prompt-specificity question: macOS is not the only high-A case;
long code generation also reaches about `A=4.02-4.03`, while prose and
explanatory tasks can naturally sit near `A=2.2-3.0` even on static `131K`.
The dynamic route is faster on every observed prompt, but the aggregate speed
gain is only indicative because the dynamic service was already warm and the
first static request included cold JIT work. The acceptance comparison is the
primary evidence.

This suite does not close the broad gate. The dynamic service had already
processed 13,909 user tokens before the suite, whereas static `131K` started
from a fresh process; both used the same prompt order, and dynamic prefill
bootstrap still ran once per request. The English backend case exceeds the
two-percent bound and should be repeated as the first request of a fresh
dynamic process. The later explicit decision enables the route by default
despite this known per-prompt miss; do not misreport the broad gate as fully
passing. A machine reboot interrupted the first simultaneous-reference
attempt, so the completed static reference was run serially. Do not run two
27B services concurrently on this host for this gate.

Artifacts:

- `bench_results/mtp_dynamic_vocab_20260710/static131k_top20_alignment_macos512.json`
- `bench_results/mtp_dynamic_vocab_20260710/alignment_static131k_top20_macos512/`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic65k_counterfactual_top20_finite_candidates_macos512.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic65k_tail512_hostproto_macos512.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic65k_tail4096_hostproto_macos512.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic65k_tail4096_refresh64_hostproto_macos512.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_tail512_hostproto_macos512.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_tail512_fused_hostlru_macos512.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_tail512_fused_rngaligned_hostlru_macos512.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_tail512_fused_rngaligned_gpulru_macos512.json`
- `bench_results/mtp_cost_breakdown_20260710/dynamic98k_gpu_lru_tp2_parallel_v2_v100.json`
- `bench_results/mtp_cost_breakdown_20260710/dynamic98k_gpu_lru_tp2_shardlocal_v3_v100.json`
- `bench_results/mtp_cost_breakdown_20260710/dynamic98k_gpu_lru_tp2_priority_shardlocal_v4_v100.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_gpulru_256k_macos12000_seed20260620.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_gpulru_256k_macos12000_seed20260620_warm2.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_shardlru1024_256k_macos12000_seed20260620_cold.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_priority_shardlru1024_256k_macos12000_seed20260620_cold.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_shardlru1024_prefilltopk4096_256k_macos12000_seed20260620_cold.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_shardlru1024_prefilltopk2048_256k_macos12000_seed20260620_cold.json`
- `bench_results/mtp_dynamic_vocab_20260710/other_prompt_suite_20260710.json`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_prefilltopk2048_other_prompts_warmmulti_seed20260620.json`
- `bench_results/mtp_static_vocab_20260710/static131k_other_prompts_seed20260620.json`
- `bench_results/mtp_cost_breakdown_20260710/fp16_dynamic65k_tail512perrank_tp2_mapped_denseq_v2_v100.json`
- `bench_results/mtp_cost_breakdown_20260710/fp16_dynamic98k_tail512perrank_tp2_mapped_denseq_topp095_v2_v100.json`
- `bench_results/mtp_cost_breakdown_20260710/fp16_dynamic98k_logical512_physical512perrank_tp2_rng99328_v100.json`

## 2026-07-10 Default Route And Calibrated MTP Cost

### Default policy

The dynamic route is now the direct default whenever
`speculative_config.method == "mtp"` and no explicit draft-vocabulary controls
are present. There is no model selector. The resolved defaults are:

| control | default |
|---|---:|
| base vocabulary | `98,304` global rows |
| dynamic tail | `512` shard-local rows per TP rank, `1,024` gathered IDs |
| proposal | fused TP2 top-20 probabilistic path |
| tail maintenance | GPU LRU, refresh interval `0` |
| one-shot final-prefill bootstrap | target-logits `topk=2048` |
| RNG/logical width | `99,328` |

The ranking is packaged as
`vllm/assets/sm70_mtp_dynamic_vocab_qwen36_27b_tp2.pt`; `setup.py` includes
`vllm/assets/*.pt` in wheel data. Existing explicit vocabulary environment
variables still override the default for controlled comparisons, and
`VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0` is a diagnostic kill switch.
The accepted implementation remains constrained to TP2, MTP4, an FP16 shared
head, and `max_num_seqs=1`; the effective default now passes through the same
validation as explicit GPU-LRU configuration.

A live non-eager service with none of the static/dynamic vocabulary variables
set logged the default route on both TP ranks and loaded the packaged asset.
The route-hit log is
`bench_results/mtp_cost_breakdown_20260710/api_default_dynamic98k_mtp_profile_port8001.log`.
Seventeen targeted spec-decode tests pass. The user-visible default decision
does not erase the six-prompt caveat: aggregate acceptance was not lower, but
the English backend prompt measured a `2.957%` loss from static `131K`, above
the requested per-prompt two-percent bound.

### Measurement contract

All requests use the official `temperature=1.0`, `top_p=0.95`, `top_k=20`,
and `seed=20260620` sampling parameters. No eager path is used.

Three measurements are combined instead of treating a host wait as GPU work:

1. The low-overhead, natural-stop long-code request provides the absolute
   decode-dominated baseline: `3,246` output tokens, quality pass,
   `A=4.019802`, `105.104589 tok/s`, and `9.514332 ms/output-token`. This is
   `38.245732 ms` per verifier round after multiplying TPOT by `A`.
2. CUDA events around disjoint runner and proposer stages provide means over
   895 steady speculative rounds. Logs are emitted in seven context bands of
   127/128 rounds; the reported p50 and range below are statistics of those
   band means, not a claim that every individual round was sampled.
3. Nsight Systems `--cuda-graph-trace=node` provides graph-correlated kernel
   composition. Its three middle rounds measured `40.877898 ms/round`,
   `6.88%` above the low-overhead baseline, so it is used for composition and
   normalized to the CUDA-event bucket, not used as the absolute speed result.

### Whole MTP round

One row here is one target verification plus the next four-token draft. It is
not one emitted token. Dividing by measured `A=4.019802` gives the final-token
column.

| disjoint stage | mean ms/round | band p50 | band range | ms/output-token | round share |
|---|---:|---:|---:|---:|---:|
| target verifier forward, 5 rows | `22.997` | `23.348` | `21.176-23.723` | `5.721` | `60.13%` |
| target full-vocabulary logits, 5 rows | `2.038` | `2.037` | `2.036-2.040` | `0.507` | `5.33%` |
| target rejection/sample plus GPU-LRU update | `1.116` | `1.115` | `1.111-1.128` | `0.278` | `2.92%` |
| MTP draft total, four sequential proposals | `7.134` | `7.105` | `6.093-8.264` | `1.775` | `18.65%` |
| GPU bookkeeping | `0.090` | `0.090` | `0.089-0.092` | `0.022` | `0.24%` |
| request/runner residual, calculated | `4.871` | - | - | `1.212` | `12.73%` |
| **total** | **`38.246`** | - | - | **`9.514`** | **`100%`** |

The current verifier cost is therefore **`26.151 ms/round`**, not only the
`22.997 ms` model forward. It amortizes to `6.505 ms/output-token` and consumes
`68.38%` of round wall time. The calculated residual is
`38.245732 - 33.375179 = 4.870553 ms/round`; it includes runner preprocessing,
state/output handoff, launch gaps, and request-level amortization not enclosed
by the five GPU event spans.

CPU diagnostics must not be added to this table. State update is
`0.580 ms/round` (`0.556 ms` in input-batch state update), while proposer loop
metadata is `0.856 ms/round`; both can overlap queued GPU work. Likewise,
`draft_wall_cpu=29.594 ms` is an inclusive synchronization wait in
`_copy_draft_token_ids_to_cpu` and contains the preceding target backlog. It is
not a 29.594 ms draft cost.

### Target verifier forward internals

The FULL graph's per-category critical sums are `23.233 ms` under Nsight. The
table scales those category means by `22.997/23.233` to the CUDA-event target
forward, preserving the measured composition. Kernel counts are TP sums over
both GPUs per round.

| verifier-forward category | calibrated ms/round | ms/output-token | forward share | kernels/round |
|---|---:|---:|---:|---:|
| TurboMind AWQ GEMM | `10.181` | `2.533` | `44.27%` | `472` |
| TP communication | `2.083` | `0.518` | `9.06%` | `256` |
| copy/cast | `1.776` | `0.442` | `7.72%` | `864` |
| GDN recurrent core plus causal conv | `1.595` | `0.397` | `6.93%` | `288` |
| TurboMind FP16 GEMM | `1.462` | `0.364` | `6.36%` | `34` |
| other Triton/elementwise | `1.447` | `0.360` | `6.29%` | `864` |
| CUTLASS/cuBLAS FP16 GEMM/GEMV | `1.192` | `0.297` | `5.18%` | `198` |
| index/reduce/scatter | `1.160` | `0.289` | `5.04%` | `480` |
| FlashAttn V100 decode | `0.967` | `0.240` | `4.20%` | `64` |
| RMSNorm/residual | `0.603` | `0.150` | `2.62%` | `226` |
| SiLU/gating | `0.456` | `0.113` | `1.98%` | `256` |
| KV-cache update | `0.076` | `0.019` | `0.33%` | `32` |
| **target forward** | **`22.997`** | **`5.721`** | **`100%`** | - |

The separate `2.038 ms` target-logits bucket is the full-vocabulary LM-head
for the verifier's five rows. The `1.116 ms` sample bucket performs official
top-k/top-p target sampling, probabilistic rejection/recovery, and the current
dynamic-tail update/refresh. Nsight attributes about `1.040 ms/round` of that
path to `_topk_topp_kernel`, softmax, recovery, and CUB selection kernels.

### MTP draft internals

| draft component | mean ms/round | band p50 | band range | ms/output-token | draft share |
|---|---:|---:|---:|---:|---:|
| first MTP forward | `1.674` | `1.659` | `0.822-2.577` | `0.416` | `23.46%` |
| loop-0 MTP forward | `0.722` | `0.718` | `0.650-0.799` | `0.180` | `10.12%` |
| loop-1 MTP forward | `0.719` | `0.721` | `0.645-0.798` | `0.179` | `10.08%` |
| loop-2 MTP forward | `0.724` | `0.722` | `0.652-0.802` | `0.180` | `10.14%` |
| **four MTP forwards subtotal** | **`3.838`** | **`3.820`** | **`2.769-4.976`** | **`0.955`** | **`53.80%`** |
| four FP16 dynamic LM-head plus samples | `2.907` | `2.907` | `2.894-2.929` | `0.723` | `40.75%` |
| proposer GPU work outside forward/head spans | `0.117` | - | - | `0.029` | `1.64%` |
| outer padded-input preparation/wrapper GPU work | `0.271` | - | - | `0.067` | `3.80%` |
| **draft total** | **`7.134`** | **`7.105`** | **`6.093-8.264`** | **`1.775`** | **`100%`** |

The four head/sample spans are stable at about `0.727 ms` each and have met
the earlier below-three-millisecond objective without W4 head quantization.
The first draft forward is now the strongly context-dependent draft cost: its
band mean rises from `0.822` to `2.577 ms` over the long request. Nsight's
draft PIECEWISE graphs account for `1.938 ms` TurboMind FP16 GEMM, `0.260 ms`
TP communication, and about `0.232 ms` norm/gating/copy/elementwise work; the
non-graph unified-attention kernel contributes another `0.480 ms/round`.

### Consequences for the 20 ms verifier target

The target is `26.151 -> <=20 ms/round`, requiring at least `6.151 ms` or
`23.52%` from the complete verifier. Eliminating target logits and rejection
sampling entirely would save only `3.154 ms`, leaving the verifier above the
goal. The target forward must therefore fall by at least about three
milliseconds even under that impossible upper bound, and realistically must
contribute most of the six-millisecond reduction.

The forward is not a one-kernel bottleneck. AWQ GEMM is the largest bucket at
`10.181 ms`, but communication, copy/index/elementwise traffic, GDN, FP16
GEMMs, and attention make up the remaining `12.816 ms`. FlashAttn alone is only
`0.967 ms`, so an attention-backend swap cannot close the target. The next
verifier work should separately prove: fewer/fused intermediate materializations
and index copies, TP communication overlap or launch reduction, and an M=5
TurboMind AWQ schedule improvement. The `4.871 ms` residual also needs explicit
NVTX spans around preprocess, state update, D2H handoff, and output scheduling
before it can be assigned to an optimization.

Artifacts:

- `bench_results/mtp_cost_breakdown_20260710/default_dynamic98k_mtp_profile_snake_seed20260620.json`
- `bench_results/mtp_cost_breakdown_20260710/api_default_dynamic98k_mtp_profile_port8001.log`
- `bench_results/mtp_dynamic_vocab_20260710/dynamic98k_prefilltopk2048_other_prompts_warmmulti_seed20260620.json`
- `bench_results/nsys_27b_awq_mtp_default_dynamic_20260710/graph_node_i512_o16.nsys-rep`
- `bench_results/nsys_27b_awq_mtp_default_dynamic_20260710/graph_node_i512_o16.sqlite`
- `bench_results/nsys_27b_awq_mtp_default_dynamic_20260710/mtp_round_kernel_breakdown.md`
- `bench_results/nsys_27b_awq_mtp_default_dynamic_20260710/mtp_round_kernel_breakdown.json`
- `benchmarks/analyze_sm70_mtp_nsys.py`

### P2: SlimSpec Low-Rank Full-Vocabulary Draft Head

SlimSpec replaces the draft head `W[V,d]` with two trained matrices
`W_up[V,r] * W_down[r,d]`. It preserves full vocabulary support and avoids
runtime shortlist construction. The paper selects `r=d/8` as its default and
reports approximately `4-5x` LM-head acceleration with acceptance ratio near
`0.99` in its tested EAGLE-3 setup.

The corresponding V100 dense-GEMM hardware bounds for this model are:

| rank | down projection | up projection | two-GEMM total | estimated draft ms/round |
|---:|---:|---:|---:|---:|
| `320 = d/16` | `0.026 ms` | `0.099 ms` | `0.126 ms` | `4.916 ms` |
| `640 = d/8` | `0.031 ms` | `0.199 ms` | `0.230 ms` | `5.332 ms` |
| `1,280 = d/4` | `0.028 ms` | `0.487 ms` | `0.515 ms` | `6.473 ms` |

Artifacts:

- `bench_results/mtp_cost_breakdown_20260710/slimspec_down_m1_hidden5120_rank320_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/slimspec_up_m1_localvocab124160_rank320_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/slimspec_down_m1_hidden5120_rank640_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/slimspec_up_m1_localvocab124160_rank640_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/slimspec_down_m1_hidden5120_rank1280_gpu0.json`
- `bench_results/mtp_cost_breakdown_20260710/slimspec_up_m1_localvocab124160_rank1280_gpu0.json`

This is not a post-training factorization of the existing tied LM-head. A naive
low-rank SVD would change draft probabilities without the acceptance-oriented
training used by the paper. Their Qwen3-30B-A3B drafter training reports about
`320 H200 GPU-hours`; it also uses EAGLE-3 rather than Qwen3.6 native MTP.

The lower-cost research probe is to freeze the existing native MTP backbone,
capture real draft hidden states and full target distributions, and train only
the draft-specific factorized head at ranks `640` and `1,280`. Proceed to a
full training run only if the held-out official-sampling acceptance model
predicts `A >= 4.241176`, targets `A_ref=4.327731`, and matches the
corresponding baseline prompt suite.

### Other Routes And Deferrals

| route | published/repository value | decision for current model |
|---|---|---|
| SpecVocab | learned per-step vocabulary router; reports up to `8.1%` throughput over EAGLE-3 | training and router/runtime complexity; revisit after the static and NanoSpec foundations |
| EvoSpec | dynamic long-tail retrieval plus online parameter adaptation; reports `1.13x` over FR-Spec in specialized domains | no P0 implementation; useful only if static coverage fails on domain shifts |
| FastMTP | shared MTP head plus language-aware vocabulary compression | requires a new trained head; not drop-in for native Qwen MTP weights |
| P-EAGLE/parallel drafting | proposes multiple draft positions in one pass | requires a parallel-capable trained drafter; current sequential native MTP weights cannot be reused directly |
| GPU n-gram/suffix draft | near-zero neural draft cost on matching text | workload-specific supplement for editing/code/repetition, not a general replacement |
| full drafter CUDA graph and inter-step fusion | removes launch/boundary overhead | secondary: measured proposer boundary is only `0.290 ms/round` |

Primary references:

- <https://github.com/thunlp/FR-Spec>
- <https://arxiv.org/abs/2502.14856>
- <https://arxiv.org/abs/2506.22694>
- <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_worker_v2.py>
- <https://github.com/csAugust/NanoSpec>
- <https://arxiv.org/abs/2605.26444>
- <https://arxiv.org/abs/2605.10453>
- <https://arxiv.org/abs/2602.13836>
- <https://arxiv.org/abs/2605.27390>
- <https://github.com/Tencent-BAC/FastMTP>
- <https://github.com/vllm-project/speculators>

## Draft Route Decision Matrix

| priority | route | retraining | conservative draft estimate | next proof |
|---|---|---:|---:|---|
| P0 | global top-`131K`, startup-balanced draft rows | no | **`8.460 ms/round` measured** | first natural-stop chat gate passed at `97.698 tok/s`; broaden prompt/domain coverage |
| P1A | TurboMind W4 full-vocabulary draft head | quantization only | `5.918 ms/round` | real-weight logit and acceptance replay |
| P1B | `98K` FP16 base + GPU dynamic tail | no | **`2.907 ms` current long-run p50 for four heads/sample; shard-local GPU refresh `0.206 ms` micro p50** | direct MTP default for the validated TP2/MTP4 single-request configuration; repeat the `2.957%` English miss cold |
| P2 | SlimSpec factorized full-vocabulary head | yes | `4.916-6.473 ms/round` | frozen-backbone head-training probe |
| P3 | SpecVocab, EvoSpec, FastMTP, P-EAGLE | yes or major architecture change | not estimated | only after P0/P1 evidence |

The `98K` FP16 plus shard-local GPU-LRU route is the direct MTP default for the
validated TP2/MTP4 single-request configuration. Its four draft heads plus
sampling remain below three milliseconds and its `topk=2048` cold bootstrap
passed the first natural-stop gate. Static `131K` remains the simpler quality
reference and explicit fallback. Keep P1A draft-head W4 independent and
disabled because it changes logits.

## Active Optimization Order

1. **Reduce the complete verifier from `26.151` to at most `20 ms/round`.**
   The `22.997 ms` M=5 forward is led by `10.181 ms` TurboMind AWQ GEMM, but
   communication and tensor movement also require work. Any GDN change must
   retain full-GDN quality behavior.
2. **Instrument the calculated `4.871 ms/round` residual.** Add NVTX/GPU event
   spans around preprocess, state update, D2H handoff, and output scheduling.
   Do not optimize an unlabeled remainder or add overlapping CPU timers to the
   disjoint GPU table.
3. **Attack the context-growing first MTP forward.** Four draft heads/sample
   already meet the below-three-millisecond objective; the first forward rises
   from `0.822` to `2.577 ms` across the long request.
4. **Broaden the default FP16 dynamic proposal quality gate.** The fused TP2
   head/sample and fixed-buffer GPU LRU paths pass microbenchmarks and exact
   short-gate parity. A one-time final-prefill target-logits bootstrap at
   `topk=2048` raises fresh-process natural-stop acceptance from `A=3.982976`
   to `A=4.017836`, a `1.6743%` loss from static `131K` and therefore inside
   the two-percent bound. `topk=4096` narrowly fails and must not be rounded up
   or repeated as the selected setting. A six-prompt matched static comparison
   shows that code tasks are also high-A and aggregate dynamic acceptance is
   not lower, but the English backend case loses `2.957%`; repeat it as a cold
   first request before extending the suite to math and long-context cases.
   Keep draft-head W4 separate and disabled.
5. **Improve acceptance only with quality proof.** Raising `A=4.01637` reduces
   the amortized verifier/draft cost, but cannot justify the unsafe split-GDN
   deep-MTP route.

Every candidate must first have a microbenchmark with an explicit go/no-go
threshold, then an MTP4 6k-token natural-stop quality run and matched no-MTP
control. Do not use a short route-hit smoke, greedy sampling, eager mode, or an
unsafe GDN override as speed evidence.

## 2026-07-12 TP4 MTP4 Bring-Up

The existing default dynamic-vocabulary asset and fused sampler were TP2-only.
The first TP4 startup stopped with `Ranking artifact TP=2, runtime TP=4`, so a
run with dynamic vocabulary silently disabled was not used as a speed result.

TP4 support now uses the same global ranking repartitioned into four 62,080-row
shards. The packed global top-20 sampler derives its candidate count from the
gathered tensor: 40 candidates for TP2 and 80 for TP4. The remaining
all-to-all, packed pair all-gather, tail all-gather, and sampled-token broadcast
were already TP-size generic. TP4 is explicit and does not replace the TP2
default asset.

The proposal microbenchmark passes TP2 and TP4 reference checks: sampled token
and sparse IDs are exact, maximum probability difference is `7.45e-9`. Four
fused proposal steps in a CUDA graph measure `2.8756 ms` for TP2 and
`1.9323 ms` for TP4, a `32.8%` TP4 reduction.

The full Qwen3.6-27B-AWQ TP4 route uses V100 GPUs 0-3, pairwise NVLink `NV2`,
TurboMind, MTP4, full-GDN, Flash-V100, non-eager CUDA graphs, and 256k max
length. With the official 74-token macOS request and natural stop, stable TP4
results are approximately `119.924 tok/s`, acceptance `4.058069`, draft
acceptance `76.4517%`, and MTP round `33.8386 ms`; all runs pass the quality
gate.

Compared with the accepted TP2 result (`100.547 tok/s`, `A=3.965217`, round
`39.4363 ms`), TP4 improves throughput `19.27%`, raises acceptance `2.34%`,
and reduces normalized round time `5.5976 ms` or `14.19%`. Acceptance is not an
exact capacity-matched comparison: keeping 512 tail rows per rank gives TP4 a
2,048-row global tail versus TP2's 1,024 rows.

Fresh-server first requests reproduce the same response hash and counters.
Later same-server requests produce different text because GPU-LRU tail state
persists across requests, while acceptance and warm throughput remain stable.

The current TP2 N64 verifier selector does not cover TP4-local GEMM dimensions.
A dedicated TP4 selector/microbenchmark is the next TP4-specific verifier
optimization, rather than reusing TP2 tactic assumptions.

Detailed evidence:
`bench_results/awq_m5_smalln_hmma_20260711/tp4_mtp4_api_20260712/summary.md`.

## 2026-07-12 Latest TP4 MTP4 Latency Update

The current TP4 route was measured again after the exact M=5 verifier selector
bridge. Two natural-stop official-sampling warm requests pass output quality at
`122.900` and `125.600 tok/s`; their mean is `124.24 tok/s`, `8.049 ms/output`
token, acceptance `4.110`, and normalized MTP round `33.077 ms`. This is a
current-route observation. It is not yet a same-binary endpoint on/off claim,
because the final installed binary also added a host-only A/B diagnostic gate.

The new CUDA-event stage profile uses calls 128--256 of a short official
sampling request as composition evidence. P50 values are target forward
`18.433 ms`, target logits `1.052 ms`, target rejection/sample `1.383 ms`,
MTP draft total `4.713 ms`, and bookkeeping `0.103 ms`. Do not add inclusive
CPU waits such as `draft_wall_cpu`; they contain queued target work.

Inside the draft, four MTP forwards are `2.302 ms` and four LM-head/sample
steps are `1.981 ms`. Target forward is therefore still the first bottleneck.
Its real-weight AWQ M=5 component is now exact and improves on every TP shard
from a `6.8065 ms` critical rank aggregate to `5.8273 ms` (`-14.39%`).

The validated TP2 to TP4 result remains `+19.27%` throughput and `-14.19%`
round latency; the latest TP4 observation implies approximately `+23.6%` and
`-16.1%` against the accepted TP2 reference. Neither is close to ideal 2x
parallel scaling. TP4's larger dynamic tail also raises acceptance, so round
time is the correct primary scaling metric. Full data and the exact scope of
each measurement are in
`bench_results/mtp4_current_binary_20260712/latest_mtp4_latency_analysis.md`.

## 2026-07-12 Current TP2 to TP4 Scaling Root Cause

Same-binary steady MTP event profiles show TP2 to TP4 measured GPU spans move
`32.560 -> 25.684 ms/round` (`-21.1%`). The target-forward event is the failed
scaling span: `22.670 -> 18.433 ms` (`-18.7%`), versus an ideal `11.335 ms`.

The all-real M=5 AWQ verifier itself scales better, `9.085 -> 5.827 ms`
(`-35.9%`), and supplies 76.9% of the total target-forward saving. After
subtracting it, the non-AWQ target-forward chain is still `13.585 -> 12.606 ms`
(`-7.2%`). This is the primary TP4 failure, not MTP acceptance or draft work.

Within AWQ, row projection scales only 15.0% and MLP-down 33.2%; both require
exact TP4 tactics. In the separate draft path, four samples scale
`2.901 -> 1.981 ms` and full draft scales `6.382 -> 4.713 ms`, so draft is
not the first scaling target. The paired target-only trace shows TP
communication grows `1.455 -> 1.860 ms` when moving to TP4 while attention
and recurrent work are nearly fixed. A current MTP graph-node trace must now
attribute the 12.606 ms non-AWQ chain before a communication rewrite.

Evidence: `bench_results/mtp4_current_binary_20260712/tp2_tp4_mtp4_scaling_analysis.md`.

## 2026-07-15 TP2 Flash-Drafter Baseline Correction

The `78.708 tok/s` Flash-drafter result from the earlier `4K/256` probe is not
an absolute TP2 MTP4 baseline. That probe used a repeated synthetic prompt,
only 256 forced output tokens, a cold first request, and
`VLLM_SM70_MTP_PROFILE=1`. It is retained only as a matched short A/B in which
Flash and Triton produced the same 256-token sequence and Flash reduced the
instrumented draft span from about `9.090` to `6.377 ms/round`.

The accepted absolute workload was reproduced with the current 2026-07-14
binary: Qwen3.6-27B-AWQ, TP2 GPUs 2/3, 256K max length, TurboMind AWQ, target
Flash-V100, non-eager `FULL_AND_PIECEWISE` CUDA graphs, dynamic `98K` draft
vocabulary, MTP profiling disabled, OpenAI chat, natural EOS, and official
`temperature=1.0`, `top_p=0.95`, `top_k=20`, `seed=20260620` sampling.

| drafter | request | output | rate | acceptance `A` | normalized round | quality |
|---|---|---:|---:|---:|---:|---|
| Triton, default | cold | 5,473 | `99.612 tok/s` | `3.965217` | `39.807 ms` | pass |
| Triton, default | warm | 5,473 | `99.896 tok/s` | `3.965217` | `39.693 ms` | pass |
| Flash-V100, explicit candidate | cold | 5,877 | `102.301 tok/s` | `3.909514` | `38.216 ms` | pass |
| Flash-V100, explicit candidate | warm | 5,877 | `102.724 tok/s` | `3.909514` | `38.058 ms` | pass |

This reproduces the historical approximately `100 tok/s` Triton baseline.
For the warm requests, Flash reduces normalized round time by `4.12%`, while
the different sampled trajectory lowers acceptance by `1.40%`; observed
throughput therefore rises only `2.83%`. At the Triton acceptance length, the
Flash round time projects to `104.188 tok/s`, but that projection is not an
endpoint result.

The two requests within each arm are text-identical and naturally stop. Both
outputs pass the repetition/corruption gate and contain a complete fenced HTML
document with balanced top-level tags; however, Triton and Flash output hashes
differ. The current evidence is therefore sufficient to show that the old
graph-collapse failure is not reproduced and that Flash has a real cost win,
but not sufficient to switch the production default. Keep Triton as the
implicit drafter backend and test Flash through explicit
`speculative_config.attention_backend=FLASH_ATTN_V100` until broader prompt and
35B-A3B quality gates pass.

Artifacts:
`bench_results/mtp_flash_drafter_20260715/tp2_api_natural_exact/`.

## 2026-07-15 FP8 E5M2 KV Cache Historical Baseline

> Superseded later on 2026-07-15. The measurements below are the pre-fast-path
> baseline and remain useful only as root-cause evidence. The implemented
> bridge/XQA path, current decision, matched results, and rollback controls are
> maintained in
> `docs/design/sm70_flash_v100_fp8_kv_long_context.md`.

FP8 KV cache is now functionally routed through Flash-V100 for both the target
and the MTP4 drafter. Align-mode page sizing requires
`max_num_batched_tokens>=1568` without MTP and `>=1616` with MTP4; the matched
matrix therefore uses 2048 for both FP16 and FP8 controls. The runs hit the C++
FP8 cache-write path, Flash-V100 paged prefill, scalar paged decode, and
non-eager CUDA graphs.

Capacity improves from 544,275 to 1,066,654 tokens without MTP and from 399,384
to 818,294 tokens with MTP4. This capacity win is not a V100 speed win in the
current implementation. At 128K, no-MTP prefill rises from `169.206` to
`450.290 s` and TPOT rises from `35.218` to `47.119 ms`. MTP4 hides most of the
decode regression at long context: its 128K TPOT moves only from `28.445` to
`28.603 ms` with identical `A=4.9231`, but its prefill still rises from
`249.040` to `425.478 s`.

The source-level cause is a fast-path gap. FP16 paged prefill has vector page
loads and D256 low-shared-memory/software-pipeline variants, while the FP8 path
uses per-element page lookup and conversion. FP16 q=1 decode can select the XQA
tensor-core kernel, but the selector excludes FP8 and falls back to generic
scalar paged decode. E5M2 can be expanded exactly into FP16 representation, so
the next kernel work is vectorized shared-memory expansion inside those
existing fast schedules, not another precision reduction.

The historical decision was to keep FP8 KV opt-in for capacity only. That
performance decision is no longer current: the prefill bridge and graph-safe
FP8 XQA recover the measured speed gap. The model-level FP8-versus-FP16 quality
question remains separate. The 1K and 4K historical MTP output hashes differ,
and the 1K acceptance length changes from `4.5893` to `3.9692`; no natural-stop
quality gate was run in this historical matrix.

Full tables and route evidence:
`bench_results/mtp_flash_drafter_20260715/long_context_tp2/fp8_kv_flash_v100_comparison.md`.

Current no-MTP result with the new default fast paths:

| context | old FP8 prefill | new FP8 prefill | old FP8 TPOT | new FP8 TPOT |
|---:|---:|---:|---:|---:|
| 16K | `14.680 s` | `10.317 s` | `21.762 ms` | `19.715 ms` |
| 64K | `127.842 s` | `62.964 s` | `32.768 ms` | `26.910 ms` |

The fixed-length 256-token hashes exactly match the old FP8 route at both
points, so the fast paths introduced no observed output regression relative to
the previous FP8 implementation. A 256K full-model confirmation remains open.

MTP4 initially exposed one additional route gap: align mode uses `M=1616`, so
the old BM32 selector rejected the whole chunk because 1616 is not divisible
by 32. The BM32 kernel now handles one guarded 16-row tail CTA while retaining
the original fast path for the first 50 full CTAs. With that fix, matched MTP4
FP8 prefill is `9.972 s` at 16K and `56.441 s` at 64K, down 28.66% and 54.27%
from the historical FP8 route. TPOT is `10.072/18.020 ms`, acceptance length
remains `4.8679/4.8113`, and both token hashes are unchanged.

## 2026-07-19 TP4 Long-Context Verifier XQA And Flash Drafter Default

The current long-context regression was reproduced with Qwen3.6-27B-AWQ,
TP4 V100, FP8 E5M2 KV, TurboMind, 256K model length, chunk size 8096,
prefix caching, Mamba align, official sampling, and non-eager CUDA graphs.
The previous release matrix explicitly pinned the MTP drafter to
`TRITON_ATTN`, even though the Flash-V100 drafter metadata/replay repair had
already passed the July 15 TP2 natural-output check.

There were two independent route gaps:

1. The target M=5 verifier already converted five causal query rows into five
   paged-decode rows, but called only the scalar kernel. It did not inherit the
   q=1 target's G6 XQA route.
2. The four serial MTP draft forwards still used Triton attention. This was the
   dominant long-context gap; target-verifier XQA alone was not sufficient.

The exact TP4 M=5 FP8-KV microbenchmark uses `Hq=6,Hkv=1,D=256`, page 1616,
five rows with sequence lengths `N-4..N`, and a fixed 262144-token workspace.

| Context | Scalar median | XQA median | XQA change | max abs diff |
|---:|---:|---:|---:|---:|
| 768 | `0.1362 ms` | `0.1423 ms` | `+4.51%` | `2.44e-4` |
| 4K | `0.2202 ms` | `0.1905 ms` | `-13.49%` | `6.10e-5` |
| 64K | `1.6835 ms` | `1.2278 ms` | `-27.07%` | `1.53e-5` |
| 128K | `3.1017 ms` | `2.1402 ms` | `-31.00%` | `7.63e-6` |
| 261888 | `6.0232 ms` | `4.1748 ms` | `-30.69%` | `7.63e-6` |

The production gate is deliberately narrower than generic q=1 XQA. It accepts
only small-query `G=6/8,D=256`, FP16 or E5M2 KV, no sliding window, and no
explicit context-bucket partition override. Existing P256/P512 bounded graphs
therefore remain on scalar; this does not reopen the rejected P256 short-graph
XQA experiment. `VLLM_FLASH_V100_SMALLQ_DECODE_USE_XQA=0` is the rollback.

Matched endpoint results are:

| Context | no MTP TPOT / TPS | Triton-drafter MTP4 TPOT / TPS | full-Flash MTP4 TPOT / TPS | full-Flash vs no MTP |
|---:|---:|---:|---:|---:|
| 64K | `17.507 / 57.121` | `19.941 / 50.149` | `9.944 / 100.564` | `+76.05%` |
| 128K | `22.009 / 45.438` | `33.676 / 29.695` | `13.281 / 75.298` | `+65.72%` |
| 261888 | `30.434 / 32.858` | `59.324 / 16.857` | `20.092 / 49.772` | `+51.47%` |

At 128K, target-verifier XQA with the old Triton drafter reached only
`30.598 ms / 32.682 tok/s`. Thus XQA saved `3.078 ms` but still lost to
no-MTP. Enabling the already repaired Flash-V100 drafter reduced TPOT by a
further `17.317 ms` and is the first-order fix.

All fixed-length repeats were deterministic, reported `is_corrupted=false`,
and retained the Triton/scalar token hashes. Acceptance was `4.981/99.52%` at
64K and 128K and `5.000/100%` at 261888.

The default-generated TP4 natural quality run also passed. It emitted 22,053
tokens, stopped naturally, produced complete closed HTML/CSS/JavaScript,
reported no bad markers or corruption, and decoded at `114.167 tok/s` with
aggregate acceptance length `3.7896`. The previous Triton-drafter natural run
decoded at `94.539 tok/s`, so the current default gains `20.77%` on that
workload. `EngineArgs` and the release matrix now default an enabled SM70 MTP
drafter to `FLASH_ATTN_V100`; explicit user backend selection remains
authoritative.

Artifacts:

- `bench_results/mtp_verifier_xqa_20260719/tp4_fp8kv_m5_scalar_xqa_workspace262144.json`
- `bench_results/mtp_verifier_xqa_20260719/tp4_fp16kv_m5_scalar_xqa_workspace262144.json`
- `bench_results/mtp_verifier_xqa_20260719/tp4_awq_fp8kv_mtp4_i128k_candidate.json`
- `bench_results/mtp_verifier_xqa_20260719/tp4_awq_fp8kv_mtp4_i128k_targetxqa_drafterflash.json`
- `bench_results/mtp_verifier_xqa_20260719/tp4_awq_fp8kv_mtp4_i64k_i256k_targetxqa_drafterflash.json`
- `bench_results/mtp_verifier_xqa_20260719/quality_tp4_awq_fp8kv_mtp4_flash_default/`

## 2026-07-19 Exact-M5 P1024 Dual-CTA Verifier

After the full-Flash repair, target attention remained the dominant source of
long-context growth. Normalizing TPOT by accepted tokens gives approximately
`49.5/66.1/100.5 ms` per MTP round at 64K/128K/261888. One rank executes 16
full-attention layers, and the exact `M=5,Hq=6,Hkv=1,D=256`, page-1616 FP8
E5M2 XQA microbenchmark accounts for almost all of the 64K-to-256K increase.

The original P1024 partition kernel used 256 threads, 186 registers/thread,
and 43.26 KiB dynamic shared memory. It could resident only one CTA per SM.
NCU measured 12.49% achieved occupancy, 0.34 eligible warps/scheduler, and
72.33% cycles with no eligible warp. DRAM throughput was only 3.05%; this was
a register/residency and latency-hiding problem, not an HBM bandwidth limit.

Two candidate families were tested:

| TP4 context | P1024 baseline | P256 dual CTA | P1024 dual CTA |
|---:|---:|---:|---:|
| 64K | `1.3865 ms` | `0.918 ms` | `1.2093 ms` |
| 128K | `2.2487 ms` | `1.626 ms` | `1.7439 ms` |
| 261888 | `4.1805 ms` | `2.993 ms` | `3.0623 ms` |

P256 was fastest, but it changes partition and final reduction order. A
natural quality run diverged into a 19,930-token single-token repetition and
did not stop at 32K. It is therefore rejected as a global default. Split
reduction tiles 16 and 32 also regressed the 128K microbenchmark and remain
disabled.

The accepted path keeps P1024 and the existing reduction order, but specializes
the exact M=5 TP4 shape to six warps (192 threads) with a two-CTA launch bound.
Across contexts 72, 97, 128, 256, 512, 768, 1K, 4K, 16K, 32K, 64K, 128K,
and 261888, direct P1024 baseline-versus-dual comparisons were bitwise equal.
NCU confirms that the change improves residency without increasing work:

| 128K partition metric | P1024 baseline | P1024 dual CTA |
|---|---:|---:|
| grid / threads | `1280 / 256` | `1280 / 192` |
| registers/thread | `186` | `168` |
| dynamic shared/CTA | `43.26 KiB` | `43.26 KiB` |
| achieved occupancy | `12.49%` | `18.67%` |
| eligible warps/scheduler | `0.34` | `0.58` |
| no eligible cycles | `72.33%` | `63.39%` |
| SM throughput | `27.28%` | `32.48%` |
| NCU partition duration | `2.30 ms` | `1.85 ms` |

The same specialization is valid for TP2. TP2 changes the per-rank shape from
`Hq=6,Hkv=1` to `Hq=12,Hkv=2`, but preserves `q_per_kv=6`; each KV head is an
independent `blockIdx.y` CTA. The production gate therefore keys on G6 rather
than a hard-coded TP4 head count. Matched exact-M5 microbenchmarks are:

| TP2 context | P1024 baseline | P1024 dual CTA | change |
|---:|---:|---:|---:|
| 64K | `2.4330 ms` | `1.7060 ms` | `-29.88%` |
| 128K | `4.3761 ms` | `3.0106 ms` | `-31.20%` |
| 261888 | `8.1997 ms` | `5.8409 ms` | `-28.77%` |

Direct same-process comparisons at all three contexts are bitwise equal. The
route test covers both TP4 `Hq=6,Hkv=1` and TP2 `Hq=12,Hkv=2`.

The matched TP2 full-model gate uses the same 128K FP8-weight/FP8-KV/MTP4
configuration as the TP4 quantization-independent gate. Only the dual-CTA
rollback changes between processes:

| TP2 128K route | TPOT | steady decode | acceptance length / rate |
|---|---:|---:|---:|
| original P1024 | `22.7660 ms` | `43.925 tok/s` | `4.94231 / 98.56%` |
| P1024 dual CTA | `18.6244 ms` | `53.693 tok/s` | `4.94231 / 98.56%` |

This is `-18.19%` TPOT and `+22.24%` throughput. Both measured repeats on
both routes have the same token hash and report `is_corrupted=false`.

The matched 128K TP4 endpoint used TurboMind AWQ, FP8 E5M2 KV, MTP4,
Flash-V100 for target and drafter, official sampling, prefix caching, Mamba
align, chunk size 8096, and non-eager CUDA graphs:

| Route | TPOT | steady decode | acceptance length |
|---|---:|---:|---:|
| P1024 baseline | `13.2808 ms` | `75.298 tok/s` | `4.98077` |
| P1024 dual CTA | `11.7291 ms` | `85.258 tok/s` | `4.98077` |

This is `-11.68%` TPOT and `+13.23%` decode throughput. Both measured repeats
have the same 256-token hash and per-position acceptance as the baseline. The
round saving is `7.73 ms`; the 16-layer microbenchmark predicts `8.08 ms`, so
the kernel gain is reflected in the endpoint rather than being hidden by a
microbenchmark mismatch.

The specialization is independent of weight quantization. A second matched
128K run used Qwen3.6-27B-FP8 weights with TurboMind FP8 GEMMs and FP8 E5M2 KV;
the log reported `0 AWQ / 35 FP8` warmup calls while all four ranks selected
the same P1024 dual-CTA verifier route:

| FP8-weight route | TPOT | steady decode | acceptance length |
|---|---:|---:|---:|
| original P1024 | `13.4579 ms` | `74.306 tok/s` | `4.94231` |
| P1024 dual CTA | `11.9242 ms` | `83.864 tok/s` | `4.94231` |

This is `-11.40%` TPOT and `+12.86%` throughput. Both measured repeats retain
the same token hash and report no corruption. The gate depends only on the
runtime attention/KV shape, not AWQ or FP8 weight metadata. Current production
scope still requires FP8 E5M2 KV; FP8 weights with FP16 KV are a separate,
not-yet-enabled kernel gate.

The natural `macos_6k_code` gate stopped normally after 26,708 tokens, produced
complete closed HTML/CSS/JavaScript plus a feature summary, and decoded at
`118.674 tok/s` with aggregate acceptance length `3.92`. The old baseline was
`114.167 tok/s` and `3.7896`, so acceptance did not pay for the speed gain.
Its only initial gate failure was an absolute 20-character repetition check:
the legitimate HTML fragment `class="settings-row` appeared 87 times in an
83,391-character application. The check now scales with output length while
the 50/100-character, same-token, same-character, malformed-tag, code-fence,
HTML-closure, and natural-EOS checks remain unchanged. Re-evaluation passes
this output and still rejects the P256 failure (`repeat20=19921`,
`repeat50=19906`, `repeat100=19881`).

Default scope is deliberately exact: five rows, G6, D256, page 1616, and FP8
E5M2 KV. It covers TP2 and TP4 by deriving head count from G6 rather than the
weight quantization or TP size. `VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA=0` restores the
original planner and P1024 kernel. `VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE`
exists for explicit experiments, but its production default is 1024; P256 is
not quality-approved as a global route.

Artifacts:

- `bench_results/mtp_verifier_xqa_20260719/m5_p1024_dual.json`
- `bench_results/mtp_verifier_xqa_20260719/m5_tp2_p1024_baseline.json`
- `bench_results/mtp_verifier_xqa_20260719/m5_tp2_p1024_dual.json`
- `bench_results/mtp_verifier_xqa_20260719/tp2_fp8weights_fp8kv_mtp4_i128k_p1024_baseline.json`
- `bench_results/mtp_verifier_xqa_20260719/tp2_fp8weights_fp8kv_mtp4_i128k_p1024_dual_default.json`
- `bench_results/mtp_verifier_xqa_20260719/m5_p256_dual.json`
- `bench_results/mtp_verifier_xqa_20260719/ncu/m5_xqa_fp8_n128k_p1024_dual_full.ncu-rep`
- `bench_results/mtp_verifier_xqa_20260719/tp4_awq_fp8kv_mtp4_i128k_p1024_dual_steady.json`
- `bench_results/mtp_verifier_xqa_20260719/tp4_fp8weights_fp8kv_mtp4_i128k_p1024_baseline.json`
- `bench_results/mtp_verifier_xqa_20260719/tp4_fp8weights_fp8kv_mtp4_i128k_p1024_dual_default.json`
- `bench_results/mtp_verifier_xqa_20260719/quality_tp4_awq_fp8kv_mtp4_p1024_dual/`
- `bench_results/mtp_verifier_xqa_20260719/quality_tp4_awq_fp8kv_mtp4_p256_dual/`
