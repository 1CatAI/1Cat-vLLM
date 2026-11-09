# DFlash2 verifier acceleration audit, 2026-09-06

## Contract and interpretation

Continue Draft PR #517 from `53be620005fb0f7664dcd27a512979191b342c73`,
integration `755baae1d075ee04fa9096b23fc0225b23589a86`. This audit measures
cost and numerical behavior separately. Production changes are limited to
two diagnostic defects; no inference arithmetic or precision default changes.

The model pair remains QUASAR Qwen3.8-27B all-NVFP4 revision
`d8e6fbfa3e3a78899b440222b827430045a05b44` and DFlash2 revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. Execution is SM70 W4A16,
FP32 logits, E4M3 target KV, FP16 draft KV, Flash-V100, probabilistic
draft7/q8, TP4, maximum length 262144, batch budget 4096, maximum requests 4,
memory 0.8, prefix cache and Mamba align. V2 runner, target full/piecewise
graphs, full draft graphs, context pipeline and context KV graphs are enabled.
The environment is CUDA 12.8, Torch 2.10.0+cu128, Python 3.12.13, Triton 3.6.0.

GPUs 4–7 are exclusively leased because GPUs 0–3 belong to another task.
Local component comparisons use logical CUDA 0, physical GPU 4, while loading
each of the four logical weight shards in turn. They exclude TP communication.
Do not compare their absolute latency to a different GPU set as a speedup.

References use identical inputs and checkpoint weights. FP32/FP64 arithmetic
references are not an unquantized QAT teacher. No dataset quality-loss
percentage is inferred from relative L2, TV, token flips or shorter output.

## Route-by-route ledger

| Acceleration path | Numerical evidence | Cost evidence and disposition |
| --- | --- | --- |
| NVFP4 QPN2 and fused gate/up/SiLU | Previous 64-layer audit covers 256 fused / 496 logical projections against the same checkpoint. Nonzero rounding and TP partition effects remain. | Retain previous source and matrix evidence; no new isolated timing. A BF16 teacher comparison remains open. |
| GDN QKV packing, split and integer metadata | Previous packing and q8 accepted-state probes match their references, including accepted selectors 1–8 and untouched states. | Retained evidence; no new isolated timing. Do not infer floating-point quality from an integer route hit. |
| GDN one-pass gated norm | New 24 real-input rank/step cases reproduce the live fused output exactly; fused/native FP16 outputs can differ. | New q8 local graph timing: 2.231 us fused vs 25.428 us eager native. Full-model attribution is qualified below. |
| Gemma fused residual/RMS | New 144 admitted real-input cases reproduce live fused outputs exactly; residual sums equal the staged reference. Normalized FP16 outputs can differ. | New local graph timing: 3.620 us fused vs 26.846 us eager reference. The reference timing is not the compiled production fallback. |
| Grouped Flash-V100 E4M3 verifier | Previous paired/scalar E4M3 conversion path is bitwise equal through 256K. Comparison to FP32 attention on the same KV has ordinary reduction error. | Retained grouped-path timings:36.25 vs45.57 us at 1032 context;1089.13 vs1617.41 us at 262144. These are not end-to-end round times. |
| TP all-reduce | Previous real partials match FP32 summation followed by FP16 output, across all ranks. Earlier local partial rounding is not recovered. | No new isolated collective timing. Do not count synchronization waits as arithmetic cost. |
| QPN8 top64 plus FP32 head rerank | New 535 target/draft rows: no local top21 or required global top-k misses; no target support changes. | New local q8 graph timing: 573.4–575.1 us vs 992.6–1010.6 us native dense FP32 head, excluding TP collectives. Retain the search with finite-coverage limits. |
| Top-k/top-p boundary protection | Previous ambiguous-cutoff regressions cover ties and scan rounding. All target groups in the new535-row head probe are unambiguous. | Retain the guard. Its earlier small-probe CPU measurement must not be confused with the preceding GPU wait. |
| Sparse target rejection | New 60 independent real q8 rounds cover emitted counts 1–8; token decisions and counts equal dense rejection and the actual captured sampler. |15.242 us sparse including top-p vs 43.530 us dense rejection alone. The dense timing excludes its separate top-k/top-p step. |
| Identity-index and padded-stride copy removal | Previous padded-sentinel tests and the new real sampler replay preserve the same inputs and decisions. | Retain; do not add component deltas to whole-round latency. |
| Output-sharded context FC | Previous real TP4 output reproduced bitwise; same-family TP2/TP4 partitions show small rounding differences. | Retained previous evidence; this is not a new TP2 endpoint. |
| Context compute/store and metadata graphs | Previous paired complete request boundaries are bitwise equal through candidates, q caches and acceptance, with matching full output. | Retain previous real-boundary and production timing evidence. |
| Lookup augmentation | Previous positive-agreement probability-bookkeeping repair preserves the deciding prefix's q. Default q8 does not use lookup. | No new lookup change or timing in this audit. |

Previous evidence and its exact limitations are in the
[operator audit](sm70_quasar_dflash2_operator_audit.md),
[E4M3/FP32 report](sm70_quasar_e4m3_fp32_logits.md),
[TP/QAT audit](sm70_quasar_tp2_tp4_quality.md), and
[proposal/context audit](sm70_dflash2_fastpath_numerics.md).

## Head coverage and real rejection

The head probe uses 465 target rows and 70 actual draft rows from the retained
fixed-prefix and real-request captures. Target support is global top20;
draft support is top16. All four local head shards use the actual QPN8
packing, top64 search, indexed FP32 rerank and dense-vocabulary ordering.
The independent oracle multiplies identical FP16-materialized weights in
FP32. All local top21 candidates are covered, required global top-k sets
match, and maximum protected target sampling TV is 1.2456439e-6.
There are zero changed target top-p supports. This extends observed coverage;
it does not prove that approximate top64 search can never miss a candidate.

The rejection probe uses actual aligned p, q, proposed IDs, request slots and
positions from the natural MBPP28 request. It scatters those supports into
the full 248320-token vocabulary, applies the public top-k/top-p reference,
and executes the production dense rejection kernels. All 60 independent
rounds reproduce sparse tokens and emitted counts, including rejection at
every depth and full acceptance. The diagnostic directory originally holds
240 files because all four workers incorrectly identified as rank0; these
are 60 rounds with four replicas, not 240 independent samples.

Head timing uses 16 graph nodes and eight replays per observation, five
observations. Rejection timing uses 32 nodes and 16 replays per observation,
seven observations. Both use resident inputs and CUDA events, with warmup.
They are component measurements, not wall-clock service improvements.

## Norm arithmetic and the attribution limit

Real norm inputs come from three q8 decode snapshots, all four ranks, seven
target layers for Gemma and two GDN layers. The first layer's post-attention
residual is FP16 and does not enter the fused Gemma gate; it is excluded.
The admitted 168 cases all reproduce their live fused outputs exactly.

| Operator | Fused vs staged FP32 reference | Correctly rounded FP64 comparison |
| --- | --- | --- |
| Gemma residual/RMS | Residual exact; max normalized relative L2 1.05e-5, max absolute 0.001953125 | 468 fused vs 444 fallback output elements differ from FP64 over 144 cases; neither path is exact |
| GDN gated RMS | Maximum relative L2 approximately 6.94e-6, max absolute 7.63e-6 over 24 cases | 32 fused vs 23 fallback elements differ from FP64; maximum absolute error is 3.05e-5 for both |

The FP64 Gemma oracle retains the FP32 residual-addition boundary and then
evaluates normalization in FP64. The GDN oracle evaluates RMS, weights and
SiLU in FP64. These finite local comparisons do not establish which full
model has better text quality. Do not simply switch to the slower reference.

Full-model experiments use identical forced prefixes for MBPP28 and MBPP3:
one prefill logit followed by 16 q8 rounds, 129 positions per case. Inputs,
positions and query boundaries are checked explicitly. Selected snapshots
also verify identical replicas within each TP configuration. Temperature 1,
top-k20 and top-p0.95 are applied to the saved native FP32 target logits.

| Diagnostic comparison | MBPP28 maximum TV | MBPP3 maximum TV | Greedy changes / top-p support changes |
| --- | ---: | ---: | --- |
| Gemma fusion off vs optimized |4.5527%|1.9551%|0/258 greedy;3 support rows |
| GDN norm fusion off vs optimized |2.6499%|1.5832%|0/258 greedy;2 support rows |
| Gemma off with tensor-copy capture disabled |4.3382%|1.7690%|0/258 greedy;4 support rows |
| Same optimized configuration, repeated startup |4.3292%|0.6145%|0/258 greedy;1 support row |

**The same-configuration control also drifts. These measurements cannot
isolate either norm switch as the cause of the full-model differences.**
For example, in the first q8 Gemma comparison, layer 0/rank 2 GDN core output
already differs at 52 elements, before any affected Gemma fusion. Its local
input norm and Z are equal; the difference spreads through projection and
all-reduce. The repeated startup has the same complete natural 297-token
MBPP28 output, while its forced-prefix logits still differ. This is a
repeatability gate for the state/prefix diagnostic, not proof that ordinary
production has a 4.33% sampling defect.

The tensor-copy-disabled variant still retains Python forward wrappers and
the forced-prefix intervention. Both corresponding natural requests reach
the diagnostic 512-token cap; they are not accepted natural-stop quality
evidence. A separate service without a worker extension, layer/selector
capture or forced prefixes is used for production closure.

The next attribution check should isolate the first GDN state divergence
with a fully reset single request and matching prefill state, then repeat
the same q8 update before changing a norm implementation. An aggregate
benchmark score or a stable top-1 cannot replace that check.

Coupled production target-Gumbel draws at seeds 0–63 also change: the Gemma
diagnostic has 20 differing draws over 16512 positions (one at seed 0), and
the GDN diagnostic has 20 (none at seed 0). These are conditional target draws
on the saved prefixes, not full DFlash rejection or a free-generation score.
The repeated-startup confound also applies to their attribution.
The same-configuration repeated startup itself changes20/16512 paired
target draws, including one at seed0, so those flip counts alone cannot
identify either norm as the cause.

## Uninstrumented production closure

A separate service uses no worker extension, layer/selector capture or
forced-prefix hook. The pinned sampling contract is temperature 1, top-k20,
top-p0.95, natural EOS, maximum output 1024; release1k uses seed 20260925 and
MBPP28 seed 0. After one warmup, three measured repetitions give:

| Request | Median round ms | Pure decode tokens/s | TTFT seconds | Output tokens / rounds | Emitted tokens per round |
| --- | ---: | ---: | ---: | ---: | ---: |
| release1k, input 1019 |19.505|154.434|0.3563|248 / 82|3.024 |
| MBPP28, input 135 |19.092|234.829|0.1187|270 / 60|4.500 |

All three repetitions of each request have identical complete token hashes,
natural stop and a nonempty final answer. MBPP28 returns the correct
`n * (n + 1) * (n + 2) // 6` implementation. The final hashes are
`0759c9a5199126539653edae14393addf3bfd8ae6e0a42a56457ea49bd8ee584`
and `e965b6253b702563d53be0a3084030304eff43a7e34c43df9c17fc15e8f46fbc`.

These physical GPUs 4–7 and output trajectories differ from the historical
GPUs 0–3 record (303/260 tokens, 18.892/18.435 ms). This is a new hardware-set
baseline, not a matched performance regression or a 17.6–18 ms restoration.
Four resolved native-library hashes match the preceding audit. Cross-startup
and cross-GPU-set output parity remains open; neither the diagnostic results
nor this short stable production repetition proves broad output quality.

## Prefill PR history and repeatability recheck

The follow-up reviewed the public PR records against main
`95205a2d9952813aa7469f63ff65b8f2813c027a` and audit runtime source
`11d6b6b9dce15d8bf89d6f4509b0f8136274a653`. The Flash-V100 companion
repository has no PR records. No applicable, validated, unmerged prefill
repair was identified for this QUASAR dense GDN/full-attention route.

| PR | Repair or evidence | Applicability and integration |
| --- | --- | --- |
| [#202](https://github.com/1CatAI/1Cat-vLLM/pull/202) | Two-phase P commit removes a paged-prefill shared-memory race for D64/D128. | Already in main and the tested audit source. D256 uses separate P storage. |
| [#226](https://github.com/1CatAI/1Cat-vLLM/pull/226) | Aligns WMMA accumulators and shared-memory base; six-replay D128 regression. | Already in both trees; 32-byte accumulator alignment and the assertion remain present. |
| [#219](https://github.com/1CatAI/1Cat-vLLM/pull/219), [#350](https://github.com/1CatAI/1Cat-vLLM/pull/350) | FP32 XQA probabilities for FP16 KV; restoration of D256 prefill operators. | Already in both trees; neither is a new E4M3/GDN-state fix. |
| [#403](https://github.com/1CatAI/1Cat-vLLM/pull/403) | Records Flash Next QSA prefill quality evidence. | Already integrated; its matched quality claim concerns a different model route. |
| [#434](https://github.com/1CatAI/1Cat-vLLM/pull/434), [#408](https://github.com/1CatAI/1Cat-vLLM/pull/408) | Legacy-runner hybrid prefill dispatch; Flash Next/cache correctness repairs. | Already integrated. The legacy-runner dispatch fix does not execute in this MRV2 run. |
| [#494](https://github.com/1CatAI/1Cat-vLLM/pull/494), [#525](https://github.com/1CatAI/1Cat-vLLM/pull/525) | QSA logical-page ordering and its NVFP4 model validation. | Both are in main; this Qwen3.5-family 27B model does not execute QSA. |
| [#524](https://github.com/1CatAI/1Cat-vLLM/pull/524) | Experimental E4M3 grouped attention with FP32 partials. | Remains Draft with a failed model token gate; not admitted as a quality repair. |

Main advanced from `4366d9d5fe80eeaf79575b51ec36a6a032673df0` when another
task merged #525 during this review. Its shared CUDA-file change is limited
to the Qwen4Exp HC down scatter kernel; it does not change this model's
ordinary TP all-reduce or the paged-prefill kernel.

The runtime library is the retained `lib-final` artifact with SHA256
`c3f3bef28a21f681d3d3d84e65d5f208b9d2c282b2c4bfe7cb5f7e221d55802e`.
Its retained compile and link logs build the paged-prefill object from this
owned source tree. The paged-prefill source is unchanged against current
main and contains both old fixes. Thus the observed 4.33% drift was already
measured with those repairs; merging them again supplies no new intervention.

CPU reanalysis of the retained captures also narrows the causal gap:

- Both runs execute the full identical 135-token MBPP28 prompt. Their final
  prefill hidden states already differ, before any forced q8 acceptance
  update. The first q8 layer-0 QKV projection is exact across starts, while
  rank 2's GDN core has 52 different elements, maximum absolute 3.8147e-6,
  before the audited output normalization.
- At position 212, the top-20 distribution's first two candidates have
  cumulative mass 0.94912833 versus 0.95171565. Crossing top-p 0.95 removes
  token 23, whose sampled mass was 0.04329225; recomputed TV is 0.04329227.
  Full-vocabulary softmax TV at that position is already 0.02030133.
- The original probe omitted prefill layer tensors and incoming conv/SSM
  state. It cannot distinguish a prefill arithmetic difference from state
  propagation or core execution. It does not prove stale state or a race.
  The fixed FP8 tuning experiment in #524 concerns another checkpoint:
  this NVFP4 prompt M135/M153 exceeds the default dense tuning maximum M16.
  Do not transfer that experiment's causal conclusion to this model.

No runtime source, production flag or integration branch changes were made
by this history recheck. No fresh GPU replay was run while all GPUs were
owned by other tasks. Preserve the pending state-replay gate; the historical
two passing D128 tests in #226 are not a new model-level validation.
PR snapshots and ancestry/library checks are retained in bundle
`v100-dflash2-prefill-pr-audit-20260906`; the CPU reanalysis is in the previous
verifier bundle's `repeatability-cause` directory and exited successfully.

## Diagnostic repairs and validation

`sm70_gdn_projection_dump` returned its input despite a non-aliasing custom-op
schema. `torch.library.opcheck(test_schema)` reports the alias violation.
Its eager/captured and fake implementations now return owned storage.
Tests cover schema/FakeTensor, AOT preservation of a live input, and changed
inputs under CUDA Graph replay. This affects enabled diagnostics only.

The alignment dumper used only environment rank variables. Multiprocess
workers without those variables all fell back to 0. It now uses the initialized
distributed rank first and preserves the environment fallback before process
group initialization. Tests cover ranks 0–3, stale environment values and the
uninitialized fallback. Three GDN dump GPU tests and seven rank tests pass.

The retained old alignment files are explicitly deduplicated; use
`sparse-real-cost-v2.json`. The first norm table included the non-admitted
first-layer residual case; use `norm-real-cost-v2.json`. The initial head
probe failed before measurements because inference buffers were updated
outside inference mode; the corrected probe uses the same inference-mode
contract as serving. These are diagnostic failures, not target-model defects.

## Reproduction

Raw bundle: `v100-dflash2-verifier-route-audit-20260906`, under the task cache
recorded in the local handoff. `queue/*.done.json` records exact commands,
environment, GPU ownership and logs; `provenance.json` retains native/JIT
hashes. Use the owned worktree's `.venv/bin/python` under an exclusive lease.

- `head_coverage_cost.py`: native candidate coverage, FP32 reference and costs.
- `sparse_real_cost.py`: independent aligned rounds, actual decisions and cost.
- `norm_real_cost.py` and `norm_fp64_oracle.py`: admitted real norm shapes.
- `run_matrix.py`, `run_minimal_matrix.py`, `run_repeat.py`: separately started
  diagnostic configurations, with request-ID-bound forced-prefix tapes.
- `analyze_routes.py`, `analyze_minimal_routes.py`, `analyze_repeat.py`: strict
  prefix validation and native target-distribution comparisons.
- `route_sample_flips.py`: paired production Gumbel draws on saved distributions,
  checked against dense sampling at seed 0; not full speculative decoding.
- `serve-production.sh` and `run_production.py`: independent uninstrumented
  production closure, natural EOS, one warmup and three measured requests.

The broader QAT teacher question remains open, and single-path full-model
attribution requires the state and prefix repeatability gate above. The user
subsequently requested mainline integration of PR #517's validated repairs
with these limits retained. That integration does not close the remaining
quality investigation or promote the optional precision/performance flags.
Merge-time checks and the exact integrated revision are recorded on #517.
