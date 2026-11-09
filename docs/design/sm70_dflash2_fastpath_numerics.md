# DFlash2 fast-path numerical audit, 2026-09-06

## Scope and frozen execution

Continue Draft PR #517 from `41a9018e9f715989b25b8f3c3ff436ffe57a13fe`,
integration `onecat/main` at `755baae1d075ee04fa9096b23fc0225b23589a86`.
The production change in this audit only repairs probability bookkeeping in
the optional lookup path. Target GEMMs, attention, KV formats, selector
arithmetic and the context-pipeline implementation are unchanged.

Target: QUASAR Qwen3.8-27B all-NVFP4 checkpoint revision
`d8e6fbfa3e3a78899b440222b827430045a05b44`, executing W4A16 on SM70.
Draft: DFlash2 revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, five
layers, hidden 5120, block8/draft7, selector rank 256/top16. The environment
is V100-SXM2-32GB GPUs 0–3, TP4, CUDA 12.8, Torch 2.10.0+cu128,
Python 3.12.13, Triton 3.6.0. Target KV is E4M3; draft KV is FP16.
`VLLM_SM70_DFLASH2_FP32_LOGITS=1` preserves the previous head precision fix.

The real request uses MBPP28, input 135, temperature 1, top-k 20, top-p 0.95,
seed 0, natural EOS, cap 512, thinking/xhigh. Model length 262144, batch
budget 4096, maximum requests 4, one active request, memory 0.8, prefix cache
and Mamba align. The V2 runner uses full/piecewise target graphs and full
draft graphs. Worker logs confirm Flash-V100 E4M3 q8 grouped verification,
FP32 candidate rerank, sparse rejection, BF16 draft emulation and TP4
output-sharded context FC. The paired context-graph flags are recorded with
each launch. Numerical capture synchronizes CPU copies and is excluded from
latency evidence.

## Confirmed lookup probability defect and repair

Affected configuration: probabilistic lookup-augmented drafting,
`ngram_assist=true`, verifier width greater than the trained draft width,
`VLLM_DFLASH2_LOOKUP_AGREE > 0`, and a match shorter than
`VLLM_DFLASH2_LOOKUP_NSTRONG` that qualifies through neural-prefix agreement.
The standard q8 request does not enable lookup. The default agreement
threshold 0 also does not exhibit this defect.

Previously, the lookup fusion decided whether to replace proposals using
the just-sampled neural prefix, then rewrote every replaced row's proposal
distribution as a point mass. The agreeing prefix already contained those
tokens, but changing its probability from the neural q to 1 invalidated the
acceptance ratio. This is an algorithmic distribution error, not FP16 noise.

A two-token counterexample uses q(A)=q(B)=0.5 and p(A)=0.8. A weak historical
continuation begins with A and requires one agreeing proposal. When the
draft draws A, the old code records q(A)=1; when it draws B, q(B) remains 0.5.
The resulting output probability for A is
`0.5 * 0.8 + 0.5 * (1 - 0.2 / 0.5) = 0.7`, rather than 0.8.

The repair preserves the original q for the agreeing random prefix. Only
positions after that prefix become point masses conditional on the already
sampled prefix. Strong matches can still replace the whole block because
their decision depends on request history. Proposed token sequences remain
identical; the correction probabilities change where necessary.

Actual production lookup, point-mass and dense-rejection kernels, 100000
independent seeds, show:

| Proposal policy | Observed P(A) | Target P(A) |
| --- | ---: | ---: |
| History-only lookup control | 0.799110 | 0.8 |
| Agreement-conditioned lookup, old bookkeeping | 0.700270 | 0.8 |
| Agreement-conditioned lookup, repaired bookkeeping | 0.800720 | 0.8 |

The committed regression repeats the repaired experiment for both dense and
sparse rejection. It also verifies strong/weak matches, greedy/probabilistic
fusion, unchanged proposed tokens, and the prefix's point-mass mask. All 12
tests in `tests/v1/spec_decode/test_dflash2_lookup.py` pass on V100.

## Real selector arithmetic and sampling

Ten actual request snapshots contain 70 conditional proposal rows. Startup
`_warmup_*` requests are excluded. The FP32 arithmetic reference uses the
same captured backbone states, FP16-materialized checkpoint codebooks and
hidden-projection weights, and the same captured FP32 unary logits. It
changes accumulation/intermediate rounding within the selector, not the
candidate support, LM-head or draft backbone. It is not a BF16 teacher.

| Comparison | Maximum or count |
| --- | ---: |
| Compiled selector vs eager, same arithmetic | Bitwise equal |
| Selector lattice vs FP32 arithmetic, max absolute | 0.0117207 |
| Bilinear edges vs FP32 arithmetic, relative L2 | 0.000362991 |
| Conditional proposal TV at temperature 1 | 0.000953654 (0.0953654%) |
| Conditional greedy top-1 changes | 0 / 70 |
| Temperature1, seeds 0–255, FP32 counterfactual draft-token changes | 3 / 17920 |
| Temperature0.6 diagnostic, seeds 0–255, counterfactual changes | 10 / 17920 |

The 13 differing draft positions occur in five proposal chains; some are
downstream consequences of an earlier selector change. Seed0 does not flip
in these samples. These are changes in the draft proposal, not 13 final
target-token errors. They can affect acceptance and latency. Correct
rejection sampling must correct the actual q drawn by the selector; a draft
FP32 conversion alone is not evidence of improved target output quality.

Separately, the existing fused prefix/tail selector and its persistent CUDA
Graph were compared against sequential full-vocabulary Gumbel sampling,
using actual candidate IDs and positions, seeds 0–63, temperatures1 and0.6.
All 8960 proposal positions match. Realized scores, request-slot sparse cache
and their dense-cache entries match exactly, including temperature scaling.
The seed 0/temperature 1 replay also matches the captured model proposal.

An additional 96-round cache-overwrite probe alternates eager/graph launches,
intersecting and reordered supports, permuted request slots, an invalid
request row, and padded dense strides. All 288 checked request-slot outputs
match an independent dense scatter; padding remains intact. No cache-scatter
ordering defect was reproduced. This finite probe is not a proof for every
supported shape or concurrent request transition.

## Context FC partition arithmetic

The FC projects concatenated target layers `[5,19,33,47,61]` from 25600 to 5120.
At M8, TP4 changes local output width to 1280 and uses TurboMind plus an
all-gather. It does not split the K reduction. Nevertheless, the changed N
can change the GEMM schedule and FP32 accumulation order before FP16 storage.

Three fresh decode captures (snapshot indices 3,7,11) reproduce the observed
TP4 projection bitwise from independently packed local checkpoint shards.
The following are local arithmetic comparisons on identical full inputs;
the TP2 variant is a counterfactual output partition, not a TP2 endpoint.

| Comparison | Maximum relative L2 |
| --- | ---: |
| TP4 TurboMind vs FP32 matrix multiplication | 0.000220945 |
| TP4 TurboMind vs replicated cuBLAS FP16 | 0.0000177354 |
| Same comparison after BF16 context normalization | 0.000105950 |
| TurboMind TP2 vs TP4 output partitions | 0.00000921025 |

TP2/TP4 maximum absolute difference is 0.0625. BF16 rounding can amplify a
small FC difference around a rounding boundary. Changed-input graph replay
matches eager bitwise over 24 replays. Prior C2 snapshots independently show
the same shape-dependent effect; the current table uses fresh E4M3 data.
No end-to-end target-quality attribution to this FC difference is established.

## Cost and quality boundaries

Two fresh services differ only in enabling
`VLLM_SM70_DFLASH2_CONTEXT_PIPELINE=1` and
`VLLM_SM70_DFLASH2_CONTEXT_KV_GRAPH=1`. Final worker logs confirm the staged
context computation, deferred writes and B1 metadata refresh graph. Across
all ten paired real-request snapshots, target/auxiliary states, FC outputs,
draft inputs and backbone states, candidates, unary/lattice scores, proposal
tokens, realized/cached q scores and acceptance/rejection counts are bitwise
equal. The snapshots include partial acceptance (2, 3, 4, 6 and 7 emitted
tokens) and full q8 emission. Both services produce the same complete
260-token output, 49 rounds and 211 accepted draft tokens. This extends the
earlier isolated context-graph check through the real selector and sampler
boundary. It remains a short B1 request, not a long-context or request-reuse
proof.

The lookup repair adds no kernel launch and affects no arithmetic in the
ordinary q8 baseline. A captured fusion-kernel benchmark uses 256 nodes per
graph, eight replays per observation and seven observations. B1 medians are
1.735–1.736 us before and 1.741–1.742 us after the repair; B4 medians are
1.818–1.829 us before and 1.818–1.834 us after. The largest paired increase
is 0.006 us. This is only fusion-kernel cost, not end-to-end round latency.

The fresh baseline request naturally emits 260 tokens over 49 draft rounds,
with 211 accepted draft tokens. Its complete token hash matches the previous
precision/recovery control:
`78648509da0a573cb79264412ffd477af13cf1be615019ab573c4975ad4ec908`.
This single code request is an execution check, not a broad quality score.

Historical unprofiled E4M3/FP32 medians remain 18.435 ms for MBPP28 and
18.892 ms for release1k, at `f22ac115d0`. This audit does not claim recovery
to 17.6–18 ms. No new long-context endpoint, QAT teacher comparison, or
acceptance-rate improvement from FP32 selector arithmetic is claimed.

## Reproduction and retained artifacts

Owned worktree:
`/home/ymzx/桌面/1cat-vllm/worktrees/v100-quasar-dflash2-operator-audit-20260905-172402`.
Raw bundle:
`/data/minimax-h3/task-cache/v100-dflash2-fastpath-numerics-20260906`.
The bundle contains exact job commands/environment in `queue/*.done.json`,
launchers and analysis scripts, logs, real snapshots, JSON results, and
native/JIT hashes in `provenance.json`. Run GPU scripts under an owned lease
with `job-env.json` and the worktree's `.venv/bin/python`.

- `scripts/cache_lookup_probe.py`: old-policy counterexample and cache probe;
  old lookup behavior is preserved by `probabilistic=False` in the wrapper.
- `scripts/cache_lookup_fixed_probe.py`: repaired probability experiment.
- `scripts/real_selector_probe.py`: current graph/cache vs sequential oracle.
- `scripts/selector_fp32_flip_probe.py`: changed-arithmetic draft sampling.
- `scripts/context_fc_fresh_probe.py`: fresh same-input FC comparisons.
- `scripts/compare_pipeline.py`: paired real boundary and full token equality.
- `scripts/lookup_cost.py`: captured fusion-kernel timing.
- `scripts/serve-capture.sh` and `scripts/serve-optimized-capture.sh`: baseline
  and context-pipeline capture services on task-owned port 18145.

The first real-selector analysis correctly rejected four-request startup
warmup data. The retained version filters request IDs, not just tensor
shapes. `context-fc-fresh-v2.json` similarly excludes warmup snapshots; do
not use the earlier warmup-containing table as real-request evidence. One
generated offline probe had a syntax error before GPU execution; the
corrected script and successful job are retained. Neither diagnostic failure
is classified as a model defect. Keep the PR Draft pending the broader
quality/performance gates.
