# SM70 DFlash DDTree Plan for Qwen3.6-27B-AWQ

This note records the DDTree reference checkout and the local integration plan
for accelerating Qwen3.6-27B-AWQ on V100/SM70. The goal is a vLLM-native
`dflash_ddtree` path that improves the DFlash16 verifier economics without
regressing exactness, long-context state, or the existing flat DFlash path.

## Reference Checkouts

The two public references were cloned under `third_party/ddtree_refs/`:

| Path | Upstream | Commit | License | Role |
|---|---|---|---|---|
| `third_party/ddtree_refs/official_ddtree` | `https://github.com/liranringel/ddtree` | `c96427a185677bf4133ed865dd1626a5041aef9b` | MIT | Algorithm reference for tree build, visibility mask, verifier walk, and cache compaction. |
| `third_party/ddtree_refs/aeon_qwen36_ddtree` | `https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DDTree` | `d2610ae5e42ddcbccbf8cd800c67644d3c7843f6` | Apache-2.0 | vLLM integration map, prototypes, tests, and Qwen3.6 hybrid-state cautions. |

Treat both as references. Do not run the AEON overlay scripts against this tree:
they patch a different GB10/Blackwell branch and include research-only DDTree
paths that are explicitly not production safe.

## Local Baseline

Current 27B-AWQ evidence, all TP2 on GPU2/3 with `FLASH_ATTN_V100`:

- No-spec greedy baseline:
  `bench_results/dflash_27b_k_sweep_20260620/27b_awq_nospec_i4096_o256_greedy_mnbt8192.json`
  measured `steady_decode_tps=55.86159315981125`.
- DFlash16 full-GDN greedy:
  `bench_results/dflash_27b_k_sweep_20260620/27b_awq_dflash16_i4096_o256_greedy_mnbt8192.json`
  measured `steady_decode_tps=34.50339715311688`,
  `acceptance_length=4.642857142857142`, and
  `overall_acceptance_rate=0.22767857142857142`.
- DFlash16 official sampling:
  `bench_results/dflash_27b_k_sweep_20260620/27b_awq_dflash16_i4096_o256_official_mnbt8192.json`
  measured `steady_decode_tps=27.732380043828957`,
  `acceptance_length=3.6956521739130435`, and
  `overall_acceptance_rate=0.16847826086956522`.
- A short DFlash profile showed the slowdown is not caused by repeatedly
  rebuilding the whole 4096-token context after the first round. The bottleneck
  is verifier/draft cost plus low accepted tokens per verifier step.

This makes DDTree the right algorithmic next step, but only if the tree verifier
and Qwen GDN state handling stay exact.

## What Transfers From The References

Useful pieces:

- Official `ddtree.py::build_ddtree_tree`: best-first heap expansion from
  per-position DFlash distributions.
- Official `compile_ddtree_tree`: root plus flattened nodes, position ids, and
  ancestor-only visibility.
- Official `follow_verified_tree`: target-logit walk through children until a
  recovery/bonus token is needed.
- AEON `prototypes/ddtree_tree.py`: cleaner vLLM-shaped tree dataclasses and
  chain-seeded best-first builder.
- AEON `prototypes/ddtree_vllm_metadata.py`: flattened tree token ids,
  parent ids, node depths, compact logit row conventions, and sibling-hiding
  attention mask tests.
- AEON `prototypes/ddtree_gdn_reference.py`: reference semantics for tree
  causal-conv and Gated DeltaNet state, where each node reads its parent state
  and writes its own scratch state.

Parts that should not be copied directly into the hot path:

- The official builder copies top-k logits/probs to CPU and builds the heap in
  Python. That is acceptable for correctness bring-up and instrumentation, but
  not a final V100 decode path.
- The official verifier uses Transformers `DynamicCache` plus a dense 4D mask.
  vLLM needs paged KV, scheduler integration, and Flash-V100-aware attention.
- AEON's patch scripts are overlay scripts for a different branch. The useful
  artifact is the design and prototype logic, not the mechanical patching.

## Local Patch Surface

The early local vLLM path was chain-shaped in the engine hot path. This section
records the bring-up surface; the current runtime defaults are summarized in
the 2026-06-25 ledger below.

- `vllm/config/speculative.py` accepts `method="dflash_ddtree"`. Current
  `dflash_ddtree` defaults enable tree verification; set
  `ddtree_disable_tree_verify=True` explicitly for the flat DFlash fallback.
- `vllm/v1/spec_decode/dflash.py::DFlashProposer` asserts
  `speculative_config.use_dflash()` and still produces one top-1 chain of
  length `num_speculative_tokens`.
- `vllm/v1/outputs.py::DraftTokenIds` carries optional per-request
  `DDTreeDraftPayload` objects aligned with `req_ids`.
- `vllm/v1/core/sched/output.py::SchedulerOutput` carries optional scheduled
  DDTree payloads after strict flat-token matching.
- `vllm/v1/spec_decode/metadata.py::SpecDecodeMetadata` describes
  linear draft tokens, target logit rows, and bonus logit rows.
- `vllm/v1/sample/rejection_sampler.py` and
  `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` both assume a
  contiguous chain. A greedy DDTree sampler now bypasses this only for a narrow
  all-greedy/no-logprobs/no-processor route.
- `vllm/v1/attention/backends/flex_attention.py` has an experimental logical
  DDTree ancestor mask. `flash_attn_v100.py` still has no native tree mask.
- Qwen GDN spec metadata now has a DDTree parent-id channel and state/cache
  slot sizing can follow `ddtree_budget`, but kernels still do not consume
  parent ids or commit selected tree-node state. The scheduler therefore
  refuses branched DDTree payloads for hybrid/Qwen3.6 models.

## Implementation Plan

### M0: Reference Capture

Status: complete. The reference repos above are cloned and this document records
their exact commits and licenses.

### M1: Local Tree Builder And Sampler Tests

Status: complete.

Add local, test-only modules under `vllm/v1/spec_decode/`:

- `ddtree_tree.py`: adapt the AEON dataclass builder, with attribution if code
  is copied.
- `ddtree_metadata.py`: compact tree metadata and CPU/torch helper functions.
- `tests/v1/spec_decode/test_ddtree_tree.py`: budget, chain seed, sibling
  branch, visibility, and greedy-walk tests.
- `tests/v1/spec_decode/test_ddtree_metadata.py`: parent offsets, compact logit
  row mapping, and sibling mask invariants.

Pass criteria:

- No engine behavior changes.
- Pure CPU tests pass in the releasecheck conda environment.
- `budget=16, top_k=1` is exactly equivalent to a DFlash16 flat chain.

### M2: Experimental Config Alias With Flat Fallback

Status: historical flat-fallback milestone complete; current DDTree runtime
defaults were changed after M6 and are listed in the 2026-06-25 ledger.

Add `method="dflash_ddtree"` and explicit DDTree fields:

- `ddtree_budget`: default `None`; if unset, use
  `num_speculative_tokens`.
- `ddtree_top_k`: current default `None`; if unset, use the DDTree budget,
  matching the reference implementation.
- `ddtree_chain_seed`: current default `False`, matching reference best-first.
- `ddtree_disable_tree_verify`: current default `False`; set it only when
  explicitly requesting the flat fallback.

Make `dflash_ddtree` instantiate the same DFlash proposer and return the same
flat `draft_token_ids` while carrying optional tree payload out-of-band. This is
a boot and regression milestone, not an acceleration milestone.

Current local state:

- `DFlashModelTypes` includes `dflash_ddtree`.
- `SpeculativeConfig.use_dflash()` covers both `dflash` and `dflash_ddtree`.
- `DFlashProposer` accepts both methods and records the DDTree knobs.
- Tree verifier remains disabled by default, so the runtime behavior is still
  the flat DFlash16 path.

Pass criteria:

- `method="dflash_ddtree"` with tree verify disabled has the same token hash and
  decode speed as `method="dflash"` within normal noise.
- Existing DFlash16 benchmark command still works unchanged.

### M3: DFlash Top-k Payload

Status: payload and scheduler bridge are present; no tree verifier is active
yet.

Extend the DFlash proposer to produce tree payloads from the same one-pass
DFlash block distribution:

- Force a logits/top-k path only for `dflash_ddtree`; current greedy DFlash can
  avoid full logits by using top-1, but DDTree needs top-k per depth.
- Keep returning the canonical top-1 chain as `draft_token_ids`.
- Attach per-request tree payloads:
  `tree_token_ids`, `parent_indices`, `node_depths`, `node_scores`,
  `top1_chain_token_ids`, `flat_draft_token_ids`, `budget`, `top_k`, and
  `chain_seed`.
- Historical flat-chain parity probe, not the current default:
  `num_speculative_tokens=16`, `ddtree_budget=16`, `ddtree_top_k=1`,
  `ddtree_chain_seed=True`.
- Historical first sweep after parity, superseded by the trace-guided guardrails
  below:
  `ddtree_budget in {16, 24, 32, 48}`, `ddtree_top_k in {2, 4}`.

Pass criteria:

- With `budget=16, top_k=1`, payload-derived flat chain equals current DFlash16
  draft ids.
- Tree payload creation overhead is measured separately from verifier time.

Current local state:

- `vllm/v1/spec_decode/ddtree_payload.py` converts
  `[batch * num_speculative_tokens, vocab]` DFlash logits into per-request
  `DDTreeDraftPayload` objects by reusing `build_ddtree()`.
- `DFlashProposer._sample_draft_tokens()` computes one logits block for
  `method="dflash_ddtree"`, reuses it for current flat sampling, and builds the
  DDTree payload from the same logits.
- `GPUModelRunner` caches the payload out-of-band via
  `take_dflash_ddtree_payloads()`. `DraftTokenIds` and scheduler output now
  carry the payload only when the flat scheduled draft token ids exactly match
  the payload's top-1 chain. Rejection sampling is still the flat-chain path.
- This path currently materializes top-k/logprob data to CPU for correctness
  bring-up. It is not the final performance implementation.

### M3.5: Scheduler And GDN Metadata Bridge

Status: complete as a data path; kernels still ignore DDTree parent metadata.

Current local state:

- `DraftTokenIds.ddtree_payloads` carries optional per-request payloads from the
  worker back to the scheduler.
- The scheduler caches payloads per request, clears them on prefill chunks,
  finish/free, preemption, structured-output trimming, and any token mismatch,
  and emits `SchedulerOutput.scheduled_ddtree_payloads` only for the matching
  scheduled speculative row.
- `ddtree_parent_metadata.py` converts payload parent indices into padded
  root-plus-tree parent-id rows aligned with the active batch.
- `SpeculativeConfig.num_speculative_state_tokens()` returns
  `max(num_speculative_tokens, ddtree_budget)` only when DDTree tree verify is
  explicitly enabled. Scheduler lookahead, Mamba speculative blocks,
  `GPUModelRunner.max_spec_state_slots`, and GDN metadata buffers use this
  state-token count.
- `GDNAttentionMetadata` carries `ddtree_parent_ids` and
  `ddtree_num_tree_tokens_cpu`, and the existing GDN diagnostic dumps include
  those tensors.

Remaining gap:

- Full-attention kernels still need ancestor-only tree visibility.
- GDN kernels still need to compute each node from its parent state and commit
  only the accepted path.

### M4: Attention-Only Verifier Correctness

Status: in progress. The vLLM runner now has the main attention-only pieces,
but the small-model forward oracle and Flash-V100 native tree mask are still
pending.

Before Qwen3.6, prove the tree verifier on a small full-attention model:

- Use a Qwen-family small dense target such as `Qwen/Qwen2.5-0.5B-Instruct`.
- Build a flattened tree, ancestor-only mask, compact logits rows, and greedy
  tree sampler.
- Compare one-pass tree verifier logits against per-path replay logits.

For V100, this can be a slower correctness path first. Do not call it a
performance route until Flash-V100 has a native ancestor-only tree verifier.

Pass criteria:

- `budget=1` equals ordinary one-token greedy verification.
- No sibling leakage in attention.
- One-pass tree logits match per-path replay top-1 for all verifier nodes.

Current local state:

- `tree_from_payload()` rebuilds a verifier-coordinate `DDTree` from
  `DDTreeDraftPayload`.
- `ddtree_verify.py` builds `DDTreeVerifierMetadata`, prompt-plus-tree
  attention verifier inputs, and greedy verification results from compact
  root-plus-node logits.
- Tests cover payload tree rebuild, compact-logit full accept, sibling-branch
  accept, and dense ancestor mask sibling hiding.
- `ddtree_sampler.py` greedily walks compact root-plus-node logits and returns
  `SamplerOutput.ddtree_accepted_node_indices` for accepted-path cache/state
  work.
- `FlexAttentionMetadata` can carry DDTree parent ids and wraps its logical
  causal mask with ancestor-only tree visibility.
- `GPUModelRunner` overrides DDTree tree-node position ids after slot mapping,
  so token storage and KV slot mapping stay linear while RoPE positions follow
  node depth.
- For non-hybrid attention-only models, `GPUModelRunner` compacts accepted
  DDTree KV rows from their tree slots back to the normal prefix slots before
  the next step can read cache.

### M5: Qwen3.6-27B-AWQ Safe Hybrid Bridge

Qwen3.6 has full-attention layers plus GDN/Mamba-style recurrent layers. A tree
attention mask alone is not enough:

- Each tree node must read GDN and conv state from its parent.
- Rejected branches must never mutate persistent state.
- Only the accepted path can be committed to normal vLLM KV and recurrent state.

The first bridge should be quality-first:

- Use tree verification for attention logits where supported.
- Commit only accepted path tokens.
- Replay the accepted path through the existing flat/full-GDN path to update
  recurrent state if tree-aware GDN state is not ready.

This may erase speedup, but it is the correctness bridge. If replay makes it
slower than flat DFlash16, keep it experimental and proceed to M6.

Pass criteria:

- Greedy output hash matches no-spec/flat-DFlash semantics for a fixed prompt.
- Long-context multi-turn smoke does not drift.
- Existing full-GDN guard remains the default unless a tree-aware state path is
  proven exact.

Current local state:

- Scheduler tree-row expansion is guarded by exact payload/scheduled-token
  matching and by token/max-len/long-prefill budget checks.
- `DDTreeDraftPayload.is_flat_chain()` now requires token ids, parent ids, and
  node depths to match a true linear chain; token equality alone is not enough.
- The scheduler rejects branched DDTree scheduling for hybrid models because
  accepted-path recurrent state compaction is not implemented yet.
- Flat-chain-equivalent DDTree payloads are still allowed on hybrid models, so
  `ddtree_budget=16, ddtree_top_k=1` remains a safe DFlash16 parity route.
- `GPUModelRunner` has a hybrid fail-fast guard before recurrent-state
  postprocess. If a branched DDTree payload reaches a hybrid model, the step
  raises instead of silently committing GDN/Mamba state from the wrong linear
  slot.
- `GPUModelRunner` now computes DDTree recurrent-state slot selectors from
  `SamplerOutput.ddtree_accepted_node_indices`. For a flat path this matches
  the old accepted-token count; for a branch it points to the last accepted
  compact tree node plus one. The guard still prevents using this on hybrid
  branched payloads until GDN/Mamba kernels consume parent state correctly.
- GDN metadata now has an internal `spec_state_slot_selectors` field. It
  defaults to the old `num_accepted_tokens` selector for flat MTP, but can
  represent the DDTree selected compact tree slot independently of generated
  token count. This is not yet part of the graph op signature.
- `GPUModelRunner` now keeps generated-token count and state-slot selector in
  separate GPU buffers before building GDN metadata. `num_accepted_tokens`
  remains the actual contiguous generated-token count, while
  `spec_state_slot_selectors` comes from
  `SamplerOutput.ddtree_accepted_node_indices` when DDTree metadata is present.
- Qwen GDN spec core now passes `spec_state_slot_selectors` to
  `causal_conv1d_update()` and `fused_recurrent_gated_delta_rule()` when the
  field is present. This selects the compact state slot independently from
  generated-token count, but it still does not compute each tree node from its
  DDTree parent.

### M6: SM70 Tree-Aware GDN And Flash-V100 Tree Verify

This is the real acceleration milestone for 27B-AWQ:

- Add tree parent ids to GDN attention metadata.
- Add scratch state buffers for per-node conv/GDN intermediate state.
- Add selected-path commit into persistent recurrent state.
- Add or adapt a Flash-V100 verifier path that supports ancestor-only tree
  visibility without falling back to a dense eager mask on every round.

Pass criteria:

- DFlash16 DDTree beats flat DFlash16 and no-spec on the same 27B-AWQ TP2
  benchmark, reporting pure decode separately from prefill.
- Token hash parity holds for greedy.
- Official sampling reports acceptance/tree-depth metrics and no sampler
  distribution shortcut is used.

Current local state, 2026-06-21:

- Scope is now narrowed to normal no-spec baseline versus DDTree tree verify.
  Flat DFlash-only speed tests are intentionally skipped because they do not
  answer whether DDTree acceleration is recovered.
- Added a fused DDTree GDN verifier path:
  `causal_conv1d_update_ddtree()` computes each verifier row from its selected
  parent/root conv state, and the fused sigmoid-gating delta-rule update now
  accepts `ddtree_parent_ids`. Qwen GDN pure-spec verification uses this path
  by default under `VLLM_DFLASH_DDTREE_FUSED_GDN=1`.
- Added `vllm/v1/attention/backends/ddtree_branch_triton.py`, a paged-KV
  Triton DDTree ancestor-mask correction kernel for Flash-V100 small-query
  verifier rows. It keeps root/prefix slot 0 visible for root children and
  works with graph capture when parent ids are already CUDA int32.
- `flash_attn_v100.py` now routes DDTree tree verifier rows through
  `prefill_ddtree_triton` before the old dense fallback. The dense fallback is
  still present only as an eager correctness bridge.
- `VLLM_DFLASH_DDTREE_ENABLE_HYBRID_TREE_STATE=1` is required for full hybrid
  tree-state runs. Without it, the scheduler rejects branched payloads and the
  run falls back to flat speculative rows; those results are invalid DDTree
  tree-verify evidence.
- Graph capture is now proven for the 27B-AWQ DDTree tree verifier by using
  `VLLM_SM70_FLASH_V100_DECODE_GRAPH_CAPTURE_SIZE=1 + ddtree_budget`.

Validated artifacts:

- Normal no-spec graph baseline:
  `bench_results/ddtree_smoke_20260621/27b_awq_nospec_flashv100_graph_i32_o16_w2r3.json`.
  It used Flash-V100 no-compile FULL_DECODE_ONLY graph, no speculative config,
  `input_len=32`, `output_len=16`, `warmup=2`, `repeat=3`, TP2, AWQ, and
  produced stable hash
  `d7e8a039cc9363c9d968bb2b1c0deea9cf6a53fecca39a7fdfdc853a869eaf7f`.
  Mean steady decode was `48.6173 tok/s`.
- Old full DDTree tree verify with dense attention fallback:
  `27b_awq_dflash_ddtree32_topk4_nochain_flashv100_i32_o16.json`, eager,
  `ddtree_budget=32`, `top_k=4`, `chain_seed=false`, spec tokens `16`.
  It produced the same hash, `draft_tokens_per_step=32`, `num_drafts=5`,
  `num_accepted_tokens=11`, acceptance length `3.2`, and mean steady decode
  `4.8949 tok/s`.
- Fused-GDN plus dense attention fallback:
  `27b_awq_dflash_ddtree32_topk4_nochain_fusedgdn_tree_eager_i32_o16.json`.
  Same hash and spec metrics, route summary included `prefill_ddtree_dense=160`,
  and mean steady decode improved to `8.8475 tok/s`.
- Fused-GDN plus Triton branch attention, eager:
  `27b_awq_dflash_ddtree32_topk4_nochain_tritonattn_fusedgdn_tree_eager_i32_o16.json`.
  Same hash and spec metrics, route summary used `prefill_ddtree_triton=160`
  with no dense DDTree route, and mean steady decode improved to
  `11.4074 tok/s`.
- Fused-GDN plus Triton branch attention, no-compile FULL_DECODE_ONLY graph:
  `27b_awq_dflash_ddtree32_topk4_nochain_tritonattn_fusedgdn_tree_graph_i32_o16_w2r3.json`.
  Capture size was `33`; graph capture completed; route summary included
  `prefill_capture_smallq=32` and `prefill_ddtree_triton=400`; all repeats
  matched the no-spec hash. Mean steady decode was `12.0542 tok/s`.
- Re-running after removing the Flash-V100 parent GPU-to-CPU branch probe did
  not show a stable speed win:
  `27b_awq_dflash_ddtree32_topk4_nochain_tritonattn_nocpuprobe_fusedgdn_tree_graph_i32_o16_w2r3.json`
  measured `11.7603 tok/s`, with identical hash/spec metrics. Keep the code
  simplification, but do not count it as a speed improvement.
- AEON's recommended first shape adjusted to the user's spec-16 requirement
  (`ddtree_budget=22`, `top_k=8`, `chain_seed=true`, capture size `23`) was
  worse on this prompt:
  `27b_awq_dflash_ddtree22_topk8_chain_tritonattn_fusedgdn_tree_graph_i32_o16_w2r3.json`
  measured `9.5538 tok/s`, acceptance length `2.8333`, and `num_drafts=6`.
  It is not the current best local DDTree setting.
- `ddtree_budget=32`, `top_k=8`, `chain_seed=true` did not produce evidence:
  startup ended with engine initialization failure and process exit `139`
  before route/sampler metrics were available.

Interpretation:

- Kernel and graph blockers are no longer the main reason DDTree is slow on
  this short 27B-AWQ smoke. The best valid DDTree graph run is still only
  `12.0542 tok/s` versus the no-spec graph baseline at `48.6173 tok/s`.
- The immediate limiter is acceptance/proposal quality and per-step DDTree
  overhead: the current best tree accepts only `11` target tokens over `5`
  drafts for 16 output tokens, so each accepted token still pays a large
  drafter plus verifier cost.
- `build_ddtree_payloads_from_logits()` still synchronizes top-k logits and
  flat draft ids to CPU and builds the tree with Python heap logic. This should
  be profiled or moved to a graph-safe GPU/semigraph path before long-output
  speed claims.

## 2026-06-21 Acceptance Debug

- Added DDTree debug payload fields for per-depth DFlash top-k token ids and
  logprobs, and sampler logging that reports each greedy verifier step as
  `(compact row, parent node, depth, target argmax, tree child, top-k rank)`.
  This is gated by `VLLM_DFLASH_DDTREE_DEBUG=1`.
- Added `VLLM_DFLASH_DDTREE_COMPACT_DRAFTER_CONTEXT=1` default-on. After a
  non-flat branch is accepted, the worker now copies accepted tree hidden-state
  rows, auxiliary hidden rows, and `input_ids` back into the committed flat
  spine before the next DFlash draft. This mirrors the accepted-attention-KV
  compaction and fixes a real DDTree/DFlash context mismatch.
- Direct tests cover the new context compaction behavior. `pytest` is not
  installed in `1cat-vllm-1.2.0-releasecheck`, so the test functions were run
  directly with `python -c`.
- Short debug run:
  `27b_awq_dflash_ddtree32_topk4_nochain_compactctx_debug_graph_i32_o16.json`.
  It still measured low acceptance length (`3.0`) and steady decode
  `11.2420 tok/s`, but now explains why: later verifier rejections were mostly
  because the target argmax was not in the current DFlash `top_k=4`, or was in
  the per-depth top-k but the budgeted tree did not allocate that parent/child.
  This points to candidate/tree coverage, not a compact-logit sampler bug.
- Reference comparison:
  official DDTree takes `topk = min(budget, vocab)`, while AEON's deployable
  Qwen3.6 profile uses `DDTREE_TOP_K=8`. The local default was temporarily
  raised from `4` to `8` during bring-up, but this was superseded on
  2026-06-25 by `ddtree_top_k=None`, which uses the DDTree budget and matches
  the reference path.
- Follow-up `top_k=8/32` validation was blocked by a separate TP4 OpenAI API
  server occupying all four V100s:
  `python -m vllm.entrypoints.openai.api_server ... --tensor-parallel-size 4`.
  Do not treat the failed `top_k=8/32` startup artifacts as DDTree evidence.

## Benchmark Gate

The first accepted 27B-AWQ DDTree benchmark must match the existing DFlash16
criterion:

```text
CUDA_VISIBLE_DEVICES=2,3
model=/home/ymzx/models/Qwen3.6-27B-AWQ
draft=/home/ymzx/models/Qwen3.6-27B-DFlash-FP16
TP=2
input_len=4096
output_len=256
max_model_len=8192
max_num_batched_tokens=8192
max_num_seqs=1
attention_backend=FLASH_ATTN_V100
mamba_cache_mode=align
num_speculative_tokens=16
```

Metrics to report:

- steady decode tok/s and output tok/s.
- prefill time and decode time.
- token hash for greedy.
- DDTree budget, verified nodes, accepted depth, sibling-branch hit count.
- tree build time, target verify time, GDN replay/commit time.
- flat DFlash16 and no-spec baselines from the same run group.

## Immediate Next Step

Recover DDTree speed by attacking acceptance and proposal overhead, not DFlash
flat benchmarking:

- Add DDTree timing instrumentation for tree payload build, drafter forward,
  target verifier forward, sampler, and accepted-state commit.
- Replace or bypass the Python/CPU tree builder in
  `build_ddtree_payloads_from_logits()` for the hot path.
- Sweep only DDTree tree-verify settings that are motivated by the reference
  plan (`budget=32/40/48`, `top_k=4/8`, chain seed on/off), using the existing
  no-spec graph baseline as the comparison.
- Keep hash parity and route summary (`prefill_ddtree_triton`, no
  `prefill_ddtree_dense`) as required acceptance checks.

## 2026-06-25 DDTree Experiment Ledger And Guardrails

This section supersedes the ad hoc DDTree runs from 2026-06-24/25. The main
API target is non-greedy Qwen3.6-27B sampling, not greedy. Greedy DDTree runs
may still be used for correctness and kernel isolation, but they must not be
used to claim normal API acceleration.

Canonical sampling policy for the 27B coding path:

- `temperature=0.6`, `top_p=0.95`, `top_k=20`.
- `min_p=0.0`, `presence_penalty=0.0`, and `repetition_penalty=1.0` are no-op
  model defaults for this comparison. `min_p` is currently disabled by vLLM
  speculative decoding, so do not pass it as an active requirement.
- For community-style calibration, use HumanEval prompts with
  `temperature=1.0`, `top_p=0.95`, `top_k=20`, `max_tokens=256`, and the
  `qwen_chat_solution_nothink` prompt style. This is now the preferred
  acceptance sweep dataset; single handcrafted coding prompts are only
  prompt-specific probes.
- Use free generation for output quality checks. Do not force extra text only
  to extend the token count; if EOS stops early, record that and compare pure
  decode separately from end-to-end output tok/s.

Required proof before a run can be counted as branched DDTree evidence:

- Qwen3.6 hybrid models must run with
  `VLLM_DFLASH_DDTREE_ENABLE_HYBRID_TREE_STATE=1`. Without this, the scheduler
  can reject branched payloads and execute the flat DFlash path.
- Logs must prove the intended route, for example `prefill_ddtree_triton` for
  tree attention and DDTree stochastic verifier trace when tracing is enabled.
- Do not enable `VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD=1` for accepted
  HumanEval evidence. The cached GDN DDTree fast-build metadata path caused
  cross-request NaN verifier logits and repeated `!` output on HumanEval. The
  default path is now the safe metadata builder; the fast-build switch is only
  a diagnostic regression toggle until its cache key/state refresh is fixed.
- Unknown environment variables in the startup log are not evidence that a
  feature is active. Either register the env var or prove the route from code
  and metrics.
- Every accepted experiment must record json/log paths, prompt, sampling
  policy, DDTree knobs, graph/capture state, acceptance metrics, and the reason
  for the next run. Console-only results are useful for debugging but are not
  final performance artifacts.

Current useful artifacts:

| Artifact | Sampling | Prompt class | Result | Use |
| --- | --- | --- | --- | --- |
| `bench_results/ddtree_smoke_20260624/baseline_official_coding_t06_o256.json` | `temp=0.6/top_p=0.95/top_k=20` | deterministic LIS code completion | `162` output tokens, total `50.24 tok/s`, steady decode `53.59 tok/s` | current normal no-spec coding baseline for the matching short prompt |
| `bench_results/ddtree_smoke_20260624/ddtree_qla_official_coding_t06_o256.json` | `temp=0.6/top_p=0.95/top_k=20` | deterministic LIS code completion | `mean_acceptance_length=10.22`, `83` accepted over `9` drafts, total `95.48 tok/s`, steady decode `163.87 tok/s` | proves official-sampling DDTree can reach the expected coding acceptance on a low-entropy code-completion prompt; pure decode is about `3.06x` the baseline, end-to-end is about `1.90x` because prefill/EOS/output length differ |
| `bench_results/ddtree_smoke_20260624/ddtree_qla_gdn_fastpath_prod_coding_o256.json` and `ddtree_qla_gdn_cg4_prod_coding_o256.json` | greedy | coding | `mean_acceptance_length=8.75`, total `114-115 tok/s` | kernel/proposer health only; not normal API evidence |
| `bench_results/ddtree_budget_sweep_20260625/ballgame_budget_sweep_summary.tsv` | `temp=0.6/top_p=0.95/top_k=20` | open-ended pygame ball-eating game code prompt, thinking disabled | formal budget sweep: `16 -> 4.16`, `22 -> 4.83`, `24 -> 4.13`, `32 -> 4.34`, `36 -> 4.34`, `40 -> 4.32`, `64 -> 3.94` mean acceptance length | current acceptance-only budget evidence; `budget=22` is the longest-AL candidate before verifier-cost optimization |
| `bench_results/ddtree_humaneval_sweep_20260625/humaneval32_qwen_chat_solution_nothink_t1p0_o256_safe_default_budget_sweep_summary.tsv` | `temp=1.0/top_p=0.95/top_k=20` | HumanEval-32, Qwen chat solution/no-think | `budget=16 -> AL 11.54, steady 122.11 tok/s`; `22 -> 10.49, 98.67`; `32 -> 10.12, 73.64`; `48 -> 9.03, 53.79`; `64 -> 8.09, 46.58`; all `bang_repeat_outputs=0` | preferred community-calibrated budget evidence; `budget=16` is the current best balance and larger budgets are slower with lower acceptance |
| `bench_results/ddtree_realcode_20260625/realcode6_ddtree_budget16_t1p0_o256_profile2.json` and `realcode6_ddtree_budget16_t1p0_o256_noprofile.json` | `temp=1.0/top_p=0.95/top_k=20` | six realistic codebase repair/implementation prompts, Qwen chat/no-think, `max_tokens=256` | profile run: `AL 7.32`, no-bonus `6.32`, total generation `57.91 tok/s`, per-request steady mean `64.06 tok/s`; no-profile run: `AL 7.36`, no-bonus `6.36`, warmed per-request steady mean excluding first JIT-contaminated request `74.18 tok/s`; target verifier profile `45.78 ms`, target+state `51.86 ms`, draft `17.45 ms` | realistic-context check; HumanEval's `122 tok/s` does not generalize to open-ended repo tasks because acceptance is lower and verifier+state cost is still above the new `<=40 ms` target |

Trace-only findings that should not be repeated without a new hypothesis:

- True branched DDTree on official sampling with `budget=16`, `top_k=20`,
  `chain_seed=false`, and `best_first` reached only about `4.9` effective
  acceptance on the open-ended coding/game prompt. Stochastic trace showed the
  sampled target token was usually present in the per-depth draft top-k but was
  not a child under the currently accepted prefix. This points to tree topology
  and prefix coverage, not simple top-k candidate quality.
- Increasing to `budget=64` made cost much worse and acceptance lower on the
  same trace family because best-first spent many nodes shallowly. Do not
  repeat large-budget sweeps until trace data shows missing shallow siblings
  are the blocker.
- The formal 2026-06-25 ballgame budget sweep confirms this non-monotonic
  behavior under the normal API sampling target. `budget=22` reached the
  highest mean acceptance length (`4.8333`, no-bonus `3.8333`), while
  `budget=32/36/40` stayed near `4.32-4.34` and `budget=64` fell to `3.9385`.
  Treat this as prompt-specific evidence only. The HumanEval calibration sweep
  supersedes it for the default DDTree budget decision.
- HumanEval-32 with the safe GDN metadata path restores coding-scene acceptance
  to the expected range: `budget=16` reached `mean_acceptance_length=11.5361`
  and `draft_acceptance_rate=0.6585`, with per-position acceptance
  `0.985/0.969/0.941/0.902/0.863/0.822/0.781/0.732/...`. Increasing budget was
  strictly worse in both acceptance length and steady decode. Do not repeat
  larger budget sweeps until the tree-builder hypothesis changes.
- Realistic repo-style code repair prompts are materially harder than
  HumanEval. The six-prompt `realcode6` check at the same `budget=16` and
  official sampling reached only `mean_acceptance_length=7.3602` with
  per-position acceptance
  `0.915/0.820/0.735/0.626/0.555/0.488/0.417/0.374/...`. The first no-profile
  request was polluted by inference-time Triton JIT
  (`copy_and_expand_dflash_inputs_kernel`, `_topk_topp_kernel`,
  `eagle_prepare_next_token_padded_kernel`, `expand_kernel`), so use the
  warmed post-first-request mean `74.18 tok/s` as the cleaner speed read for
  this artifact.
- The `realcode6` profile run fixed a profile-only `NameError` in
  `DFlashProposer._sample_draft_tokens`: the payload-stage log must read
  `logits.shape`, not a stale `payload_logits` name. This changed only
  instrumentation and does not alter sampling.
- Current realistic-context verifier cost is still high: the last
  `SM70 spec runner profile avg_ms` line reported `target_forward=40.540 ms`,
  `target_logits=2.848 ms`, `target_rejection_sample=2.394 ms`,
  `state_update_wall_cpu=6.082 ms`, and `draft_wall_cpu=17.450 ms`. Use
  `45.78 ms` for target verifier cost and `51.86 ms` when state update is
  included. The active near-term target is now `target verifier + state update
  <=40 ms`, so improvements may come from AWQ, attention, Torch small kernels,
  all-reduce, sampler, GDN, or accepted-state update rather than one kernel
  family alone.
- HumanEval diagnosis root cause: with the GDN DDTree fast-build cache enabled,
  the first request could finish normally, then the next request's first
  verifier step saw NaN compact logits and sampled token id `0`, causing
  repeated `!`. Disabling that fast-build path produced normal HumanEval code
  and no NaN trace events.
- `chain_seed=true` and `spine_leaf` preserved a deeper top-1 path but did not
  recover the open-ended prompt to the official coding target. Do not keep
  toggling these options without a changed tree-building hypothesis.
- Horizon `15` versus `16` did not materially change acceptance in earlier
  DFlash tests, so block-size-minus-one is not the current primary suspect.

Current code-state notes:

- `ddtree_top_k` now defaults to `None`, meaning the builder uses the DDTree
  budget as the reference top-k. `ddtree_chain_seed` defaults to `False` to
  match the official best-first expansion, and tree verify defaults to enabled
  for `dflash_ddtree`.
- The sampler now has stochastic DDTree trace events, including sampled target
  tokens, accepted nodes, child matches, top-k rank, and tree topology.
- A pending sampling-alignment change warps draft logits with the request
  `temperature/top_k/top_p` before building the DDTree payload. This is the
  next thing to validate; do not start another parameter sweep before checking
  whether this fixes the open-ended coding acceptance collapse.

Next experiments:

- Treat `budget=16` as the current default HumanEval-calibrated DDTree setting.
  The next work item is total verifier-plus-state reduction on this setting,
  not another baseline run.
- Split `state_update_wall_cpu=6.082 ms` before optimizing it. Record whether
  it is CPU waiting on GPU, accepted-state copy/commit kernels, metadata
  construction, or graph replay synchronization.
- `gpu_model_runner.py` now reports state-update sub-buckets in the existing
  `SM70 spec runner profile avg_ms` line:
  `state_update_validate_cpu`, `state_update_attn_compact_cpu`,
  `state_update_mamba_compact_cpu`, `state_update_input_batch_cpu`, and
  `state_update_drafter_context_cpu`.
- Use the selected target-forward Nsight hierarchy to target non-AWQ buckets
  as well: DDTree attention (`~4.65 ms/rank`), Torch native small kernels
  (`~3.9 ms/rank`), TP all-reduce rank imbalance (`3.86 ms` vs `1.90 ms`),
  and GDN delta/gating (`~3.30 ms/rank`).
- Once verifier cost is materially lower, rerun a narrow balance sweep
  (`budget=16/22/32` first) and choose by accepted tokens per verifier
  millisecond, not by acceptance length alone.
- Compare only against the already-recorded baseline unless the baseline code
  path or sampling policy changes.

## 2026-06-30 AWQ Verifier GEMM Microbench

The next optimization target is the DDTree target verifier plus state-update
wall time:

```text
target verifier + state update: 51.86 ms -> <= 40 ms
```

Nsight graph replay still attributed about `18.6 ms/rank` of the `~40 ms`
target-forward wall time to TurboMind AWQ uint4 GEMM kernels, so AWQ remains
the largest bucket. It is no longer the only accepted route to the target:
DDTree attention, Torch native small kernels, TP all-reduce imbalance, GDN
delta/gating, sampler work, and accepted-state update are also in scope.

New microbench harness:

- `benchmarks/benchmark_sm70_awq_verifier_micro.py`
- Default model: `/home/ymzx/models/Qwen3.6-27B-AWQ`
- Default shape: `m=17`, matching `budget=16` plus root for the verifier
  dense pass.
- Default weighted suite models one Qwen3.6-27B-AWQ TP rank verifier forward:
  `63` gate/up MLP calls, `63` down MLP calls, `47` linear-attention qkv
  calls, `47` linear-attention z calls, and `16` full-attention o-proj calls.
- The harness times only `awq_gemm_sm70_out` after `awq_sm70_prepare` and warmup.
  It intentionally excludes Python, tree build, logits, sampling, and state
  update overhead.

Use this command shape for candidate validation:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$PWD \
VLLM_SM70_AWQ_TP2_FAST_SELECTOR=1 \
VLLM_SM70_AWQ_TUNE_SMALL_SHAPES=1 \
VLLM_SM70_AWQ_DENSE_TUNE_MAX_M=17 \
VLLM_SM70_AWQ_REUSE_IMPORTED_CACHE=0 \
VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS=0 \
VLLM_SM70_AWQ_PRESERVE_DEFAULT_SPLITS_ONLY=0 \
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python \
  benchmarks/benchmark_sm70_awq_verifier_micro.py \
  --json-out bench_results/ddtree_awq_micro_20260630/nopreserve_m17.json \
  --csv-out bench_results/ddtree_awq_micro_20260630/nopreserve_m17.csv
```

Current microbench result:

| Artifact | Total weighted AWQ GEMM time | Notes |
| --- | ---: | --- |
| `bench_results/ddtree_awq_micro_20260630/preserve_m17.json` | `34.04 ms` | Default split preservation is too conservative for verifier small-M. |
| `bench_results/ddtree_awq_micro_20260630/nopreserve_m17.json` | `25.03 ms` | Current best dynamic-selector microbench total. |
| `bench_results/ddtree_awq_micro_20260630/all_shapes_m17_dynamic_nopreserve_recheck.json` | `24.80 ms` | Current best rechecked microbench comparison point. |
| `bench_results/ddtree_awq_micro_20260630/restored_default_recheck_200.json` | `27.97 ms` | Restored non-forced default after failed experiments were removed. |
| `bench_results/ddtree_awq_micro_20260630/all_shapes_m17_forced_mgroup_c32x128_recheck.json` | `28.14 ms` | `c32x128` mgroup candidate; useful Nsight evidence but reverted. |

Current best breakdown from `all_shapes_m17_dynamic_nopreserve_recheck.json`:

| Bucket | Mean kernel time | Weighted time |
| --- | ---: | ---: |
| `mlp_gate_up` (`17x17408x5120`) | `132.20 us` | `8.33 ms` |
| `mlp_down` (`17x5120x17408`) | `129.22 us` | `8.14 ms` |
| `linear_attn_in_proj_qkv` (`17x10240x5120`) | `88.28 us` | `4.15 ms` |
| `linear_attn_in_proj_z` (`17x6144x5120`) | `65.32 us` | `3.07 ms` |
| `full_attn_o_proj` (`17x5120x6144`) | `69.25 us` | `1.11 ms` |

Important interpretation:

- The microbench's absolute weighted total is higher than the Nsight graph
  replay AWQ bucket. Use it for per-shape breakdown and candidate ranking, not
  as a replacement for graph replay.
- The main bottlenecks are the two MLP buckets. Together they account for about
  two thirds of the microbench AWQ time.
- The current dynamic selector's best gate/up route is still the existing
  `32x256x32` mgroup family. A formal recheck measured `131.44 us` for the
  gate/up bucket.
- Nsight Compute on the gate/up shape shows the useful mgroup route improves
  grid fill (`68` CTAs -> `136` CTAs, waves/SM `0.47` -> `0.94`) and SM
  throughput (`44.31%` -> `58.42%`), but remains register-limited at allocated
  `192` registers/thread. It is not a pure DRAM bandwidth bottleneck
  (`~33%` DRAM on the mgroup route).
- The `c32x128` mgroup candidate halves shared memory versus `c32x256`
  (`~16.4 KiB` vs `~32.8 KiB`) but keeps the same register cap and does not
  improve the full weighted suite.
- A dense-fp16-output epilogue diagnostic was also tested and reverted. It
  measured `140.65 us` CUDA-event mean for the gate/up shape, while NCU showed
  `124.99 us`, `190` registers/thread, SM `59.12%`, and DRAM `33.91%`. This
  proves epilogue-only cleanup does not remove the verifier GEMM limiter.

Negative results that should not be repeated without a changed hypothesis:

- Smaller CTA-M candidates did not help. The best `16x256x32` candidate was
  about `166 us` for gate/up, slower than the current `~131 us`.
- A `24x256x32` verifier-focused CTA-M experiment and the required temporary
  `12B` helper support selected and ran, but was slower on gate/up
  (`~183 us` for `24x256x32`, worse for `24x128x32`). It was reverted.
- CTA-N `128` and `8x*` small-M candidates were also slower in the gate/up
  sweep.
- Experimental `TG_N=2` and `TG_N=8` registry variants did not improve the
  dynamic gate/up result and were reverted.
- Ordinary AWQ Marlin GEMM was much slower on the same synthetic 27B shapes
  (`~284 us` gate/up and `~674 us` down in the direct comparison), so routing
  the DDTree verifier dense AWQ path away from TurboMind is not the fix.
- A direct verifier launch/epilogue branch without changing the mainloop was
  slower overall: `m17_direct_dense_verifier_probe.json` measured `40.09 ms`
  weighted and regressed MLP down to `293.79 us`. It was reverted.
- Forcing `__launch_bounds__(..., 3)` to chase three active blocks per SM was
  counterproductive: `gate_up_m17_mgroup_c32x128_launchbounds3.json` measured
  `187.48 us`. It was reverted. Register pressure must be reduced at the
  source/live-range level, not by compiler pressure alone.
- A no-prefetch low-live-range mainloop probe compiled and selected correctly,
  but `gate_up_m17_mgroup_c32x128_noprefetch_probe.json` measured
  `189.07 us`. It was reverted. The next low-register design must preserve
  the current prefetch/transform overlap instead of making the loop fully
  synchronous.

Current code support:

- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/gemm.cu` has a
  default-off environment override,
  `VLLM_SM70_AWQ_TP2_FAST_TARGETS`, for exact descriptor-to-kernel candidate
  testing.
- The override is a microbench/profiling tool only. Production behavior remains
  unchanged unless the environment variable is set.
- Format:
  `<desc>|<cta_m>x<cta_n>x<cta_k>:<splits>:<swizzle>:<require_mgroup>`, with
  multiple entries separated by `;`.
- No experimental `c32x128` or no-prefetch candidate is retained in
  `sm70_884_4.cu`. Their artifacts remain useful because they show shared
  memory reduction and naive live-range reduction do not recover the target
  while register allocation stays near `192` registers/thread or pipeline
  overlap is lost.
- The shared operator maintenance ledger is
  `docs/design/sm70_nvfp4_turbomind_operator_optimization.md`; despite the
  historical file name, it now also tracks this AWQ verifier operator route.

Decision:

- Existing TurboMind selector/config sweeps and epilogue-only changes are not
  enough to move the full `51.86 ms` path to `<=40 ms`.
- The next implementation work should be chosen by expected total-path
  contribution. AWQ mainloop work remains valid, especially verifier-specific
  V-scale reuse or `CTA_K=64` experiments, but it no longer has to supply the
  entire reduction alone.
- Before coding another AWQ kernel, split `state_update_wall_cpu` and identify
  whether it can remove several milliseconds through fewer syncs, fewer
  accepted-state kernels, or graph-safe batching.

2026-06-30 state-update split and DDTree compact probes:

- The runner now reports `state_update_validate_cpu`,
  `state_update_attn_compact_cpu`, `state_update_mamba_compact_cpu`,
  `state_update_input_batch_cpu`, and
  `state_update_drafter_context_cpu` in the SM70 spec profile line.
- Text-only official-sampling profile:
  `bench_results/ddtree_total_target_20260630/realcode2_o128_mamba_fused_cache_profile.json`
  and `.log`. Setup: Qwen3.6-27B-AWQ TP2, DDTree `budget=16/top_k=20`,
  `temperature=1.0/top_p=0.95/top_k=20`, `max_tokens=128` for two prompts,
  `limit_mm_per_prompt={"image":0,"video":0}`, Flash-V100, CUDA graph enabled.
  Result: `256` output tokens in `3.4808 s`, mean acceptance `12.86`,
  draft acceptance rate `0.741`. The profile did not meet the `<=40 ms` target:
  `target_forward=56.583 ms`, `target_logits=3.120 ms`,
  `target_rejection_sample=4.031 ms`, and
  `state_update_wall_cpu=5.941 ms` at the `calls=20` average.
- Active 27B-AWQ serving uses `mamba_cache_mode=none`, not `align`. Therefore
  the align-mode DDTree-aware fused Mamba postprocess can safely defer legacy
  compact only for align-mode tests/runs; it does not remove the current
  `none`-mode explicit compact path.
- A whole-block Mamba compact batch-memcpy experiment was implemented, tested,
  and reverted. Artifact:
  `bench_results/ddtree_total_target_20260630/realcode2_o128_mamba_batchcopy_cache_profile.json`.
  It preserved acceptance (`mean_acceptance_length=12.85`) but worsened
  profile cost (`state_update_wall_cpu=16.311 ms`,
  `state_update_mamba_compact_cpu=6.696 ms`,
  `state_update_drafter_context_cpu=7.350 ms` at `calls=20`). Do not use this
  as the state-update optimization route.
- Current profiling points to target-forward GPU wait as the dominant remaining
  limiter. `full_logits_split` repeatedly shows `pre_sync_ms` around
  `31-36 ms` for `rows=17`; this is the target forward completing before
  verifier logits, and it is larger than all state-update sub-buckets combined
  in the non-experimental run.

2026-06-30 verifier+state <=40 ms milestone:

- Runner state update now keeps DDTree accepted rows, sampled-token counts,
  current request indices, and current positions in CPU sidecars. This lets
  accepted-state update avoid several D2H reads from GPU metadata in the hot
  path.
- `_update_states_after_model_execute` uses the CPU sidecar to build
  `num_accepted_tokens` and `spec_state_slot_selectors` on CPU. It stages those
  tensors back to GPU only when align-mode GPU postprocess needs them.
- DDTree clamp has a no-clamp fast path based on sidecar sampled-token counts
  and request `max_tokens`, avoiding the GPU contiguous-token count and
  `torch.equal` synchronization when every request is already within limits.
- Attention KV compaction has a CPU slot-pair fast path for the common CP/DCP=1
  case. It computes physical KV slots from CPU positions and the CPU block
  table instead of calling `slot_mapping.detach().cpu().tolist()` every spec
  step. The old D2H slot-mapping path remains as fallback.
- Validation passed:
  `python -m py_compile vllm/v1/worker/gpu_model_runner.py`,
  `pytest -q tests/v1/spec_decode/test_ddtree_sampler.py tests/v1/worker/test_gpu_input_batch.py`,
  and
  `pytest -q tests/v1/spec_decode/test_ddtree_gpu_runner_positions.py tests/v1/worker/test_mamba_utils.py -k 'not shared_storage'`.
- Official-sampling, text-only, 256-output-token coding benchmark artifacts:

  | artifact | mode | final hot verifier+state |
  | --- | --- | --- |
  | `bench_results/ddtree_total_target_20260630/realcode2_o256_statecpu_slotsidecar_16k.*` | default `mamba_cache_mode=none` | best hot `40.100 ms`, last hot `40.489 ms`; `state_update_wall_cpu` fell to `2.108-2.709 ms` |
  | `bench_results/ddtree_total_target_20260630/realcode2_o256_slotsidecar_mambadirect_16k.*` | direct Mamba compact A/B | worse: last hot `41.422 ms`, `state_update_wall_cpu=3.992 ms`; do not use |
  | `bench_results/ddtree_total_target_20260630/realcode2_o256_slotsidecar_align_16k.*` | `enable_prefix_caching=true`, `mamba_cache_mode=align` | last hot `39.388 ms`: `target_forward=33.270`, `target_logits=2.890`, `target_rejection_sample=1.290`, `state_update_wall_cpu=1.938` |

- The accepted milestone configuration for this target-cost gate is:
  DDTree `budget=16/top_k=20`, official Qwen sampling
  `temperature=1.0/top_p=0.95/top_k=20`, `max_tokens=256`,
  `max_num_batched_tokens=16384`, visual disabled with
  `limit_mm_per_prompt={"image":0,"video":0}`, Flash-V100, CUDA graph enabled,
  `enable_prefix_caching=true`, and `mamba_cache_mode=align`.
- Caveat: the `<=40 ms` goal is satisfied for the steady verifier+state hot
  interval. End-to-end generation time did not yet improve in the align run
  (`generate_seconds=8.5076` for `512` output tokens) because warm/capture and
  early align costs are still high. The next serving-speed gate should measure
  steady throughput after warmup and reduce align staging/warm costs before
  claiming an end-to-end speed win.
