# SM70 MTP Output Quality Audit - 2026-06-16

## Superseded Note - 2026-07-01

The June rows that treated `VLLM_SM70_QWEN_GDN_INPUT_CORE_OP=1` as a possible
quality fix are superseded by the July 1 fixed-quality gate. Under official
sampling (`temperature=1.0/top_p=0.95/top_k=20`, seed `20260620`), the
input-core route fails long output, while the accepted 27B-AWQ native MTP4
route is:

```text
VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD=1
VLLM_SM70_QWEN_GDN_INPUT_CORE_OP=0
VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP=0
VLLM_SM70_QWEN_GDN_003_SPEC_CORE_OP=1
VLLM_SM70_QWEN_GDN_SPEC_CORE_OP=0
```

Primary artifact:
`bench_results/verifier20_quality_20260701/noinputcore_003_fixed_quality4_seed20260620/summary.md`.
See `docs/design/sm70_v100_migration_control.md` for the full route table and
the rejected input-core/output-op evidence.

## Scope

This log records the latest-vLLM Qwen3.6 27B AWQ MTP output-collapse
investigation. Keep it updated before running another variant so future work
does not repeat the same exclusions.

Primary model:

```text
/home/ymzx/models/Qwen3.6-27B-AWQ
```

Primary runtime:

```text
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python
PYTHONPATH=/home/ymzx/桌面/1cat-vllm/vllm
```

Primary serving shape:

```text
TP2 on V100
max_model_len=262144
max_num_batched_tokens=8096
max_num_seqs=1
gpu_memory_utilization=0.88
OpenAI chat API with Qwen3 reasoning/tool parser enabled
```

Primary prompt used to reproduce the failure:

```text
帮我用html做一个macos，功能要尽可能全，要尽可能还
```

For long-output reproduction, requests used `max_tokens=6000` unless otherwise
noted. Sampling used the server/model defaults unless a row explicitly states
otherwise. Do not change official sampling parameters as a final fix.

## Current Conclusion

The current failure is a real MTP/spec-decode quality bug:

- Without MTP, the same model/backend/request style is stable across repeated
  long outputs.
- With MTP enabled, output initially looks normal, then later collapses into
  repeated CSS/text fragments or repeated characters such as `555...`.
- Around the collapse, speculative decoding metrics rise to all-position
  acceptance rate `1.000`; this is a symptom of target and draft having already
  fallen into the same degenerate stream, not proof that the output is correct.
- The issue reproduces with `num_speculative_tokens=1`, so MTP4 multi-position
  drafting is not required.
- The issue reproduces when the target attention backend is `TRITON_ATTN`, so
  Flash-V100 target verification is not the sole root cause.

The likely root is now narrowed to the latest compile/FULL graph split of the
Qwen GDN target-verifier forward. The failing latest path splits Qwen GDN into
compiled input projection, an opaque recurrent core custom op, and compiled
RMSNorm/output projection. Keeping only input projection plus recurrent core
behind one opaque custom-op boundary is not sufficient: it passed one long run
but failed a second same-prompt run with an early `UTF-UTF...` collapse and
all-position MTP acceptance. Isolating input projection alone or output
projection alone also fails. The only passing compile/FULL graph localizer so
far is the full Qwen GDN forward behind one opaque custom-op boundary. Do not
treat this as a Flash-V100-only issue and do not close it by disabling MTP or
changing official sampling parameters.

Latest high-confidence finding:

- Latest `VLLM_SM70_QWEN_GDN_CONTEXT_CORE=1` still fails, so explicit-cache
  custom-op arguments are not the sole root.
- Latest `VLLM_SM70_QWEN_GDN_FULL_FORWARD=1` passes two long reproducer runs
  without the repeated-output collapse or all-position acceptance plateau. This
  is now the selected quality guard for SM70 compile/FULL graph.
- The default SM70 compile/FULL graph path now automatically uses the full
  Qwen GDN forward opaque boundary. One default-startup long run passed:
  `/tmp/latest_default_fullgdn_mtp4_macos_6000.json`.
- Latest default automatic `input_projection_core` boundary is not sufficient:
  the second same-prompt run collapsed into `UTF-UTF...` from character 148 and
  had sustained all-position acceptance `1.000`.
  keeps compile/FULL graph and MTP4 enabled, so the important delta is the GDN
  split compile boundary around projection/core/output-projection.
- Latest `VLLM_SM70_QWEN_GDN_INPUT_CORE_OP=1` passes while
  `VLLM_SM70_QWEN_GDN_INPUT_PROJECTION_OP=1` and
  `VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP=1` both fail. This makes the
  projection-to-core boundary the current minimal root, not the output
  projection and not projection materialization alone.
- A deep alignment run shows the target verifier logits are already degenerate
  before the rejection sampler can be blamed. Around alignment step 535 the
  target argmax, draft tokens, and emitted tokens all agree on repeated
  `px3px3...`.
- The first stable repeated token pair begins at generated token index `1583`
  (`[1705, 18]`, decoded as `px3`). With the 42-token prompt this corresponds
  to sequence position about `1625`, immediately before the GDN/Mamba state
  block boundary at `2 * 816 = 1632`.
- GDN state diagnostics show the state window crossing that boundary from
  `[2, 3, 4, 5, 17]` to `[3, 4, 5, 17, 22]` between seq 1632 and 1638.
- 1Cat-vLLM 0.0.3 has a known-good `[Bugfix] Stabilize MTP state handling`
  commit (`acd2a3150`) and stays clean under the same MTP4 reproducer. The next
  fix path is therefore the latest-vLLM MTP verifier recurrent-state rollover
  semantics versus 0.0.3, not more sampler or prompt testing.

## Experiment Matrix

| ID | Variant | Artifact | Result | Conclusion |
| --- | --- | --- | --- | --- |
| A | no-MTP, target `FLASH_ATTN_V100`, 3x 6000-token macOS prompt | `/tmp/nomtp_macos_1.json`, `/tmp/nomtp_macos_2.json`, `/tmp/nomtp_macos_3.json` | All completed normally; content chars about 13.4K-13.8K; max digit run 4/5/4; no block repetition | Target Flash-V100, compile graph, Qwen3 parser, and tool parser are not sufficient to reproduce |
| B | MTP4, target `FLASH_ATTN_V100`, alignment dump enabled | `/tmp/spec_alignment_pid1722373_*.pt`, `/tmp/spec_alignment_pid1722374_*.pt` | Collapse around output token ~1984; diagnostics show earlier recovered tokens already pushing CSS context into `rgba(25555...)` / `#2c2c...`; later target and draft become degenerate and all acceptance becomes 1.000 | All-1 acceptance is late symptom; drift begins before the metric fully saturates |
| C | MTP4 with `VLLM_MAMBA_ALIGN_CPU_POSTPROCESS=1` | `/tmp/mtp_cpu_post_macos_1.json`, `/tmp/mtp_cpu_post_macos_2.json`, `/tmp/mtp_cpu_post_macos_3.json` | Still bad; max digit run up to 2769; repeated CSS/text blocks | Fused GPU postprocess is not the sole root |
| D | MTP1, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN` | `/tmp/mtp1_macos_1.json`, `/tmp/mtp1_macos_2.json`, `/tmp/mtp1_macos_3.json` | Still bad; two requests repeat `555...`; one repeats `UTF-UTF-...` | MTP4 multi-step drafting and per-position step index are not required |
| E | MTP1, `top_p=1.0`, target `FLASH_ATTN_V100` | `/tmp/mtp1_topp1_macos.json` | Still repeats; `body{}body{}` / repeated `lockDate...` | Top-p truncation alone is not the root |
| F | MTP1, target `TRITON_ATTN`, drafter `TRITON_ATTN` | `/tmp/mtp1_target_triton_macos_1.json`, `/tmp/mtp1_target_triton_macos_2.json` | Still bad; one repeats `background: 0.9;`; one repeats `UTF-UTF-...`; metrics again rise to 99-100% acceptance | Target Flash-V100 verifier/small-query path is not required |
| G | MTP greedy draft sampling | `/tmp/mtp_greedy_macos_2.json`, `/tmp/mtp_greedy_macos_3.json` | Still repeats | Random draft sampling is not the sole root |
| H | MTP with `--no-async-scheduling` | `/tmp/mtp_noasync_macos_1.json`, `/tmp/mtp_noasync_macos_2.json` | Still repeats | Async scheduling is not the sole root |
| I | 1Cat-vLLM-0.0.3, TP2, 27B-AWQ, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, MTP4, same macOS prompt, 3x 6000-token requests | `/tmp/003_mtp4_macos_1.json`, `/tmp/003_mtp4_macos_2.json`, `/tmp/003_mtp4_macos_3.json` | All three completed at 6000 tokens with no detected repeated block; max digit run was 4 for all runs; server metrics stayed in normal ranges instead of saturating all positions at 1.000 | 0.0.3 does not reproduce this current long-output collapse under the matched MTP4 shape; current latest-vLLM regression is not an unavoidable model/MTP behavior |
| I2 | 1Cat-vLLM-0.0.3 recheck, same TP2 27B-AWQ MTP4 shape, same macOS prompt, 2x 6000-token requests | `/tmp/003_recheck_mtp4_macos_1.json`, `/tmp/003_recheck_mtp4_macos_2.json` | Both completed at 6000 tokens with no repeated block; max digit run 4 and 5; metrics stayed normal and never saturated all positions to 1.000 | Reconfirmed 0.0.3 does not have the latest long-output MTP repetition/collapse under this reproducer |
| J | latest-vLLM, TP2, 27B-AWQ, explicit `--enable-prefix-caching --mamba-cache-mode align`, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, MTP4, same macOS prompt, 3x requests | `/tmp/latest_prefix_align_mtp4_macos_1.json`, `/tmp/latest_prefix_align_mtp4_macos_2.json`, `/tmp/latest_prefix_align_mtp4_macos_3.json` | Request 2 collapsed into repeated `-webkit-app-region: no-drag;`; server metrics saturated to per-position `1.000` after initially normal acceptance. Request 1 stopped early; request 3 had numeric/CSS oddities but no long repeated block. | Latest still reproduces even under the 0.0.3-style prefix+align production shape; no-prefix/align mismatch is not the root |
| K | latest-vLLM, same as J, with `VLLM_SM70_MTP_DENSE_F16_FASTPATH=0` | `/tmp/latest_no_mtp_densefast_mtp4_macos_1.json`, `/tmp/latest_no_mtp_densefast_mtp4_macos_2.json`, `/tmp/latest_no_mtp_densefast_mtp4_macos_3.json` | All three still failed: request 1 repeated `UTF-UTF-...`, request 2 had max digit run 2740, request 3 repeated `UTF-UTF-...`; server metrics again saturated to 100% acceptance after an initially normal phase | The latest-only SM70 MTP draft dense fp16 TurboMind fast path is not required for the collapse |
| L | latest-vLLM, same as J, with diagnostic `VLLM_SM70_MTP_SYNC_ACCEPT_COUNTS=1` to use 0.0.3-style synchronous accepted-count correction | `/tmp/latest_sync_accept_mtp4_macos_1.json` | Still failed. Request 1 finished at 6000 completion tokens, all in reasoning, then collapsed into a run of 4825 `0` characters. Server metrics still moved from normal acceptance to all-position `1.000`. | Latest optimistic/deferred accepted-count correction is not required for the collapse |
| M | latest-vLLM, same as J, with diagnostic `VLLM_SM70_MTP_LEGACY_OUTPUT_TOKEN_REPAIR=1` to use 0.0.3-style async output-token repair | `/tmp/latest_legacy_output_repair_mtp4_macos_1.json` | Still failed. Request 1 finished at 6000 completion tokens and collapsed into a run of 4792 repeated digit tokens, mostly `2`; server metrics again saturated to all-position `1.000`. | Latest async output-token repair mutation of `token_ids_cpu` / `num_tokens_no_spec` is not required for the collapse |
| N | latest-vLLM, same as J, with diagnostic `VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS=1` to restore 0.0.3-style materialized Qwen3.5 GDN `mixed_qkv` before the custom-op boundary | `/tmp/latest_gdn_mixedqkv_contig_mtp4_macos_1.json` | Still failed. Request 1 finished at 6000 completion tokens and repeated `UTF-UTF-...` from char 3714; server metrics again saturated to all-position `1.000`. | Projection-slice view plus in-place `causal_conv1d_update()` mutation at the GDN custom-op boundary is not required for the collapse |
| O | latest-vLLM, same as J, with `VLLM_SM70_FLA_WARPS=4 VLLM_SM70_FLA_STAGES=2` to restore the 0.0.3-style FLA recurrent verifier launch shape | `/tmp/latest_fla_recurrent_003sched_mtp4_macos_1.json`, `/tmp/latest_fla_recurrent_003sched_mtp4_macos_2.json` | Request 1 stopped early and was not a pass sample. Request 2 finished at 6000 completion tokens and repeated `<html lang="zh-CN">`; server metrics again ramped from normal acceptance to all-position `1.000`. | The latest FLA recurrent verifier `num_warps`/`num_stages` schedule difference is not required for the collapse |
| P | latest-vLLM, same as J, with explicit `draft_sample_method="greedy"` to avoid latest-only probabilistic draft-probability rejection path | `/tmp/latest_explicit_greedy_mtp4_macos_1.json` | Still failed. Request finished at 6000 completion tokens; output collapsed around `text-shadow: ... rgba(255, 255, 255, ...)` into repeated `5, 55, 555...` fragments. Server metrics stayed in normal acceptance ranges instead of saturating all positions to `1.000`. | Probabilistic draft probabilities are not required for the quality collapse. They can change the failure symptom to all-1 acceptance, but the primary root remains in MTP verifier/state handling. |
| Q | latest-vLLM with restored token-matching code, explicit `draft_sample_method="greedy"`, `VLLM_MTP_STOCHASTIC_TOKEN_MATCHING=1`, same TP2 27B-AWQ MTP4 macOS prompt | `/tmp/latest_tokenmatch_mtp4_macos_1.json`, `/tmp/mtp_tokenmatch_8011.log` | Still failed. Output repeated CSS fields such as `display: 0;` / `padding: 0;`; metrics ramped to `Per-position acceptance rate: 1.000, 0.996, 0.996, 0.996` then all `1.000`. Logs showed `sample_recovered_tokens_kernel` and `rejection_random_sample_kernel`, proving token-matching did not actually run. | The first token-matching patch was blocked by latest-vLLM's spec-decode `MinTokensLogitsProcessor` gating. This is not a proof that token matching is ineffective. |
| R | latest-vLLM with inactive `MinTokensLogitsProcessor` allowed through token-matching gate, explicit `draft_sample_method="greedy"`, `VLLM_MTP_STOCHASTIC_TOKEN_MATCHING=1`, same TP2 27B-AWQ MTP4 macOS prompt | `/tmp/latest_tokenmatch_mintokens_mtp4_macos_1.json`, `/tmp/mtp_tokenmatch_mintokens_8011.log` | Still failed. Output collapsed early around `.menu-bar` CSS into repeated `height: 25px;` / `min-height: 25px;` and then `50px`. Metrics stayed high but not fully all-1: later windows around mean acceptance length `4.71-4.73`, avg draft acceptance `92.7-93.2%`. Log no longer showed `sample_recovered_tokens_kernel` or `rejection_random_sample_kernel` JIT during this request. | Token matching likely ran and still did not fix quality. This excludes recovered-token standard rejection as the sole root; continue with verifier multi-token recurrent state/accounting versus 0.0.3. |
| S | latest-vLLM, dynamic partition forced off, per-step MTP dump enabled, same TP2 27B-AWQ MTP4 macOS prompt | `/tmp/latest_mtp4_dynoff_stepdump_macos_1.json`, dump dir `/tmp/mtp_stepdump_latest_mtp4_macos` | Still failed. Output collapsed around token index ~2500 into repeated CSS such as `font-weight`; metrics did not necessarily saturate all positions to 1.000. Dump showed target and draft becoming self-consistent after the stream was already bad. | Dynamic partition is not required for this MTP collapse. Per-step dump confirms all-1 acceptance is a late symptom, not the first bad transition. |
| T | latest-vLLM, same as S, plus diagnostic `VLLM_SM70_MTP_EXACT_DRAFT_SEQ_LENS_CPU=1` to force exact draft CPU seq-lens upper bound | `/tmp/latest_mtp4_exactseq_dynoff_macos_1.json`, dump dir `/tmp/mtp_stepdump_latest_exactseq_mtp4_macos` | Still failed, with a long `55, 55, 55...` / `555...` run. Metrics had high-acceptance windows but not a clean fix. | `seq_lens_cpu_upper_bound` optimism is not sufficient root cause. Do not repeat this A/B as a proposed fix. |
| U | code audit: current vs 0.0.3 GDN state dtype rule | `mamba_utils.py` current and 0.0.3 comparison | 0.0.3 GDN auto -> fp32 SSM state; current generic auto -> conv/model dtype, fp16 on SM70 | Strong current-only long-output recurrent-state drift candidate. Patch restores 0.0.3 fp32 GDN SSM auto rule; serving validation pending. |
| V | latest-vLLM, dynamic partition off, 0.0.3-style context-resolved GDN core custom op (`VLLM_SM70_QWEN_GDN_CONTEXT_CORE=1`) | `/tmp/latest_contextcore_mtp4_macos_1.json`, `/tmp/latest_contextcore_mtp4_macos_2.json`, `/tmp/latest_contextcore_mtp4_macos_3.json` | Still failed. Request 2 collapsed into a 5261-character run of `0`; request 3 had corrupted CSS numerics such as `grid-template-columns: repeat(auto-fill, 0; 35 2px...)`. | Latest explicit-cache custom-op boundary is not the sole root. Do not keep pursuing context-core as a fix by itself. |
| W | latest-vLLM, same as V, plus explicit `draft_sample_method="greedy"` | `/tmp/latest_contextcore_greedy_mtp4_macos_1.json` | Still failed. Output did not produce a simple long digit run, but collapsed into generated key/value sequences like `'cv80=1' ... '536=`. Metrics ramped from normal acceptance to `Per-position acceptance rate: 1.000, 1.000, 1.000, 1.000`. | Draft probabilistic sampling is not the root; the all-1 acceptance collapse can occur with greedy draft as well. |
| X | latest-vLLM, TP2 27B-AWQ MTP4, `--enable-prefix-caching --mamba-cache-mode align`, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, explicit `draft_sample_method="greedy"`, deep sampler+GDN state dumps | `/tmp/latest_align_deep_mtp4_macos_6000.json`, `/tmp/mtp_stepdump_alignment_deep`, `/tmp/mtp_gdn_state_alignment_deep`, `/tmp/spec_alignment_pid2512743_*.pt` | Collapse starts as repeated `px3`. First stable repeated token index is `1583`; alignment 535 has target argmax, draft, and output all locked to `px3px3`. GDN seq positions straddle the 1632 state-block boundary and `current_state_block_ids` rolls from `[2, 3, 4, 5, 17]` to `[3, 4, 5, 17, 22]`. | The first bad self-consistent stream is in target verifier recurrent state around block rollover. Rejection sampler/all-1 acceptance is downstream. Compare latest rollover metadata and state-copy semantics directly against 0.0.3 `acd2a3150`. |
| Y | latest-vLLM, TP2 27B-AWQ MTP4, compile/FULL graph, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, plus drafter `TRITON_ATTN` persistent `slot_mapping` fix | `/tmp/latest_slotmapfix_mtp4_macos_6000.json` | Still failed. Request finished at 6000 completion tokens; no long digit run, but tail repeated `title-bar-btn:hover { background: var(--red); }`. Metrics again moved from normal acceptance to `Per-position acceptance rate: 1.000, 1.000, 1.000, 1.000`. | Missing drafter `slot_mapping` stabilization was a real compile-graph risk and remains patched, but it is not sufficient root cause. Continue on target FULL graph recurrent metadata/state path. |
| Z | latest-vLLM, TP2 27B-AWQ MTP4, compile/FULL graph, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, drafter `slot_mapping` fix, and actually wired `VLLM_SM70_QWEN_GDN_CONTEXT_CORE=1` | `/tmp/latest_context_core_mtp4_macos_6000.json` | Still failed. Request finished at 6000 completion tokens; tail repeated `transform: translateY(-50%);` with `max_same_line_run=227`. Metrics again moved from initially normal acceptance to near/all-1 windows such as `1.000, 0.995, 0.995, 0.985`. | The latest explicit-cache/custom-op metadata boundary is not sufficient root cause even after the context-core switch is wired. Continue outward to the full GDN forward/projection compile boundary and target FULL graph recurrent state replay. |
| AA | latest-vLLM, TP2 27B-AWQ MTP4, compile/FULL graph, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, drafter `slot_mapping` fix, and `VLLM_SM70_QWEN_GDN_FULL_FORWARD=1` | `/tmp/latest_gdn_fullforward_mtp4_macos_6000.json`, `/tmp/latest_gdn_fullforward_mtp4_macos_6000_2.json` | Both requests finished at 6000 completion tokens with no repeated-output collapse. Run 1: 13,684 chars, max digit run 4, `max_same_line_run=1`. Run 2: 13,556 chars, max digit run 5, `max_same_line_run=1`. Metrics stayed in normal acceptance ranges around mean acceptance length 4.2-4.6 / avg draft acceptance about 80-89%, without the sustained all-position `1.000` plateau during generation. | First positive localizer that keeps MTP4 and compile/FULL graph enabled. The remaining root is inside the latest split Qwen GDN compile boundary: input projection, recurrent core custom op, RMSNorm/output projection, or their graph dependency ordering. Do not mark final until speed impact is measured and the minimal bad sub-boundary is identified. |
| AB | latest-vLLM, TP2 27B-AWQ MTP4, compile/FULL graph, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, with only Qwen GDN RMSNorm/output projection isolated by `VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP=1` | `/tmp/latest_outputproj_mtp4_macos_6000.json` | Still failed. Request finished at 6000 completion tokens; `repeat20=220`, `repeat50=216`, `repeat100=209`, tail repeated `<html lang="zh-CN">` / `<!DOCTYPE html>` fragments. Metrics again ramped to near/all-position acceptance: `1.000, 1.000, 1.000, 0.996` and then `1.000, 1.000, 1.000, 1.000` after completion. | Isolating only RMSNorm/out-projection is not sufficient. The minimal bad boundary is earlier than output projection alone: input projection, recurrent core, or dependency ordering between those and the core custom op. Next useful A/B is input-projection+core opaque while keeping output projection compiled. |
| AC | latest-vLLM, TP2 27B-AWQ MTP4, compile/FULL graph, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, with Qwen GDN input projection plus recurrent core isolated by `VLLM_SM70_QWEN_GDN_INPUT_CORE_OP=1` and output projection left in the compiled path | `/tmp/latest_inputcore_mtp4_macos_6000.json` | Passed this reproducer. Request finished at 6000 completion tokens; 12,867 chars; `repeat20=1`, `repeat50=1`, `repeat100=1`, `max_same_line_run=2`; no repeated-output collapse. Metrics stayed normal during generation, ending around `0.926, 0.811, 0.747, 0.568`, not all-position `1.000`. | Root is now inside the input-projection-to-core half of the split GDN compile boundary. Output projection can remain compiled. Next useful A/B is input-projection-only opaque: if it passes, the projection/split products are the root; if it fails, the dependency ordering between projection outputs and recurrent core custom op is the root. |
| AD | latest-vLLM, TP2 27B-AWQ MTP4, compile/FULL graph, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, with only Qwen GDN input projection/splitting isolated by `VLLM_SM70_QWEN_GDN_INPUT_PROJECTION_OP=1`; recurrent core and output projection left in the latest split path | `/tmp/latest_inputproj_mtp4_macos_6000.json` | Still failed. Request finished at 6000 completion tokens; 15,429 chars; `repeat20=221`, `repeat50=215`, `repeat100=205`; tail repeated `title-bar-btn:hover { background: var(--red); }`. Metrics again saturated during generation: `1.000, 1.000, 1.000, 1.000`, then `1.000, 0.995, 0.995, 0.995`. | Projection/splitting products alone are not sufficient to fix quality. Since row AC passes and row AD fails, the root is the boundary/dependency ordering between Qwen GDN input projection outputs and the recurrent core custom op under compile/FULL graph. The narrow quality candidate is `VLLM_SM70_QWEN_GDN_INPUT_CORE_OP=1`, not full-forward and not output-only. |
| AE | latest-vLLM, TP2 27B-AWQ MTP4, compile/FULL graph, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, automatic SM70 Qwen GDN `input_projection_core` boundary enabled by default without explicitly setting `VLLM_SM70_QWEN_GDN_INPUT_CORE_OP` | `/tmp/latest_autoinputcore_mtp4_macos_6000.json`, `/tmp/latest_autoinputcore_mtp4_macos_6000_2.json` | Mixed result, therefore not sufficient. Run 1 passed: 6000 completion tokens, 13,451 chars, `repeat20=0`, `repeat50=0`, `repeat100=0`, `max_same_line_run=1`, no bad markers. Run 2 failed: 6000 completion tokens, 11,100 chars, `repeat20=10874`, `repeat100=10554`, `UTF-UTF` starts at char 148, and metrics entered sustained all-position acceptance around `1.000, 1.000, 1.000, 1.000`. | Row AE invalidates row AC as a complete fix. The split from `core_attn_out`/`z` into compiled RMSNorm/output projection is still part of the bad compile boundary. The next quality candidate is full Qwen GDN forward behind one opaque custom op, then measure speed. Do not default input+core alone. |
| AF | latest-vLLM, TP2 27B-AWQ MTP4, compile/FULL graph, target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, `VLLM_SM70_QWEN_GDN_FULL_FORWARD=1` explicit localizer | `/tmp/latest_fullgdn_mtp4_macos_6000_auto_1.json`, `/tmp/latest_fullgdn_mtp4_macos_6000_auto_2.json` | Passed both 6000-token runs. Run 1: 13,621 chars, `repeat20=0`, `repeat50=0`, `repeat100=0`, `max_digit_run=4`, `max_same_line_run=1`, no bad markers, overall 85.65 completion tok/s including first-request JIT. Run 2: 13,808 chars, same repeat/bad-marker result, overall 91.61 completion tok/s. Server windows stayed around 89-101 tok/s and did not enter sustained all-position `1.000`. | Full Qwen GDN forward opaque boundary is the first stable candidate that preserves MTP4, target Flash-V100, drafter Triton, and compile/FULL graph. |
| AG | latest-vLLM after defaulting the full Qwen GDN forward boundary for SM70 compile/FULL graph, same 27B-AWQ MTP4 conditions, without setting `VLLM_SM70_QWEN_GDN_FULL_FORWARD` explicitly | `/tmp/latest_default_fullgdn_mtp4_macos_6000.json` | Passed default-startup validation. Request finished at 6000 completion tokens; 13,722 chars; `repeat20=0`, `repeat50=0`, `repeat100=0`, `max_digit_run=4`, `max_same_line_run=1`; no `UTF-UTF`, long digit, `rgba(rgba`, or `showMenu: false` markers. Overall 86.25 completion tok/s including first-request JIT; server generation windows reached 89-101 tok/s with normal high-but-not-locked acceptance. | Final default gate should be full Qwen GDN forward boundary under `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH`. The helper must not call `current_platform.is_device_capability()` inside compiled forward; that caused a Dynamo unsupported graph break during default startup. |
| AH | latest-vLLM, TP2 27B-AWQ MTP4, keep `FULL_AND_PIECEWISE` compile policy but set `VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD=1` and `VLLM_SM70_QWEN_GDN_SPEC_DECODE_PIECEWISE=1` so active-MTP Mamba/GDN verifier runtime dispatch excludes `FULL` and selects the captured `PIECEWISE` graph | `bench_results/spec_piecewise_dispatch_20260619/27b_awq_mtp4_spec_piecewise_dispatch_macos512.json`, `bench_results/spec_piecewise_dispatch_20260619/27b_awq_mtp4_spec_piecewise_dispatch_macos128.json` | Runtime dispatch was confirmed as `PIECEWISE` with `BatchDescriptor(num_tokens=5, num_reqs=None, uniform=False, ...)`. Short 128/512 macOS probes did not show repeat20/50/100 or same-token runs, but speed fell to about 35 tok/s decode (`512`: 35.62 tok/s, `128`: 34.89 tok/s), much slower than the full-GDN guard 68-91 tok/s range. Initial implementation also proved capture-time FULL must not be forced to PIECEWISE, or `_dummy_run` asserts with `Expected PIECEWISE, but got FULL`; the runtime-only gate fixes that startup issue. | Do not default whole verifier PIECEWISE dispatch. It can avoid the known FULL split quality hazard, but it throws away too much of the verifier/full-graph speed path. The viable default remains full Qwen GDN forward guard; a faster future fix must repair the split GDN state/dependency contract or make a narrower opaque boundary that passes repeated 6k tests. |
| AI | external comparison: `/tmp/vllm-2080ti-definitive` at `ac2df76`, fork `v0.1.10`, base vLLM `0.21.0`, dual RTX 2080 Ti / SM75 MTP profiles | `/tmp/vllm-2080ti-definitive/vllm/config/compilation.py`, `/tmp/vllm-2080ti-definitive/launcher.sh`, `/tmp/vllm-2080ti-definitive/docs/mtp-task-sensitivity.md`, `/tmp/vllm-2080ti-definitive/docs/qwen36-kv-throughput-sweep.md` | The branch does not contain our Qwen-specific `qwen_gdn_attention_core_spec_commit` / `qwen_gdn_full_forward` split machinery. Its normal MTP launch path sets `cudagraph_mode=PIECEWISE` and `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0`; fast/aggressive modes opt into `FULL_AND_PIECEWISE` with a documented output-stability risk. Its GDN metadata builder uses upstream-style `block_table[:, :num_spec+1]` state rows and does not implement our SM70 align-mode `current_state_block_ids` / accepted-slot rollover contract. Published Qwen3.6-27B GPTQ-INT4 figures show MTP3/MTP4 plateau around `60-61 tok/s` on LongGen3, and separate profile tables show `~85-100 tok/s` on selected FP16KV/TurboQuant routes. | Reference the branch for deployment policy, not as a direct kernel fix: separate safe/normal/fast modes, keep Mamba/GDN spec-decode FULL replay opt-in only, validate MTP with real prompts, and treat MTP3 as the mixed-workload default. Do not port its generic GDN metadata into the SM70 Qwen3.6 path, and do not replace the current full-GDN quality guard with whole-verifier PIECEWISE, which measured only ~35 tok/s on this V100 tree. |

## Observed Failure Shape

The failure is not immediate. A typical bad request starts with plausible HTML,
CSS, or reasoning text. After a later drift point, the text falls into repeated
syntax or repeated characters. Server-side SpecDecoding metrics then commonly
show:

```text
Mean acceptance length: maxed out
Per-position acceptance rate: 1.000, 1.000, ...
Avg Draft acceptance rate: 100.0%
```

This means target verification is accepting the degenerate draft stream. It
does not mean quality is acceptable.

Representative bad fragments observed:

```text
rgba(255555555555...
UTF-UTF-UTF-...
body{}body{}...
background: 0.9; background: 0.9; ...
```

## Current Root-Cause Candidates

These are ordered by current likelihood. Each candidate must be closed with a
specific code audit or targeted diagnostic before moving on.

Current compile-path finding: latest drafter `TRITON_ATTN` graph stabilization
did not pin `slot_mapping`, even though draft KV cache update uses it to choose
write slots. That was patched and validated by experiment Y, but the long
output collapse still reproduced. The remaining compile-specific focus is
target FULL graph recurrent metadata/state replay for Qwen GDN/Mamba, not
generic MTP, Flash attention, or sampling parameters.

1. MTP state advancement / accepted-token accounting for hybrid GDN/Mamba
   state. The key question is whether the value passed into recurrent state
   postprocess represents scheduled verifier rows that have real state, not
   merely emitted output tokens.
2. 0.0.3 versus latest recurrent-state block rollover semantics. 0.0.3 is the
   positive control and contains `acd2a3150 [Bugfix] Stabilize MTP state
   handling`. Latest has a much more complex proposer/runner path and a fused
   GPU align postprocess; the collapse lands exactly on the 1632 state-block
   boundary.
3. Recovered-token handling after rejection. When a draft token is rejected,
   the recovered token is sampled from target distribution but may not have a
   recurrent state for the same step; the next scheduler/drafter transition must
   not treat it as already computed.
4. Draft/target token probability alignment after request-state transitions.
   Existing dump validation catches row/token mismatches, but the collapse
   suggests a later state mismatch can make target and draft agree on a wrong
   degenerate continuation.
5. Official MTP path interaction with Qwen3.6 hybrid recurrent layers on SM70.
   This must be compared against 1Cat-vLLM-0.0.3 before assuming the newest
   official implementation is clean for this model family.
6. Latest-only optimistic/deferred speculative-token accounting. 0.0.3 waits
   for `valid_sampled_token_count` at the start of `update_requests()` and
   immediately subtracts rejected draft tokens from `num_computed_tokens`.
   Latest-vLLM adds `use_async_spec_decode`, optimistic CPU lengths, GPU-side
   correction in `_prepare_inputs()`, and a deferred CPU correction after the
   forward. For hybrid recurrent/GDN state this is a high-risk difference
   because some metadata is built before or around the correction point. A/B L
   restored the synchronous correction timing and still failed, so this is no
   longer a primary suspect.
7. Latest async `output_token_ids` repair mutates `token_ids_cpu` and
   `num_tokens_no_spec`; 0.0.3 only repairs `output_token_ids`. This can affect
   the drafter's next context directly. A/B M restored the 0.0.3 behavior and
   still failed, so this is no longer a primary suspect.
8. Latest-only MTP drafter dense fp16 TurboMind fast path. A/B K disabled this
   path and the collapse still reproduced, so it is no longer a primary
   suspect unless a later code change specifically targets draft dense kernels.
9. MTP verifier multi-token decode through SM70 GDN/FLA recurrent fast paths.
   No-MTP decode is single-token and clean; MTP target verification schedules
   `num_spec_tokens + 1` rows per step. If any recurrent decode fast path or
   recurrent-state movement assumes single-token decode, target and draft can
   eventually become self-consistent on a wrong state, producing all-1
   acceptance. This is the next root-cause path to audit.
10. 0.0.3 versus latest MTP/GDN state-table semantic delta. 0.0.3 is now the
    positive control: same model, TP2, MTP4, prompt, and sampling stays clean.
    The next useful diagnostic is not another long-output proof, but a direct
    comparison of `current_state_block_ids`, `spec_state_indices_tensor`,
    `num_accepted_tokens`, `num_computed_tokens`, and block rollover behavior
    around the first corrupted-token region.

## Do Not Repeat

- Do not rerun the exact-CPU `seq_lens_cpu_upper_bound` A/B unless the related
  code changes. It failed and is not sufficient root.
- Do not treat all-position acceptance `1.000` as a pass. In this bug it is a
  late symptom of target and draft agreeing on a degenerate stream.
- Do not attribute this MTP bug to Flash-V100 alone. Target `TRITON_ATTN`
  also reproduced the collapse.
- Do not rerun context-core or explicit-greedy A/B as a proposed fix. The
  original context-core row was discovered to have been only half-wired; the
  wired re-run is row Z and still failed under the long-output MTP reproducer.
- Do not treat `VLLM_SM70_QWEN_GDN_FULL_FORWARD=1` as only "turning off
  compile". It keeps the outer compile/FULL graph active and changes only the
  Qwen GDN custom-op boundary. It is a root localizer and quality candidate,
  but still needs speed measurement and minimal sub-boundary reduction.
- Do not rerun `VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP=1` as a proposed fix.
  Row AB shows output projection isolation alone still collapses.
- Do not rerun `VLLM_SM70_QWEN_GDN_INPUT_PROJECTION_OP=1` as a proposed fix.
  Row AD shows projection-only isolation still collapses. The passing boundary
  is input projection plus recurrent core together, row AC.
- Do not rerun generic MTP/no-MTP or short-probe tests. The decisive reproducer
  is long-output MTP with dumps around the first bad region.
- Do not rerun sampler-only fixes until the target verifier state issue is
  closed. Deep dump X shows target argmax itself is already degenerate.

## 1Cat-vLLM-0.0.3 Control

The 0.0.3 control was launched on GPUs 2,3 with:

```bash
PYTHONPATH=/home/ymzx/桌面/1cat-vllm/1Cat-vLLM-0.0.3/vllm:${PYTHONPATH:-} \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
/home/ymzx/miniconda3/envs/1cat-vllm-0.0.3/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8012 \
  --model /home/ymzx/models/Qwen3.6-27B-AWQ \
  --served-model-name qwen36-27b-awq \
  --quantization awq \
  --tensor-parallel-size 2 \
  --dtype half \
  --max-model-len 262144 \
  --max-num-batched-tokens 8096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 \
  --attention-backend FLASH_ATTN_V100 \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"use_local_argmax_reduction":true,"attention_backend":"TRITON_ATTN"}' \
  --trust-remote-code
```

Notes:

- 0.0.3 does not accept `draft_sample_method`; the first attempt failed before
  engine creation with a pydantic error for that unknown field. This is not a
  quality result.
- 0.0.3 automatically applied its SM70 defaults:
  `enable_prefix_caching=True`, `mamba_cache_mode=align`,
  `cudagraph_mode=FULL_AND_PIECEWISE`, and MTP verifier capture shape 5.

## Historical 35B MTP Acceptance Probe Re-Review

The 2026-06-12/13 35B-AWQ MTP acceptance probes are useful for graph-route
isolation, but they are not sufficient long-output quality evidence for the
current 27B-AWQ macOS/code-generation collapse.

Key re-read results:

- The artifact named `nocompile`
  (`decode_latest_qwen36_35b_a3b_awq_mtp4_acceptance_probe_nocompile_i1024_o128_w1_r1_tp2_20260612.json`)
  was not a full eager run. It recorded `engine_kwargs.enforce_eager=false`,
  `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=0`, and
  `VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE=1`, with the graph policy
  indicating `mode=NONE, cudagraph_mode=FULL_DECODE_ONLY`.
- That no-compile/decode-graph run had healthy MTP activity:
  acceptance length `4.48`, overall acceptance `0.87`, steady decode
  `89.52 tok/s`.
- The bad compile graph run without forced drafter backend had collapsed
  acceptance: acceptance length `1.10`, overall acceptance `0.026`, steady
  decode `32.73 tok/s`; the `max_num_batched_tokens=2048` retry was worse
  with overall acceptance `0.004`.
- Keeping target compile/FULL graph and setting only
  `speculative_config.enforce_eager=true` restored acceptance in the short
  probe: acceptance length `5.0`, overall acceptance `1.0`, steady decode
  `123.60 tok/s`.
- Keeping target compile/FULL graph while using drafter `TRITON_ATTN` with
  drafter graph enabled also restored short-probe acceptance:
  acceptance length `4.85`, overall acceptance `0.962`, steady decode
  `150.38 tok/s`.
- A later regression showed that an external/model-level drafter
  `CUDAGraphWrapper` was the wrong path: acceptance dropped to `0.172`. Leaving
  the drafter on the proposer dispatcher's PIECEWISE graph recovered acceptance
  to `0.746`.

Important limitation:

- These probes used a repeated synthetic benchmark prompt and `ignore_eos=True`
  with only 96/128/256 output tokens. Decoding the stored token ids shows large
  repeated fragments such as
  `This fixed benchmark prompt is used to create a deterministic tokenized input...`
  in the "healthy acceptance" artifacts. Therefore, the historical acceptance
  probes cannot be used as proof that latest MTP has long-output quality.

Conclusion:

- The historical probes do support one routing conclusion: the target
  compile/FULL graph was not the sole acceptance-collapse root in that short
  35B experiment; the drafter graph/backend route mattered.
- They do **not** answer whether current latest MTP with target+drafters fully
  eager is quality-clean, and they do **not** clear the current 27B-AWQ
  long-output repetition bug.
- The server used model generation defaults `top_k=20` and `top_p=0.95`.

Result summary:

| Artifact | finish | completion tokens | reasoning chars | content chars | max digit run | repeated block |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `/tmp/003_mtp4_macos_1.json` | length | 6000 | 1596 | 13566 | 4 | no |
| `/tmp/003_mtp4_macos_2.json` | length | 6000 | 971 | 14011 | 4 | no |
| `/tmp/003_mtp4_macos_3.json` | length | 6000 | 736 | 14246 | 4 | no |

Observed 0.0.3 metrics during the first two requests stayed in normal ranges,
for example:

```text
Mean acceptance length: 4.40-4.49
Per-position acceptance rate: about 0.97, 0.89-0.92, 0.83-0.85, 0.70-0.76
Avg Draft acceptance rate: about 85-87%
```

This excludes the prompt, model, official sampling defaults, and native MTP4
concept as unavoidable causes of the current latest-vLLM collapse.

## 1Cat-vLLM-0.0.3 Recheck

A second 0.0.3 control was run on GPUs 2,3 at 2026-06-16 19:00 with the same
server shape:

```bash
PYTHONPATH=/home/ymzx/桌面/1cat-vllm/1Cat-vLLM-0.0.3/vllm:${PYTHONPATH:-} \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
/home/ymzx/miniconda3/envs/1cat-vllm-0.0.3/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8011 \
  --model /home/ymzx/models/Qwen3.6-27B-AWQ \
  --served-model-name qwen36-27b-awq \
  --quantization awq \
  --tensor-parallel-size 2 \
  --dtype half \
  --max-model-len 262144 \
  --max-num-batched-tokens 8096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 \
  --attention-backend FLASH_ATTN_V100 \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"use_local_argmax_reduction":true,"attention_backend":"TRITON_ATTN"}' \
  --trust-remote-code
```

The 0.0.3 startup log again confirmed the production defaults:

```text
enable_prefix_caching=True
mamba_cache_mode=align
cudagraph_mode=FULL_AND_PIECEWISE
cudagraph_capture_sizes=[1, 2, 4, 5, 8, 9]
target attention backend=FLASH_ATTN_V100
drafter attention backend=TRITON_ATTN
generation_config overrides: top_k=20, top_p=0.95
```

Result summary:

| Artifact | finish | completion tokens | reasoning chars | content chars | max digit run | repeated block |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `/tmp/003_recheck_mtp4_macos_1.json` | length | 6000 | 1596 | 13684 | 4 | no |
| `/tmp/003_recheck_mtp4_macos_2.json` | length | 6000 | 642 | 13996 | 5 | no |

Observed metrics stayed in the normal acceptance band during both requests:

```text
Request 1: Mean acceptance length 4.01-4.52 after warmup,
           Avg Draft acceptance rate about 75-88%.
Request 2: Mean acceptance length 3.59-4.58,
           Avg Draft acceptance rate about 65-89%.
No window saturated to Per-position acceptance rate 1.000,1.000,1.000,1.000.
```

Conclusion: 0.0.3 was rechecked and still does not show the current latest-vLLM
MTP long-output repetition/collapse. Future work should stop spending startup
time proving this again unless a new reproducer differs materially from this
prompt/model/backend/MTP shape.

## Latest Prefix+Align Reproduction

The latest-vLLM control was launched with explicit prefix caching and Mamba
align to match the 0.0.3 production shape:

```bash
PYTHONPATH=/home/ymzx/桌面/1cat-vllm/vllm:${PYTHONPATH:-} \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8011 \
  --model /home/ymzx/models/Qwen3.6-27B-AWQ \
  --served-model-name qwen36-27b-awq \
  --quantization awq \
  --tensor-parallel-size 2 \
  --dtype half \
  --max-model-len 262144 \
  --max-num-batched-tokens 8096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 \
  --attention-backend FLASH_ATTN_V100 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","use_local_argmax_reduction":true,"attention_backend":"TRITON_ATTN"}' \
  --trust-remote-code
```

Result summary:

| Artifact | finish | completion tokens | reasoning chars | content chars | notable failure |
| --- | --- | ---: | ---: | ---: | --- |
| `/tmp/latest_prefix_align_mtp4_macos_1.json` | stop | 709 | 2497 | 322 | early stop before full code output |
| `/tmp/latest_prefix_align_mtp4_macos_2.json` | length | 6000 | 445 | 17187 | repeated `-webkit-app-region: no-drag;` tail |
| `/tmp/latest_prefix_align_mtp4_macos_3.json` | length | 6000 | 674 | 14204 | odd CSS numerics such as `rgba(250, 25, 25, 25, 0.1)` |

Request 2 server metrics showed the same collapse signature:

```text
Mean acceptance length: 4.01, Avg Draft acceptance rate: 75.2%
then
Mean acceptance length: 5.00, Per-position acceptance rate: 1.000, 1.000, 1.000, 1.000
```

Important latest-only log difference:

```text
SM70 MTP draft dense fp16 TurboMind fast path requested for
down_proj,fc,gate_up_proj,o_proj,qkv_proj.
```

This path did not exist in the 0.0.3 control and is now the first A/B target.

## Latest Dense-Fastpath A/B

The dense fp16 drafter fast path was disabled with:

```bash
VLLM_SM70_MTP_DENSE_F16_FASTPATH=0
```

The usual latest TP2 MTP4 command was otherwise unchanged. The expected startup
log line:

```text
SM70 MTP draft dense fp16 TurboMind fast path requested ...
```

was absent, so the A/B was effective.

Result summary:

| Artifact | finish | completion tokens | reasoning chars | content chars | notable failure |
| --- | --- | ---: | ---: | ---: | --- |
| `/tmp/latest_no_mtp_densefast_mtp4_macos_1.json` | length | 6000 | 2756 | 10609 | repeated `UTF-UTF-...` at char 3811 |
| `/tmp/latest_no_mtp_densefast_mtp4_macos_2.json` | length | 6000 | 848 | 8417 | max digit run 2740 (`555...`) |
| `/tmp/latest_no_mtp_densefast_mtp4_macos_3.json` | length | 6000 | 837 | 11234 | repeated `UTF-UTF-...` at char 956 |

Conclusion: this drafter dense fast path is not required for the output
collapse. Do not spend more time on this path until another diagnostic points
back to draft dense layer numerics.

## Code-Path Delta: Latest vs 0.0.3

The strongest structural delta found so far is speculative-token accounting:

- 0.0.3: `gpu_model_runner.py` synchronously calls
  `_get_valid_sampled_token_count()` at the start of request update, then
  immediately computes `num_rejected = prev_num_draft_len - num_accepted` and
  subtracts it from `num_computed_tokens` before building the next inputs.
- latest-vLLM: `update_requests()` optimistically assumes all previous draft
  tokens were accepted, extends placeholders, records `prev_num_draft_tokens`,
  then corrects token counts later through `use_async_spec_decode` GPU-side
  correction in `_prepare_inputs()` and a deferred CPU correction after the
  forward.
- latest `_prepare_inputs()` still computes several CPU-side structures before
  the GPU correction point, including CPU positions, optimistic seq lengths,
  discard mask, and hybrid recurrent accepted-token staging.

This delta matches the observed failure shape: the stream starts normal, then a
state-position mismatch can slowly push target and draft into the same
degenerate continuation, after which all-position acceptance becomes 1.000.

Next diagnostic: add an env-gated synchronous correction fallback in latest so
the latest tree can run the same request-state accounting style as 0.0.3
without changing model, sampling, attention backend, or MTP4 configuration.

Result: diagnostic L still failed. The accepted-count correction path is now
recorded as excluded.

Next diagnostic: run `VLLM_SM70_MTP_LEGACY_OUTPUT_TOKEN_REPAIR=1` to make
`InputBatch.update_async_output_token_ids()` match the 0.0.3 behavior and avoid
mutating `token_ids_cpu` / `num_tokens_no_spec` during async output-id repair.

Result: diagnostic M still failed. The async output-token repair delta is now
recorded as excluded.

## Code Audit: GDN Spec Path Delta

After the 0.0.3 recheck passed, the next audit compared the latest
Qwen3.5/Qwen3.6 GDN spec path against 0.0.3.

Findings:

- `causal_conv1d_update()` spec kernel is effectively the same between
  0.0.3 and latest. The meaningful diff is stride specialization and validate
  asserts; the accepted-token sliding-window math itself matches. This is now
  lower priority unless a tensor-level compare points back to conv output.
- Pure one-request MTP decode goes through the pure spec branch in
  `gdn_attn.py`; the latest mixed spec/non-spec routing change is not the first
  suspect for the current `max_num_seqs=1` reproducer.
- Latest pure spec GDN does not use the non-spec packed/FlashQLA decode route,
  because those routes require `spec_sequence_masks is None`. The remaining
  high-value area is the GDN custom-op boundary and the tensors passed through
  it under compile/FULL graph.
- A concrete 0.0.3/latest delta was found: 0.0.3 materializes
  `mixed_qkv`, `z`, `b`, and `a` with `.contiguous()` before the GDN custom-op
  boundary. Latest materializes `z/b/a`, but leaves Qwen3.5 `mixed_qkv` as a
  projection slice view. The GDN core then calls `causal_conv1d_update()`, which
  overwrites that `mixed_qkv` tensor in place.

Diagnostic added:

```text
VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS=1
```

This restores only the 0.0.3-style `mixed_qkv.contiguous()` materialization
before the GDN custom-op boundary. It does not disable MTP, Flash-V100,
compile/FULL graph, official sampling, or recurrent verification.

Validation status: pending. If this A/B passes the same 6000-token macOS
reproducer, the likely root is a strided projection view being mutated across
the GDN custom-op boundary under the latest compile/FULL graph path. If it
fails, record it in the experiment matrix and move on; do not keep retesting
the same black-box prompt.

Result: diagnostic N still failed. The `mixed_qkv` materialization delta is now
recorded as excluded. Keep the env only as a diagnostic switch; do not treat it
as a fix.

## Code Audit: FLA Recurrent Schedule Delta

The Qwen3.5/Qwen3.6 MTP spec branch calls
`fused_recurrent_gated_delta_rule()` for target verification after
`fused_gdn_gating()`. The gating kernel itself matches 0.0.3: fixed
`BLK_HEADS=8`, `num_warps=1`, no latest SM70 schedule gate.

The recurrent kernel does differ:

- 0.0.3 SM70 path selected `num_stages=1` for `T<=1`, otherwise
  `num_stages=2`; for the MTP verifier shape `T=num_spec+1=5`, this means
  `num_stages=2`.
- 0.0.3 selected `num_warps` from `BV`; for the observed Qwen GDN verifier
  shape this is expected to be `BV=16`, `num_warps=4`.
- latest default `VLLM_SM70_FLA_RECURRENT_SCHEDULE=1` selects
  `num_stages=3` and `num_warps=1` unless env overrides are supplied.

Next diagnostic:

```text
VLLM_SM70_FLA_WARPS=4
VLLM_SM70_FLA_STAGES=2
```

This restores the 0.0.3 recurrent-kernel launch shape for the MTP verifier
without disabling MTP, Flash-V100, compile/FULL graph, official sampling, or
the recurrent kernel itself. If this passes the long-output reproducer, the
likely root is a bad latest SM70 recurrent schedule/codegen combination rather
than MTP bookkeeping.

## Code Audit: Token-Matching Sampling Delta

The latest tree was missing a reachable 0.0.3-style MTP stochastic
token-matching path in `vllm/v1/sample/rejection_sampler.py`.

Observed latest behavior before the patch:

- SM70 arg handling could set `draft_sample_method="probabilistic"` by default,
  which supplies `draft_probs` and changes the failure symptom into all-position
  `1.000` acceptance.
- Even with explicit `draft_sample_method="greedy"`, the request still failed.
  In that shape `draft_probs is None`, but latest fell through to the standard
  no-draft-probs rejection sampler. That path can emit a recovered target token
  after a draft rejection.
- For Qwen3.6 hybrid recurrent MTP, recovered-token handling is high risk:
  `_count_contiguous_spec_tokens()` counts non-placeholder emitted tokens, while
  recurrent state movement must only advance through verifier rows that have
  valid state. A recovered token can therefore stress the exact state/accounting
  boundary that matches the delayed-collapse symptom.

0.0.3 had an alternate token-matching route:

- sample target tokens for each verifier row using the normal sampler with
  expanded temperature/top-k/top-p;
- accept matching draft tokens in order;
- stop at the first sampled target token that differs from the draft token;
- emit the bonus token only if all draft tokens match.

Patch added:

- `_token_matching_sampling_enabled()` gated by
  `VLLM_MTP_STOCHASTIC_TOKEN_MATCHING=1` and safe sampling conditions;
- an early `draft_probs is None and not all_greedy` branch into
  `_sample_by_token_matching()`;
- `token_match_sample()` and `token_match_sample_kernel()`;
- a targeted unit test proving this branch is reachable.

Verification so far:

```bash
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python \
  -m py_compile \
  vllm/v1/sample/rejection_sampler.py \
  tests/v1/sample/test_rejection_sampler.py

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/ymzx/桌面/1cat-vllm/vllm:${PYTHONPATH:-} \
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python \
  -m pytest -q \
  tests/v1/sample/test_rejection_sampler.py::test_token_matching_path_is_used_for_safe_mtp_stochastic_sampling
```

Result:

```text
py_compile passed
1 passed, 16 warnings
```

First serving result: failed, but the logs showed the token-matching branch did
not run. `sample_recovered_tokens_kernel` and `rejection_random_sample_kernel`
were JIT-compiled during inference, and the output still collapsed:

```text
/tmp/latest_tokenmatch_mtp4_macos_1.json
finish=length, completion_tokens=6000
repeated tail: display: 0; / padding: 0;
metrics: acceptance eventually reached 0.996-1.000 and then all 1.000
```

The reason is a latest-vs-0.0.3 logits-processor delta:

- 0.0.3 returned empty `LogitsProcessors()` when speculative decoding was
  enabled.
- latest-vLLM installs `MinTokensLogitsProcessor` under speculative decoding.
- The original copied 0.0.3 token-matching gate rejected any logits processor.
  Therefore current latest never entered token matching even though
  `min_tokens=0` makes this processor inactive for the reproducer.

Follow-up patch: allow inactive `MinTokensLogitsProcessor` to pass the
token-matching gate. This preserves safety: active min-tokens, penalties,
bad-words, allowed-token masks, seeded generators, and other logits processors
still disable token matching.

Verification after follow-up patch:

```text
py_compile passed
test_token_matching_path_is_used_for_safe_mtp_stochastic_sampling:
1 passed, 16 warnings
```

Follow-up serving result: failed. The request completed in 57.46s and still
collapsed:

```text
/tmp/latest_tokenmatch_mintokens_mtp4_macos_1.json
finish=length, completion_tokens=6000
first repeated block around char 3425:
  height: 25px; / min-height: 25px;
later tail:
  height: 25px; ... height: 50px; ...
```

The request log did not show `sample_recovered_tokens_kernel` or
`rejection_random_sample_kernel` JIT, unlike the first token-matching serving
run. That means the inactive-MinTokens patch likely let token matching run.
Because output still failed, recovered-token standard rejection is no longer
the sole root.

Conclusion: token-matching restoration is useful for parity analysis, but it is
not sufficient. Continue with verifier multi-token recurrent state/accounting
versus 0.0.3.

## 0.0.3 Clean vs Latest Failing: Current Root-Cause Direction

The same 27B-AWQ TP2 MTP4 6000-token macOS prompt is clean on 1Cat-vLLM 0.0.3
and fails on the latest tree. This means the remaining work should compare
versioned code-path differences instead of rerunning generic MTP black-box
checks.

Current highest-probability difference is GDN/Mamba recurrent-state metadata
under speculative decode:

- 0.0.3 keeps ordinary `query_len == 1` rows on the decode path even when a
  speculative verification row is present in the batch. Latest routes non-spec
  rows through the prefill side during mixed spec/non-spec batches.
- 0.0.3 non-spec state selection effectively uses state slot 0. Latest can
  select a state slot derived from accepted-token count. That is a plausible
  MTP-only corruption path because accepted count changes over time and the
  failure only appears after a long run.
- Earlier alignment dumps showed many bad steps where accepted draft tokens and
  target argmax agree. That rules out simple sampler randomness as the complete
  root: by the time the text degenerates, the target model is also self-
  consistent on the wrong continuation.

Added diagnostic A/B gates for this exact version delta:

```bash
VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0=1
VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING=1
```

These gates are intentionally off by default. The next serving run should enable
both gates first. If the long-output collapse disappears, split the gates one at
a time to identify the minimal fix. If the collapse remains, this GDN
state-routing hypothesis is excluded and the next highest-probability path is
accepted-token accounting/CPU-token vs GPU-position mismatch versus 0.0.3.

Result of combined GDN A/B:

```bash
VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0=1
VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING=1
```

Artifact: `/tmp/latest_gdn_legacy_ab_mtp4_macos_1.json`.

The run still failed. The output no longer had a long numeric run
(`max_digit_run=13`), but it entered a deterministic CSS repetition:
`display: display: display: ...`. Server metrics showed the same collapse
signature:

```text
20:57:41 Mean acceptance length 3.08, Avg Draft acceptance rate 52.1%
20:57:51 Mean acceptance length 3.96, Avg Draft acceptance rate 74.1%
20:58:01 Mean acceptance length 2.76, Avg Draft acceptance rate 44.0%
20:58:11 Mean acceptance length 5.00, Per-position acceptance rate 1.000...
20:58:21 Mean acceptance length 5.00, Per-position acceptance rate 1.000...
20:58:31 Mean acceptance length 5.00, Per-position acceptance rate 1.000...
```

Conclusion: the two GDN state-routing deltas are not sufficient to explain the
collapse. Do not keep iterating on these two gates as the primary root. Move to
the 0.0.3-vs-latest accepted-count and token/position/seq-len synchronization
delta.

Result of accepted-count sync A/B:

```bash
VLLM_SM70_MTP_SYNC_ACCEPT_COUNTS=1
```

Artifact: `/tmp/latest_sync_accept_mtp4_macos_1.json`.

The run still failed. Failure shape changed to repeated `0 0 0 ...`, but server
metrics kept the same collapse signature:

```text
21:05:55 Mean acceptance length 3.12, Avg Draft acceptance rate 53.0%
21:06:05 Mean acceptance length 3.26, Avg Draft acceptance rate 56.4%
21:06:15 Mean acceptance length 3.85, Avg Draft acceptance rate 71.2%
21:06:25 Mean acceptance length 5.00, Per-position acceptance rate 1.000...
21:06:35 Mean acceptance length 5.00, Per-position acceptance rate 1.000...
21:06:45 Mean acceptance length 5.00, Per-position acceptance rate 1.000...
```

Conclusion: the async accepted-count correction path is not the sole root. It
can change the failure surface, but latest still reaches a state where draft and
target agree on a degenerate continuation. Continue with token/draft input
alignment and placeholder/stale-token ingress.

## Do Not Repeat

Do not rerun these as generic proof:

- no-MTP `FLASH_ATTN_V100` long-output control. It is already clean.
- MTP4-only checks. MTP1 already reproduces.
- CPU vs GPU Mamba align postprocess as the first suspect. CPU postprocess still
  fails.
- Flash-V100 target vs Triton target as the first suspect. Target Triton still
  fails.
- `top_p=1.0` as a fix. It still fails.
- `--no-async-scheduling` as a fix. It still fails.
- `VLLM_SM70_MTP_DENSE_F16_FASTPATH=0` as a fix. It still fails.
- `VLLM_SM70_MTP_SYNC_ACCEPT_COUNTS=1` as a fix. It still fails.
- `VLLM_SM70_MTP_LEGACY_OUTPUT_TOKEN_REPAIR=1` as a fix. It still fails.
- `VLLM_SM70_GDN_MIXED_QKV_CONTIGUOUS=1` as a fix. It still fails.
- `VLLM_SM70_FLA_WARPS=4 VLLM_SM70_FLA_STAGES=2` as a fix. It still fails and only restores one 0.0.3 recurrent launch-shape detail.
- explicit `draft_sample_method="greedy"` as a fix. It still fails; the
  probabilistic draft-probability path is an amplifier/symptom changer, not the
  necessary root.
- the first token-matching serving run `/tmp/latest_tokenmatch_mtp4_macos_1.json`
  as proof against token matching. It never hit the token-matching branch
  because latest's inactive `MinTokensLogitsProcessor` blocked the copied
  0.0.3 gate.
- token matching as the complete fix. After inactive MinTokens was allowed, the
  serving reproducer still failed.
- token-matching unit checks as quality proof. The branch is reachable, but only
  the serving reproducer can prove whether it fixes the long-output collapse.
- 0.0.3 long-output black-box checks for this exact reproducer. The original
  3-run control and the 2-run recheck are both clean.
- `VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0=1` together with
  `VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING=1` as a complete fix. The
  combined A/B still collapses to `display:` repetition and acceptance rate
  100%.
- `VLLM_SM70_MTP_SYNC_ACCEPT_COUNTS=1` as a complete fix. The sync accepted-
  count A/B still collapses, in that run to repeated `0 0 0 ...`, with
  acceptance rate 100%.

Only rerun a previous variant if a code change specifically targets that
variant and the run is used as a regression check.

## Required Next Checks

1. Compare the latest-vLLM MTP state/accounting semantics against 1Cat-vLLM
   0.0.3, especially accepted-token count, recovered-token scheduling, and
   GDN/Mamba state postprocess.
2. The 0.0.3 control has passed the same 6000-token long-output reproducer.
   Next comparisons should focus on code-path differences, not repeated
   black-box serving checks.
3. Add a small diagnostic that logs, per request step, generated output count,
   accepted draft count, rejected count, scheduled spec count, and recurrent
   state advance count. This should prove or exclude the off-by-one/state
   mismatch path without relying on manual chat inspection.
4. Audit latest-vLLM MTP verifier decode through hybrid GDN/FLA recurrent
   layers. This is the next highest-value path because no-MTP single-token
   decode is clean, while MTP verifier decode advances multiple target rows per
   step and then eventually makes target and draft self-consistent on a
   degenerate continuation.

## Latest MTP Dense-F16 Fastpath Disabled

The latest-vLLM A/B was launched with the same production shape as the
prefix+align reproduction, but with the latest-only MTP draft dense fp16
TurboMind fastpath disabled:

```bash
PYTHONPATH=/home/ymzx/桌面/1cat-vllm/vllm:${PYTHONPATH:-} \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
VLLM_SM70_MTP_DENSE_F16_FASTPATH=0 \
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8011 \
  --model /home/ymzx/models/Qwen3.6-27B-AWQ \
  --served-model-name qwen36-27b-awq \
  --quantization awq \
  --tensor-parallel-size 2 \
  --dtype half \
  --max-model-len 262144 \
  --max-num-batched-tokens 8096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 \
  --attention-backend FLASH_ATTN_V100 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic","use_local_argmax_reduction":true,"attention_backend":"TRITON_ATTN"}' \
  --trust-remote-code
```

The expected log line
`SM70 MTP draft dense fp16 TurboMind fast path requested ...` disappeared, so
the control variable was effective.

Result summary:

| Artifact | finish | completion tokens | reasoning chars | content chars | notable failure |
| --- | --- | ---: | ---: | ---: | --- |
| `/tmp/latest_no_mtp_densefast_mtp4_macos_1.json` | length | 6000 | 2756 | 10609 | repeated `UTF-UTF-...`, first repeat detector at char 3811 |
| `/tmp/latest_no_mtp_densefast_mtp4_macos_2.json` | length | 6000 | 848 | 8417 | max digit run 2740, repeated `55555555...` |
| `/tmp/latest_no_mtp_densefast_mtp4_macos_3.json` | length | 6000 | 837 | 11234 | repeated `UTF-UTF-...`, first repeat detector at char 956 |

Server metrics again showed the collapse signature:

```text
Normal early window: Mean acceptance length about 3.3-4.5,
Avg Draft acceptance rate about 58-88%.
Collapse window: Mean acceptance length 5.00,
Per-position acceptance rate 1.000, 1.000, 1.000, 1.000.
```

Conclusion: the MTP draft dense fp16 fastpath is not a necessary condition for
the output-quality collapse. Keep it in the suspect list only as a possible
amplifier; do not spend more time treating it as the primary root. The next
root-cause work should focus on latest-vLLM MTP state/accounting/recovered-token
semantics versus 0.0.3.

## Latest GDN Auto SSM State fp32 A/B

The 0.0.3 tree kept GDN recurrent SSM state in fp32 when
`mamba_ssm_cache_dtype=auto`; latest inherited the generic Mamba rule and used
the model dtype. We patched latest to restore the 0.0.3 fp32 rule for GDN and
ran the long-output MTP4 reproducer on GPUs 2,3 with target
`FLASH_ATTN_V100`, drafter `TRITON_ATTN`, prefix cache align, and dynamic
partition off.

Artifacts:

| Artifact | finish | completion tokens | notable failure |
| --- | --- | ---: | --- |
| `/tmp/latest_gdn_fp32_mtp4_macos_1.json` | stop | 709 | early stop; no repeat, not a long-run proof |
| `/tmp/latest_gdn_fp32_mtp4_macos_2.json` | length | 6000 | repeated `UTF-UTF-...`, first repeat detector around char 820 |
| `/tmp/latest_gdn_fp32_mtp4_macos_3.json` | length | 6000 | repeated `content: 0;`, first repeat detector around char 3640 |

The server metrics again showed the collapse signature after a normal early
window:

```text
Normal early window examples:
Mean acceptance length 3.70, Per-position acceptance 0.884, 0.714, 0.631, 0.473.
Mean acceptance length 4.06, Per-position acceptance 0.886, 0.811, 0.719, 0.640.

Collapse window examples:
Mean acceptance length 5.00, Per-position acceptance 1.000, 1.000, 1.000, 1.000.
```

Conclusion: restoring GDN SSM state fp32 is directionally correct for matching
0.0.3, but it is not a sufficient fix. Do not keep testing dtype-only changes
as the primary root. The remaining high-value comparison is current MTP's
post-accept recurrent state row selection and probability-rejection semantics
versus 0.0.3.

## 0.0.3 Clean Control: What It Rules Out

The 0.0.3 tree passed the same long macOS/code-generation reproducer with
MTP4, while latest-vLLM still collapses into repeated CSS/text/digits. This is
now the strongest control point.

It rules out these as primary causes:

- The prompt itself.
- Qwen3.6 native MTP being inherently unstable.
- Official sampling parameters by themselves.
- The final sampler choosing one bad token after a long accumulation.
- Flash attention alone: latest target `TRITON_ATTN` also fails, and the GDN
  recurrent path is still present under both attention backends.

The current failure signature is instead:

- Early windows have plausible acceptance rates, commonly 60-90%.
- After the output has already drifted into a degenerate continuation, metrics
  often become `Mean acceptance length 5.00` with per-position acceptance
  `1.000, 1.000, 1.000, 1.000`.
- Step dumps around visible repetition show `valid_count`, `positions`,
  `seq_lens`, `next_token_ids`, and `num_rejected_tokens_gpu` are internally
  consistent. The repeated text is not a single stuck token; repeated n-grams
  such as `font-weight: 3;` or `outline: none;` recur while the scheduler
  metadata still advances normally.

Interpretation: 100% acceptance is a late symptom. By the time the logs show
all-1 acceptance, target/verifier and drafter are already self-consistent on a
bad continuation. The remaining root-cause search should focus on earlier
hybrid recurrent-state corruption or state-row selection drift, not on proving
again that the final sampler repeats.

## Latest-vs-0.0.3 Code Delta: Current Suspect

The Qwen MTP model class itself is close between 0.0.3 and latest: hidden-state
injection, `fc`, per-step MTP layer cycling, and the basic proposer loop are
largely equivalent. Latest adds dense-f16 fastpath, metadata cloning, AOT
plumbing, and profiling, but disabling dense-f16 did not fix the collapse.

The sharper delta is in Qwen3.5/Qwen3.6 GDN state handling:

- 0.0.3 `Qwen3_5GatedDeltaNet` inherits the older Qwen3-Next GDN path and calls
  `torch.ops.vllm.gdn_attention_core(mixed_qkv, b, a, core_attn_out, prefix)`.
  The core op resolves metadata and KV state through the layer/context path.
- Latest `Qwen3_5GatedDeltaNet` uses
  `QwenGatedDeltaNetAttention` and calls
  `torch.ops.vllm.qwen_gdn_attention_core_standard(...)` with explicit
  `conv_state_cache`, `ssm_state_cache`, `non_spec_query_start_loc`, and
  `non_spec_state_indices_tensor`.
- 0.0.3 `gdn_attn.py` uses `block_table[:, 0]` for non-spec state rows in the
  normal non-spec path. Latest defaults to selecting a state row from
  `num_accepted_tokens - 1`, and in mixed spec/non-spec batches also routes
  non-spec rows through this accepted-token-based selection.
- A coarse A/B with `VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0=1` and
  `VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING=1` was not sufficient. That
  means the bug is likely not only the simple non-spec slot choice; the full
  spec/non-spec state table needs to be checked against 0.0.3 semantics.

Next targeted diagnostic:

1. Dump `GDNAttentionMetadataBuilder` state tables in latest around the known
   collapse window.
2. Inspect `spec_state_indices_tensor`, `non_spec_state_indices_tensor`,
   `current_state_block_ids`, `num_accepted_tokens`, `num_decode_draft_tokens`,
   and `seq_lens`.
3. Verify whether the running state block chosen for verifier decode advances
   by accepted tokens in a way 0.0.3 does not.

Do not repeat generic MTP/no-MTP or sampler-only tests before this state-table
check; they no longer move the root-cause search forward.

## 2026-06-16 Late Audit Notes

### Existing GDN state-table dump review

Reviewed `/tmp/mtp_gdn_ab_gdn/gdn_state_table_pid2267766_*.pt` and matching
rank-1 dumps. This run produced 700 dumps per rank from prefill/early decode,
covering roughly seq 5 through seq 731. It did **not** cover the later visible
collapse window around token/seq ~1800-2600.

Findings from this early range:

- Runtime pure-spec rows have `num_spec_decodes=1`,
  `num_spec_decode_tokens=5`, `num_actual_tokens=5`.
- Full graph pads metadata to 5 rows, but only row 0 is live. Padding rows are
  filled with `PAD_SLOT_ID=-1`.
- For normal runtime spec rows, `spec_state_indices_tensor[0]` matches
  `current_state_block_ids[0]`; examples include `[1, 2, 3, 4, 5]`,
  `[6, 7, 8, 9, 10]`, and `[11, 12, 13, 14, 15]` for different GDN/cache
  groups.
- The only `spec_state_indices_tensor != current_state_block_ids` entries were
  cudagraph capture/dummy rows at seq 5, where `current_state_block_ids=None`
  and `spec_state_indices_tensor` is intentionally all `-1`.
- `num_accepted_tokens[0]` stays in the expected 1-5 range.

Conclusion: the early state table does not show obvious padding contamination
or an immediate block-table mismatch. This does **not** clear the MTP/GDN state
path, because the dump stops before the actual output collapse. The next
diagnostic must capture the later window where repetition begins.

### Updated root-cause priority

The 0.0.3-vs-latest comparison now points to a narrower class of bugs:

1. Mamba/GDN align-mode state movement at block boundaries:
   latest has fused GPU postprocess, async accepted-count copies, and
   optimistic seq lengths around the older 0.0.3 CPU reference semantics.
2. Verifier recurrent state row selection when `num_accepted_tokens > 1`:
   the formula kernels are close to 0.0.3, so the remaining question is whether
   the row/slot metadata feeding them is exactly equivalent at late steps.
3. Sampler/rejection remains a secondary suspect only if state metadata is
   proven correct around the collapse window.

Required next capture:

- Enable `VLLM_SM70_DUMP_GDN_STATE_TABLE_DIR` with
  `VLLM_SM70_DUMP_GDN_STATE_TABLE_START_SEQ=1800` and
  `VLLM_SM70_DUMP_GDN_STATE_TABLE_END_SEQ=2600`.
- Enable `VLLM_MAMBA_ALIGN_DEBUG=1` in the same run to log `pre_copy`,
  `post_copy`, accepted counts, `accept_token_bias`, and source/destination
  block indices.
- Use the same 27B-AWQ MTP4 macOS/code-generation reproducer, with official
  sampling defaults. Do not use `ignore_eos`.

## 0.0.3 Stabilizing Commit Review

The current useful control is not just "0.0.3 passes"; 0.0.3 contains a
specific prior MTP state fix:

```text
acd2a3150 [Bugfix] Stabilize MTP state handling
```

The important semantic change in that commit was not to reset every request's
Mamba/GDN state mapping after speculative decode. The stable path only resets
accepted-count state when `src_block_idx == dest_block_idx`, and it avoids the
old unconditional `mamba_state_idx[req_id] = dest_block_idx` update. Latest's
fused postprocess is written to match this rule, and the CPU-postprocess A/B
already failed, so this old bug is not sufficient by itself. It remains useful
as a reference for what "correct state movement" must mean.

Two latest-vs-0.0.3 differences remain high value:

- **Default MTP cache mode**: 0.0.3 SM70 Qwen/GDN MTP defaulted to
  `enable_prefix_caching=True` and `mamba_cache_mode=align`. Latest did not
  force this when the user omitted prefix caching. This is now patched so
  hybrid linear-attention MTP uses the validated 0.0.3 production default when
  the user did not explicitly choose otherwise.
- **Qwen GDN core boundary**: 0.0.3 called the GDN core through a custom op that
  resolved cache tensors and metadata through the forward context/layer object.
  Latest Qwen3.5/Qwen3.6 calls `qwen_gdn_attention_core_standard(...)` with
  explicit `conv_state_cache`, `ssm_state_cache`, and non-spec metadata in the
  custom-op signature. A diagnostic switch was added:
  `VLLM_SM70_QWEN_GDN_CONTEXT_CORE=1`. It routes Qwen GDN through the
  0.0.3-style context-resolved core boundary without disabling MTP, Flash-V100,
  full graph, or official sampling.

Patch status:

- `vllm/engine/arg_utils.py`: restore 0.0.3-style prefix-caching default for
  SM70 hybrid linear-attention MTP when unset.
- `vllm/model_executor/models/config.py`: if prefix caching and speculative
  decoding are enabled, default Mamba cache mode to `align` instead of latest's
  generic `all`.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` and
  `vllm/model_executor/models/qwen3_5.py`: add the context-core diagnostic path.
- `vllm/config/compilation.py`: add `vllm::qwen_gdn_attention_core_context` to
  the V1 splitting ops list. Without this, the context-core A/B does not have
  the same graph-boundary semantics as the 0.0.3 control.
- `py_compile` passed for all touched Python files.

Next validation: run the usual 27B-AWQ TP2 MTP4 macOS/code-generation
reproducer with explicit `--enable-prefix-caching --mamba-cache-mode align` and
`VLLM_SM70_QWEN_GDN_CONTEXT_CORE=1`. If this passes while previous latest
prefix+align failed, the root is the latest explicit-cache custom-op boundary.
If it fails, move deeper into the actual GDN `_forward_core`/metadata contents;
do not repeat generic sampler or no-MTP tests.

### Context-Core A/B Result

The context-core diagnostic server was launched on GPUs 2,3 with:

```text
VLLM_SM70_QWEN_GDN_CONTEXT_CORE=1
VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS=0
--enable-prefix-caching
--mamba-cache-mode align
target attention backend FLASH_ATTN_V100
drafter attention backend TRITON_ATTN
MTP4
```

Startup confirmed `vllm::qwen_gdn_attention_core_context` was present in
`splitting_ops`. The first attempt before adding it to `splitting_ops` was not
counted as a quality result because graph capture did not reach a clean request
phase.

Result summary:

| Artifact | finish | completion tokens | notable result |
| --- | --- | ---: | --- |
| `/tmp/latest_contextcore_mtp4_macos_1.json` | length | 6000 | no long repeated block, but numeric/CSS corruption such as `rgba(2020202020.05)` and `font-weight: 50` |
| `/tmp/latest_contextcore_mtp4_macos_2.json` | length | 6000 | failed with a 5261-character run of `0` |
| `/tmp/latest_contextcore_mtp4_macos_3.json` | length | 6000 | no single huge digit run, but CSS is corrupted into repeated/invalid numeric fragments such as `grid-template-columns: repeat(auto-fill, 0; 35 2px 22255...)` |

Observed metrics in the available windows did not always saturate to
all-position `1.000`; for example the final windows were around avg draft
acceptance `74%`. This is important: all-1 acceptance is a common late symptom,
but the output can already be broken without an all-1 metrics window.

Conclusion: the latest explicit-cache Qwen GDN custom-op boundary is not the
sole root. It may change the failure shape, but it is not sufficient to restore
0.0.3 quality. The next 0.0.3 delta to test is draft sampling/rejection
semantics, especially latest's default `draft_sample_method=probabilistic`
versus 0.0.3 not accepting that field.

### FULL Graph Spec Tail Padding Correction

The earlier note that replay tails should use `NULL_BLOCK_ID == 0` was wrong.
0.0.3 control dumps showed padded rows as zero, but that does not make zero a
safe sentinel in the latest GDN/FULL graph path. In the latest implementation
state slot 0 is a real recurrent-state slot; replay padding that reaches a GDN
state consumer must be `PAD_SLOT_ID == -1`, matching the capture/dummy metadata
path and the state kernels that explicitly skip negative slots.

Patch:

- `vllm/v1/attention/backends/gdn_attn.py`: FULL graph replay-only padded tails
  for both `spec_state_indices_tensor` and `non_spec_state_indices_tensor` now
  use `PAD_SLOT_ID`, not `NULL_BLOCK_ID`.
- `tests/v1/attention/test_gdn_metadata_builder.py`: the FULL graph MTP replay
  test now asserts padded spec state rows are `PAD_SLOT_ID`.

Verification:

```text
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python -m pytest -q \
  tests/v1/attention/test_gdn_metadata_builder.py -q
```

Result: `16 passed`.

Quality impact:

- This fixes a real padding-state corruption risk without disabling MTP, Flash,
  compile, or FULL graph.
- Earlier quality experiments that used replay-tail `NULL_BLOCK_ID` are not
  sufficient to exclude FULL graph padding as a contributor; they should not be
  treated as final evidence after this correction.
- End-to-end MTP quality still needs to be rerun after this patch with dynamic
  Flash partitions disabled.

### Qwen MTP Step-Index Compatibility A/B

0.0.3 uses the Qwen MTP drafter through `spec_decode/eagle.py`. In the
multi-token draft loop it passes `draft_index=token_index + 1` to attention
metadata, but it does not pass `spec_step_idx` into the Qwen MTP model or its
draft logits/top-token helpers. In practice, Qwen3_5/Qwen3_5MTP uses the
default `spec_step_idx=0` for every draft forward.

Latest `llm_base_proposer.py` does pass `spec_step_idx=1..num_spec_tokens` into
the model, logits, top-token path, and drafter graph variant selection. This is
a real 0.0.3/latest semantic difference, so a narrow diagnostic gate was added:

```bash
VLLM_SM70_MTP_LEGACY_QWEN_STEP_IDX=1
```

Run:

```text
27B-AWQ TP2, target FLASH_ATTN_V100, drafter TRITON_ATTN, MTP4
--enable-prefix-caching --mamba-cache-mode align
VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS=0
temperature=0, max_tokens=3000
```

Artifact:

```text
/tmp/latest_legacy_stepidx_mtp4_macos_temp0_3000.json
```

Result: still failed. The output did not enter a contiguous `000000...` run,
but collapsed into a repeated CSS zero pattern:

```text
#menu-bar .menu-item {
  padding: 0 0 0 0 0 0 0 ...
```

Server metrics still ramped into a high-acceptance phase during the request and
logged an all-position `1.000` window immediately after completion. Conclusion:
Qwen MTP step-index drift is a real code-path difference, but it is not the
complete root cause of the latest long-output MTP collapse. Do not rerun this
variant as a proposed fix unless a new patch specifically changes Qwen MTP
layer/step semantics.

### CPU Postprocess Re-Test After Tail Padding Fix

The old CPU postprocess A/B was run before the FULL graph tail padding bug was
fixed, so it was not a clean exclusion. It was repeated after the tail-padding
patch with:

```text
VLLM_MAMBA_ALIGN_CPU_POSTPROCESS=1
VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS=0
--enable-prefix-caching
--mamba-cache-mode align
target attention backend FLASH_ATTN_V100
drafter attention backend TRITON_ATTN
MTP4
```

Result artifact:

| Artifact | finish | completion tokens | result |
| --- | --- | ---: | --- |
| `/tmp/latest_cpu_postprocess_mtp4_macos_temp0_3000.json` | length | 3000 | failed with a long tail of repeated `0` |

The server metrics still rose toward very high acceptance near the failure:

```text
Mean acceptance length: 4.87
Per-position acceptance rate: 0.996, 0.984, 0.960, 0.935
Avg Draft acceptance rate: 96.9%
```

Conclusion: the fused GPU Mamba/GDN postprocess is not the sole remaining root.
It may still be worth optimizing later, but the current correctness bug also
reproduces with the 0.0.3-style CPU reference state-copy path. Do not keep
cycling on fused postprocess until new evidence appears.

Updated hypothesis: the all-1 acceptance windows are likely a late symptom after
the model/drafter token stream has already entered a degenerate loop. The next
useful comparison is 0.0.3-vs-latest MTP proposer/verifier token plumbing:
`draft_token_ids`, `sampled_token_ids`, `valid_sampled_count`,
`target_positions`, `next_token_ids`, and draft probability alignment around
the first corrupted segment.

### Legacy Output Repair Re-Test After Tail Padding Fix

The earlier `VLLM_SM70_MTP_LEGACY_OUTPUT_TOKEN_REPAIR=1` A/B was run before the
FULL graph GDN tail-padding bug was fixed, so it was rerun after the padding
patch.

Run shape:

```text
27B-AWQ TP2, GPUs 2,3
target attention backend FLASH_ATTN_V100
drafter attention backend TRITON_ATTN
--enable-prefix-caching --mamba-cache-mode align
MTP4, draft_sample_method=greedy, use_local_argmax_reduction=true
VLLM_MTP_STOCHASTIC_TOKEN_MATCHING=1
VLLM_SM70_MTP_LEGACY_OUTPUT_TOKEN_REPAIR=1
VLLM_SM70_MTP_DUMP_STEP_DIR=/tmp/mtp_stepdump_legacyrepair_afterpad_macos
```

Artifact:

```text
/tmp/latest_legacyrepair_afterpad_mtp4_macos_6000.json
```

Result:

- `finish_reason=length`, `completion_tokens=6000`.
- Output still failed quality. It did not become one long pure digit run
  (`max_digit_run=13`), but it entered corrupted CSS/value repetition around
  character 8188:

```text
padding:px
border-bottom:px solid rgba(0.0)
.prefs-row
.cal-day:hover
```

- Late server metrics stayed around normal/high but did not saturate to all
  positions `1.000`:

```text
Mean acceptance length: 3.42
Per-position acceptance rate: 0.828, 0.650, 0.528, 0.417
Avg Draft acceptance rate: 60.6%
```

Conclusion: latest async output-token repair mutation of `token_ids_cpu` /
`num_tokens_no_spec` can change the failure symptom and the all-1 metrics
behavior, but it is not the root cause of the MTP long-output corruption after
the GDN padding fix. The primary corruption must occur earlier in the
drafter/verifier model-state or hidden/logit path.

### Packed Recurrent Decode A/B After Tail Padding Fix

The latest tree differs from 0.0.3 because it defaults
`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1` for the SM70 Flash-V100 baseline,
while 0.0.3 did not have this packed recurrent decode route. Previous migration
notes already showed that packed recurrent decode writes slightly different GDN
SSM state than the stable recurrent path, so it was a plausible MTP-only
amplification source.

Targeted A/B run:

```text
latest-vLLM, Qwen3.6-27B-AWQ, TP2 on GPUs 2,3
target attention FLASH_ATTN_V100, drafter attention TRITON_ATTN
prefix caching enabled, mamba_cache_mode=align
compile/full graph enabled
MTP4, draft_sample_method=greedy, use_local_argmax_reduction=true
VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0
same 6000-token macOS/code-generation prompt
```

Artifact:

```text
/tmp/latest_packedoff_mtp4_macos_6000.json
```

Result:

- `finish_reason=length`, `completion_tokens=6000`.
- Output still collapsed into a long run of zeros:
  `max_digit_run=3752`.
- Server metrics again moved from normal acceptance to all-position 1.000:
  `Per-position acceptance rate: 1.000, 1.000, 1.000, 1.000`.

Conclusion: packed recurrent decode is not the sole or necessary root of the MTP
long-output collapse. Packed recurrent remains a real numeric drift source and
should not be treated as strict max-diff-zero, but the MTP corruption also
reproduces on the stable/non-packed recurrent route. Do not spend another model
startup testing `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0` as a proposed fix
for this MTP bug.

Updated root-cause direction: keep the investigation on latest-vs-0.0.3
speculative state/cache advancement and accepted-token bookkeeping, especially
the point where the target begins accepting every draft token after the output
stream is already corrupted.

### Default MTP Step Dump After Packed-Off A/B

Targeted diagnostic run:

```text
latest-vLLM, Qwen3.6-27B-AWQ, TP2 on GPUs 2,3
target attention FLASH_ATTN_V100, drafter attention TRITON_ATTN
prefix caching enabled, mamba_cache_mode=align
compile/full graph enabled
MTP4, draft_sample_method=greedy, use_local_argmax_reduction=true
default VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1
VLLM_SM70_MTP_DUMP_STEP_DIR=/tmp/mtp_stepdump_late_default
VLLM_SM70_DUMP_GDN_STATE_TABLE_DIR=/tmp/mtp_gdn_state_late_default
same 6000-token macOS/code-generation prompt
```

Artifact:

```text
/tmp/latest_diag_mtp4_macos_6000.json
/tmp/mtp_stepdump_late_default
/tmp/mtp_gdn_state_late_default
```

Result:

- `finish_reason=length`, `completion_tokens=6000`.
- Output collapsed almost immediately in the HTML header:

```text
<meta name="viewport" viewBox="0 800 800 450000000...
```

- Reconstructed output tokens from rank0 step dump showed the first long zero
  run starts at generated token index 568. Around the transition:

```text
step 636 input/draft:  name="viewport" content
step 636 output:       ="viewport" viewBox
step 639 output:       ="0 8
step 642 output:       00 8
step 645 output:       00 4
step 648 output:       5
step 651 output:       00
step 654 output:       0
step 657 onward:       00000 per MTP iteration
```

- The later all-1 acceptance window is therefore a symptom after the stream is
  already bad. In this run the first bad transition is the verifier/sample path
  replacing normal `content` continuation with `viewBox`, then entering a
  self-consistent zero loop.
- The GDN state dump window was configured for sequence 1800-2600, so this
  early-collapse run did not capture the first bad GDN state transition.

Conclusion: the next diagnostic must dump sampler alignment/top-k around the
early verifier transition, not another packed recurrent or dynamic-partition
A/B. Start with `VLLM_SPEC_DUMP_ALIGNMENT=1` and a high enough
`VLLM_SPEC_DUMP_ALIGNMENT_LIMIT` to include the step-636 region, then inspect
whether `content`, `viewBox`, and token `0` are coming from target logits or
from rejection/recovered-token handling.

### CPU Postprocess Full 6000-Token Re-Test

After the tail-padding and GDN dtype fixes, the latest tree was rerun with the
0.0.3-style CPU postprocess path enabled:

```text
latest-vLLM, Qwen3.6-27B-AWQ, TP2 on GPUs 2,3
target attention FLASH_ATTN_V100, drafter attention TRITON_ATTN
prefix caching enabled, mamba_cache_mode=align
dynamic Flash-V100 decode partitions disabled
MTP4, official sampling defaults
VLLM_MAMBA_ALIGN_CPU_POSTPROCESS=1
VLLM_SPEC_DUMP_ALIGNMENT=1
VLLM_SPEC_DUMP_ALIGNMENT_LIMIT=900
same macOS/code-generation prompt, max_tokens=6000
```

Artifact:

```text
/tmp/latest_cpu_postprocess_mtp4_macos_6000.json
/tmp/spec_alignment_pid2564448_*.pt
/tmp/spec_alignment_pid2564449_*.pt
```

Result:

- `finish_reason=length`, `completion_tokens=6000`, output sha256
  `ff22f03b3d46ee4cbc34973033f351c677eafa96700d7c0abfab55364b3b9316`.
- Output collapsed into repeated CSS:
  `border-radius: 3;` first appears around char index 2507 and appears 557
  times. The tail is a stable `border-radius: 3;` loop.
- The corrupted transition already contains invalid CSS before the final loop:

```text
line-height: 13px;
cursor: 13px;
cursor: pointer;
white-space: 13;
white-space: 13;
color: 13;
font-size: 13;
font-size: 13;
padding: 2px 8;
border-radius: 3;
```

- Alignment dumps show the loop token ids
  `[17183, 25, 220, 18, 26, 198, 262, 3755]`, decoded as
  `-radius: 3;\n    border`.
- Around dump steps 600-900, target, draft, and output tokens are already
  self-consistent on this loop. `draft_token_probs`, `draft_target_probs`, and
  `draft_acceptance_caps` are all `1.0`; target top-k dumps usually contain
  only one finite entry and the remaining entries are `-inf`.
- Server metrics again saturated to all-position acceptance `1.000`.

Conclusion: the CPU postprocess/reference state-copy path does not fix the
latest MTP collapse. Dynamic partition is also disabled in this run. The root
must be earlier than rejection sampling and earlier than the all-1 acceptance
metric, in the latest-vs-0.0.3 MTP verifier token/state plumbing that first
pushes target logits into the invalid CSS/digit continuation.

## 2026-06-17 Code Audit: Hidden Spec Metadata Dependency

After re-reading the already recorded exclusions, the strongest remaining
root-cause path is compile/FULL-graph dependency tracking for Qwen GDN spec
metadata, not another sampler or attention-backend toggle.

Reasoning:

- no-MTP uses non-spec GDN metadata and is stable.
- MTP verifier uses spec GDN metadata:
  `spec_query_start_loc`, `spec_state_indices_tensor`, `spec_token_indx`,
  `spec_sequence_masks`, and `num_accepted_tokens`.
- Latest Qwen3.5/Qwen3.6 GDN fastpath already passed non-spec metadata as
  explicit custom-op tensor inputs, but left spec metadata hidden behind
  `forward_context.attn_metadata`.
- Under compile/FULL graph this is a dangerous asymmetry: the graph/op boundary
  can see the non-spec metadata dependency, but the verifier's spec-state
  tensors are implicit Python-context reads. This fits the observed shape:
  no-MTP clean, MTP-only corruption, target logits becoming self-consistently
  degenerate, and failures clustered around recurrent-state/block rollover.
- The earlier `VLLM_SM70_QWEN_GDN_CONTEXT_CORE=1` A/B did not disprove this,
  because that path also kept spec metadata implicit. It only changed cache
  resolution style.

Patch:

- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
  now exposes `_qwen_gdn_metadata_tensors()` that returns both non-spec and
  spec GDN metadata tensors.
- `qwen_gdn_attention_core_standard(...)` now takes the spec metadata tensors
  as explicit custom-op inputs and temporarily patches the `GDNAttentionMetadata`
  object with those tensors during `_forward_core()`, restoring the fields
  afterward.
- `vllm/model_executor/models/qwen3_5.py` now passes the same explicit spec
  metadata tensors from the Qwen3.5 wrapper path.

Why this is not a speed-sacrifice workaround:

- It does not disable MTP, Flash-V100, compile/FULL graph, GDN kernels, or
  official sampling.
- It only makes the verifier-state metadata that the existing kernels already
  consume visible at the custom-op boundary, matching the dependency discipline
  already used for non-spec metadata.

Static verification:

```text
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python -m py_compile \
  vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py \
  vllm/vllm/model_executor/models/qwen3_5.py
```

Result: passed.

Required next validation:

- Run the standard 27B-AWQ TP2 MTP4 long-output macOS/code prompt on GPUs 2,3
  with official sampling defaults, prefix caching enabled, `mamba_cache_mode=align`,
  target `FLASH_ATTN_V100`, drafter `TRITON_ATTN`, and dynamic partition disabled
  unless explicitly testing that separate feature.
- Pass criterion: no repeated digit/CSS/text collapse through 6000 generated
  tokens, and SpecDecoding metrics must not enter a sustained all-position
  `1.000` phase while content is degrading.

## 2026-06-17 After-Reboot Validation And Draft Proposal Fix

After the machine reboot, the latest tree was retested on GPUs 2,3 with the
same 27B-AWQ TP2 production MTP shape:

```text
target attention FLASH_ATTN_V100
drafter attention TRITON_ATTN
MTP4
draft_sample_method=probabilistic
official model generation defaults: top_k=20, top_p=0.95
--enable-prefix-caching --mamba-cache-mode align
VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS=0
compile/FULL graph enabled through the SM70 0.0.3 compatibility policy
```

### Greedy Draft Sanity After Explicit Spec Metadata Patch

Artifact:

```text
/tmp/mtp_greedy_live_snake_6000_after_reboot.json
```

Result:

- `finish_reason=length`, `completion_tokens=6000`.
- No catastrophic `000...`, `555...`, `showMenu: false`, `UTF-UTF`, or
  `rgba(rgba` collapse.
- `max_digit_run=4`.
- Some normal stochastic code imperfections remain, e.g.
  `rgba(255, 2555, 255, 0.05)`. Do not confuse these local code-generation
  mistakes with the earlier MTP collapse signature.

### Probabilistic Draft Failure Before Proposal Fix

Artifact:

```text
/tmp/mtp_prob_live_snake_6000_after_reboot.json
```

Result:

- `finish_reason=length`, `completion_tokens=6000`.
- No long zero run, but output collapsed into repeated
  `showMenu: false,`.
- Alignment dumps:

```text
/tmp/spec_alignment_pid53602_*.pt
/tmp/spec_alignment_pid53603_*.pt
/tmp/mtp_prob_stepdump_after_reboot
/tmp/mtp_prob_gdn_after_reboot
```

Key diagnostics:

- TP rank0/rank1 dumps matched exactly for `draft_token_ids`,
  `output_token_ids`, `output_valid_counts`, `target_argmax`,
  `draft_token_probs`, `draft_target_probs`, and `draft_acceptance_caps`.
  Rank divergence is excluded.
- Late dumps showed target, draft, and output already self-consistent on the
  repeated stream; `draft_token_probs`, `draft_target_probs`, and acceptance
  caps were all `1.0`.

### 0.0.3 Draft Proposal Semantic Delta

A concrete 0.0.3/latest delta was found in probabilistic Qwen MTP draft
sampling:

- 0.0.3: when `top_k` is present, the draft proposal distribution applies
  top-k only and intentionally skips top-p:

```python
draft_top_p = None if sampling_metadata.top_k is not None else sampling_metadata.top_p
```

- latest before the patch: the draft proposal applied both top-k and top-p.

This does not change target/model sampling. The final target sampler still uses
the official `top_k=20, top_p=0.95`; only the speculative draft proposal `q`
is restored to the validated 0.0.3 shape.

Patch:

- `vllm/v1/spec_decode/llm_base_proposer.py`: restore 0.0.3 top-k-only draft
  proposal semantics when `top_k` is configured.
- `tests/v1/spec_decode/test_tp_draft_sampling_sync.py`: add a regression test
  proving that `top_p` passed into draft proposal is `None` when `top_k` is
  present.

Verification:

```text
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python -m py_compile \
  vllm/v1/spec_decode/llm_base_proposer.py \
  tests/v1/spec_decode/test_tp_draft_sampling_sync.py

/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python -m pytest -q \
  tests/v1/spec_decode/test_tp_draft_sampling_sync.py
```

Result: `4 passed`.

### Probabilistic Draft After Top-K-Only Proposal Fix

Artifacts:

```text
/tmp/latest_topkonly_prob_mtp4_macos_6000.json
/tmp/latest_topkonly_prob_mtp4_snake_6000.json
/tmp/spec_alignment_pid60774_*.pt
/tmp/spec_alignment_pid60775_*.pt
/tmp/mtp_topkonly_stepdump_after_reboot
```

Results:

| Prompt | finish | completion tokens | max digit run | repeated-collapse result |
| --- | --- | ---: | ---: | --- |
| macOS/code prompt | length | 6000 | 13 | no long all-zero/all-five/fixed-fragment collapse |
| snake game prompt | length | 6000 | 7 | no long all-zero/all-five/fixed-fragment collapse |

The old catastrophic signatures were absent:

```text
00000000000000000000: not found
showMenu: false: not found in the fixed runs
UTF-UTF: not found
rgba(rgba: not found
```

Alignment summary:

```text
macOS run: 1604 rank0 dumps, mean output_valid_count=3.74,
           mean acceptance cap=0.824, all-cap-1 steps=297.
snake run: 996 rank0 dumps, mean output_valid_count=3.84,
           mean acceptance cap=0.837, all-cap-1 steps=288.
```

Interpretation:

- The top-k-only proposal fix materially changes the failure mode: the
  probabilistic path no longer enters the unusable repeated-output collapse
  seen before the patch.
- Some local stochastic code-generation imperfections remain, such as invalid
  CSS color literals (`#5555555`) or repeated common CSS layout lines. The
  greedy-draft sanity artifact also contains small CSS numeric mistakes
  (`rgba(255, 2555, 255, 0.05)`), so these are not by themselves proof of the
  earlier MTP collapse.
- Do not mark the whole MTP quality issue closed solely from these two runs.
  The next useful check is a matched no-MTP or 0.0.3-after-reboot artifact for
  the same prompts if we need to decide whether the remaining local CSS
  mistakes are model sampling noise or still MTP-specific.

### Step-Index Delta Recheck

The model configs for both 27B-AWQ and 35B-AWQ have no
`mtp_num_hidden_layers`; Qwen MTP defaults to one MTP layer. Therefore
`spec_step_idx % self.num_mtp_layers` is always zero for these target models.
The latest-vs-0.0.3 `spec_step_idx` plumbing is not a plausible root for the
27B/35B quality issue and should not be rerun as the next A/B.

### Compile-Path Root Re-Focus After End-to-End Repro

User end-to-end testing after the top-k-only proposal patch still reproduced
the same production failure class when MTP was enabled: output starts normally,
then later collapses into repeated text / repeated numeric fragments, while
the MTP metrics can jump to all-position acceptance `1.000`. No-MTP runs for
the same interactive use case were normal. This supersedes the earlier
interpretation that the top-k-only proposal patch closed the quality issue.

The useful historical A/B remains:

- latest MTP with compile / CUDA graph enabled: bad long-output quality.
- latest MTP with compile disabled or drafter `enforce_eager=true`: acceptance
  and output quality recover in the recorded probes.
- 0.0.3 MTP did not show the same repeated-output collapse under its validated
  graph setup.

Conclusion: keep the investigation centered on the latest compile/CUDA-graph
path, especially the drafter/proposer graph replay path. Do not spend more time
on generic MTP sampling or Flash-vs-Triton target-attention tests unless they
directly explain the compile-only failure.

### Triton Drafter Metadata Address-Stability Fix

Concrete latest-vs-0.0.3 compile-path risk:

- latest proposer clones mutable drafter metadata (`seq_lens`,
  CPU shadows, etc.) so target persistent metadata is not mutated by draft
  rejection accounting.
- 0.0.3 did not have the same cloned-metadata path.
- `CUDAGraphWrapper` explicitly does not persist/copy forward-context metadata;
  it only captures the addresses used by the runnable. Its debug address check
  sees tensor arguments, but not `ForwardContext.attn_metadata` tensors such as
  `attn_metadata.seq_lens`.
- MTP drafter defaults to `TRITON_ATTN` on SM70. The Triton metadata builder
  used `common_attn_metadata.seq_lens`, `query_start_loc`, and `block_table`
  directly for drafter metadata. Unlike the Flash-V100 draft builder, it did
  not copy those tensors into stable persistent buffers before CUDA graph
  replay.

This creates a compile-only failure mode: CUDA graph capture can bind a
one-off cloned `seq_lens` / metadata tensor address, while later replay should
use the current drafter metadata. Eager/no-compile paths read the current
Python metadata and are not affected.

Patch:

- `vllm/v1/attention/backends/triton_attn.py`: add speculative-drafter
  persistent metadata buffers for `block_table`, `seq_lens`, and
  `query_start_loc`, then override `build_for_drafting()` to copy live metadata
  into those buffers and expose stable tensor views to graph replay. This
  mirrors the existing Flash-V100 draft metadata stabilization and keeps the
  drafter CUDA graph enabled.

Verification so far:

```text
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python -m py_compile \
  vllm/v1/attention/backends/triton_attn.py \
  vllm/v1/spec_decode/llm_base_proposer.py \
  vllm/v1/worker/gpu_model_runner.py
```

Result: syntax/bytecode compile passed.

Next required validation when GPUs are free: rerun the 27B-AWQ MTP4 compile-on
long prompt (`帮我用html做一个macos，功能要尽可能全，要尽可能还`, max output
around 6000) and confirm that the previous late repeated-output collapse and
all-position acceptance plateau do not recur. If it still recurs, the next
compile-specific suspect is stale recurrent/GDN metadata in the target
verifier graph, not generic MTP sampling.
