# Inkling-2-Mini from Unsloth GGUF on SM70 (stacked on #1598)

This note covers the GGUF route for the `Inkling2` model added in #1598. That model serves
the official `thinkingmachines/Inkling-2-Mini` checkpoint on Volta. This work lets the same
model load the Unsloth GGUF files (`unsloth/Inkling-2-Mini-GGUF`, `UD-Q4_K_XL` and
`UD-Q8_K_XL`), the format most people use to run the model on consumer and older
datacenter GPUs. It also records the measurements behind the follow-up work on single-card
decode speed and KV memory.

It builds on a standalone GGUF port (TP=1, own model code) that served these files on V100.
That port is not proposed for merging, because it would duplicate #1598. Its results are
the reference for what a single V100 can do with these weights, and its parts move into
#1598's model here.

## Why

On V100 the GGUF files ran only through llama.cpp. Unsloth's build b12588 gave:
- Q4 on one card: about 34 tok/s decode and 400–500 tok/s prefill.
- Q8: two cards needed, about 28 tok/s.

The model is looped: its 24 stored layers make 48 layer visits per token, each with many
small operations. llama.cpp spends most of each token launching kernels. At ~3 GB of weight
reads per token (Q4), V100 bandwidth would allow ~250 tok/s.

## Same-machine measurements (2× V100-SXM2-32GB, NVLink)

**Setup:**
- #1598 at `f5d1a9a57`, with #1599 merged for the parsers.
- Official checkpoint, `--dtype float16 --kv-cache-dtype float16 --enable-prefix-caching
  --disable-custom-all-reduce --max-num-batched-tokens 2048 --max-num-seqs 4`.
- `--language-model-only`: with transformers 5.20.2, multimodal profiling fails at startup
  on the checkpoint's image processor.
- The standalone GGUF port on 1Cat main `2d218ab037`, without prefix caching.
- Same request files for both, with 512 output tokens and prompts not shared between
  requests.

| | standalone port, GGUF Q4, 1 GPU | standalone port, GGUF Q8, 1 GPU | #1598 TP=2 | #1598 TP=1 |
| --- | --- | --- | --- | --- |
| decode tok/s (short / 4k / 16k prompt) | 87.1 / 78.3 / 68.5 | 71.5 / 65.4 / 58.4 | 47.2 / 44.8 / 42.9 | 45.3 / 42.5 / 39.3 |
| prefill tok/s (4k / 16k) | 1201 / 1253 | 1308 / 1267 | 1331 / 1316 | 793 / 773 |
| 4 concurrent, total tok/s | 168 | 160 | 115 | 112 |
| largest context on one card | 262k | ~164k served (~175k KV capacity) | n/a | ~28k |
| time to first token, next turn of a 41k conversation | 40 s | 40 s | 0.4 s | n/a |
| HumanEval / HumanEval+ (164) | 97.6 / 94.5 | 98.2 / 93.3 | 97.6 / 93.9 | n/a |
| LiveCodeBench v6, 60 problems | 80.0 | 90.0 | 85.0 | n/a |

How the scores and measurements were taken:
- **Coding settings:** one sample per problem, a 65,536-token budget, temperature 1.0 and
  top_p 1.0, run in a network-less sandbox. On LiveCodeBench, one problem is 1.7 points.
- **Long context:** needle retrieval at 32k, 128k and 250k passed on both.
- **Sanity:** 12/12 cases passed on all four configurations.

What this shows:

1. **Quality is equivalent.** The official checkpoint and both GGUFs score within sampling
   noise. The Q8 GGUF holds essentially the official weights (next section).
2. **Decode speed comes from the model and kernels, not TP.** #1598 decodes at nearly the
   same rate on one GPU as on two. With equivalent weights on one card (GGUF Q8), the
   standalone port is 1.5–1.6× faster.
3. **KV memory per token is much higher in #1598 on one card.** Weights are similar in
   size (22.6 GiB for #1598, 21.5 GiB for the GGUF Q8), but #1598 has 4.66 GiB of KV
   giving 25,239 tokens. The standalone port with the GGUF Q8 has ~6.1 GiB of KV, giving
   ~175k tokens (served at 164k). Through #1598 (this PR), both GGUFs take #1598's memory
   and KV capacity: 25,239 tokens on one card (see Implementation status).
   - At TP=2, #1598 uses ~30 KB per token per GPU, where its half of the 8 full-attention
     visits needs 16 KB.
   - Suspected cause: the hybrid KV allocator sizing the sliding-window groups like the
     full-attention group, plus per-sequence window reserves. **To be confirmed.**
4. **Prefix caching in #1598 removes the per-turn re-prefill** that dominates agent loops
   on the standalone port.

## Weight formats

GGUF shapes are listed by llama.cpp in reverse order (`ne0` first). The table uses PyTorch
order.

| tensor group | official checkpoint | GGUF Q8 (`UD-Q8_K_XL`) | GGUF Q4 (`UD-Q4_K_XL`) |
| --- | --- | --- | --- |
| attention projections (q, gate, rel, k, v, o) | BF16 | BF16 | **Q8_0** |
| embedding, unembedding | BF16 | BF16 | **Q8_0** |
| experts, layers 0–19 | NVFP4: U8 `[E, N, K/2]`, F8 E4M3 block scales `[E, N, K/16]`, F32 `amax [E]`, `outer_scale_denominator [1]` | NVFP4 (GGML `NVFP4` blocks), F32 per-expert `scale [E]` | same as Q8 |
| experts, layers 20–23 | MXFP8: F8 E4M3 `[E, N, K]`, U8 E8M0 scales `[E, N, K/32]` | **BF16** | **Q8_0** |
| router, norms, short-conv taps, clips, sink / conv projections, loop controllers | BF16 / F32 | F32 / BF16 | F32 / BF16 |

Loaded on one V100:
- Q4 GGUF: 17.8 GiB.
- Q8 GGUF: 21.5 GiB (layers 20–23 experts in FP16).
- Official checkpoint in #1598: 22.6 GiB. #1598 dequantises MXFP8 to FP16 for its SM70
  expert GEMV.

## Name map (GGUF → official)

The two are almost one-to-one. `blk.N` is physical layer N. Prefix
`model.llm.layers.N.` on the right.

| GGUF `blk.N.` | official |
| --- | --- |
| `attn_q` / `attn_gate` / `attn_rel` / `attn_k` / `attn_v` / `attn_output` | `attn.wq_du` / `attn.gate_attn` / `attn.wr_du` / `attn.wk_dv` / `attn.wv_dv` / `attn.wo_ud` |
| `attn_sink_proj` | `attn.attn_sink_proj` |
| `attn_q_clip` `[24,128]`, `attn_rel_clip` `[24,16]` | `attn.q_l2_clip.scale`, `attn.rel_l2_clip.scale` |
| `attn_rel_bias` `[1024,16]` | `attn.rel_logits_proj.proj` `[16,1024]` (transpose) |
| `attn_k_conv` / `attn_v_conv` `[1024,4]` | sliding layers: `attn.k_sconv.weight` / `attn.v_sconv.weight` `[1024,1,4]`; full layers: `attn.k_sconv.static_w` / `attn.v_sconv.static_w` `[4,1024]` |
| `attn_k_conv_proj` / `attn_v_conv_proj` (full layers) | `attn.k_sconv.weight_proj` / `attn.v_sconv.weight_proj` |
| `attn_post_conv` / `ffn_post_conv` `[2304,4]` | `attn_sconv.weight` / `mlp_sconv.weight` `[2304,1,4]` |
| `attn_norm` (sliding) / `attn_q,k,v_inp_norm` (full) | `attn_norm` / `q_input_norm`, `k_input_norm`, `v_input_norm` |
| `ffn_norm` | `mlp_norm` |
| `ffn_gate_inp`, `exp_probs_b` | `mlp.gate.weight`, `mlp.gate.bias` |
| `ffn_gate_up_exps`, `ffn_down_exps` (+ `.scale`) | `mlp.experts.w13_weight*`, `mlp.experts.w2_weight*` |
| `loop_{attn,k,v,mlp}_{down,gain,forget}` (core layers 6–17) | `controllers.{attn,k,v,mlp}.{down,gain,forget}_weight` |
| `attn_res_query` (`[2304]`, or `[3,2304]` on core layers), `ffn_res_query` | `model.llm.attn_res.attn_queries.i`, `model.llm.attn_res.mlp_queries.i`, indexed by layer visit |

Globals:

| GGUF | official |
| --- | --- |
| `token_embd`, `token_embd_norm` | `model.llm.embed`, `model.llm.embed_norm` |
| `output`, `output_norm` | `model.llm.unembed`, `model.llm.norm` |
| `output_res_query` | `model.llm.attn_res.output_query` |
| `loop_res_query` / `loop_kv_res_query` `[24,2304]` | `model.llm.middle_cycle_cross_pass_queries` / `..._kv_queries` |

The per-visit index of the residual queries, and every transpose, are checked in this PR
by comparing the adapter's output with the official checkpoint tensors
(`tests/models/inkling2/test_gguf_adapter.py`). The standalone port also has an FP32
reference forward of the GGUF model; that tool is not part of this PR.

## Design of the GGUF route

1. **Config.**
   - Map the GGUF `inkling2.*` metadata to #1598's `Inkling2` text config. `block_count`
     counts visits (48); 24 physical layers.
   - Add the o200k pre-tokenizer for `tokenizer.ggml.pre = inkling2`.
   - Both pieces exist in the standalone port.
2. **Quantisation config.**
   - #1598's `model.py` required `Inkling2QuantConfig`. It now also accepts the GGUF quant
     config that 1Cat's GGUF loader sets.
   - #1598's dense layers are built without a quant method and its attention reads
     `qkvr.weight` directly, so in this PR the dense and embedding layers stay
     unquantized: Q8_0 tensors are dequantized at load. Experts use #1598's SM70 expert
     path.
   - Keeping Q8_0 quantized (less memory and faster dense GEMV) needs changes to #1598's
     attention forward. It is a follow-up.
3. **Weights.** An adapter yields official parameter names (table above), converting:
   - **Dense, BF16 / F32:** as is, with transposes where noted.
   - **Dense, Q8_0:** dequantized to FP16 at load, in row chunks (no full FP32 copy).
   - **Experts, NVFP4:** repacked into the layout #1598's `Fp4MoECodec` path expects.
     - Same element format (E2M1, 16-element blocks, E4M3 block scales).
     - The per-expert global scale must be translated between GGUF `scale` and
       `amax` / `outer_scale_denominator`.
     - Verified by dequantising both and comparing.
   - **Experts, layers 20–23:** BF16 (Q8) or Q8_0 (Q4), fed to the FP16 expert GEMV path
     that #1598 already uses after MXFP8 dequantisation (a dense `"fp16"` expert
     format). Keeping Q8_0 expert storage is a follow-up.
4. **TP.**
   - TP=1 and TP=2 both work. Dense weights arrive unquantized, so #1598's own weight
     loaders shard them as they do for the official checkpoint (measured at TP=2 below).

## Validation plan

- **Next-token log-probs** against the FP32 GGUF reference: within 0.02 nats, as the
  standalone port achieved.
- **Greedy continuations** against llama.cpp b12588.
- **Sanity** 12/12, and needle retrieval at 32k / 128k.
- **Coding:** HumanEval+ and LiveCodeBench subset, Q4 and Q8, against the table above.
- **Speed:** the same `stream_bench` and concurrency runs, at TP=1 (and TP=2 if
  supported).
- **#1598's own path unchanged:** its tests, plus one official-checkpoint run with
  identical speed.
- **Repo checks:** `pre-commit`, layering, env registration, config inventory,
  architecture reference, route snapshots.

## Implementation status (this branch)

The GGUF route loads both Unsloth files into #1598's
`Inkling2LoopForConditionalGeneration` at TP=1. Dense layers are dequantized
to FP16; keeping Q8_0 is a decode-speed follow-up.

| piece | where | check |
| --- | --- | --- |
| GGML NVFP4 → official layout | `vllm/models/inkling2/gguf_weights.py` | dequantizes bit-identically (random data and real expert banks) |
| config from GGUF metadata | `vllm/models/inkling2/gguf_config.py` | equals the official `config.json` field by field |
| tensor adapter | `model_loader/gguf_adapters/inkling2.py` | covers every official text tensor. The UD-Q8_K_XL GGUF reproduces the official tensors exactly on sampled layers: dense weights, norms, queries, NVFP4 experts byte for byte, BF16 experts equal to the dequantized MXFP8 banks. UD-Q4_K_XL dense weights are within Q8_0 error. |
| registration, o200k tokenizer | `gguf_config.py`, `gguf_adapters/__init__.py`, `tokenizers/gguf.py` | the native GGUF tokenizer equals the official `tokenizer.json` on mixed-script samples |
| model | `nvidia/model.py`, `nvidia/moe.py`, `nvidia/ops/moe_sm70.py`, `config/model.py` | the GGUF quant config is accepted; dense (`"fp16"`) experts on SM70; float casts for GGUF only; a GGUF tokenizer is allowed with `--language-model-only` |

Layout details found while matching the official tensors:
- **Gate and up rows:** GGUF stores `ffn_gate_up_exps` as all gate rows then all up rows;
  the official `w13` interleaves them (gate₀, up₀, gate₁, …).
- **Short-conv taps:** full layers' GGUF K/V conv taps run newest-first; the official
  `static_w` is `[taps, C]` oldest-first.
- **Residual queries:** they map one-to-one: `blk.N.attn_res_query` →
  `attn_queries.{N-1}` (`[3, 2304]` on full layers), and `blk.N.ffn_res_query` →
  `mlp_queries.{N}`.

End-to-end parity through the model (`tools/inkling2/gguf_parity.py`):
- **Setup:** 6 prompts (English, code, translation, arithmetic, Japanese, a list), greedy,
  24 tokens, top-5 log-probs, one V100, eager.

| | greedy tokens equal to official | max top-5 log-prob difference (aligned steps) |
| --- | --- | --- |
| UD-Q8_K_XL | 6 / 6 prompts, 24 / 24 tokens each | 0.0000 |
| UD-Q4_K_XL | 5 / 6 prompts (one diverges after 16 tokens) | 0.16–1.19 on tail candidates; top token within ~0.05 |

Serving a GGUF:

```bash
vllm serve Inkling-2-Mini-UD-Q4_K_XL.gguf --tokenizer Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --served-model-name inkling-2-mini --dtype float16 --kv-cache-dtype float16 \
  --language-model-only --max-model-len 32768 --max-num-seqs 4 \
  --max-num-batched-tokens 2048 --enable-prefix-caching
# with #1599: --reasoning-parser inkling2 --enable-auto-tool-choice --tool-call-parser inkling2
```

Serving through the API (#1599 parsers merged locally, one V100 unless noted):

| | official (#1598) | GGUF Q8 | GGUF Q4 | GGUF Q4, TP=2 |
| --- | --- | --- | --- | --- |
| model memory per GPU | 22.6 GiB | 22.6 GiB | 22.6 GiB | 11.4 GiB |
| KV capacity (tokens) | 25,239 | 25,239 | 25,239 | 562,555 |
| decode tok/s (short / 4k / 16k) | 45.3 / 42.5 / 39.3 | 44.2 / 41.4 / 38.4 | 45.0 / 42.2 / 39.1 | 47.2 / 44.8 / 42.9 |
| prefill tok/s (16k) | 773 | 753–760 | 765–773 | 1289–1314 |
| 4 concurrent, total tok/s | 112 | 109 | 112 | — |
| sanity (12 cases, exact text / JSON / tool calls) | 12/12 | 12/12 | 12/12 | — |
| next turn of a 41k-token conversation | — | — | — | 40.0 s → 0.43 s |

What this shows:
- **Speed matches #1598**, as expected for a loading change.
- **Prefix caching now works for GGUF files.**
- **On one card, the GGUFs take the official checkpoint's memory**, because dense Q8_0 is
  dequantized. Keeping it quantized is a follow-up.

Static checks on the changed files:
- **Pass:** ruff, typos, layering, architecture reference, config inventory, env
  registration, SPDX headers, forbidden imports, `torch.cuda` calls.
- **mypy:** no new errors over #1598.
- **SM70 route snapshots:** `moe` and `nvfp4` show no changes against #1598.

## Decode profile: where the time goes

Per-kernel GPU time per decode step, from `torch.profiler`:
- Eager mode, one sequence, 16 steps, one V100.
- #1598 ran the official checkpoint at TP=1 (`max_num_seqs=4`, because `1` trips a warm-up
  assertion).
- The standalone port ran the GGUF Q8, which holds the same weights.

| | #1598 | standalone GGUF Q8 | standalone GGUF Q4 |
| --- | --- | --- | --- |
| total GPU time per step | 23.5 ms | 14.4 ms | 12.0 ms |
| kernels per step | 1,597 | 1,085 | 1,177 |

By component (ms per step; groupings from kernel names):

| component | #1598 kernels | #1598 | standalone kernels | standalone Q8 | gap |
| --- | --- | --- | --- | --- | --- |
| routed experts and routing | TurboMind NVFP4 GEMM, FP16 MoE GEMV (layers 20–23), radix sort, select, permute / unpermute | ~8.2 | `_w13_situ`, `_w2` (dp4a NVFP4 GEMV, fused SiTU), `_router`, `_router_logits` | ~4.7 | ~3.5 |
| loop controllers | `_k_group_kernel` (K-grouped linear), `_blend_rms_norm` | ~3.2 | `_ctrl_down`, `_ctrl_apply` | ~0.8 | ~2.4 |
| attention and short convs | `_rel_attention`, `_attn_prologue`, `_rel_decode_combine`, `_sconv` | ~3.6 | `_inkling2_attn`, `_inkling2_attn_g3`, `_prep`, `_sconv` | ~1.4 | ~2.2 |
| residual mixing and norms | `_attn_res_row`, `_pair_rms_norm` | ~1.2 | `_mix`, `_merge`, `_rows_norm` | ~1.4 | ~0 |
| dense projections and LM head | cuBLAS GEMV, CUTLASS | ~5.6 | cuBLAS GEMV, CUTLASS | ~5.3 | ~0 |

What this means:
- **Three components account for nearly all of the 9 ms gap**: routed experts, loop
  controllers, and attention with short convs.
- **Dense projections cost the same**; the Q4 GGUF saves a further ~0.9 ms there with Q8_0
  GEMV.
- **Kernel count matters less.** CUDA graphs remove most launch overhead in serving, so
  these GPU-time gaps are the ones to close.

## Follow-ups (separate stacked PRs)

1. **Single-card decode speed**, in the order of the profile:
   - routed-expert decode: dp4a NVFP4 GEMV with fused SiTU, and a routing path without
     sort / permute for small batches;
   - loop controllers: fused down and apply;
   - decode attention and short convs: fused prep with the K/V convs, and grouped-query
     split-KV attention.

   Each kernel only replaces #1598's decode path where measured faster at TP=1 and TP=2,
   behind #1598's existing dispatch.
2. **KV memory on one card.**
   - 1Cat's `SlidingWindowSpec.max_memory_usage_bytes` is already window-bounded.
   - By that sizing, #1598 at 64k context should need ~2.6 GiB, but startup asked for
     5.79 GiB.
   - The cause is not yet identified. Next: log the per-group admission sizes from
     `get_kv_cache_configs`, and compare with prefix caching on and off.
