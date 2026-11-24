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

| | GGUF Q4, 1 GPU | GGUF Q8, 1 GPU | #1598 TP=2 | #1598 TP=1 |
| --- | --- | --- | --- | --- |
| decode tok/s (short / 4k / 16k prompt) | 87.1 / 78.3 / 68.5 | 71.5 / 65.4 / 58.4 | 47.2 / 44.8 / 42.9 | 45.3 / 42.5 / 39.3 |
| prefill tok/s (4k / 16k) | 1201 / 1253 | 1308 / 1267 | 1331 / 1316 | 793 / 773 |
| 4 concurrent, total tok/s | 168 | 160 | 115 | 112 |
| largest context on one card | 262k | ~164k | n/a | ~28k |
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
   giving 25,239 tokens. The GGUF Q8 has ~6.1 GiB giving ~175k.
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

The per-visit index of the residual queries, and every transpose, are checked against the
FP32 GGUF reference forward (`tools/inkling2/gguf_reference.py` in the standalone port).

## Design of the GGUF route

1. **Config.**
   - Map the GGUF `inkling2.*` metadata to #1598's `Inkling2` text config. `block_count`
     counts visits (48); 24 physical layers.
   - Add the o200k pre-tokenizer for `tokenizer.ggml.pre = inkling2`.
   - Both pieces exist in the standalone port.
2. **Quantisation config.**
   - #1598's `model.py` requires `Inkling2QuantConfig`.
   - The GGUF route needs dense and embedding layers through 1Cat's GGUF linear methods
     (Q8_0 or BF16) and experts through #1598's SM70 expert path.
   - Plan: a GGUF variant of the Inkling2 quant config that returns the GGUF linear
     method for dense layers and the existing expert method for experts. This avoids new
     dispatch.
3. **Weights.** An adapter yields official parameter names (table above), converting:
   - **Dense, BF16 / F32:** as is, with transposes where noted.
   - **Dense, Q8_0:** raw GGML rows, kept quantised (GGUF linear).
   - **Experts, NVFP4:** repacked into the layout #1598's `Fp4MoECodec` path expects.
     - Same element format (E2M1, 16-element blocks, E4M3 block scales).
     - The per-expert global scale must be translated between GGUF `scale` and
       `amax` / `outer_scale_denominator`.
     - Verified by dequantising both and comparing.
   - **Experts, layers 20–23:** BF16 (Q8) or Q8_0 (Q4), fed to the FP16 expert GEMV path
     that #1598 already uses after MXFP8 dequantisation. Q8_0 can keep its 8-bit storage
     later (memory), dequantising on load at first.
4. **TP.**
   - TP=1 is the target.
   - TP=2 needs GGUF row sharding of the Q8_0 dense layers, which 1Cat's GGUF linears
     support for other models. It will be checked, but it is not required for the first
     PR.

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

## Follow-ups (separate stacked PRs)

1. **Single-card decode speed.** Profile #1598 per op at TP=1 against the standalone port.
   Bring over only the kernels that win (candidates: dp4a NVFP4 expert GEMV, fused mix /
   loop controller / prep / router kernels, split-KV decode attention for full and
   sliding-window visits).
2. **KV memory on one card.** Confirm the cause of the per-token overhead. If it is
   allocator-level, discuss with maintainers before changing shared code.
