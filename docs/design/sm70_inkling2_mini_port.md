# Inkling 2 Mini on SM70 (V100)

Port of `thinkingmachines/Inkling-2-Mini` to SM70 (V100): TP4, long context,
prefix caching, tool and reasoning parsing, vision, DSpark speculative decoding.
Current speeds are in the pull request description and the perf plans next to
this file.

## Model (text)

`Inkling2LoopForConditionalGeneration`, 26.4B total / 3.4B active.

- 24 layers, hidden 2304, vocab 201024 (200058 used). Layers 0-5 prelude,
  6-17 loop core run 3 times, 18-23 coda: 48 layer visits per token.
- Attention, all layers: 24 q heads, 8 KV heads, head_dim 128, per-token
  attention sinks (`sigmoid(lse - sink)` on the output), sigmoid output gate,
  L2-clipped q (scaled, threshold 22.6) and k, RMS-normed v (no gamma),
  relative-position bias: `r = clip(scale * W_r x)` (16 dims/head) times a
  `[16, 1024]` profile bank gives a bias for backward distances 0..1023.
  - 20 sliding layers (window 1024): static depthwise K/V short convs.
  - 4 global layers (5, 11, 17, 23): dynamic K/V short convs (static taps +
    per-token taps from a projection of the K/V input), log-scaled q and r
    (`tau = 1 + 0.1 * log(max((pos + 1) / 8192, 1))`), separate q/k/v
    input norms.
- Short convs (kernel 4) on K, V, the attention output and the MLP output,
  each `(conv(x) + x) / sqrt(5)` (`/ sqrt(9)` for dynamic convs).
- Block attention residuals (blocks of 6 layers): every sublayer input is a
  softmax mix, over the completed blocks plus the running partial block, of
  RMS-normalized states, with a learned query per sublayer (3 for the q/k/v
  inputs of global layers).
- Loop passes 2 and 3 restart from the pre-loop states. Each loop layer gets
  4 messages (q, k, v, mlp) from an attention over the previous pass's blocks
  (`middle_cycle_cross_pass_*queries`), mixed in by low-rank tanh gates
  (`controllers.{attn,k,v,mlp}`). Each pass has its own KV and conv caches.
- MoE: 128 experts, top-8, sigmoid router with correction bias, weights
  normalized and scaled by 8, SwiGLU with softcaps (gate 4, up 25):
  `4 tanh(g/4) * sigmoid(g) * 25 tanh(u/25)`.
- Final: block mix with the output query, RMSNorm, fp32 logits, softcap 30.

Checkpoint: experts of layers 0-19 NVFP4 (uint8 codes, per-16 E4M3 scales,
per-expert `amax`, per-tensor denominator 2688 = 6*448), layers 20-23 MXFP8
(E4M3 + per-32 E8M0). Everything else bf16 / fp32. 19.5 GB plus a 636 MB DSpark
drafter (`mtp.safetensors`).

## SM70 constraints and decisions

| Piece | Upstream (Hopper/Blackwell) | SM70 port |
| --- | --- | --- |
| Activations | bf16 | fp16; mixes, routers, convs, softmax stats fp32 |
| Dense weights | bf16 | fp16 (largest |w| 76.5; 0.18% of nonzeros below fp16 normal range) |
| NVFP4 experts | TRT-LLM / Humming | fork `ModelOptNvFp4SM70MoEMethod` (TurboMind W4A16), converted at load: `weight_scale_2 = amax / denominator`, w13 rows de-interleaved (upstream stores gate/up interleaved) |
| MXFP8 experts | MXFP8 GEMM | dequantized to fp16 at load (exact for this checkpoint), grouped per-expert GEMMs; group-32 FP8 TurboMind kernel later |
| MoE activation | fused | new softcapped-SwiGLU stage in the SM70 FP4 codec |
| Attention | FA4 with sheared bias / Triton SM90 | phase 1: Triton (FMA on SM70, correctness first); phase 2: Flash-V100 CUDA decode + one paged-prefill variant with relative bias and LSE output |
| Sinks + gate | `gate_sink` after attention, needs LSE | same; LSE from the attention kernels |
| Conv states | sliding-window KV spec (window = 4) | fork `SlidingWindowSpec(block_size=4, head_size=pow2, head_size_v=0)` |
| KV cache API | `get_kv_cache_config` | fork `get_kv_cache_spec` (`FullAttentionSpec` / `SlidingWindowSpec`) |
| TP all-reduce | FlashInfer fused | plain NCCL all-reduce (non-fused upstream path) |
| DSpark | MRV2 `spec_decode/dspark` | port upstream's MRV2 package (the fork only has V1 DSpark for DeepSeek) |

## Findings that shape the implementation

- Triton 3.x on SM70 lowers `tl.dot` (fp16 and bf16) to scalar FP32 FMA, with no
  `mma.sync` (fork docs `sm70_flash_next_mtp4_batch_gdn.md`, confirmed by
  compiling the upstream kernels for sm_70). Every dense GEMM therefore goes
  through cuBLAS; Triton is kept for memory-bound work (norms, convs, AttnRes
  mixes, routing, decode attention).
- Upstream kernels compile for sm_70 after: removing `tl.extra.cuda.gdc_wait` /
  `gdc_launch_dependents` and `launch_pdl` (SM90 `griddepcontrol`), replacing
  `tanh.approx.f32` in the softcap (SM75+) with libdevice `tanh`, replacing the
  bf16 inline-asm rounding in vision, and fp16 instead of hard-coded bf16.
  `rel_attention` (upstream's SM90 Triton path) is correct on SM70 but needs
  small tiles (BM 16-32, BN 32) to avoid register spills at D = 128.
- The fork's FLASH_ATTN_V100 kernels have paged KV, GQA and sliding windows but
  no additive bias and no LSE output from paged prefill; Inkling's sinks only
  need the LSE (`out * sigmoid(gate) * sigmoid(lse - sink)`).
- SM70 NVFP4 MoE (`ModelOptNvFp4SM70MoEMethod`) needs: Inkling's contract
  `(2304, 1152, 128, 8)` added, w13 de-interleaved, an activation hook in the
  FP4 codec (only SiLU-mul exists), and later Inkling prefill shapes in the
  grouped-prefill whitelist (otherwise one GEMM launch per expert). Its output
  buffer is reused per layer: consume it before the next loop pass.
- No SM70 MXFP8 MoE kernel exists; the fp16 Triton fused MoE is FMA-only.
  Layers 20-23 use fp16 weights (exact) with cuBLAS per-expert GEMMs first.
- The conv-state cache fits the fork's `SlidingWindowSpec` (page unification
  scales block sizes); conv kernels must index blocks by the bound tensor's
  block dim.
- DSpark in the fork exists only for DeepSeek V4/V4.1 in the V1 runner; the
  Qwen setups run MRV2. Inkling DSpark means porting upstream's MRV2
  `spec_decode/dspark` package.
- fp16 risk: stored block/partial residual sums are raw (not normalized).
  Check their max on real prompts; fall back to an fp32 stack if needed
  (5 x tokens x 2304 x 4 bytes).

## Phase 1 implementation (TP1, eager, correctness)

`vllm/models/inkling2/sm70/`:
- `ops/` - upstream Triton kernels with the SM70 fixes above: `norm`, `sconv`
  (prologue: q/r clip, rel logits, K/V convs, clip/norm, KV + conv-state
  writes), `attn_res`, `loop` (blend), `attention` (unsheared `(T, H, E)` rel
  logits, small-tile `rel_attention` for every batch shape, `gate_sink`),
  `moe_route` (`select_experts`).
- `conv_state.py` - `InklingConvState` on `SlidingWindowSpec`.
- `attention.py` - `Inkling2Attention` + pass caches on the fork's spec API
  and a metadata builder based on the Triton/V100 builder.
- `moe.py` - router + NVFP4 experts through the SM70 TurboMind method with the
  softcapped SwiGLU stage; MXFP8 layers in fp16.
- `model.py`, `loop.py` - forward identical to upstream's non-fused path.
Registered as `Inkling2LoopForConditionalGeneration` (text only at first).

## Verification

Reference: HF transformers `main` (`Inkling2LoopForConditionalGeneration`,
PR #51050) in fp32 on CPU, saved for four
prompts (5, 20, 25 and 1600 tokens; the last crosses the 1024-token window
and bias extent): per-position top-20 logits, full last-position logits and a
12-token greedy continuation.

Port checks, in order: per-op unit tests against the reference math; full
forward logits vs reference (top-1 agreement, KL); greedy continuation
match; prefix-cache and chunked-prefill consistency; TP1 vs TP4; then serving
benchmarks with kewmark against Qwen 3.8 27B.

## Status (2026-11-21)

Text model works at TP1 and TP4 (eager and decode CUDA graphs), fp16 KV.

Accuracy vs the fp32 CPU reference (TP4, TurboMind NVFP4 + fp16 MXFP8 experts):

| prompt | tokens | top-1 agreement | mean logit-gap error | greedy (12) |
| --- | --- | --- | --- | --- |
| short | 5 | 4/4 | 0.014 | 12/12 |
| code | 20 | 19/19 | 0.018 | 12/12 |
| chat | 25 | 24/24 | 0.059 | 12/12 |
| long | 1600 | 1577/1599 | 0.094 | 7/12 |

All long-prompt disagreements are near-ties (reference top-2 gap at most
0.21 nats, the port picks the reference's #2/#3), spread evenly before and
after position 1024: fp16 rounding, not a window/extent bug. Chunked prefill
(512) and prefix-cache hits give identical results.

Gotchas found:
- ModelOpt's NVFP4 selection picks the skinny-QPN oracle route for some
  shapes (TP1 here), which has no activation hook and runs plain SiLU; the
  Inkling quant config constructs `ModelOptNvFp4SM70MoEMethod` directly.
- The fork defaults V100 KV caches to `int8_per_token_head`; Inkling needs
  `--kv-cache-dtype float16`.
- A hybrid-model prefix hit only lands where a previous request ended (the
  sliding-window conv/KV blocks behind a request's end are freed).
