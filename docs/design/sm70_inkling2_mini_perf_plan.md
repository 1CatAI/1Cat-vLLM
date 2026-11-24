# Inkling 2 Mini on SM70: decode and prefill performance plan

Status: written as a plan on 2026-11-22; the work is implemented in this
branch. Kept as the design record, so the timings below are the "before".
Companion to `sm70_inkling2_mini_port.md`.

## 1. Why Inkling is slow today

Inkling 2 Mini is a mixture-of-experts model with 3.4B active parameters.
Per decode token it reads about 1.46 GB of weights per GPU at TP4:

| Component | Bytes per GPU |
|---|---|
| Dense attention and conv weights over 48 layer visits | ~650 MB |
| NVFP4 experts (44 visits × 8 experts) | ~394 MB |
| MXFP8 experts (4 visits, held as fp16) | ~127 MB |
| lm_head shard | 232 MB |
| fp32 router (replicated) | ~57 MB |

At a practical 750 GB/s that is a floor of about 2 ms per token, or ~500 tok/s
for one stream. Qwen 3.8 27B (dense, 4-bit) reads about 2.5× more per token.
So the user's expectation is right on bandwidth grounds.

The port is about 8× above that floor. The cause is not CPU launch overhead,
and it is not all-reduce latency: decode already runs as a full CUDA graph,
and the GPU is busy 95% of each step. The cause is 1,839 small kernels per
token (about 38 per layer visit), each running far below memory bandwidth.

## 2. Measurements

Setup: TP4 on GPUs 0-3 (NVLink), decode CUDA graphs (`FULL_DECODE_ONLY`, sizes
1/2/4/8), fp16 KV, TurboMind NVFP4 MoE. Torch-profiler traces from the live
server, rank 0.

### 2.1 Decode step time

| Case | Step period | Graph span | GPU busy in graph | Kernels per step |
|---|---|---|---|---|
| batch 1, 1.3K context | 16.6 ms (60 tok/s) | 16.0 ms | 15.2 ms | 1,839 |
| batch 1, 25.5K context | 18.5 ms (54 tok/s) | 17.3 ms | 16.5 ms | 1,839 |
| batch 8, 1.3K context | 37.4 ms (214 tok/s total) | 32.3 ms | 31.3 ms | 2,174 |

Outside the graph, each step spends about 0.5 ms on lm_head and sampling.
On top of that come block-table host-to-device copies (section 3, D8).

### 2.2 Decode breakdown per step (ms)

| Class | batch 1 | batch 8 | Main kernels |
|---|---|---|---|
| Attention | 4.34 | 6.18 | `_rel_attention_kernel` 52 µs × 48 (b1), 83 µs (b8); `_attn_res_row` × 121; prologue |
| Dense GEMM/GEMV (cuBLAS) | 3.49 | 7.83 | `gemv2T` qkvr 24 µs (~210 GB/s); b8: a 3-CTA `volta_fp16_s884gemm_64x64` takes 109 µs × 24 |
| MoE experts | 2.22 | 10.71 | TurboMind w13 29 µs + w2 16 µs (b1); w13 86 µs + w2 142 µs (b8) |
| Torch glue (elementwise) | 1.92 | 2.23 | ~11 copy, cast, fill, add and gather kernels per visit |
| Short conv | 0.97 | 0.99 | `_sconv_kernel` × 96 |
| Router top-k / sigmoid | 0.78 | 1.27 | `gatherTopK` 11 µs × 48 |
| Loop norms | 0.64 | 0.76 | `pair_rms_norm`, `blend_rms_norm` |
| All-reduce | 0.51 | 0.76 | fork SM70 push all-reduce, 5.3 µs per call |

### 2.3 One layer visit at batch 1 (sliding layer, outside the loop)

30 kernels, about 272 µs:

| Kernels | Time |
|---|---|
| qkvr GEMV | 23.8 µs |
| 2 memsets (`torch.zeros`) | 4 µs |
| Attention prologue | 10.8 µs |
| `rel_attention` split-K | 57.5 µs |
| Combine | 3.9 µs |
| wo GEMV | 10.7 µs |
| All-reduce + short conv + attn_res | 25 µs |
| Router chain: copy, fp32 GEMV, split-K reduce, sigmoid, add, top-k, gather, sum, div, mul, cast, slot prep (12 kernels) | 62 µs |
| MoE: w13 + activation + w2 + reduce (4 kernels) | 53 µs |
| All-reduce + short conv + attn_res | 21 µs |

Loop passes 2-3 add about 50 µs per visit (24 visits): message attn_res,
`pair_rms_norm`, two small GEMMs and `blend_rms_norm`.

The decode attention kernel stands out. It pads one query row to
`BLOCK_M = 64` and runs `tl.dot` as FP32 FMA. It reads about 1.3 MB of KV in
57 µs, roughly 30× slower than bandwidth allows.

### 2.4 Prefill

| Prompt | Time | Rate | Breakdown |
|---|---|---|---|
| 1,283 tokens | 725 ms | 1,770 tok/s | MoE 274 ms (per-expert host loop: ~118 GEMM launches per visit), attention 76 ms, GPU idle (CPU-bound) 277 ms |
| 25,511 tokens | 12.6 s | 2,030 tok/s | attention 5.1 s (40%), MoE 4.8 s (38%), GEMM 0.63 s, all-reduce 0.44 s, idle 0.75 s |

## 3. Options

Savings are estimates from the traces above and from fork-documented kernel
costs. Every change must pass the existing accuracy gate before it ships:
`e2e_compare.py` against the fp32 reference, plus prefix-cache and chunked
prefill equality.

### Decode

| ID | Change | Saving b1 | Saving b8 | Effort | Risk |
|---|---|---|---|---|---|
| D1 | Fused router: fp32 row GEMV plus one select kernel (sigmoid + bias + top-8 + renormalise + route_scale + slot prep). Use the fork `grouped_topk` op with sigmoid scoring, or port upstream `_select_kernel`. 12 kernels per visit become 2. | ~2.4 ms | ~3 ms | 1-2 d | low |
| D2 | Decode attention kernel: GQA-packed (3 query heads per KV head), split-K, relative bias, window, sinks and gate fused; prologue folded in; no memsets. 76 µs per visit becomes ~15 µs. | ~2.7 ms | ~4.5 ms | 3-5 d | medium |
| D3 | Dense projections on the fork fp16 row GEMV / `sm70_f16_gemm` (qkvr 24 → ~8 µs, wo 11 → ~6 µs). Replace the cuBLAS loop-controller GEMM that runs 3 CTAs for 109 µs at batch 8. | ~1 ms | ~3.5 ms | 1-2 d | low |
| D4 | MoE experts: route batch 1 through the TurboMind codec QPN M1 path for Inkling's shape (2304 / 288 / 128 / 8), with W2 direct-reduce. Tune the M=2..16 grouped path (w2 is slower than w13 at M=8). | ~1 ms | ~5 ms | 3-5 d | medium |
| D5 | Fused all-reduce + short conv + AttnRes mix (+ next RMSNorm): an SM70 rewrite of upstream `allreduce_conv_mix.cu` on the fork's push buffers and NaN sentinel. 3 kernels per site become 1, at 96 sites. A Triton conv+mix kernel is a 1-day intermediate worth ~0.35 ms. | ~1.1 ms | ~1.2 ms | 4-6 d | medium-high |
| D6 | Loop blend fusion: message mix, `pair_rms_norm`, two GEMVs and `blend_rms_norm`, 9 kernels down to ~4 on 24 visits. | ~0.5 ms | ~0.5 ms | 2 d | low |
| D7 | Glue cleanup: `zeros` to `empty`, drop redundant casts and copies. | ~0.4 ms | ~0.5 ms | 1 d | low |
| D8 | Block-table copies. The conv-state spec uses block size 4, so at 262K max-model-len each block table is 65,536 entries wide. Full copies cost 0.5 ms every 4th step at b1 and ~4.3 ms per step at b8. Fix: larger conv-state blocks or incremental table updates. | ~0.1 ms | ~4.7 ms | 1 d | low |
| D9 | DSpark speculative decoding with the bundled drafter (7 drafts). Needs the target on model runner V2 first; V1 is the fallback. Verify width is limited by the TurboMind MoE (≤ 10 tokens on the compact path), so the width policy must depend on batch size. | 1.2-1.7× on today's kernels; ~2-2.5× after D1-D8 | small unless the MoE 16-64-token path is fixed | 8-10 d + 2-5 d tuning | medium-high |
| D10 | Layout experiment: TP2 × DP2 or DP4 (TP1) instead of TP4. Trades single-stream speed for total throughput. | measure | measure | hours | low |

Not available: running fewer loop passes or sharing the loop KV cache. Both
are rejected for fixed-loop checkpoints like Inkling 2 Mini, and they cost
quality.

### Prefill

| ID | Change | Effect | Effort | Risk |
|---|---|---|---|---|
| P1 | Grouped NVFP4 MoE prefill: one launch each for w13 and w2 across all experts, replacing the per-expert host loop. Needs Inkling's shape whitelisted in the TurboMind grouped path (C++ rebuild of the fork image). | MoE prefill ~5-8× faster; removes most CPU idle. 1.3K: 725 → ~350 ms. 25K: about −4 s. | 3-5 d + image build | medium |
| P2 | Tensor-core prefill attention: SM70 `mma.sync` flash kernel with relative bias, sliding window, sinks, gate, log scaling and LSE output, derived from the fork's Flash-V100 kernels. | 25K: attention 5.1 s → ~0.5 s | 5-8 d | medium-high |
| P3 | Larger prefill chunks (`max-num-batched-tokens` 2048 → 8192) and a grouped MXFP8 GEMM. | ~10-20% after P1 | hours | low |

P1 + P2 + P3 together: 25K-token prompt 12.6 s → ~3 s (~8K tok/s);
1.3K-token prompt 725 ms → ~150-250 ms.

## 4. Projected results (estimates)

| Stage | Batch 1 | Batch 8 total |
|---|---|---|
| Today | 16.6 ms (60 tok/s) | 37.4 ms (214 tok/s) |
| After D1, D2, D3, D7, D8 (~2 weeks) | ~10 ms (~100 tok/s) | ~21 ms (~380 tok/s) |
| Plus D4, D5, D6 (~2 more weeks) | ~7.5 ms (~135 tok/s) | ~14.5 ms (~550 tok/s) |
| Plus DSpark (τ ≈ 3-3.5) | ~220-300 tok/s | depends on the MoE verify path |

For comparison, Qwen 3.8 27B Quality with DFlash2 runs at about 216 tok/s for
a single stream on the same GPUs.

## 5. Recommended order

1. **Phase A (D1, D2, D3, D7, D8).** These are cheap, low-risk and independent.
   They also speed up the 8-token verify step DSpark needs, so they come
   before it.
2. **P1 in parallel.** It is the largest prefill win, and agent workloads are
   prefill-heavy.
3. **Phase B (D4, D5, D6) and P2.**
4. **D9 (DSpark) last**, on top of the fast kernels.
