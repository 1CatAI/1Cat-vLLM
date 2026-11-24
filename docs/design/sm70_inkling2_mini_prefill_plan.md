# Inkling 2 Mini on SM70: prefill plan

Status: written as a plan on 2026-11-22; implemented in this branch (the
timings below are the "before"). Companion to `sm70_inkling2_mini_perf_plan.md`
(decode) and `sm70_inkling2_mini_port.md`.

## 1. Where prefill time goes (merged decode build, TP4, rank 0)

Torch-profiler traces of 1300 and 16000 token prompts.

| | 1,385 tokens | 16,984 tokens (9 chunks of 2048) |
|---|---|---|
| Total (GPU span) | 697 ms | 7.76 s (2,190 tok/s) |
| NVFP4 MoE, TurboMind per-expert GEMMs | 306 ms, 11,352 launches | 3.16 s, 101,376 launches (128 experts x 2 stages per visit, 31 us each) |
| MoE permute/finalize/sort | 12 ms | 0.17 s |
| `_rel_attention` (Triton, FP32 FMA) | 62 ms | 2.46 s |
| `_qr_rel` + prologue + gate/sink | 11 ms | 0.12 s |
| All-reduce | 119 ms (custom 2-stage, 1.2 ms per call) | 0.27 s (NCCL ring, 281 us per 9.4 MB call) |
| cuBLAS fp16 GEMMs | 41 ms | 0.43 s |
| attn_res, loop norms, sconv, copies, fp32 router SGEMM | ~35 ms | ~0.43 s |
| GPU idle (gaps > 20 us) | 92 ms | 0.59 s (MXFP8 per-expert Python loop with `.tolist()` sync ~0.25 s; NVFP4 C++ host loop ~0.1 s; chunk boundaries) |

Attention by layer type, per 2048-token chunk number c (from the trace): 40 sliding-window
visits cost a constant 115 ms; 8 global visits cost 23 + 44*c ms. The growing part is keys at
distance >= 1024, where the relative bias is exactly zero (`rel_attention.py:181`,
`decode_attention.py`, upstream `rel_extent` = 1024). `_rel_attention` uses 230 registers per
thread (1 CTA per SM).

FLOP view: Inkling needs about 10 GFLOP per token at 25K context (MoE ~1.5, projections
~0.45, attention ~1.1-1.75 per GPU at TP4), 5x less than Qwen 3.8 27B, which prefills at about
2,000 tok/s on this box at ~27 TFLOP/s per GPU. Inkling runs at ~5 TFLOP/s per GPU. The gap is
kernel efficiency, not work: at Qwen-like efficiency the ceiling is ~10K tok/s.

Real traffic (gateway log, `hs_perf.py`, `hs_cache_loss.py`): Inkling's prefix cache works.
Only 1.2% of the tokens it prefilled were prefix it had seen before; 11% of prompt tokens
were prefilled at all. The gateway already pins each conversation to the replica holding its
cache. Prefill speed under load (~570 tok/s p50) is therefore kernel-bound, not cache-bound.

## 2. In progress (sessions ink-pf-moe, ink-pf-attn, ink-pf-chunk; rules in `work/PF_BRIEF.md`)

| ID | Change | Sharpened by research | Expected (17K prompt) |
|---|---|---|---|
| P1 | Grouped NVFP4 MoE prefill | First try N2: the image's `_C.abi3.so` exports TurboMind's grouped GEMM `nvfp4_moe_gemm_sm70_out_impl(...)`; only the Python/C++ caller's shape whitelist (`awq_sm70_gemm.cu:9867-9889`) keeps Inkling on the per-expert loop. A JIT C++ shim can call it with Inkling's shapes (w13 k2304 n576, w2 k288 n2304). Fork docs: bitwise equal to the loop, 20x (W13) / 9.5x (W2) faster at M=4096. Fallback: own JIT WMMA grouped GEMM. | MoE 3.3 s -> ~0.3-0.6 s |
| P2 | Tensor-core prefill attention | N1: split global layers into a bias-free far field (distance >= 1024) on the fork's existing `flash_attn_v100_cuda.fwd` (D128, WMMA, LSE, bottom-right causal, GQA) and a near band with bias, merged by LSE before sink and gate. N3: base the biased kernel on the FA2-SM70 hdim128 source (copied to `work/third_party/vllm-flash-attn-src`; `SM70_8x8x4` MMA, paged KV, window, split-KV, LSE, ALiBi hook in `mask.h:132`). N5: split-KV for global prefill and a decode-row split for mixed batches (upstream `plan_attention`). | Attention 2.6 s -> ~0.3-0.5 s |
| P3 | MXFP8 grouped experts, prefill glue, chunk size | Remove the `moe.py:195` sync and Python expert loop (no `torch._grouped_mm`: it falls back to a per-group loop on SM70). N4: prefill all-reduces above ~1 MB via pynccl (custom 2-stage is slower per byte; 119 ms of 697 ms at 1.3K). N9: pad the q-side GEMM N to %8 (align2 -> align8), fp16 HMMA router GEMM if exact enough, conv+mix fusion at prefill sizes. Chunk budget 4096-8192 once MoE is grouped (KV pool shrinks ~1 GB/rank at 8192), multiples of 64/128, one long prefill at a time. | idle + glue ~1 s -> ~0.4 s; AR at short prompts -0.06 s |

Projected with P1-P3: 17K prompt 7.8 s -> ~2-2.5 s (~7-8K tok/s); 1.3K prompt 0.7 s ->
~0.2-0.25 s. For reference, 1Cat reports 6-7K tok/s TP4 prefill for its Qwen3.8 NVFP4 MoE
after the same grouped-MoE work (upstream PR 390). These are estimates.

## 3. New options, not started

| ID | Option | Gain | Effort / risk | Recommendation |
|---|---|---|---|---|
| T1 | Triton 3.6 fork with tensor-core `tl.dot` on sm_70 (Peuqui/triton `v3.6.0-sm7x`, used by 1Cat's open PR 1282). Our image matches (Triton 3.6, torch 2.10). Stock Triton has been FMA-only on SM70 since late 2024 (triton PR 5066); confirmed: our `tl.dot` PTX has 0 `mma.sync`. Fork GEMM on V100: 6.8 -> 47.7 TFLOPS at 128x4096x4096. | Existing Triton kernels (`rel_attention`, MXFP8 `moe_gemv`, norms) on tensor cores with no code change: possibly 5-10x on `rel_attention` | Low if built from reviewed source; 1-ulp drift on some paths | **User decision**: it is a third-party build, not installed. |
| N7 | Sequence-parallel residual stream (reduce-scatter, per-rank glue on 1/4 of tokens, all-gather); upstream disables its version for loop models | ~0.3 s at 17K, ~10% after P1/P2 | 5-8 d / high | After P1-P3 |
| N8 | Comm/compute overlap (two micro-batches per chunk) | <= 27 ms per 2048 chunk | 4-6 d / high | Later |
| N6 | Sliding-window retention checkpoints without Mamba (`kv_cache_coordinator.py:102-110`), so a prompt edited mid-way does not re-prefill from 0 | Inkling loses only 1.2% of prefill to cache misses today | 1-2 d / medium | Not now; re-measure with more traffic |

Rejected: KV offload (LMCache, Mooncake: hit rate is not capacity-bound; hybrid SWA
unsupported), prefill-decode disaggregation (halves prefill capacity), context parallel / DCP
(no per-GPU FLOP saving at TP4 with 8 KV heads), loop-pass KV sharing (changes the model),
xFormers >= 0.0.31 / FlashInfer (no SM70), MegaBlocks (sm80), EP for the MoE (fails closed in
the fork's route; dispatch cost).

Side finding: on the historical Qwen 3.8 traffic, 66.5% of prefilled tokens re-read a prefix
the conversation's previous turn already had (cache misses), versus 1.2% for Inkling.
