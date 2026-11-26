# Inkling2 on SM70: single-card decode speed (stacked on the GGUF route)

Follow-up to `docs/design/inkling2_gguf_sm70.md`. Per decode step on one V100 (eager GPU
time, one sequence), #1598's model takes 23.5 ms and a standalone single-card port 14.4 ms
on the same weights. The profile attributes the gap to three components:

| component | #1598 | standalone | gap | needs a different weight layout? |
| --- | --- | --- | --- | --- |
| routed experts and routing | ~8.2 ms | ~4.7 ms | ~3.5 ms | yes (see below) |
| loop controllers | ~3.2 ms | ~0.8 ms | ~2.4 ms | no (FP16 weights) |
| attention and short convs | ~3.6 ms | ~1.4 ms | ~2.2 ms | no (FP16 weights) |

## Routed experts: what reuse is possible

- **Where #1598's decode time goes.** At decode, #1598 runs the general SM70 NVFP4 MoE
  stage (`nvfp4_moe_dense_stage_sm70_out`) on TurboMind banks:
  - TurboMind GEMM, ~5.7 ms per step;
  - a sort / permute / select / unpermute pipeline, ~1.85 ms per step.
- **The shared codec's decode modes are model-specific.** `Fp4MoECodec` has
  small-batch modes (`qpn`, `direct_reduce`, `batch_reduce`, `fused_qpn`), but their kernels
  (`csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu`) are specialised to Qwen3.8 TP4 shapes:
  - 512 experts;
  - top-10 routing;
  - fixed output shapes `{512, 160, 320}` / `{512, 10, 2560}`.

  Calling `nvfp4_moe_qpn_m1_sm70_out` with Inkling2's shapes (128 experts, top-8, H 2304,
  I 1152) fails with "expected 10..160 routes in 10-route token groups and 512 experts".
  Generalising those kernels is native work in shared code, outside this PR series.
- **The standalone port's decode kernels need a different layout.** They are Triton dp4a
  GEMVs with SiTU fused, one program per (token, expert slot), and need no permutation.
  But they read NVFP4 in the GGML / official layout, while #1598 converts experts at load
  into TurboMind's swizzled tensor-core layout (`nvfp4_sm70_prepare`) and drops the
  original. Keeping both would duplicate ~11.5 GB of experts on one card.

The candidate approaches for the experts are therefore:
1. a Triton decode GEMV that reads TurboMind's layout directly (needs the swizzle
   documented and tested);
2. keeping experts in the official layout, with a graph-safe prefill path over it.

They are investigated separately, after the layout-independent parts.

## Order of work

1. **Loop controllers:** fused down and apply kernels over the existing FP16 weights.
2. **Decode attention and short convs:** fused prep (q/k/v/r with the K/V convs) and
   grouped-query split-KV decode attention.
3. **Experts:** after the layout question above is settled.

Each step replaces a #1598 decode kernel only where it is measured faster at TP=1 and TP=2,
behind #1598's existing call sites, with tests against #1598's current kernels as the
reference.
