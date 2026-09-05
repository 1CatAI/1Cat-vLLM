# Qwen3.8 exact single-request decode follow-up

Integration line: `public/main`, base `755baae1d075ee04fa9096b23fc0225b23589a86`.
This task is stacked on HC PR #506 (`4ae6a0005a`) to preserve the measured
single-request contract; it does not duplicate the M4/M8/M16 work in #504 or
the Page4 relocation-order fix in #494. Human review is required. AI assistance
was used (OpenAI Codex).

## Baseline and scope

TP4 V100-SXM2-32GB GPU0–3, Torch2.10.0+cu128, Qwen3.8-Flash-Next-NVFP4,
FP16 activations/KV, native NVFP4 experts, no MTP/prefix cache, hybrid PLE,
V2 dual CUDA graph, max length262144, prefill chunk8192, single request.
8K/513 before-capture engine baseline: 96.394713/96.417843/96.420508 tok/s,
aggregate96.411020, TPOT10.372258ms. Configured256K is not an input-quality gate.
Three greedy baseline token sequences equal the previous baseline; two short
official-sampling/thinking/natural-EOS smokes passed. Full256K quality pending.

The matching 8K/32 graph-node trace has 29 middle replay windows/rank. Complete
HC is2.033180ms; QSA1.219981ms including top-k0.307393ms; router top-k0.486903ms.
These are GPU service times, not additive end-to-end latency.

Raw baseline directory (outside Git):
`/home/ymzx/桌面/1cat-vllm/worktrees/v100-qwen38-nomtp-token-trace-20260903-173451/.artifacts/hc_trace_vector_20260905_retry1/`.

## Three bounded directions

1. Build the existing QSA single-row decode top-k from the frozen source and
   verify that the actual runtime uses it. The baseline trace contains the
   generic kernel even at grid1; source selects the specialized kernel for M1.
2. Screen a lossless 32-bit router sort key for FP16 logits/E512/K10/M1.
   Keep max/exp/normalization reduction, tie rules and invalid-row semantics.
   Other dtypes and widths retain the original 64-bit-key path.
3. Screen precomputed sparse KV addresses, keeping logical token order,
   split/merge arithmetic, dtype and FP16 boundaries. Do not assume a gather
   or persistent-kernel rewrite is profitable.

No lower precision, changed accumulation order, top-k truncation, MTP or
top1-only LM-head shortcuts. First prove numerical/graph/shape gates on small
operators, then use a combined whole-model run when justified. Never preempt
unrelated GPU owners. Keep failed paths and raw evidence.

## Current status

Router 32-bit-key candidate added behind an internal constexpr, default off
pending GPU exactness and paired speed screening. QSA runtime build and sparse
address candidate pending. No new performance or model-quality claim yet.
