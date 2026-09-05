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

Draft PR #507 is stacked on #506. The three operator paths are implemented;
matching whole-model performance and long-context health checks are pending.

The router keeps the original 8-warp FP32 max/exp/normalization tree. Its
32-bit key is selected only for existing FP16 E512/K10 M1 routing. Signed-zero,
NaN/Inf, ties, all raw half encodings and finite shuffled inputs are covered:
1,280 rows are bitwise equal, plus16 changed-input/poisoned graph replays.
After warmup, eight alternating-order A/B pairs measure48 calls at
**0.252652 ->0.203530ms** (save0.049121ms). An earlier non-interleaved screen
drifted in clock state and is retained but not used for the accepted delta.

The QSA sidecar builder compiles the production header, exposes a version
marker and a literal old generic-kernel control.72 cases cover M1/M2, lengths
0/1/511/512/2048/2304/2305/4096/65536, ties, signed-zero and non-finite scores;
all selected IDs are exact.16 changed-input/poisoned graph replays pass.
Twelve calls at live lengths2048–2169 measure **0.258196 ->0.111665ms**.
Source-overlay services must pin the freshly built library through existing
`VLLM_SM70_QSA_TOPK_LIBRARY`; do not infer native-binary freshness from Python
source or an old unversioned log message. The endpoint trace must prove the
decode-specialized symbol is present. Header SHA256:
`e09d4af611894d2c3613ea1d5ac50e1fd2606f729e6fa7c45eb8079fcc6b9508`.
Sidecar SHA256:
`e56d7874877dddb88589a7e08ecbe9074f7f7532af1be764ab5f47a9b8aa8165`.

QSA address resolution retains logical ordering/duplicates and validity,
precomputes physical token slots, and removes the dependent page-table load
from the unchanged partial attention arithmetic. It is limited to SM70,
FP16 M1/Q6/KV1/D256/page400/selection2051 and signed-int32-safe physical slots.
Other shapes, cache formats and prefill keep the original path. Eight changing
graph scenarios per length test page relocation, invalid pages/indices and
requests, duplicates and poisoned outputs; all outputs are bitwise identical.
Public production-dispatch screen (12 attention+merge calls, resolver included):

| Cache context | Original ms | Resolved ms |
|---|---:|---:|
|8192|0.345016|0.318909|
|32768|0.384205|0.367094|
|262144|0.365860|0.323968|

These are operator service times, not additive endpoint savings. The256K row
is an operator cache-size check, not a256K model-input quality acceptance.
CPU dispatcher/QSA launch suites:49 passed,1 skipped (GPU-only). Targeted Ruff
checks pass. No new lower-precision weights, KV or arithmetic introduced.

Reproduction (project Python environment, SM70 GPU ownership required):

```bash
CUDA_HOME=/usr TORCH_CUDA_ARCH_LIST=7.0 .venv/bin/python \
  benchmarks/kernels/build_sm70_qsa_topk_sidecar.py --build-dir .artifacts/qsa-build
.venv/bin/python benchmarks/kernels/verify_sm70_qsa_router_exact.py \
  --qsa-library .artifacts/qsa-build/vllm_qsa_decode_topk_sm70.so \
  --out .artifacts/operators.json
.venv/bin/python benchmarks/kernels/verify_sm70_qsa_resolved.py \
  .artifacts/address.json
```

Raw task artifacts: `.artifacts/three_paths/operators_interleaved.json`,
`address_production.json`, build/queue logs. All standalone GPU screens exited.
One combined endpoint is prepared in `.artifacts/endpoint/`: same8K/513r3
baseline and8K/32 trace, plus261632-token natural-output health and262143+1
context-boundary request. These long tests are explicitly not retrieval scores.
