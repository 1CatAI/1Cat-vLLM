# SM70/V100 Concurrent Throughput Ledger

Date: 2026-08-01

This is the decision ledger for concurrent decode throughput on V100/SM70. It
is separate from the single-token operator and MTP latency ledgers so future
work does not confuse B=1 latency wins with multi-request throughput wins.

## Objective

Measure and improve aggregate decode throughput at active request counts
`B=1,2,4,8,16` for the current non-eager TurboMind + Flash-V100 routes.
The initial acceptance matrix is Qwen3.6-27B-AWQ and Qwen3.6-27B-FP8, TP4,
FP8 E5M2 KV, Mamba `align`, CUDA graphs, and official Qwen sampling:

```text
temperature=1.0, top_p=0.95, top_k=20
```

Each result must state model, quantization, TP/GPU topology, context/input and
output cap, KV dtype, target and drafter attention backends, graph policy,
prefix-cache state, and exact environment. TurboMind remains the quantized
linear route; Marlin does not count as a replacement result.

## Metrics

There are two required views and they must not be merged:

1. Exact offline batch: all `B` requests enter one `LLM.generate` call. Report
   aggregate steady decode TPS, per-request TPOT p50/p90/p99, output-length
   distribution, and `TPS(B) / TPS(1) / B` efficiency.
2. API continuous batching: `vllm bench serve --request-rate inf
   --max-concurrency B`. Report output TPS, TTFT, TPOT, queueing, and the same
   scaling efficiency. This is the production confirmation, not a substitute
   for the exact-batch route diagnosis.

For MTP, additionally report accepted length, draft acceptance rate, target
verifier, drafter, LM-head/sample/state, and host/scheduler cost per emitted
token. A request that naturally finishes early changes active batch size and
must be called out rather than treated as a full-B steady-state sample.

The offline entrypoint is:

```text
benchmarks/benchmark_sm70_concurrency.py
```

It preserves natural EOS by default and records a failed startup in JSON. It
does not claim output-quality validation; a normal natural-stop text-quality
gate remains mandatory for any runtime change.

## Confirmed Static Route Map

| Area | B=1 state | B=2..16 state | Consequence |
| --- | --- | --- | --- |
| Generic native MTP proposer | Supports batched rows | Supports batched rows; unit coverage exists at B=2 | The MTP architecture itself is not intrinsically serial across requests. |
| Dynamic GPU-LRU vocabulary | Accepted fused default | Explicitly rejects `max_num_seqs != 1` | Current default MTP cannot start at B>1. |
| Fused dynamic proposal | Scratch tensors are `[1, ...]`; one hidden row; one generator at index 0 | Top-20 LM-head is capped at M=8 and merge/sample are one-row, one-CTA kernels | Primary functional and throughput gap for MTP concurrency; B16 needs a new top-20 bound as well as batched merge/sample. |
| MTP verifier CUDA graphs | M=5 | Graph keys can cover `M=5*B` | Graph infrastructure is batch-capable once proposal is. |
| Flash-V100 small-query verifier | Exact M=5 dual-CTA specialization | Generic XQA supports expanded rows, but the exact dual-CTA specialization is not selected | B>1 needs an exact M=5B route study. |
| TurboMind dense warmup/tuning | M through 16 is warmed/tuned | No-MTP B<=16 is covered; MTP B>=4 has M=20/40/80 and falls back to default dispatch | High-M verifier dense tuning is missing. |
| TurboMind MoE warmup | Default warmup through 8 tokens | B16 no-MTP and MTP B>=2 are not prewarmed before graph capture | High-M MoE tuning/capture coverage is missing. |
| AWQ/FP8 MoE | B=1 takes an active-expert compact route | B>1 switches to batched/per-expert dispatch | Do not extrapolate B=1 kernel cost to batch throughput. |

The relevant source guards are `vllm/v1/spec_decode/llm_base_proposer.py`,
`vllm/v1/spec_decode/static_draft_vocab.py`,
`vllm/model_executor/warmup/awq_sm70_warmup.py`, and
`csrc/sm70_turbomind/ops/awq_sm70_gemm.cu`.

## Initial Test Order

1. Run non-MTP exact-batch B1/B2/B4/B8/B16 on AWQ and FP8. Capture runtime
   route logs and establish the scaling curve before changing a kernel.
2. Record the current default MTP B1 result and the expected B2 startup
   rejection from the dynamic GPU-LRU guard. Do not silently disable the
   default path and call that a MTP throughput baseline.
3. Use the existing exact `M=5B` AWQ/FP8 microbenchmarks to compare default
   dispatch against prewarmed/tuned M=10/20/40/80 shapes. This decides whether
   high-M TurboMind tuning earns a source change.
4. Add a B-aware dynamic vocabulary fused proposal only after its row-wise
   random-number, probability, LRU-update order, and rejection-sampler
   contracts are specified and tested. The required fallback for mixed sampling
   parameters must retain the existing generic full-probability path.
5. Add exact M=5B Flash-V100 verifier microbenchmarks after proposal is
   functional. Promote only bitwise-equivalent or otherwise classified
   quality-safe variants.
6. Confirm every winning route with API continuous batching and a natural-stop
   output-quality gate.

## Current State

No B2/B4/B8/B16 measurements are claimed yet. The current workstation has a
long-lived TP4 service using all four V100 devices, so the first matrix is
pending a task-owned GPU allocation. The existing service must not be stopped
or used as a benchmark target. The current NVML user-space and kernel driver
versions also differ, so `nvidia-smi` cannot be used as the sole device-health
oracle; task startup logs and PyTorch route checks are required when GPUs are
available.

## Rejected Shortcuts

- Do not use a Marlin fallback to improve a TurboMind concurrency result.
- Do not force `ignore_eos`, synthetic fixed output lengths, or changed
  sampling as the only throughput evidence.
- Do not remove the GPU-LRU B=1 guard without batching its buffers, per-row
  sampler state, and LRU update semantics.
- Do not infer B>1 performance from B=1 operator microbenchmarks.
