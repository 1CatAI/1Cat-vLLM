# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant dense GEMMs for the SM70 DeepSeek-V4.1 port (``VLLM_DS41_BATCH_INVARIANT_GEMM``; default off).

DeepInfer's ``DEEPINFER_BATCH_INVARIANT`` makes its kernels' reduction schedules independent of the batch. On V100 the
dense GEMMs are cuBLAS, which selects the kernel (and so the reduction order) from M; measured on this box
(scripts/gemm_det_test.py): rows 0..12 of a 14-row fp16 GEMM differ from the 13-row call in half of the elements, and the
kernel class changes at M = 14 and 17 (and 64 for some N). Through 40 MoE layers that ulp became up to 2.5 nats between a
prompt prefilled alone and the same prompt prefilled alongside other requests' decode rows (mean KL 0.12 on short
prompts; the pack's quantisation KL is 0.036). Layer bisect 2026-10-20: first difference after layer 0, embeddings
identical; with the rows padded the two were bit-identical through all 40 layers.

This module pads the ROW count of a GEMM to a fixed bucket (<= 16 rows -> 16, one HMMA tile and about the cost of 1 row;
otherwise the next multiple of 64) and returns an OWNING tensor of the real rows (``clone``, never a slice view: the
earlier global torch.mm wrapper returned views and made lone prefills history-dependent). It is applied at the port's own
GEMM call sites only — compressor.mm_fp32 / mm_fp32_full (attention q/kv, indexer, compressor, DSpark drafter), the
wq_b projections, the MoE gate and shared expert, the LM head — not to torch globally. It pads under CUDA-graph capture
too: steps of <= 16 tokens go through the per-layer attention graphs, captured on first use and replayed afterwards, so a
capture-time skip made the FIRST request (eager, padded) differ from every later one (replay, unpadded) — the
"alone-vs-alone" effect of 2026-10-20. Prefill chunks above 64 rows can still change bucket between alone and batched; the
residual there measured quantisation-noise sized (KL 0.027 at 47 tokens).
"""

from __future__ import annotations

import os

import torch

ENV = "VLLM_DS41_BATCH_INVARIANT_GEMM"
ENABLED = os.environ.get(ENV, "0") == "1"
SMALL_M = int(os.environ.get("VLLM_DS41_BATCH_INVARIANT_SMALL_M", "16"))
BUCKET = int(os.environ.get("VLLM_DS41_BATCH_INVARIANT_BUCKET", "64"))


def target_rows(m: int) -> int:
    if SMALL_M > 0 and m <= SMALL_M:
        return SMALL_M
    return -(-m // BUCKET) * BUCKET


def mm(a: torch.Tensor, b: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """``torch.mm(a, b[, out_dtype])`` with a's row count padded to the bucket; returns an owning [m, N] tensor.
    With the knob off (or under graph capture, or for non-CUDA / non-2-D inputs) it is plain torch.mm."""
    kw = {} if out_dtype is None else {"out_dtype": out_dtype}
    if not ENABLED or a.dim() != 2 or not a.is_cuda:
        return torch.mm(a, b, **kw)
    m = a.shape[0]
    t = target_rows(m)
    if t == m:
        return torch.mm(a, b, **kw)
    a2 = torch.cat([a, a.new_zeros((t - m, a.shape[1]))], 0)
    return torch.mm(a2, b, **kw)[:m].clone()
