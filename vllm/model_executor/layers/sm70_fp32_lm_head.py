# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Candidate LM-head dots without an intermediate FP16 logit rounding."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _indexed_fp32_logits(
    X,
    W,
    IDS,
    OUT,
    K: tl.constexpr,
    C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    candidate = tl.program_id(1)
    token = tl.load(IDS + row * C + candidate)
    k = tl.arange(0, BLOCK_K)
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    weight = tl.load(W + token * K + k, k < K, 0).to(tl.float32)
    value = tl.sum(x * weight, 0)
    tl.store(OUT + row * C + candidate, value)


def indexed_fp32_logits(
    x: torch.Tensor,
    weight: torch.Tensor,
    candidate_ids: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Evaluate contiguous FP16 rows into an owned FP32 candidate buffer.

    The caller gates the supported SM70 local shape and candidate bounds. Each
    program computes one dot, avoiding both expanded cross-row products and
    FP16 storage of the logits used for top-k/top-p.
    """
    _indexed_fp32_logits[(x.shape[0], candidate_ids.shape[1])](
        x,
        weight,
        candidate_ids,
        out,
        x.shape[1],
        candidate_ids.shape[1],
        triton.next_power_of_2(x.shape[1]),
        num_warps=4,
    )
