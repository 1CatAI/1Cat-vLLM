# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused V4.1 router epilogue: sqrt(softplus) scores, biased top-k, renormalize."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _route(
    logits,
    bias,
    probs,
    ids,
    E: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    e = tl.arange(0, BLOCK)
    valid = e < E
    g = tl.load(logits + t * E + e, valid, 0.0)
    # softplus as in torch (threshold 20), then sqrt.
    soft = tl.where(g > 20.0, g, tl.log(1.0 + tl.exp(tl.minimum(g, 20.0))))
    score = tl.sqrt(soft)
    key = tl.where(valid, score + tl.load(bias + e, valid, 0.0), float("-inf"))
    k = tl.arange(0, 8)
    chosen = tl.zeros((8,), tl.int32)
    values = tl.zeros((8,), tl.float32)
    for j in tl.static_range(K):
        best = tl.max(key, 0)
        index = tl.min(tl.where(key == best, e, BLOCK), 0)
        value = tl.sum(tl.where(e == index, score, 0.0), 0)
        chosen = tl.where(k == j, index, chosen)
        values = tl.where(k == j, value, values)
        key = tl.where(e == index, float("-inf"), key)
    total = tl.sum(tl.where(k < K, values, 0.0), 0)
    out = values / (total + 1e-20) * SCALE
    tl.store(probs + t * K + k, out, k < K)
    tl.store(ids + t * K + k, chosen.to(tl.int64), k < K)


def route_topk(logits, bias, top_k=6, scale=1.5):
    tokens, experts = logits.shape
    if logits.dtype != torch.float32 or bias.dtype != torch.float32 or top_k > 8:
        raise ValueError("V4.1 fused routing requires FP32 logits/bias, top_k<=8")
    probs = torch.empty(tokens, top_k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(tokens, top_k, dtype=torch.int64, device=logits.device)
    if tokens:
        _route[(tokens,)](
            logits.contiguous(),
            bias,
            probs,
            ids,
            E=experts,
            K=top_k,
            SCALE=scale,
            BLOCK=triton.next_power_of_2(experts),
            num_warps=4,
        )
    return probs, ids
