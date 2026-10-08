# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 debug attention oracle, independent of KV storage encoding."""

from __future__ import annotations

import torch


def _torch_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    window_size: tuple[int, int],
    softmax_scale: float,
) -> torch.Tensor:
    """Small debug-only fp32 attention reference for prefix/paged checks."""
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()

    num_q_heads = query_f.shape[1]
    num_kv_heads = key_f.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "num attention heads must be divisible by num KV heads for debug "
            f"reference, got {num_q_heads=} {num_kv_heads=}"
        )
    if num_q_heads != num_kv_heads:
        repeat = num_q_heads // num_kv_heads
        key_f = key_f.repeat_interleave(repeat, dim=1)
        value_f = value_f.repeat_interleave(repeat, dim=1)

    # [H, M, N]
    scores = torch.einsum("mhd,nhd->hmn", query_f, key_f) * softmax_scale
    q_len = query_f.shape[0]
    k_len = key_f.shape[0]
    q_pos = torch.arange(q_len, device=query.device) + max(k_len - q_len, 0)
    k_pos = torch.arange(k_len, device=query.device)
    valid = torch.ones((q_len, k_len), device=query.device, dtype=torch.bool)
    if causal:
        valid &= k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    window_left, window_right = window_size
    if window_left >= 0:
        valid &= k_pos.unsqueeze(0) >= q_pos.unsqueeze(1) - window_left
    if window_right >= 0:
        valid &= k_pos.unsqueeze(0) <= q_pos.unsqueeze(1) + window_right
    scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hmn,nhd->mhd", probs, value_f)
    return out.to(dtype=query.dtype).unsqueeze(0)
