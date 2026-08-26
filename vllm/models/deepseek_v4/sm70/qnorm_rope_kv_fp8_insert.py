# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Volta-safe Q normalization, RoPE, and DeepSeek FP8 KV insertion."""

import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    quantize_and_insert_k_cache,
)
from vllm.triton_utils import tl, triton

_HEAD_DIM = 512
_ROPE_DIM = 64
_NOPE_DIM = _HEAD_DIM - _ROPE_DIM
_HALF_ROPE = _ROPE_DIM // 2


@triton.jit
def _sm70_qnorm_rope_kernel(
    q_ptr,
    kv_ptr,
    kv_out_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    eps: tl.constexpr,
    num_tokens,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens:
        return

    pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)
    pair_idx = tl.arange(0, HALF_ROPE)
    cos = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + pair_idx).to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + HALF_ROPE + pair_idx).to(
        tl.float32
    )

    if head_idx < num_heads:
        q_base = q_ptr + token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM
        offsets = tl.arange(0, HEAD_DIM)
        values = tl.load(q_base + offsets).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(values * values, axis=0) / HEAD_DIM + eps)
        values *= rrms

        tl.store(
            q_base + offsets,
            values.to(q_ptr.type.element_ty),
            mask=offsets < NOPE_DIM,
        )

        even_offsets = NOPE_DIM + pair_idx * 2
        odd_offsets = even_offsets + 1
        even = tl.load(q_base + even_offsets).to(tl.float32) * rrms
        odd = tl.load(q_base + odd_offsets).to(tl.float32) * rrms
        tl.store(
            q_base + even_offsets,
            (even * cos - odd * sin).to(q_ptr.type.element_ty),
        )
        tl.store(
            q_base + odd_offsets,
            (even * sin + odd * cos).to(q_ptr.type.element_ty),
        )
    else:
        kv_base = kv_ptr + token_idx * HEAD_DIM
        kv_out_base = kv_out_ptr + token_idx * HEAD_DIM
        offsets = tl.arange(0, HEAD_DIM)
        tl.store(
            kv_out_base + offsets,
            tl.load(kv_base + offsets, mask=offsets < NOPE_DIM),
            mask=offsets < NOPE_DIM,
        )

        even_offsets = NOPE_DIM + pair_idx * 2
        odd_offsets = even_offsets + 1
        even = tl.load(kv_base + even_offsets).to(tl.float32)
        odd = tl.load(kv_base + odd_offsets).to(tl.float32)
        tl.store(
            kv_out_base + even_offsets,
            (even * cos - odd * sin).to(kv_out_ptr.type.element_ty),
        )
        tl.store(
            kv_out_base + odd_offsets,
            (even * sin + odd * cos).to(kv_out_ptr.type.element_ty),
        )


def sm70_qnorm_rope_kv_fp8_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> torch.Tensor:
    """Apply the V4 Q/KV transform using FP16 compute on exact SM70."""
    assert q.dtype == torch.float16 and kv.dtype == torch.float16, (
        f"SM70 DeepSeek V4 requires FP16 Q/KV, got {q.dtype} and {kv.dtype}"
    )
    assert q.ndim == 3 and q.shape[-1] == _HEAD_DIM
    assert kv.ndim == 2 and kv.shape == (q.shape[0], _HEAD_DIM)
    assert q.is_contiguous() and kv.is_contiguous()

    num_tokens, num_heads, _ = q.shape
    kv_roped = torch.empty_like(kv)
    _sm70_qnorm_rope_kernel[(num_tokens, num_heads + 1)](
        q,
        kv,
        kv_roped,
        positions,
        cos_sin_cache,
        eps,
        num_tokens,
        num_heads=num_heads,
        HEAD_DIM=_HEAD_DIM,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM=_NOPE_DIM,
        HALF_ROPE=_HALF_ROPE,
        num_warps=4,
    )
    quantize_and_insert_k_cache(
        kv_roped,
        swa_kv_cache.view(swa_kv_cache.shape[0], -1),
        slot_mapping,
        block_size=block_size,
    )
    return q
