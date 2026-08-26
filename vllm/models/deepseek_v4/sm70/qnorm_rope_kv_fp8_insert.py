# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Volta-safe Q normalization, RoPE, and DeepSeek FP8 KV insertion."""

import torch

import vllm.envs as envs
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp32_to_fp8_e4m3fn_bits,
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


@triton.jit
def _sm70_qnorm_rope_parallel_kv_insert_kernel(
    q_ptr,
    kv_ptr,
    cache_ptr,
    slot_mapping_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    cache_stride0,
    eps: tl.constexpr,
    num_tokens,
    num_heads: tl.constexpr,
    cache_block_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TOKEN_DATA_SIZE: tl.constexpr,
):
    """Fuse TP4 Q transform with eight exact packed-KV writer CTAs."""
    token_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    if token_idx >= num_tokens:
        return

    if worker_idx < num_heads:
        q_position = tl.load(positions_ptr + token_idx).to(tl.int64)
        q_pair_idx = tl.arange(0, HALF_ROPE)
        q_cos = tl.load(cos_sin_cache_ptr + q_position * ROPE_DIM + q_pair_idx).to(
            tl.float32
        )
        q_sin = tl.load(
            cos_sin_cache_ptr + q_position * ROPE_DIM + HALF_ROPE + q_pair_idx
        ).to(tl.float32)
        q_base = q_ptr + token_idx * num_heads * HEAD_DIM + worker_idx * HEAD_DIM
        q_offsets = tl.arange(0, HEAD_DIM)
        q_values = tl.load(q_base + q_offsets).to(tl.float32)
        q_rrms = tl.rsqrt(tl.sum(q_values * q_values, axis=0) / HEAD_DIM + eps)
        q_values *= q_rrms
        tl.store(
            q_base + q_offsets,
            q_values.to(q_ptr.type.element_ty),
            mask=q_offsets < NOPE_DIM,
        )

        q_even_offsets = NOPE_DIM + q_pair_idx * 2
        q_odd_offsets = q_even_offsets + 1
        q_even = tl.load(q_base + q_even_offsets).to(tl.float32) * q_rrms
        q_odd = tl.load(q_base + q_odd_offsets).to(tl.float32) * q_rrms
        tl.store(
            q_base + q_even_offsets,
            (q_even * q_cos - q_odd * q_sin).to(q_ptr.type.element_ty),
        )
        tl.store(
            q_base + q_odd_offsets,
            (q_even * q_sin + q_odd * q_cos).to(q_ptr.type.element_ty),
        )
    else:
        part_idx = worker_idx - num_heads
        slot_idx = tl.load(slot_mapping_ptr + token_idx)
        if slot_idx < 0:
            return

        block_idx = slot_idx // cache_block_size
        pos_in_block = slot_idx % cache_block_size
        cache_block = cache_ptr + block_idx.to(tl.int64) * cache_stride0
        token_data = cache_block + pos_in_block * TOKEN_DATA_SIZE
        token_scales = (
            cache_block + cache_block_size * TOKEN_DATA_SIZE + pos_in_block * 8
        )
        kv_row = kv_ptr + token_idx * HEAD_DIM
        kv_offsets = tl.arange(0, 64)
        if part_idx < 7:
            input_offsets = part_idx * 64 + kv_offsets
            kv_values = tl.load(kv_row + input_offsets)
            block_max = tl.maximum(tl.max(tl.abs(kv_values), axis=0), 1.0e-4)
            exponent = tl.ceil(tl.log2(block_max / 448.0))
            scale = tl.exp2(exponent)
            scaled = tl.clamp(kv_values / scale, -448.0, 448.0)
            packed = fp32_to_fp8_e4m3fn_bits(scaled.to(tl.float32))
            tl.store(token_data + input_offsets, packed)
            encoded = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0)
            tl.store(token_scales + part_idx, encoded.to(tl.uint8))
        else:
            tl.store(token_scales + 7, tl.zeros((), dtype=tl.uint8))
            kv_position = tl.load(positions_ptr + token_idx).to(tl.int64)
            kv_pair_idx = tl.arange(0, HALF_ROPE)
            kv_cos = tl.load(
                cos_sin_cache_ptr + kv_position * ROPE_DIM + kv_pair_idx
            ).to(tl.float32)
            kv_sin = tl.load(
                cos_sin_cache_ptr + kv_position * ROPE_DIM + HALF_ROPE + kv_pair_idx
            ).to(tl.float32)
            kv_even = tl.load(kv_row + NOPE_DIM + kv_pair_idx * 2).to(tl.float32)
            kv_odd = tl.load(kv_row + NOPE_DIM + kv_pair_idx * 2 + 1).to(tl.float32)
            rotated_even = (kv_even * kv_cos - kv_odd * kv_sin).to(tl.float16)
            rotated_odd = (kv_even * kv_sin + kv_odd * kv_cos).to(tl.float16)
            bf16_out = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            tl.store(bf16_out + kv_pair_idx * 2, rotated_even.to(tl.bfloat16))
            tl.store(
                bf16_out + kv_pair_idx * 2 + 1,
                rotated_odd.to(tl.bfloat16),
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
    if envs.VLLM_SM70_DSV4_QNORM_KV_FUSED_TP4 and num_tokens == 1 and num_heads == 16:
        cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
        _sm70_qnorm_rope_parallel_kv_insert_kernel[(num_tokens, num_heads + 8)](
            q,
            kv,
            cache_2d,
            slot_mapping,
            positions,
            cos_sin_cache,
            cache_2d.stride(0),
            eps,
            num_tokens,
            num_heads=num_heads,
            cache_block_size=block_size,
            HEAD_DIM=_HEAD_DIM,
            ROPE_DIM=_ROPE_DIM,
            NOPE_DIM=_NOPE_DIM,
            HALF_ROPE=_HALF_ROPE,
            TOKEN_DATA_SIZE=576,
            num_warps=4,
        )
        return q

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
