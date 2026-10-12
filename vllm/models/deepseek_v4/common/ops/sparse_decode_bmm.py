# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA decode attention on the grouped byte cache as one indexed
dequantizing gather and two batched matmuls.

Same idea as sparse_prefill_bmm: every head of a token reads the same keys, so
they are gathered once per token and the rest goes to the BLAS library. The
gather reads the declared mixed MLA cache by slot index and dequantizes on the
way. Against the split-K QK-D Triton kernel at 6 query tokens and 64 heads:
C4 (512 + 128 keys) 1.14 to 0.19 ms on RTX 8000 and 0.68 to 0.19 ms on V100,
C128 0.38 to 0.20 and 0.28 to 0.20 ms, same output to 1.2e-4. All shapes are
static, so it captures into CUDA graphs.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.kv_codec import decode_mla_cache_group

from .cache_utils import mixed_mla_cache_view, needs_software_fp8
from .sparse_prefill_bmm import WorkspaceSpec, _fit, attend_gathered_keys

# Mixed MLA: 448 byte values in seven 64-wide scale groups and 64 BF16 RoPE
# fields. The common view supplies the UE8M0 or FP32 scale region separately.
_NOPE_DIM = 448
_ROPE_DIM = 64
_HEAD_DIM = _NOPE_DIM + _ROPE_DIM
_SCALE_GROUP = 64
_TOKEN_DATA_BYTES = _NOPE_DIM + 2 * _ROPE_DIM
_TOKEN_SCALE_BYTES = 8


def sparse_decode_bmm_blocker(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    extra_cache: torch.Tensor | None,
    *,
    kv_cache_dtype: str = "fp8_ds_mla",
) -> str | None:
    """Admission for the actual packed gather layout, including padded blocks."""
    if q.ndim != 3 or q.dtype != torch.float16 or q.shape[-1] != _HEAD_DIM:
        return "requires FP16 queries with 448 FP8 + 64 RoPE dimensions"
    for cache in (main_cache, extra_cache):
        if cache is None:
            continue
        try:
            mixed_mla_cache_view(cache, kv_cache_dtype)
        except (ValueError, RuntimeError) as error:
            return f"incompatible mixed MLA cache view: {error}"
    return None


@triton.jit
def _dequant_gather_rows_kernel(
    out_ptr,
    cache_ptr,
    scales_ptr,
    scale_stride_block,
    scale_stride_token,
    slots_ptr,
    num_cache_rows,
    cache_stride_block,
    slots_stride_t,
    out_stride_t,
    out_col_offset,
    BLOCK_SIZE: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    SCALE_GROUP: tl.constexpr,
    TOKEN_DATA_BYTES: tl.constexpr,
    TOKEN_SCALE_BYTES: tl.constexpr,
    SOFTWARE_FP8: tl.constexpr,
    CODEC: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1)
    slot = tl.load(slots_ptr + token * slots_stride_t + col)
    # An unused slot (-1) reads row 0; the caller zeroes what it gathered.
    valid = (slot >= 0) & (slot < num_cache_rows)
    safe = tl.where(valid, slot, 0).to(tl.int64)
    block = cache_ptr + (safe // BLOCK_SIZE) * cache_stride_block
    pos = safe % BLOCK_SIZE
    data = block + pos * TOKEN_DATA_BYTES
    scales = (
        scales_ptr
        + (safe // BLOCK_SIZE) * scale_stride_block
        + pos * scale_stride_token
    )
    out_row = (
        out_ptr
        + token.to(tl.int64) * out_stride_t
        + (out_col_offset + col) * (NOPE_DIM + ROPE_DIM)
    )

    for group in tl.static_range(NOPE_DIM // SCALE_GROUP):
        offsets = group * SCALE_GROUP + tl.arange(0, SCALE_GROUP)
        bits = tl.load(data + offsets, mask=valid, other=0)
        encoded = tl.load(scales + group, mask=valid, other=127 if CODEC == 1 else 1.0)
        value = decode_mla_cache_group(
            bits,
            encoded,
            CODEC,
            OUTPUT_DTYPE=out_ptr.type.element_ty,
            FP16_SCALE_FIRST=False,
            SOFTWARE_FP8=SOFTWARE_FP8,
        )
        tl.store(out_row + offsets, value)

    rope_offsets = tl.arange(0, ROPE_DIM)
    rope = tl.load(
        (data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16)) + rope_offsets,
        mask=valid,
        other=0,
    )
    tl.store(out_row + NOPE_DIM + rope_offsets, rope.to(out_ptr.dtype.element_ty))


def dequant_gather_rows(
    keys: torch.Tensor,
    col_offset: int,
    cache: torch.Tensor,
    slots: torch.Tensor,
    *,
    kv_cache_dtype: str = "fp8_ds_mla",
) -> None:
    """Fill keys[:, col_offset : col_offset + W] with the cache rows named by
    slots [T, W]. cache is the packed [blocks, block_size, bytes] uint8 cache."""
    view = mixed_mla_cache_view(cache, kv_cache_dtype)
    assert view.key_scale is not None
    assert keys.shape[2] == _HEAD_DIM and keys.is_contiguous()
    assert slots.dtype == torch.int32 and slots.stride(1) == 1
    num_tokens, width = slots.shape
    if width == 0:
        return
    _dequant_gather_rows_kernel[(num_tokens, width)](
        keys,
        view.key,
        view.key_scale,
        view.key_scale.stride(0),
        view.key_scale.stride(1),
        slots,
        cache.shape[0] * cache.shape[1],
        cache.stride(0),
        slots.stride(0),
        keys.stride(0),
        col_offset,
        BLOCK_SIZE=cache.shape[1],
        NOPE_DIM=_NOPE_DIM,
        ROPE_DIM=_ROPE_DIM,
        SCALE_GROUP=_SCALE_GROUP,
        TOKEN_DATA_BYTES=view.key.stride(1),
        TOKEN_SCALE_BYTES=_TOKEN_SCALE_BYTES,
        SOFTWARE_FP8=needs_software_fp8(),
        CODEC=view.codec.native_id,
    )


def sparse_decode_bmm_workspace_specs(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    main_width: int,
    extra_width: int,
    dtype: torch.dtype,
) -> list[WorkspaceSpec]:
    """Buffers of `sparse_attn_decode_bmm`, in its argument order."""
    width = main_width + extra_width
    return [
        ((num_tokens, width, head_dim), dtype),
        ((num_tokens, num_heads, width), torch.float32),
        ((num_tokens, num_heads, width + 1), torch.float32),
        ((num_tokens, num_heads, width + 1), torch.float32),
    ]


def sparse_attn_decode_bmm(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_lengths: torch.Tensor,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lengths: torch.Tensor | None,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    keys: torch.Tensor,
    scores: torch.Tensor,
    logits: torch.Tensor,
    probs: torch.Tensor,
    *,
    kv_cache_dtype: str = "fp8_ds_mla",
) -> None:
    """q [T, H, D]; main/extra indices are global slot ids into their packed
    caches (-1 = unused) with per-token lengths; attn_sink [H]. The four
    buffers come from the workspace specs and may be larger than needed."""
    num_tokens, num_heads, head_dim = q.shape
    main_slots = main_indices.reshape(num_tokens, -1)
    main_width = main_slots.shape[1]
    has_extra = extra_cache is not None
    if has_extra:
        assert extra_indices is not None and extra_lengths is not None
        extra_slots = extra_indices.reshape(num_tokens, -1)
        extra_width = extra_slots.shape[1]
    else:
        extra_width = 0
    width = main_width + extra_width

    keys = _fit(keys, (num_tokens, width, head_dim))
    scores = _fit(scores, (num_tokens, num_heads, width))
    logits = _fit(logits, (num_tokens, num_heads, width + 1))
    probs = _fit(probs, (num_tokens, num_heads, width + 1))

    columns = torch.arange(max(main_width, extra_width), device=q.device)[None, :]
    unused = torch.empty((num_tokens, width), dtype=torch.bool, device=q.device)
    unused[:, :main_width] = (
        (main_slots < 0)
        | (main_slots >= main_cache.shape[0] * main_cache.shape[1])
        | (columns[:, :main_width] >= main_lengths.reshape(-1, 1))
    )
    dequant_gather_rows(keys, 0, main_cache, main_slots, kv_cache_dtype=kv_cache_dtype)
    if has_extra:
        assert extra_cache is not None and extra_lengths is not None
        unused[:, main_width:] = (
            (extra_slots < 0)
            | (extra_slots >= extra_cache.shape[0] * extra_cache.shape[1])
            | (columns[:, :extra_width] >= extra_lengths.reshape(-1, 1))
        )
        dequant_gather_rows(
            keys, main_width, extra_cache, extra_slots, kv_cache_dtype=kv_cache_dtype
        )

    attend_gathered_keys(
        q, keys, unused, scale, attn_sink, output, scores, logits, probs
    )
