# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 FP16 Q/K norm, full NeoX RoPE and paged KV publication on SM70."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType, direct_register_custom_op


@triton.jit
def _norm_rope_cache(
    QKV,
    Q,
    K,
    QW,
    KW,
    POSITIONS,
    COS_SIN,
    CACHE_K,
    CACHE_V,
    SLOTS,
    QKV_STRIDE: tl.constexpr,
    CK_BLOCK: tl.constexpr,
    CK_TOKEN: tl.constexpr,
    CK_HEAD: tl.constexpr,
    CV_BLOCK: tl.constexpr,
    CV_TOKEN: tl.constexpr,
    CV_HEAD: tl.constexpr,
    PAGE: tl.constexpr,
    EPS: tl.constexpr,
    STORE_CACHE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    CACHE_TOKENS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    lanes = tl.arange(0, 16)
    variance = tl.full((16,), 0, tl.float32)
    base = tl.where(head < 8, head * 128, 1024 + (head - 8) * 128)
    for part in tl.static_range(8):
        offset = lanes * 8 + part
        value = tl.load(QKV + row * QKV_STRIDE + base + offset).to(tl.float32)
        variance += value * value
    inv = tl.rsqrt(tl.sum(variance, 0) / 128 + EPS)
    d = tl.arange(0, 128)
    mate = (d + 64) % 128
    weight = tl.load(tl.where(head < 8, QW, KW) + d).to(tl.float32)
    mate_weight = tl.load(tl.where(head < 8, QW, KW) + mate).to(tl.float32)
    value = (
        (tl.load(QKV + row * QKV_STRIDE + base + d).to(tl.float32) * inv * weight)
        .to(tl.float16)
        .to(tl.float32)
    )
    other = (
        (
            tl.load(QKV + row * QKV_STRIDE + base + mate).to(tl.float32)
            * inv
            * mate_weight
        )
        .to(tl.float16)
        .to(tl.float32)
    )
    position = tl.load(POSITIONS + row)
    position = tl.where(position < 0, position + CACHE_ROWS, position)
    valid_position = (position >= 0) & (position < CACHE_ROWS)
    cosine = tl.load(COS_SIN + position * 128 + d % 64, valid_position, other=0).to(
        tl.float32
    )
    sine = tl.load(COS_SIN + position * 128 + d % 64 + 64, valid_position, other=0).to(
        tl.float32
    )
    rotated = value * cosine + tl.where(d < 64, -other, other) * sine
    if head < 8:
        tl.store(Q + row * 1024 + head * 128 + d, rotated)
    else:
        kv_head = head - 8
        tl.store(K + row * 256 + kv_head * 128 + d, rotated)
        if STORE_CACHE:
            slot = tl.load(SLOTS + row, row < NUM_SLOTS, other=-1)
            page = slot // PAGE
            token = slot % PAGE
            tl.store(
                CACHE_K + page * CK_BLOCK + token * CK_TOKEN + kv_head * CK_HEAD + d,
                rotated,
                (slot >= 0) & (slot < CACHE_TOKENS),
            )
            val = tl.load(QKV + row * QKV_STRIDE + 1280 + kv_head * 128 + d)
            tl.store(
                CACHE_V + page * CV_BLOCK + token * CV_TOKEN + kv_head * CV_HEAD + d,
                val,
                (slot >= 0) & (slot < CACHE_TOKENS),
            )


def qk_norm_rope_cache(
    raw,
    qw,
    kw,
    positions,
    cos_sin,
    key_cache=None,
    value_cache=None,
    slots=None,
    q_out=None,
    k_out=None,
    epsilon=1e-6,
):
    q = q_out if q_out is not None else raw.new_empty((raw.shape[0], 1024))
    k = k_out if k_out is not None else raw.new_empty((raw.shape[0], 256))
    store_cache = key_cache is not None
    _norm_rope_cache[(raw.shape[0], 10)](
        raw,
        q,
        k,
        qw,
        kw,
        positions,
        cos_sin,
        key_cache,
        value_cache,
        slots,
        raw.stride(0),
        *(key_cache.stride()[:3] if store_cache else (0, 0, 0)),
        *(value_cache.stride()[:3] if store_cache else (0, 0, 0)),
        key_cache.shape[1] if store_cache else 1,
        epsilon,
        store_cache,
        slots.numel() if store_cache else 0,
        cos_sin.shape[0],
        key_cache.shape[0] * key_cache.shape[1] if store_cache else 0,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q, k


def sm70_dflash2_qk_norm_rope_cache(
    raw: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    epsilon: float,
    layer_name: LayerNameType,
) -> None:
    from vllm.model_executor.layers.attention.attention import get_attention_context
    from vllm.utils.torch_utils import _resolve_layer_name
    from vllm.v1.attention.backends.flash_v100.kv_layout import (
        _split_paged_kv_cache,
    )

    _, layer, kv_cache, slots = get_attention_context(_resolve_layer_name(layer_name))
    cache_args = {}
    if slots is not None and kv_cache.numel() > 0:
        kc, vc = _split_paged_kv_cache(kv_cache)
        if (
            kc.dtype == vc.dtype == torch.float16
            and kc.ndim == vc.ndim == 4
            and kc.shape == vc.shape
            and kc.shape[2:] == (2, 128)
            and kc.stride(3) == vc.stride(3) == 1
            and slots.ndim == 1
            and slots.dtype == torch.int64
        ):
            cache_args = dict(key_cache=kc, value_cache=vc, slots=slots)
    qk_norm_rope_cache(
        raw,
        qw,
        kw,
        positions,
        cos_sin,
        q_out=q_out,
        k_out=k_out,
        epsilon=epsilon,
        **cache_args,
    )
    if slots is not None and not cache_args:
        layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            layer,
            k_out.view(-1, 2, 128),
            raw[:, 1280:].view(-1, 2, 128),
            kv_cache,
            slots,
        )


def sm70_dflash2_qk_norm_rope_cache_fake(
    raw: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    epsilon: float,
    layer_name: LayerNameType,
) -> None:
    pass


direct_register_custom_op(
    op_name="sm70_dflash2_qk_norm_rope_cache",
    op_func=sm70_dflash2_qk_norm_rope_cache,
    fake_impl=sm70_dflash2_qk_norm_rope_cache_fake,
    mutates_args=["q_out", "k_out"],
)
