# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gated Qwen3.5 query/key norm, rotary embedding and SM70 KV publication."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import LayerNameType, direct_register_custom_op


@triton.jit
def _e4m3_satfinite(value):
    bits = value.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 0x80
    absolute_bits = bits & 0x7FFFFFFF
    absolute = absolute_bits.to(tl.float32, bitcast=True)
    limited = tl.minimum(absolute, 448.0)
    limited_bits = limited.to(tl.uint32, bitcast=True)
    # RNE to three fraction bits. The normal exponent bias is 7.
    normal = (limited_bits - 0x3C000000 + 0x7FFFF + ((limited_bits >> 20) & 1)) >> 20
    # Adding 2**14 gives an FP32 ULP of 2**-9, the FP8 subnormal step.
    subnormal = (limited + 16384.0).to(tl.uint32, bitcast=True) - 0x46800000
    code = tl.where(limited < 0.015625, subnormal, normal)
    code = tl.where(absolute_bits > 0x7F800000, 0x7F, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _qk_norm_rope(
    QKV,
    QW,
    KW,
    Pos,
    Cache,
    QOut,
    KOut,
    GateOut,
    STORE_GATE: tl.constexpr,
    Slots,
    KCache,
    VCache,
    KScale,
    VScale,
    STORE_CACHE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    POS_ROW: tl.constexpr,
    POS_COL: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    TOKENS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    ROW: tl.constexpr,
    POS_PLANES: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, 256)
    if head < 6:
        source = QKV + token * ROW + head * 512
        weight = QW
        destination = QOut + token * 1536 + head * 256
    else:
        source = QKV + token * ROW + 3072
        weight = KW
        destination = KOut + token * 256
    if STORE_GATE:  # noqa: SIM102 - eliminate the optional pointer at compile time
        if head < 6:
            gate = tl.load(QKV + token * ROW + head * 512 + 256 + col)
            tl.store(GateOut + token * 1536 + head * 256 + col, gate)
    values = tl.load(source + col).to(tl.float32)
    variance = tl.sum(values * values, 0) / 256.0
    inverse = tl.rsqrt(variance + EPS)
    w = tl.load(weight + col).to(tl.float32) + 1.0
    normalized = values * inverse * w
    # Reload only the rotary partner. Its norm shares the same variance,
    # while its learned channel weight can differ.
    partner = tl.where(col < 32, col + 32, tl.where(col < 64, col - 32, col))
    pv = tl.load(source + partner).to(tl.float32)
    pw = tl.load(weight + partner).to(tl.float32) + 1.0
    pn = pv * inverse * pw
    frequency = col % 32
    plane = tl.full((256,), 0, tl.int32)
    if POS_PLANES == 3:
        plane = tl.where((frequency % 3 == 1) & (frequency < 33), 1, plane)
        plane = tl.where((frequency % 3 == 2) & (frequency < 30), 2, plane)
    position = tl.load(Pos + plane * POS_ROW + token * POS_COL)
    position = tl.where(position < 0, position + CACHE_ROWS, position)
    cosine = tl.load(Cache + position * 64 + frequency)
    sine = tl.load(Cache + position * 64 + 32 + frequency)
    # The compiled target keeps normalized rotary channels in FP32 until
    # the final rotary store. Do not insert an intermediate FP16 rounding.
    first = normalized.to(tl.float32) * cosine.to(tl.float32)
    second = pn.to(tl.float32) * sine.to(tl.float32)
    sign = tl.where(col < 32, -1.0, 1.0)
    rotated = (first.to(tl.float32) + sign * second.to(tl.float32)).to(tl.float16)
    # Cache encoding must consume the same FP16 value published to KOut,
    # including the non-rotary channels whose normalization stays in FP32.
    processed = tl.where(col < 64, rotated, normalized).to(tl.float16)
    tl.store(destination + col, processed)
    if STORE_CACHE:
        slot = tl.load(Slots + token, mask=token < NUM_SLOTS, other=-1)
        if slot >= 0:
            offset = (
                (slot // BLOCK_SIZE) * CACHE_BLOCK
                + (slot % BLOCK_SIZE) * CACHE_TOKEN
                + col
            )
            if head == 6:
                scaled = tl.div_rn(processed.to(tl.float32), tl.load(KScale))
                tl.store(KCache + offset, _e4m3_satfinite(scaled))
            elif head == 0:
                value = tl.load(QKV + token * ROW + 3328 + col).to(tl.float32)
                tl.store(
                    VCache + offset, _e4m3_satfinite(tl.div_rn(value, tl.load(VScale)))
                )


def qk_norm_rope(
    qkv,
    q_weight,
    k_weight,
    positions,
    cache,
    eps=1e-6,
    key_cache=None,
    value_cache=None,
    slots=None,
    k_scale=None,
    v_scale=None,
    q_out=None,
    k_out=None,
    gate_out=None,
):
    assert qkv.ndim == 2 and qkv.shape[1] == 3584 and qkv.dtype == torch.float16
    tokens = qkv.shape[0]
    assert qkv.stride(1) == 1
    assert q_weight.shape == k_weight.shape == (256,)
    assert q_weight.is_contiguous() and k_weight.is_contiguous()
    assert cache.ndim == 2 and cache.shape[1] == 64 and cache.is_contiguous()
    assert cache.dtype == torch.float16
    assert positions.shape in ((tokens,), (3, tokens))
    assert positions.dtype == torch.int64
    store_cache = key_cache is not None
    if store_cache:
        assert key_cache.shape == value_cache.shape and key_cache.ndim == 4
        assert key_cache.shape[2:] == (1, 256)
        assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
        assert key_cache.dtype == value_cache.dtype == torch.uint8
        assert slots.ndim == 1 and slots.dtype == torch.int64
        assert k_scale.numel() == v_scale.numel() == 1
        assert k_scale.dtype == v_scale.dtype == torch.float32
    q = (
        q_out
        if q_out is not None
        else torch.empty((tokens, 1536), device=qkv.device, dtype=qkv.dtype)
    )
    k = (
        k_out
        if k_out is not None
        else torch.empty((tokens, 256), device=qkv.device, dtype=qkv.dtype)
    )
    assert q.shape == (tokens, 1536) and k.shape == (tokens, 256)
    assert q.is_contiguous() and k.is_contiguous()
    _qk_norm_rope[(tokens, 7)](
        qkv,
        q_weight,
        k_weight,
        positions,
        cache,
        q,
        k,
        gate_out,
        gate_out is not None,
        slots,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        store_cache,
        key_cache.shape[1] if store_cache else 0,
        key_cache.stride(0) if store_cache else 0,
        key_cache.stride(1) if store_cache else 0,
        positions.stride(0) if positions.ndim == 2 else 0,
        positions.stride(-1),
        cache.shape[0],
        tokens,
        slots.numel() if store_cache else 0,
        qkv.stride(0),
        1 if positions.ndim == 1 else 3,
        eps,
        num_warps=4,
        enable_fp_fusion=True,
    )
    return q, k


def _prepare_with_cache(
    qkv,
    q_out,
    k_out,
    gate_out,
    q_weight,
    k_weight,
    positions,
    cache,
    epsilon,
    layer_name,
):
    from vllm.model_executor.layers.attention.attention import get_attention_context
    from vllm.utils.torch_utils import _resolve_layer_name
    from vllm.v1.attention.backends.flash_attn_v100 import _split_paged_kv_cache

    _, layer, kv_cache, slots = get_attention_context(_resolve_layer_name(layer_name))
    cache_args = {}
    if slots is not None and kv_cache.numel() > 0:
        key_cache, value_cache = _split_paged_kv_cache(kv_cache)
        eligible = (
            layer.kv_cache_dtype in ("fp8", "fp8_e4m3")
            and key_cache.dtype == value_cache.dtype == torch.uint8
            and key_cache.ndim == value_cache.ndim == 4
            and key_cache.shape == value_cache.shape
            and key_cache.shape[2:] == (1, 256)
            and key_cache.stride() == value_cache.stride()
            and key_cache.stride(3) == 1
            and slots.ndim == 1
            and slots.dtype == torch.int64
            and layer._k_scale.numel() == layer._v_scale.numel() == 1
        )
        if eligible:
            cache_args = dict(
                key_cache=key_cache,
                value_cache=value_cache,
                slots=slots,
                k_scale=layer._k_scale,
                v_scale=layer._v_scale,
            )
    qk_norm_rope(
        qkv,
        q_weight,
        k_weight,
        positions,
        cache,
        epsilon,
        q_out=q_out,
        k_out=k_out,
        gate_out=gate_out,
        **cache_args,
    )
    if slots is not None and not cache_args:
        # Keep the standard writer for a cache layout not handled above.
        layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            layer,
            k_out.view(-1, 1, 256),
            qkv[:, 3328:].view(-1, 1, 256),
            kv_cache,
            slots,
        )


def sm70_qwen38_qk_norm_rope_cache(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    gate_out: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    epsilon: float,
    layer_name: LayerNameType,
) -> None:
    # Keep compiler-owned destinations, as in unified attention. Query/key
    # dependencies order the context-owned KV update before attention.
    _prepare_with_cache(
        qkv,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        positions,
        cache,
        epsilon,
        layer_name,
    )


def sm70_qwen38_qk_norm_rope_cache_fake(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    gate_out: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    epsilon: float,
    layer_name: LayerNameType,
) -> None:
    pass


direct_register_custom_op(
    op_name="sm70_qwen38_qk_norm_rope_cache",
    op_func=sm70_qwen38_qk_norm_rope_cache,
    fake_impl=sm70_qwen38_qk_norm_rope_cache_fake,
    mutates_args=["q_out", "k_out", "gate_out"],
)
