# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton tile encoders for the existing KV writer contracts.

Page addressing and launch scheduling stay in cache writers. These helpers
preserve legacy casts, scale floors and operation order, including truncation
at the existing INT8 store. Native CUDA/QSA writers are migrated separately.
"""

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def encode_fp8_satfinite(value, E5M2: tl.constexpr = False):
    """Software RNE encoding on devices without FP8 conversion instructions."""
    bits = value.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 0x80
    absolute_bits = bits & 0x7FFFFFFF
    absolute = absolute_bits.to(tl.float32, bitcast=True)
    if E5M2:
        limited = tl.minimum(absolute, 57344.0)
        limited_bits = limited.to(tl.uint32, bitcast=True)
        normal = (
            limited_bits - 0x38000000 + 0xFFFFF + ((limited_bits >> 21) & 1)
        ) >> 21
        subnormal = (limited + 128.0).to(tl.uint32, bitcast=True) - 0x43000000
        code = tl.where(limited < 0.00006103515625, subnormal, normal)
    else:
        limited = tl.minimum(absolute, 448.0)
        limited_bits = limited.to(tl.uint32, bitcast=True)
        normal = (
            limited_bits - 0x3C000000 + 0x7FFFF + ((limited_bits >> 20) & 1)
        ) >> 20
        subnormal = (limited + 16384.0).to(tl.uint32, bitcast=True) - 0x46800000
        code = tl.where(limited < 0.015625, subnormal, normal)
    code = tl.where(absolute_bits > 0x7F800000, 0x7F, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def kv_token_head_scale(
    values, QUANT_MAX: tl.constexpr, ROUND_INT8_EVEN: tl.constexpr = False
):
    if ROUND_INT8_EVEN:
        scale = tl.maximum(tl.div_rn(tl.max(tl.abs(values)), QUANT_MAX), 1e-6)
        invalid = (tl.abs(values) == float("inf")) | (values != values)
        return tl.where(tl.sum(invalid.to(tl.int32)) == 0, scale, float("nan"))
    return tl.maximum(tl.max(tl.abs(values)) / QUANT_MAX, 1e-6)


@triton.jit
def encode_kv_per_token_head(
    values,
    scale,
    QUANT_MIN: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    ROUND_INT8_EVEN: tl.constexpr = False,
):
    if ROUND_INT8_EVEN:
        return tl.clamp(
            tldevice.nearbyint(tl.div_rn(values, scale)), QUANT_MIN, QUANT_MAX
        )
    # Cast/rounding is performed by the typed store, exactly as before extraction.
    return tl.clamp(values * (1.0 / scale), QUANT_MIN, QUANT_MAX)


@triton.jit
def encode_kv_cache_row(
    values, tensor_scale, CODEC: tl.constexpr, ROUND_INT8_EVEN: tl.constexpr
):
    """Common row encoder for ordinary and fused cache writers.

    Codec IDs match the public storage contract. The caller owns addressing
    and stores the returned FP32 scale only for a token/head codec.
    """
    if CODEC == 3:
        values = values.to(tl.float32)
        scale = kv_token_head_scale(values, 127.0, ROUND_INT8_EVEN)
        return (
            encode_kv_per_token_head(values, scale, -128.0, 127.0, ROUND_INT8_EVEN),
            scale,
        )
    elif CODEC == 1 or CODEC == 2:
        scaled = tl.div_rn(values.to(tl.float32), tl.load(tensor_scale))
        return encode_fp8_satfinite(scaled, CODEC == 2), 1.0
    else:
        return values, 1.0
