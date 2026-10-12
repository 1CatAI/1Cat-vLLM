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
    values,
    QUANT_MAX: tl.constexpr,
    ROUND_INT8_EVEN: tl.constexpr = False,
    AXIS: tl.constexpr = None,
):
    if ROUND_INT8_EVEN:
        scale = tl.maximum(tl.div_rn(tl.max(tl.abs(values), AXIS), QUANT_MAX), 1e-6)
        invalid = (tl.abs(values) == float("inf")) | (values != values)
        return tl.where(tl.sum(invalid.to(tl.int32), AXIS) == 0, scale, float("nan"))
    return tl.maximum(tl.max(tl.abs(values), AXIS) / QUANT_MAX, 1e-6)


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


@triton.jit
def fp8_e4m3fn_bits_to_fp32(bits):
    """Decode E4M3FN bytes exactly using scalar arithmetic."""
    value_bits = bits.to(tl.int32)
    sign = tl.where((value_bits & 0x80) != 0, -1.0, 1.0)
    exponent = (value_bits >> 3) & 0x0F
    mantissa = value_bits & 0x07
    subnormal = mantissa.to(tl.float32) * (2.0**-9)
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(
        exponent.to(tl.float32) - 7.0
    )
    decoded = tl.where(exponent == 0, subnormal, normal) * sign
    return tl.where((exponent == 15) & (mantissa == 7), float("nan"), decoded)


@triton.jit
def fp8_e4m3fn_bits_to_fp32_bitcast(bits):
    """Decode E4M3FN bytes by constructing exact IEEE FP32 normal values."""
    value_bits = bits.to(tl.int32)
    exponent = (value_bits >> 3) & 0x0F
    mantissa = value_bits & 0x07
    sign_bits = (value_bits & 0x80) << 24
    normal_bits = sign_bits | ((exponent + 120) << 23) | (mantissa << 20)
    normal = normal_bits.to(tl.float32, bitcast=True)

    sign = tl.where((value_bits & 0x80) != 0, -1.0, 1.0)
    subnormal = mantissa.to(tl.float32) * (2.0**-9) * sign
    decoded = tl.where(exponent == 0, subnormal, normal)
    return tl.where((exponent == 15) & (mantissa == 7), float("nan"), decoded)


@triton.jit
def decode_kv_payload(values, CODEC: tl.constexpr, E4M3_BITS: tl.constexpr = False):
    """Unscaled payload conversion shared by compatible algorithm bindings."""
    if CODEC == 1:
        if E4M3_BITS:
            return fp8_e4m3fn_bits_to_fp32_bitcast(values)
        return fp8_e4m3fn_bits_to_fp32(values)
    elif CODEC == 3:
        return values.to(tl.uint8).to(tl.int8, bitcast=True).to(tl.float32)
    else:
        tl.static_assert(CODEC == 0, "Missing unscaled KV codec binding")
        return values.to(tl.float32)


@triton.jit
def encode_history_cache_rows(values, CODEC: tl.constexpr, AXIS: tl.constexpr):
    """History representation: preserve E4M3's row-scale arithmetic order."""
    if CODEC == 3:
        scale = kv_token_head_scale(values, 127.0, True, AXIS)
        encoded = encode_kv_per_token_head(
            values, tl.expand_dims(scale, AXIS), -128.0, 127.0, True
        )
        return encoded.to(tl.int8), scale
    elif CODEC == 1:
        scale = tl.maximum(tl.div_rn(tl.max(tl.abs(values), AXIS), 448.0), 2.0**-126)
        encoded = fp32_to_fp8_e4m3fn_bits(
            tl.div_rn(values, tl.expand_dims(scale, AXIS))
        )
        return encoded, scale
    else:
        tl.static_assert(CODEC == 0, "Missing history cache writer binding")
        return values.to(tl.float16), 1.0


@triton.jit
def _round_nearest_even_i32(x):
    return tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        constraints="=r,f",
        args=[x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def fp32_to_fp8_e4m3fn_bits(x):
    """Encode finite FP32 values to saturated E4M3FN round-to-nearest-even."""
    sign_bit = (x.to(tl.int32, bitcast=True) >> 24) & 0x80
    value = tl.minimum(tl.abs(x), 448.0)

    subnormal_mantissa = _round_nearest_even_i32(value * 512.0)
    # A rounded value of eight is the minimum normal encoding (0x08).
    subnormal_mantissa = tl.minimum(tl.maximum(subnormal_mantissa, 0), 8)

    safe_value = tl.maximum(value, 2.0**-126)
    exponent_unbiased = tl.floor(tl.log2(safe_value)).to(tl.int32)
    exponent_scale = tl.exp2(exponent_unbiased.to(tl.float32))
    mantissa = _round_nearest_even_i32((safe_value / exponent_scale - 1.0) * 8.0)
    carry = mantissa >= 8
    exponent = exponent_unbiased + 7 + carry.to(tl.int32)
    mantissa = tl.where(carry, 0, mantissa)

    overflow = (exponent > 15) | ((exponent == 15) & (mantissa > 6))
    exponent = tl.minimum(tl.maximum(exponent, 1), 15)
    mantissa = tl.minimum(tl.maximum(mantissa, 0), 7)
    normal_bits = (exponent << 3) | mantissa
    normal_bits = tl.where(overflow, 0x7E, normal_bits)

    magnitude_bits = tl.where(value < (2.0**-6), subnormal_mantissa, normal_bits)
    magnitude_bits = tl.where(value == 0.0, 0, magnitude_bits)
    return (sign_bit | magnitude_bits).to(tl.uint8)


@triton.jit
def encode_indexer_cache_group(
    values,
    CODEC: tl.constexpr,
    ROUND_FP8_SCALE: tl.constexpr,
    SOFTWARE_FP8: tl.constexpr,
    INVERSE_FP8_SCALE: tl.constexpr = False,
):
    """Encode one index group; retain each FP8 writer's scaling order."""
    if CODEC == 3:
        encoded, scale = encode_kv_cache_row(values, 1.0, 3, True)
        return encoded.to(tl.int8).to(tl.uint8, bitcast=True), scale
    else:
        tl.static_assert(CODEC == 1, "Missing indexer cache encoder binding")
        absolute = tl.maximum(tl.max(tl.abs(values)), 1e-4)
        raw_scale = absolute * (1.0 / 448.0)
        if ROUND_FP8_SCALE:
            exponent = tl.ceil(tl.log2(raw_scale))
            scale = tl.exp2(exponent)
            if INVERSE_FP8_SCALE:
                normalized = values * tl.exp2(-exponent)
            else:
                normalized = values / scale
        else:
            scale = raw_scale
            normalized = values / scale
        normalized = tl.clamp(normalized, -448.0, 448.0)
        if SOFTWARE_FP8:
            encoded = fp32_to_fp8_e4m3fn_bits(normalized)
        else:
            encoded = normalized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        return encoded, scale


@triton.jit
def encode_mla_cache_group(
    values,
    CODEC: tl.constexpr,
    AXIS: tl.constexpr = -1,
    SOFTWARE_FP8: tl.constexpr = True,
    FP8_SCALE_RECIPROCAL: tl.constexpr = False,
    FP8_VALUE_RECIPROCAL: tl.constexpr = False,
):
    """G64 latent encoder; FP8 retains its UE8M0 division and rounding order."""
    values = values.to(tl.float32)
    if CODEC == 3:
        scale = kv_token_head_scale(values, 127.0, True, AXIS)
        return (
            encode_kv_per_token_head(
                values, tl.expand_dims(scale, AXIS), -128.0, 127.0, True
            )
            .to(tl.int8)
            .to(tl.uint8, bitcast=True),
            scale,
        )
    else:
        tl.static_assert(CODEC == 1, "Missing MLA grouped encoder binding")
        absolute = tl.maximum(tl.max(tl.abs(values), AXIS), 1e-4)
        if FP8_SCALE_RECIPROCAL:
            raw_scale = absolute * (1.0 / 448.0)
        else:
            raw_scale = absolute / 448.0
        exponent = tl.ceil(tl.log2(raw_scale))
        if FP8_VALUE_RECIPROCAL:
            normalized = values * tl.expand_dims(tl.exp2(-exponent), AXIS)
        else:
            normalized = values / tl.expand_dims(tl.exp2(exponent), AXIS)
        normalized = tl.clamp(normalized, -448.0, 448.0)
        if SOFTWARE_FP8:
            encoded = fp32_to_fp8_e4m3fn_bits(normalized)
        else:
            encoded = normalized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        scale = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.uint8)
        return encoded, scale


@triton.jit
def decode_mla_cache_group(
    payload,
    encoded_scale,
    CODEC: tl.constexpr,
    E4M3_BITS: tl.constexpr = False,
    OUTPUT_DTYPE: tl.constexpr = None,
    FP16_SCALE_FIRST: tl.constexpr = True,
    SOFTWARE_FP8: tl.constexpr = True,
):
    """Decode a latent group; unquantized representation fields stay separate."""
    result_dtype: tl.constexpr = tl.float16 if OUTPUT_DTYPE is None else OUTPUT_DTYPE
    if CODEC == 1 and not SOFTWARE_FP8:
        raw = payload.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        raw = decode_kv_payload(payload, CODEC, E4M3_BITS)
    if CODEC == 3:
        value = raw.to(tl.float32) * encoded_scale.to(tl.float32)
        finite_scale = (encoded_scale == encoded_scale) & (
            tl.abs(encoded_scale) != float("inf")
        )
        limit: tl.constexpr = (
            65504.0
            if tl.float16 == result_dtype
            else (
                3.3895313892515355e38
                if tl.bfloat16 == result_dtype
                else 3.4028234663852886e38
            )
        )
        value = tl.where(finite_scale, tl.clamp(value, -limit, limit), value)
        return value.to(result_dtype)
    else:
        tl.static_assert(CODEC == 1, "Missing MLA grouped decoder binding")
        # Preserve the existing FP8 reader's FP16 multiply by exponent scale.
        scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
        if FP16_SCALE_FIRST:
            return (raw.to(tl.float16) * scale.to(tl.float16)).to(result_dtype)
        return (raw.to(tl.float32) * scale).to(result_dtype)
