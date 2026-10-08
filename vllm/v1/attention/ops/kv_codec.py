# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton tile encoders for the existing KV writer contracts.

Page addressing and launch scheduling stay in cache writers. These helpers
preserve legacy casts, scale floors and operation order, including truncation
at the existing INT8 store. Native CUDA/QSA writers are migrated separately.
"""

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    FP8_DTYPE,
    get_fp8_min_max,
)
from vllm.triton_utils import tl, triton

FP8_MIN, FP8_MAX = get_fp8_min_max()
_PER_TOKEN_HEAD_QUANT_PARAMS: dict[torch.dtype, tuple[float, float]] = {
    torch.int8: (127.0, -128.0),
    FP8_DTYPE: (FP8_MAX, FP8_MIN),
}


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
def scale_kv_e4m3_per_tensor(values, scale_ptr):
    """Keep precise scaling separate from the fused writer's byte cast.

    Moving K scaling inside its store expression changes pointer evaluation
    order, and ptxas can then reorder the earlier RMSNorm accumulation.
    """
    return tl.div_rn(values.to(tl.float32), tl.load(scale_ptr))


@triton.jit
def encode_kv_per_tensor(values, scale_ptr, QUANTIZED: tl.constexpr):
    """Pass native/already-FP8 tiles through; scale other tiles before store."""
    if QUANTIZED:
        encoded = values if values.dtype.is_fp8() else values / tl.load(scale_ptr)
    else:
        encoded = values
    return encoded


@triton.jit
def kv_token_head_scale(values, QUANT_MAX: tl.constexpr):
    return tl.maximum(tl.max(tl.abs(values)) / QUANT_MAX, 1e-6)


@triton.jit
def encode_kv_per_token_head(
    values, scale, QUANT_MIN: tl.constexpr, QUANT_MAX: tl.constexpr
):
    # Cast/rounding is performed by the typed store, exactly as before extraction.
    return tl.clamp(values * (1.0 / scale), QUANT_MIN, QUANT_MAX)
