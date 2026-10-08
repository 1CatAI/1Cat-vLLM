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
