# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton tile encoders for the existing KV writer contracts.

Page addressing and launch scheduling stay in cache writers. These helpers
preserve legacy casts, scale floors and operation order, including truncation
at the existing INT8 store. Native CUDA/QSA writers are migrated separately.
"""

from vllm.triton_utils import tl, tldevice, triton


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
