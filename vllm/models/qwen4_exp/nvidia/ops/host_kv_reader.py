# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 hot values with exact host-format fallback for the QSA main loop."""

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast,
)
from vllm.triton_utils import tl, triton


@triton.jit
def load_host_kv(
    Hot,
    History,
    Scales,
    hot_tokens,
    pages,
    offsets,
    valid,
    dims,
    PAGE: tl.constexpr,
    DIM: tl.constexpr,
    FP8: tl.constexpr,
):
    hits = valid & (hot_tokens >= 0)
    safe_hot = tl.maximum(hot_tokens, 0).to(tl.int64)
    cached_k = tl.load(
        Hot + safe_hot[None, :] * 2 * DIM + dims[:, None],
        hits[None, :],
        other=0,
    )
    cached_v = tl.load(
        Hot + (safe_hot[:, None] * 2 + 1) * DIM + dims[None, :],
        hits[:, None],
        other=0,
    )
    misses = valid & ~hits
    source_k = tl.load(
        History + (pages[None, :] * 2 * PAGE + offsets[None, :]) * DIM + dims[:, None],
        misses[None, :],
        other=0,
    )
    source_v = tl.load(
        History
        + ((pages[:, None] * 2 + 1) * PAGE + offsets[:, None]) * DIM
        + dims[None, :],
        misses[:, None],
        other=0,
    )
    if FP8:
        tokens = pages * PAGE + offsets
        k_scale = tl.load(Scales + tokens * 2, misses, other=0)
        v_scale = tl.load(Scales + tokens * 2 + 1, misses, other=0)
        source_k = (fp8_e4m3fn_bits_to_fp32_bitcast(source_k) * k_scale[None, :]).to(
            tl.float16
        )
        source_v = (fp8_e4m3fn_bits_to_fp32_bitcast(source_v) * v_scale[:, None]).to(
            tl.float16
        )
    return (
        tl.where(hits[None, :], cached_k, source_k),
        tl.where(hits[:, None], cached_v, source_v),
    )
