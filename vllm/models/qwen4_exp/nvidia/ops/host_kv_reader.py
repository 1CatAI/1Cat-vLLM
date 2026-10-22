# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read protected hot values as FP16 and separately decoded misses."""

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32_bitcast,
)
from vllm.triton_utils import tl, triton


@triton.jit
def load_host_kv(
    Hot,
    Fallback,
    hot_tokens,
    row,
    columns,
    valid,
    dims,
    PADDED: tl.constexpr,
    DIM: tl.constexpr,
    HotScales=None,
    HOT_FP8: tl.constexpr = False,
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
    if HOT_FP8:
        k_scales = tl.load(HotScales + safe_hot * 2, hits, other=0)
        v_scales = tl.load(HotScales + safe_hot * 2 + 1, hits, other=0)
        cached_k = (fp8_e4m3fn_bits_to_fp32_bitcast(cached_k) * k_scales[None, :]).to(
            tl.float16
        )
        cached_v = (fp8_e4m3fn_bits_to_fp32_bitcast(cached_v) * v_scales[:, None]).to(
            tl.float16
        )
    misses = valid & ~hits
    source_k = tl.load(
        Fallback + (row * 2 * PADDED + columns[None, :]) * DIM + dims[:, None],
        misses[None, :],
        other=0,
    )
    source_v = tl.load(
        Fallback + ((row * 2 + 1) * PADDED + columns[:, None]) * DIM + dims[None, :],
        misses[:, None],
        other=0,
    )
    return (
        tl.where(hits[None, :], cached_k, source_k),
        tl.where(hits[:, None], cached_v, source_v),
    )
