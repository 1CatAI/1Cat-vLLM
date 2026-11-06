# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream MTP tile and native-route contracts, owned by the SM70 feature."""

from typing import Any

import torch

_SM70_MTP_MOE_MAX_M = 80


def get_decode_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    *,
    force_legacy: bool,
    tuned_enabled: bool,
) -> dict[str, int] | None:
    """Return graph-tuned exact-shape SM70 MTP tiles."""
    if force_legacy or not tuned_enabled:
        return None
    if (E, N, K, topk) == (256, 128, 2048, 8) and 2 <= M <= 16:
        # Qwen3.6 TP4 shards the checkpoint-global I512 expert width to I128
        # per rank. M2-M16 all select this tile in the exact local-shape graph
        # oracle.
        return {
            "BLOCK_SIZE_M": 8,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    if (E, N, K, topk) == (512, 160, 2560, 10) and 1 <= M <= _SM70_MTP_MOE_MAX_M:
        # Qwen3.8-Flash-Next TP4 keeps the BF16 MTP expert weights and shards
        # the checkpoint-global I640 expert width to I160 per rank. M5 is the
        # first proposer pass over the verifier window; M1 covers its three
        # autoregressive continuation passes. Concurrent requests multiply
        # both (M5c and Mc); the exact kernel computes every route alike.
        return {
            "BLOCK_SIZE_M": 2,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    return None


def native_shape_supported(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    weights: torch.Tensor | None,
    ids: torch.Tensor,
    top_k: int,
    weighted: bool,
) -> bool:
    if weights is None or weights.ndim != 2 or weights.shape[1] != 10:
        return False
    m = weights.shape[0]
    if not 1 <= m <= _SM70_MTP_MOE_MAX_M:
        return False
    n, k = (2560, 160) if weighted else (320, 2560)
    return (
        top_k == (1 if weighted else 10)
        and a.shape == (m * 10 if weighted else m, k)
        and b.shape == (512, n, k)
        and c.shape == (m, 10, n)
        and ids.shape == (m * 10,)
        and a.dtype == b.dtype == c.dtype == torch.float16
        and weights.dtype == torch.float32
        and ids.dtype == torch.int32
        and all(
            t.device == a.device and t.is_contiguous() for t in (a, b, c, weights, ids)
        )
    )


def use_naive_routes(
    config: dict[str, Any], num_tokens: int, top_k_num: int, global_num_experts: int
) -> bool:
    # The exact SM70 MTP expert kernel (the only BLOCK_SIZE_M=2 tile) reads
    # one route per CTA, so it keeps the naive assignment at every draft batch.
    return (
        config.get("BLOCK_SIZE_M") == 2
        and global_num_experts == 512
        and top_k_num == 10
        and num_tokens <= _SM70_MTP_MOE_MAX_M
    )
