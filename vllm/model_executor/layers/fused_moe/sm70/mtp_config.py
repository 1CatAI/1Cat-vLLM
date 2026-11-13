# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-shape MTP tiles and their shared startup warmup override."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from vllm.config.sm70_moe import unquantized_moe_policy

# Concurrent MTP4 drafts: the first proposer pass covers every request's
# five-row verifier window (16 requests -> 80 rows); continuation passes have
# one row per request.
_SM70_MTP_MOE_MAX_M = 80

_force_sm70_mtp_moe_legacy_config = ContextVar("sm70_moe_legacy_warmup", default=False)


@contextmanager
def force_sm70_mtp_moe_legacy_config() -> Iterator[None]:
    """Temporarily select the legacy SM70 MoE tile during startup warmup."""
    token = _force_sm70_mtp_moe_legacy_config.set(True)
    try:
        yield
    finally:
        _force_sm70_mtp_moe_legacy_config.reset(token)


def _get_sm70_mtp_moe_decode_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> dict[str, int] | None:
    """Return graph-tuned exact-shape SM70 MTP tiles."""
    if _force_sm70_mtp_moe_legacy_config.get() or not unquantized_moe_policy().value(
        "mtp_tuned"
    ):
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
    if (
        E == 512
        and N in (160, 320, 640)
        and K == 2560
        and topk == 10
        and 1 <= M <= _SM70_MTP_MOE_MAX_M
    ):
        # Qwen3.8-Flash-Next TP1/TP2/TP4 keeps the BF16/FP16 MTP expert weights
        # and shards the checkpoint-global I640 expert width to I640/I320/I160
        # per rank. M5 is the first proposer pass over the verifier window; M1
        # covers its three autoregressive continuation passes. Concurrent
        # requests multiply both (M5c and Mc); the exact kernel computes every
        # route alike.
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
