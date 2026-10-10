# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-shape MTP tile selection, preserving upstream native M1/M5 admission."""

# Qwen3.8-Flash-Next TP4 drafter MoE (BF16 MTP experts served as FP16, the
# checkpoint-global I640 expert width sharded to I160 per rank): tiles from an
# exact-shape CUDA-graph sweep on V100 (1Cat perf STEP-38). M is the drafter row
# count: 5 per request in the first proposer pass over the verifier window and 1
# per request in the three continuation passes, padded to a CUDA-graph size.
# M1 and M5 retain the upstream tile required by sm70_mtp_moe_fp16_out's
# admission guard; the residual tuning extends the other batch sizes only.
# Values: (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages); None
# keeps the SM70 0.0.3 default tile. The nearest key wins, as for a per-device
# config file; M > 40 keeps the defaults.
_SM70_QWEN38_MTP_MOE_TILES: dict[int, tuple[int, int, int, int, int] | None] = {
    1: (2, 128, 64, 4, 3),
    2: (2, 128, 64, 4, 3),
    3: (2, 64, 64, 2, 4),
    4: (2, 64, 64, 2, 3),
    5: (2, 128, 64, 4, 3),
    6: (2, 64, 64, 2, 4),
    7: (2, 64, 64, 2, 4),
    8: (2, 64, 32, 2, 3),
    9: (2, 128, 64, 4, 3),
    10: (2, 128, 64, 4, 3),
    12: (2, 64, 64, 2, 4),
    13: (2, 64, 64, 2, 4),
    15: (2, 64, 64, 2, 4),
    16: (2, 64, 64, 2, 3),
    18: (2, 64, 64, 2, 2),
    20: (2, 64, 64, 2, 4),
    24: (2, 64, 64, 2, 2),
    25: (2, 64, 64, 2, 2),
    30: (2, 64, 64, 2, 4),
    32: (2, 64, 64, 2, 4),
    35: (2, 64, 64, 2, 3),
    40: (2, 64, 64, 2, 3),
}
_SM70_QWEN38_MTP_MOE_MAX_M = 40


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
    if (E, N, K, topk) == (512, 160, 2560, 10) and M <= _SM70_QWEN38_MTP_MOE_MAX_M:
        key = min(_SM70_QWEN38_MTP_MOE_TILES, key=lambda m: abs(m - M))
        tile = _SM70_QWEN38_MTP_MOE_TILES[key]
        if tile is None:
            return None
        block_m, block_n, block_k, num_warps, num_stages = tile
        return {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    return None
