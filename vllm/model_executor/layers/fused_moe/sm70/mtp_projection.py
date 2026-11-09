# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact FP16 expert projection eligibility and native dispatch."""

from typing import Any

import torch

import vllm.envs as envs
from vllm.config.sm70_moe import unquantized_moe_policy
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.sm70.mtp_config import (
    _SM70_MTP_MOE_MAX_M,
    _force_sm70_mtp_moe_legacy_config,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl

logger = init_logger(__name__)


def _sm70_mtp_moe_fp16_shape_supported(
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


def try_mtp_fp16_projection(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
) -> bool:
    if (
        unquantized_moe_policy().value("mtp_fp16_exact")
        and not envs.VLLM_BATCH_INVARIANT
        and not _force_sm70_mtp_moe_legacy_config.get()
        and unquantized_moe_policy().value("mtp_tuned")
        and sorted_token_ids is None
        and not (use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
        and A_scale is None
        and B_scale is None
        and B_zp is None
        and B_bias is None
        and block_shape is None
        and compute_type == tl.float16
        and config.get("BLOCK_SIZE_M") == 2
        and config.get("BLOCK_SIZE_N") == 128
        and config.get("BLOCK_SIZE_K") == 64
        and config.get("num_warps") == 4
        and A.is_cuda
        and current_platform.is_device_capability(70)
        and _sm70_mtp_moe_fp16_shape_supported(
            A, B, C, topk_weights, expert_ids, top_k, mul_routed_weight
        )
        and B.data_ptr() % 16 == 0
        and num_tokens_post_padded.device == A.device
        and num_tokens_post_padded.dtype == torch.int32
        and num_tokens_post_padded.numel() == 1
        and num_tokens_post_padded.is_contiguous()
        and hasattr(torch.ops._C, "sm70_mtp_moe_fp16_out")
    ):
        logger.info_once(
            "Using exact SM70 FP16 MTP MoE projections (M1..80, E512, H2560, I160)."
        )
        torch.ops._C.sm70_mtp_moe_fp16_out(
            C,
            A,
            B,
            expert_ids,
            topk_weights,
            num_tokens_post_padded,
            mul_routed_weight,
        )
        return True

    return False
