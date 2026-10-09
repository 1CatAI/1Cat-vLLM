# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape-gated native-g32 TP4 AWQ QPN M1 admission.

The changed FP32 reduction order is not a bitwise-equivalence guarantee.
"""

import torch

from vllm.config.sm70_moe import capture_sm70_moe_config


def _has_native_op() -> bool:
    return hasattr(torch.ops._C, "awq_moe_qpn_m1_sm70_out")


def initialize_qpn_m1(layer, shape_contract: bool) -> bool:
    policy = getattr(layer, "sm70_moe_policy", None)
    if policy is None:
        policy = capture_sm70_moe_config("awq")
    if not policy.qpn_m1:
        return False
    explicit = "qpn_m1" in policy.explicit_fields
    if not (
        shape_contract
        and layer.sm70_awq_moe_batched_gemm
        and layer.sm70_awq_moe_w13_interleaved
        and layer.sm70_awq_moe_legacy_single_token_compact
        and layer.sm70_awq_checkpoint_group_size == 32
        and layer.sm70_awq_group_size == 32
    ):
        if explicit:
            raise RuntimeError("AWQ QPN M1 requires native-g32 TP4 E512 C1")
        return False
    if not _has_native_op():
        if explicit:
            raise RuntimeError("AWQ QPN M1 requires a native build with CUDA arch 7.0")
        return False
    return True


def use_qpn_m1(layer, x, topk_weights, topk_ids) -> bool:
    return bool(
        getattr(layer, "sm70_awq_qwen38_qpn_m1", False)
        and x.shape == (1, 2560)
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.shape == (1, 10)
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and topk_weights.shape == (1, 10)
        and topk_weights.dtype == torch.float32
        and topk_weights.is_contiguous()
    )
