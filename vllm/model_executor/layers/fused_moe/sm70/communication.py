# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eligibility for the shared+routed two-input collective."""

from typing import TYPE_CHECKING

import torch

from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.config.execution_policy import communication_policy, graph_policy
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


def can_use_sum2_allreduce(
    runner: "MoERunner",
    shared_output: torch.Tensor | None,
    fused_output: torch.Tensor,
) -> bool:
    if shared_output is None:
        return False
    if graph_policy().dual_compile and not use_sm70_decode_graph_semantics():
        return False
    if not current_platform.is_cuda():
        return False
    if runner.moe_config.is_sequence_parallel:
        return False
    if runner._fused_output_is_reduced:
        return False
    if runner.moe_config.tp_size <= 1:
        return False
    if runner.moe_config.ep_size != 1 or runner.moe_config.pcp_size != 1:
        return False
    if (
        shared_output.shape != fused_output.shape
        or shared_output.dtype != fused_output.dtype
    ):
        return False

    tp_size = runner.moe_config.tp_size
    glm53_q8 = bool(
        communication_policy().value("moe_sum2_q8")
        and tp_size == 8
        and shared_output.dtype == torch.float16
        and tuple(shared_output.shape) == (8, 4096)
    )
    if not communication_policy().moe_add_allreduce and not glm53_q8:
        return False
    return tp_size in (2, 4, 6, 8)
