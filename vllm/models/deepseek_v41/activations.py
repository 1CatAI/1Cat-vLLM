# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 clamped SwiGLU with the original FP16 output boundary."""

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(gate)
    torch.ops._C.deepseek_v41_swiglu_sm70_out(output, gate, up)
    return output


def _fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(gate)


direct_register_custom_op(
    op_name="deepseek_v41_swiglu", op_func=_swiglu, fake_impl=_fake
)
swiglu = torch.ops.vllm.deepseek_v41_swiglu
