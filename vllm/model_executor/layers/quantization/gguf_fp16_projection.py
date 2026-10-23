# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime selection for already converted floating GGUF projection shards."""

import torch

from vllm.model_executor.kernels.gguf import (
    GGUFOperatorCapability,
    decoder_family,
)
from vllm.model_executor.kernels.linear.fp16_gemv_silu import Sm70Fp16GemvSiluKernel
from vllm.platforms import current_platform
from vllm.transformers_utils.gguf_tensor_reader import quant_type_name
from vllm.utils.torch_utils import direct_register_custom_op

FP16_SOURCE_TYPES = frozenset((1, 30))
_OPERATOR = "prepared_gguf_fp16_projection"


def fp16_projection_capabilities(
    source_type: int,
    weight: torch.Tensor,
    act_dtype: torch.dtype,
    enabled: bool = True,
) -> tuple[GGUFOperatorCapability, ...]:
    """Admit M8 row GEMV without changing source or operand precision."""
    reason = None
    if not enabled:
        reason = "disabled_by_kernel_config"
    elif source_type not in FP16_SOURCE_TYPES:
        reason = "requires_f16_or_bf16_source"
    elif act_dtype != torch.float16:
        reason = "requires_fp16_activations"
    elif weight.dtype != torch.float16:
        reason = "requires_converted_fp16_weight"
    elif weight.ndim != 2 or tuple(weight.shape) not in ((12, 5120), (24, 5120)):
        reason = "floating_projection_shape_not_admitted"
    elif weight.device.type != "cuda":
        reason = "device_not_cuda"
    elif current_platform.get_device_capability(weight.device.index) != (7, 0):
        reason = "requires_sm70"
    elif not weight.is_contiguous():
        reason = "requires_contiguous_weight"
    elif not hasattr(torch.ops.vllm, _OPERATOR):
        reason = f"operator_missing:{_OPERATOR}"
    return (
        GGUFOperatorCapability(
            decoder_family(source_type),
            quant_type_name(source_type),
            _OPERATOR,
            True,
            min_m=8,
            max_m=8,
            reason=reason,
        ),
    )


def _prepared_gguf_fp16_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    source_type: int,
    enabled: bool = True,
) -> torch.Tensor:
    # vLLM range compilation can first see prefill. Resolve actual M only
    # inside this opaque op so later graph replay retains the decode choice.
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if rows.shape[0] == 8 and all(
        capability.reason is None
        for capability in fp16_projection_capabilities(
            source_type, weight, x.dtype, enabled
        )
    ):
        output = rows.new_empty((rows.shape[0], weight.shape[0]))
        if Sm70Fp16GemvSiluKernel.can_implement(
            rows, weight, output, weight.shape[0], 0, 0, 0, 1.0
        ):
            # No activated prefix: use the existing FP32 row reduction and
            # final FP16 projection boundary as a plain GEMV.
            Sm70Fp16GemvSiluKernel.apply_out(rows, weight, output, weight.shape[0], 0)
            return output.reshape(*x.shape[:-1], weight.shape[0])
    return torch.mm(rows, weight.T).reshape(*x.shape[:-1], weight.shape[0])


def _prepared_gguf_fp16_projection_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    source_type: int,
    enabled: bool = True,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name=_OPERATOR,
    op_func=_prepared_gguf_fp16_projection,
    fake_impl=_prepared_gguf_fp16_projection_fake,
)
