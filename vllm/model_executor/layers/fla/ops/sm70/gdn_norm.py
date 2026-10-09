# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN gated norm provider with an explicit per-engine calculation policy."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _sm70_gdn_rmsnorm_gated_onepass_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    out_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, D)
    x = tl.load(x_ptr + row * D + offsets).to(tl.float32)
    z = tl.load(z_ptr + row * D + offsets).to(tl.float32)
    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    sumsq = tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(sumsq / D + EPS)
    gated = z * tl.sigmoid(z)
    tl.store(out_ptr + row * D + offsets, x * rstd * weight * gated)


def _sm70_qwen_gdn_rmsnorm_gated_configured_impl(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    activation: str,
    onepass: bool,
) -> torch.Tensor:
    if (
        onepass
        and current_platform.is_device_capability(70)
        and x.is_cuda
        and x.dtype == torch.float16
        and x.shape == (12, 128)
        and x.is_contiguous()
        and z.is_cuda
        and z.device == x.device
        and z.dtype == x.dtype
        and z.shape == x.shape
        and z.is_contiguous()
        and weight.is_cuda
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.shape == (128,)
        and weight.is_contiguous()
        and group_size <= 0
        and norm_before_gate
        and activation in ("silu", "swish")
    ):
        out = torch.empty_like(x)
        _sm70_gdn_rmsnorm_gated_onepass_kernel[(12,)](
            x,
            z,
            weight,
            out,
            D=128,
            EPS=eps,
            num_warps=2,
        )
        return out

    from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

    return rmsnorm_fn(
        x,
        weight,
        None,
        z=z,
        eps=eps,
        group_size=None if group_size <= 0 else group_size,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )


def _sm70_qwen_gdn_rmsnorm_gated_impl(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    """Old standalone schema; engine calls use the configured entry."""
    from vllm import envs

    return _sm70_qwen_gdn_rmsnorm_gated_configured_impl(
        x,
        z,
        weight,
        eps,
        group_size,
        norm_before_gate,
        activation,
        envs.VLLM_SM70_GDN_RMSNORM_ONEPASS,
    )


def _sm70_qwen_gdn_rmsnorm_gated_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    return torch.empty_like(x)


def _sm70_qwen_gdn_rmsnorm_gated_configured_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    activation: str,
    onepass: bool,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="sm70_qwen_gdn_rmsnorm_gated",
    op_func=_sm70_qwen_gdn_rmsnorm_gated_impl,
    fake_impl=_sm70_qwen_gdn_rmsnorm_gated_fake,
)

direct_register_custom_op(
    op_name="sm70_qwen_gdn_rmsnorm_gated_configured",
    op_func=_sm70_qwen_gdn_rmsnorm_gated_configured_impl,
    fake_impl=_sm70_qwen_gdn_rmsnorm_gated_configured_fake,
)
