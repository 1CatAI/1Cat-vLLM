# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 common bindings and their fake implementations."""

from typing import TYPE_CHECKING

import torch

from . import loader as loader

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake as register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake  # noqa: F401


def _op(name: str):
    if not hasattr(torch.ops._C, name):
        raise RuntimeError(
            f"SM70 TurboMind op _C::{name} is not available. "
            "Build vLLM with CUDA arch 7.0 to enable it."
        )
    return getattr(torch.ops._C, name)


def _qwen38_qpn8_op(name: str):
    """Prefer the task sidecar, then fall back to the production namespace."""
    sidecar = torch.ops._C_qwen38
    if hasattr(sidecar, name):
        return getattr(sidecar, name)
    return _op(name)


def has_fp8_qpn8_hc_dispatch() -> bool:
    return hasattr(torch.ops._C_qwen38, "fp8_qpn8_hc_dispatch_sm70_out") or hasattr(
        torch.ops._C, "fp8_qpn8_hc_dispatch_sm70_out"
    )


def has_nvfp4_qpn_m1_dispatch() -> bool:
    return hasattr(torch.ops._C_qwen38, "nvfp4_moe_qpn_m1_sm70_out") or hasattr(
        torch.ops._C, "nvfp4_moe_qpn_m1_sm70_out"
    )


def has_nvfp4_qpn_raw_scale_dispatch() -> bool:
    names = (
        "nvfp4_expand_raw_scales_sm70_out",
        "nvfp4_moe_qpn_raw_scale_sm70_out",
        "nvfp4_moe_qpn_raw_w13_swiglu_batch_sm70_out",
        "nvfp4_moe_qpn_raw_w2_reduce_sm70_out",
    )
    return all(
        hasattr(torch.ops._C_qwen38, name) or hasattr(torch.ops._C, name)
        for name in names
    )


def has_nvfp4_qwen38_w2_direct_reduce() -> bool:
    return hasattr(torch.ops._C_qwen38, "nvfp4_qwen38_w2_direct_reduce_out") or hasattr(
        torch.ops._C, "nvfp4_qwen38_w2_direct_reduce_out"
    )


def has_nvfp4_qwen38_w13_fused_swiglu() -> bool:
    return hasattr(torch.ops._C_qwen38, "nvfp4_qwen38_w13_fused_swiglu_out") or hasattr(
        torch.ops._C, "nvfp4_qwen38_w13_fused_swiglu_out"
    )


def has_qwen38_shared_gate_exact() -> bool:
    return hasattr(torch.ops._C_qwen38, "qwen38_shared_gate_exact_out") or hasattr(
        torch.ops._C, "qwen38_shared_gate_exact_out"
    )


def has_qwen38_shared_gate_sigmoid_mul() -> bool:
    return hasattr(torch.ops._C, "qwen38_shared_gate_sigmoid_mul_out")


def has_nvfp4_qpn_mtp5_dispatch() -> bool:
    """Reject extensions that only implement the legacy ten-route kernel."""
    return hasattr(torch.ops._C_qwen38, "nvfp4_moe_qpn_mtp5_sm70_out") or hasattr(
        torch.ops._C, "nvfp4_moe_qpn_mtp5_sm70_out"
    )


def has_nvfp4_qpn_w13_swiglu_batch_dispatch() -> bool:
    return hasattr(
        torch.ops._C_qwen38,
        "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out",
    ) or hasattr(torch.ops._C, "nvfp4_moe_qpn_w13_swiglu_batch_sm70_out")


def has_nvfp4_grouped_decode_dispatch() -> bool:
    return all(
        hasattr(torch.ops._C, name)
        for name in ("nvfp4_grouped_w13_sm70_out", "nvfp4_grouped_w2_sm70_out")
    )


def has_nvfp4_grouped_batch_reduce_dispatch() -> bool:
    return hasattr(torch.ops._C, "nvfp4_grouped_w2_batch_reduce_sm70_out")


def has_nvfp4_qpn_w2_reduce_dispatch() -> bool:
    return hasattr(
        torch.ops._C_qwen38,
        "nvfp4_moe_qpn_w2_reduce_sm70_out",
    ) or hasattr(torch.ops._C, "nvfp4_moe_qpn_w2_reduce_sm70_out")
