# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode and prefill from one resident native QPN2 weight layout."""

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.kernels.linear.scaled_mm.sm70_fp8 import (
    _get_sm70_fp8_prefill_exact_dense_workspace,
)
from vllm.utils.torch_utils import direct_register_custom_op


def _dispatch(
    out: torch.Tensor,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    split_k: int,
    accumulator_chains: int,
    gated_silu: bool,
) -> None:
    # Keep M dispatch opaque: compiled token ranges include both decode and
    # prefill. Native QPN2 codes and E4M3 scales are the only resident layout.
    if x.shape[0] <= 32:
        op = (
            sm70_ops.nvfp4_qpn2_gated_sm70_out
            if gated_silu
            else sm70_ops.nvfp4_qpn2_gemm_sm70_out
        )
        op(out, x, codes, scales, global_scale, split_k, accumulator_chains)
        return
    k = x.shape[1]
    n = out.shape[1] * (2 if gated_silu else 1)
    workspace = _get_sm70_fp8_prefill_exact_dense_workspace(codes)
    if workspace is None or workspace.numel() < k * n:
        raise RuntimeError("Native QPN2 prefill workspace is unavailable")
    # Resolve the pointer inside the operator, never in an AOT artifact. The
    # serialized layer chain shares FP8's bounded per-device FP16 scratch.
    sm70_ops.nvfp4_qpn4_prefill_sm70_out(
        out,
        workspace.data_ptr(),
        x,
        codes.view(k, n // 2),
        scales.view(k // 16, n),
        global_scale,
        True,
        gated_silu,
    )


def _dispatch_fake(
    out: torch.Tensor,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    split_k: int,
    accumulator_chains: int,
    gated_silu: bool,
) -> None:
    return None


direct_register_custom_op(
    "sm70_nvfp4_native_dispatch",
    _dispatch,
    mutates_args=["out"],
    fake_impl=_dispatch_fake,
)
