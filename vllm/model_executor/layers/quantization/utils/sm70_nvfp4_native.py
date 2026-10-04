# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode and prefill from one resident native QPN2 weight layout."""

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.kernels.linear.scaled_mm.sm70_fp8 import (
    _get_sm70_fp8_prefill_exact_dense_workspace,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_qpn2_dequant import (
    _e4m3_value,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_scale_workspaces: dict[int, torch.Tensor] = {}


@triton.jit
def _restore_prefill_scales(Codes, Out, GlobalScale, Count: tl.constexpr):
    index = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    raw = tl.load(Codes + index, index < Count, other=0)
    # Match decode's FP32 group/global product followed by FP16 rounding.
    # QPN4's existing scale-code converter assumes normal, nonzero E4M3
    # scales and combines the global factor differently. Padded zero scales
    # and subnormals must retain their exact weights on the new prefill route.
    effective = (_e4m3_value(raw) * GlobalScale).to(tl.float16).to(tl.float32)
    tl.store(Out + index, effective * 16384.0, index < Count)


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
        # Retune only the single-request gated shape; preserve the admitted
        # two-chain batch reductions for concurrent requests.
        if (
            x.shape[0] <= 8
            and gated_silu
            and x.shape[1] == 5120
            and out.shape[1] == 4352
        ):
            accumulator_chains = 1
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
    device = codes.device.index
    assert device is not None
    scale_workspace = _scale_workspaces.get(device)
    if scale_workspace is None:
        scale_workspace = torch.empty(
            workspace.numel() // 16, dtype=torch.float16, device=codes.device
        )
        _scale_workspaces[device] = scale_workspace
    _restore_prefill_scales[(triton.cdiv(scales.numel(), 1024),)](
        scales.view(torch.uint8),
        scale_workspace,
        global_scale,
        scales.numel(),
    )
    # Resolve the pointer inside the operator, never in an AOT artifact. The
    # serialized layer chain shares FP8's bounded per-device FP16 scratch.
    sm70_ops.nvfp4_qpn4_prefill_sm70_out(
        out,
        workspace.data_ptr(),
        x,
        codes.view(k, n // 2),
        scale_workspace[: k * n // 16].view(k // 16, n),
        global_scale,
        False,
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
