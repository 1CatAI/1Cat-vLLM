# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode and prefill from one resident native QPN2 weight layout."""

import torch

from vllm import _sm70_ops as sm70_ops
from vllm._sm70.policy import register_policy_op
from vllm.model_executor.kernels.linear.qpn.fp8 import (
    _get_sm70_fp8_prefill_exact_dense_workspace,
)
from vllm.model_executor.kernels.linear.qpn.nvfp4_dequant import (
    _e4m3_value,
)
from vllm.triton_utils import tl, triton

_scale_workspaces: dict[int, torch.Tensor] = {}
_code_workspaces: dict[tuple[int, int], torch.Tensor] = {}


def clear_sm70_nvfp4_native_workspaces() -> None:
    _scale_workspaces.clear()
    _code_workspaces.clear()


def _get_bundled_prefill_code_workspace(
    weight: torch.Tensor, dense_workspace: torch.Tensor
) -> torch.Tensor | None:
    # Reserve before loading bundled weights. All serialized layers and their
    # captured graphs share one compact code buffer, rather than retaining a
    # second weight layout for each layer/graph.
    device = weight.device.index
    assert device is not None
    elements = dense_workspace.numel() // 2
    key = (device, elements)
    workspace = _code_workspaces.get(key)
    if workspace is None:
        try:
            workspace = torch.empty(elements, dtype=torch.uint8, device=weight.device)
        except torch.OutOfMemoryError:
            return None
        _code_workspaces[key] = workspace
    return workspace


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


@triton.jit
def _restore_bundled_prefill_operands(
    Codes,
    Scales,
    OutCodes,
    OutScales,
    GlobalScale,
    CodeCount: tl.constexpr,
    ScaleCount: tl.constexpr,
):
    index = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    code = tl.load(Codes + index // 256 * 288 + index % 256, index < CodeCount, other=0)
    tl.store(OutCodes + index, code, index < CodeCount)
    if tl.program_id(0) * 1024 < ScaleCount:
        raw = tl.load(
            Scales + index // 32 * 288 + index % 32, index < ScaleCount, other=0
        )
        effective = (_e4m3_value(raw) * GlobalScale).to(tl.float16).to(tl.float32)
        tl.store(OutScales + index, effective * 16384.0, index < ScaleCount)


def _dispatch(
    out: torch.Tensor,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    split_k: int,
    accumulator_chains: int,
    gated_silu: bool,
    native_policy: list[str] | None = None,
) -> None:
    from vllm._sm70.policy import call_native

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
        call_native(
            op,
            native_policy,
            out,
            x,
            codes,
            scales,
            global_scale,
            split_k,
            accumulator_chains,
        )
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
    if codes.is_contiguous():
        compact_codes = codes
        _restore_prefill_scales[(triton.cdiv(scales.numel(), 1024),)](
            scales.view(torch.uint8), scale_workspace, global_scale, scales.numel()
        )
    else:
        code_workspace = _get_bundled_prefill_code_workspace(codes, workspace)
        if code_workspace is None:
            raise RuntimeError("Bundled QPN2 prefill workspace is unavailable")
        compact_codes = code_workspace[: codes.numel()]
        _restore_bundled_prefill_operands[(triton.cdiv(codes.numel(), 1024),)](
            codes,
            scales,
            compact_codes,
            scale_workspace,
            global_scale,
            codes.numel(),
            scales.numel(),
        )
    # Resolve the pointer inside the operator, never in an AOT artifact. The
    # serialized layer chain shares FP8's bounded per-device FP16 scratch.
    call_native(
        sm70_ops.nvfp4_qpn4_prefill_sm70_out,
        native_policy,
        out,
        workspace.data_ptr(),
        x,
        compact_codes.view(k, n // 2),
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
    native_policy: list[str] | None = None,
) -> None:
    return None


register_policy_op(
    "sm70_nvfp4_native_dispatch",
    "(Tensor(a!) out, Tensor x, Tensor codes, Tensor scales, float global_scale, "
    "int split_k, int accumulator_chains, bool gated_silu, "
    "str[]? native_policy=None) -> ()",
    _dispatch,
    _dispatch_fake,
)
