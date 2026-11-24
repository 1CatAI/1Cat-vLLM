# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode and prefill from one resident native QPN2 weight layout."""

import torch

from vllm import _sm70_ops as sm70_ops
from vllm._sm70.policy import register_policy_op
from vllm._sm70_27b_prefill_ops import (
    clear_sm70_27b_prefill_workspaces,
    drain_prefill_prefetcher,
    fast_nvfp4_prefill_out,
    has_sm70_27b_prefill_ops,
)
from vllm.model_executor.kernels.linear.qpn.nvfp4_dequant import (
    _e4m3_value,
)
from vllm.triton_utils import tl, triton

_dense_workspaces: dict[int, torch.Tensor] = {}
_retained_dense_workspaces: list[torch.Tensor] = []
_scale_workspaces: dict[int, torch.Tensor] = {}
_code_workspaces: dict[tuple[int, int], torch.Tensor] = {}


def clear_sm70_nvfp4_native_workspaces() -> None:
    clear_sm70_27b_prefill_workspaces()
    _dense_workspaces.clear()
    _retained_dense_workspaces.clear()
    _scale_workspaces.clear()
    _code_workspaces.clear()


def _engine_prefill_ceiling_elements() -> int:
    from vllm.config import get_current_vllm_config_or_none
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
        model_parallel_is_initialized,
    )

    cfg = get_current_vllm_config_or_none()
    model_cfg = getattr(cfg, "model_config", None)
    text_cfg = getattr(model_cfg, "hf_text_config", None)
    if text_cfg is None:
        return 0
    hidden = int(getattr(text_cfg, "hidden_size", 0) or 0)
    inter = int(getattr(text_cfg, "intermediate_size", 0) or 0)
    tp = (
        get_tensor_model_parallel_world_size() if model_parallel_is_initialized() else 1
    )
    if hidden <= 0 or inter <= 0 or tp <= 0:
        return 0
    return hidden * ((2 * inter + tp - 1) // tp)


def _shared_fp8_prefill_workspace(
    device: int, min_elements: int
) -> torch.Tensor | None:
    from vllm.model_executor.kernels.linear.qpn import fp8 as qpn_fp8
    from vllm.model_executor.layers.quantization.utils import sm70_layer_workspaces

    pool = sm70_layer_workspaces.workspace_pool(
        "_sm70_fp8_prefill_dense_workspaces",
        qpn_fp8._sm70_fp8_prefill_dense_workspaces,
    )
    best: torch.Tensor | None = None
    for key, candidate in pool.items():
        if (
            isinstance(key, tuple)
            and len(key) >= 2
            and key[0] == device
            and key[1] == torch.float16
            and isinstance(candidate, torch.Tensor)
            and candidate.is_cuda
            and candidate.dtype == torch.float16
            and candidate.numel() >= min_elements
            and (best is None or candidate.numel() > best.numel())
        ):
            best = candidate
    return best


def _get_native_prefill_dense_workspace(
    codes: torch.Tensor, elements: int
) -> torch.Tensor | None:
    device = codes.device.index
    assert device is not None
    target_elements = max(elements, _engine_prefill_ceiling_elements())
    shared = _shared_fp8_prefill_workspace(device, target_elements)
    if shared is not None:
        return shared
    workspace = _dense_workspaces.get(device)
    if workspace is not None and workspace.numel() >= target_elements:
        return workspace
    capturing = torch.cuda.is_current_stream_capturing()
    if workspace is not None and not capturing:
        # Release any smaller eager buffer before allocating the wider ceiling
        # so multi-projection layers never leave an orphaned scratch tensor.
        del _dense_workspaces[device]
        workspace = None
    elif workspace is not None and capturing:
        _retained_dense_workspaces.append(workspace)
    try:
        workspace = torch.empty(
            target_elements, dtype=torch.float16, device=codes.device
        )
    except torch.OutOfMemoryError:
        return None
    _dense_workspaces[device] = workspace
    return workspace


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


@torch.compiler.assume_constant_result
def _native_decode_rows() -> int:
    capacity = getattr(torch.ops._C, "nvfp4_qpn2_max_rows_sm70", None)
    return int(capacity()) if capacity is not None else 32


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
    if x.shape[0] <= _native_decode_rows():
        drain_prefill_prefetcher(x.device)
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
    workspace = _get_native_prefill_dense_workspace(codes, k * n)
    if workspace is None or workspace.numel() < k * n:
        raise RuntimeError("Native QPN2 prefill workspace is unavailable")
    device = codes.device.index
    assert device is not None
    if codes.is_contiguous() and scales.is_contiguous():
        if (
            x.shape[0] >= 256
            and x.dtype == torch.float16
            and x.is_contiguous()
            and out.is_contiguous()
            and not torch.cuda.is_current_stream_capturing()
            and has_sm70_27b_prefill_ops()
        ):
            fast_nvfp4_prefill_out(
                out, workspace, x, codes, scales, float(global_scale), gated_silu
            )
            return
        drain_prefill_prefetcher(x.device)
        call_native(
            sm70_ops.nvfp4_qpn4_prefill_sm70_out,
            native_policy,
            out,
            workspace.data_ptr(),
            x,
            codes.view(k, n // 2),
            scales.view(torch.float8_e4m3fn).view(k // 16, n),
            global_scale,
            False,
            gated_silu,
        )
        return
    scale_workspace = _scale_workspaces.get(device)
    if scale_workspace is None or scale_workspace.numel() < workspace.numel() // 16:
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


def _bundled_down_dispatch(
    out,
    x,
    codes,
    bundled_scales,
    tm_weight,
    scales,
    global_scale,
    split_k,
    accumulator_chains,
    tm_scales,
    group_size,
    k_ld,
    q_ld,
    gated_silu,
    min_prefill_m,
    prescaled_scales,
    native_policy=None,
):
    if 1 <= x.shape[0] <= 8 and not gated_silu:
        sm70_ops.nvfp4_qpn2_gemm_sm70_out(
            out,
            x,
            codes,
            bundled_scales,
            global_scale,
            split_k,
            accumulator_chains,
            native_policy=native_policy,
        )
        return
    sm70_ops.nvfp4_qpn2_tm_dispatch_sm70_out(
        out,
        x,
        tm_weight,
        scales,
        global_scale,
        split_k,
        accumulator_chains,
        tm_scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
        min_prefill_m,
        prescaled_scales,
        native_policy=native_policy,
    )


def _bundled_down_dispatch_fake(*args, **kwargs):
    return None


register_policy_op(
    "sm70_nvfp4_bundled_down_dispatch",
    "(Tensor(a!) out, Tensor x, Tensor codes, Tensor bundled_scales, "
    "Tensor tm_weight, Tensor scales, float global_scale, int split_k, "
    "int accumulator_chains, Tensor tm_scales, int group_size, int k_ld, "
    "int q_ld, bool gated_silu, int min_prefill_m, bool prescaled_scales, "
    "str[]? native_policy=None) -> ()",
    _bundled_down_dispatch,
    _bundled_down_dispatch_fake,
)
