# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scratch workspaces of prepared SM70 linear layers, resolved at run time.

Several SM70 kernels take the address of a bounded FP16 scratch workspace.
apply() used to pass `workspace.data_ptr()` as an int; TorchDynamo records it
as a constant and the AOT artifact then carries the compiling process's
address, so the next process writes through a stale pointer. Prepared layers
register their workspace here under their prefix, which is the same in every
process, and apply() calls the opaque ops below with the prefix.
"""

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# Layer prefix -> the workspace its kernels write through.
_layer_workspaces: dict[str, torch.Tensor] = {}


def register_layer_workspace(layer: torch.nn.Module, workspace: torch.Tensor) -> None:
    """Make `workspace` the scratch the opaque SM70 ops use for `layer`."""
    prefix = getattr(layer, "prefix", "")
    if not prefix:
        raise RuntimeError("SM70 workspaces are bound by layer prefix")
    bound = _layer_workspaces.get(prefix)
    if bound is not None and bound is not workspace:
        raise RuntimeError(f"{prefix} is already bound to another SM70 workspace")
    _layer_workspaces[prefix] = workspace


def clear_layer_workspaces() -> None:
    _layer_workspaces.clear()


def _sm70_fp8_qpn8_dispatch(
    out: torch.Tensor,
    layer_name: str,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    prefetch_codes: bool,
    gated_silu: bool,
) -> None:
    from vllm import _sm70_ops as sm70_ops

    sm70_ops.fp8_qpn8_dispatch_sm70_out(
        out,
        _layer_workspaces[layer_name].data_ptr(),
        x,
        codes,
        scales,
        split_k,
        accumulator_chains,
        prefetch_codes,
        gated_silu,
    )


def _sm70_fp8_qpn8_dispatch_fake(
    out: torch.Tensor,
    layer_name: str,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    prefetch_codes: bool,
    gated_silu: bool,
) -> None:
    return None


direct_register_custom_op(
    "sm70_fp8_qpn8_dispatch",
    _sm70_fp8_qpn8_dispatch,
    mutates_args=["out"],
    fake_impl=_sm70_fp8_qpn8_dispatch_fake,
)


def _sm70_fp8_qpn8_dispatch_ba_split(
    qkv_out: torch.Tensor,
    z_out: torch.Tensor,
    b_out: torch.Tensor,
    a_out: torch.Tensor,
    qkvz_staging: torch.Tensor,
    ba_staging: torch.Tensor,
    layer_name: str,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    ba_weight: torch.Tensor,
) -> None:
    from vllm import _sm70_ops as sm70_ops

    sm70_ops.fp8_qpn8_dispatch_ba_split_sm70_out(
        qkv_out,
        z_out,
        b_out,
        a_out,
        qkvz_staging,
        ba_staging,
        _layer_workspaces[layer_name].data_ptr(),
        x,
        codes,
        scales,
        ba_weight,
    )


def _sm70_fp8_qpn8_dispatch_ba_split_fake(
    qkv_out: torch.Tensor,
    z_out: torch.Tensor,
    b_out: torch.Tensor,
    a_out: torch.Tensor,
    qkvz_staging: torch.Tensor,
    ba_staging: torch.Tensor,
    layer_name: str,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    ba_weight: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    "sm70_fp8_qpn8_dispatch_ba_split",
    _sm70_fp8_qpn8_dispatch_ba_split,
    mutates_args=["qkv_out", "z_out", "b_out", "a_out", "qkvz_staging", "ba_staging"],
    fake_impl=_sm70_fp8_qpn8_dispatch_ba_split_fake,
)


def _sm70_fp8_prefill_dispatch(
    out: torch.Tensor,
    layer_name: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool,
    min_prefill_m: int,
) -> None:
    from vllm import _sm70_ops as sm70_ops

    sm70_ops.fp8_gemm_sm70_prefill_dispatch_out(
        out,
        _layer_workspaces[layer_name].data_ptr(),
        x,
        weight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
        min_prefill_m,
    )


def _sm70_fp8_prefill_dispatch_fake(
    out: torch.Tensor,
    layer_name: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool,
    min_prefill_m: int,
) -> None:
    return None


direct_register_custom_op(
    "sm70_fp8_prefill_dispatch",
    _sm70_fp8_prefill_dispatch,
    mutates_args=["out"],
    fake_impl=_sm70_fp8_prefill_dispatch_fake,
)


def _sm70_nvfp4_qpn4_dispatch(
    out: torch.Tensor,
    layer_name: str,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    use_scale_code: bool,
    gated_silu: bool,
) -> None:
    from vllm import _sm70_ops as sm70_ops

    sm70_ops.nvfp4_qpn4_dispatch_sm70_out(
        out,
        _layer_workspaces[layer_name].data_ptr(),
        x,
        codes,
        scales,
        global_scale,
        use_scale_code,
        gated_silu,
    )


def _sm70_nvfp4_qpn4_dispatch_fake(
    out: torch.Tensor,
    layer_name: str,
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_scale: float,
    use_scale_code: bool,
    gated_silu: bool,
) -> None:
    return None


direct_register_custom_op(
    "sm70_nvfp4_qpn4_dispatch",
    _sm70_nvfp4_qpn4_dispatch,
    mutates_args=["out"],
    fake_impl=_sm70_nvfp4_qpn4_dispatch_fake,
)
