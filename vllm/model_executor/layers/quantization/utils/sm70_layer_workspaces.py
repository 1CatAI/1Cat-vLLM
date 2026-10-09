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

from collections.abc import MutableMapping
from dataclasses import dataclass

import torch

from vllm._sm70.policy import NativeBindings
from vllm.utils.torch_utils import direct_register_custom_op

# Layer prefix -> the workspace its kernels write through.
_layer_workspaces: dict[str, torch.Tensor] = {}


class LayerWorkspaceView:
    """Name a family's layer-owned buffers without copying their bindings.

    The layer remains the sole owner: legacy attribute rebinding and persistent
    tensor addresses stay visible to every caller. This view creates no tensor
    allocation or global registration and does not change the single-workspace
    registry used by opaque linear operators.
    """

    __slots__ = ("_layer", "_prefix")
    _layer: torch.nn.Module
    _prefix: str

    def __init__(self, layer: torch.nn.Module, prefix: str):
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name: str):
        return getattr(self._layer, self._prefix + name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._layer, self._prefix + name, value)


@dataclass
class _WorkspaceBinding:
    workspace: torch.Tensor
    native: NativeBindings


_WORKSPACE_PREFIX = "sm70_workspace:"
_legacy_native: dict[str, NativeBindings] = {}


def _engine_workspace_registry():
    from vllm.config import get_current_vllm_config_or_none
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if is_forward_context_available():
        return get_forward_context().no_compile_layers
    config = get_current_vllm_config_or_none()
    return config.compilation_config.static_forward_context if config else None


def workspace_pool(name: str, legacy: MutableMapping) -> MutableMapping:
    """Share bounded scratch within an engine, never between live engines."""
    registry = _engine_workspace_registry()
    if registry is None:
        return legacy
    return registry.setdefault(_WORKSPACE_PREFIX + "pool:" + name, {})


def register_layer_workspace(
    layer: torch.nn.Module, workspace: torch.Tensor, *, family: str | None = None
) -> None:
    """Bind addresses and native policy to this engine's existing context."""
    from vllm.config.sm70_native import capture_linear_native_config

    prefix = getattr(layer, "prefix", "")
    if not prefix:
        raise RuntimeError("SM70 workspaces are bound by layer prefix")
    values = capture_linear_native_config(family).values if family else ()
    binding = _WorkspaceBinding(workspace, NativeBindings(values))
    registry = _engine_workspace_registry()
    if registry is None:
        bound = _layer_workspaces.get(prefix)
        if bound is not None and bound is not workspace:
            raise RuntimeError(f"{prefix} is already bound to another SM70 workspace")
        _layer_workspaces[prefix] = workspace
        _legacy_native[prefix] = binding.native
        return
    key = _WORKSPACE_PREFIX + prefix
    bound = registry.get(key)
    if bound is not None and bound.workspace is not workspace:
        raise RuntimeError(f"{prefix} is already bound to another SM70 workspace")
    registry[key] = binding


def _workspace_binding(prefix: str) -> _WorkspaceBinding:
    registry = _engine_workspace_registry()
    if registry is not None:
        return registry[_WORKSPACE_PREFIX + prefix]
    return _WorkspaceBinding(_layer_workspaces[prefix], _legacy_native[prefix])


def clear_layer_workspaces() -> None:
    registry = _engine_workspace_registry()
    if registry is not None:
        for key in tuple(registry):
            if key.startswith(_WORKSPACE_PREFIX):
                del registry[key]
    else:
        _layer_workspaces.clear()
        _legacy_native.clear()


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
    binding = _workspace_binding(layer_name)

    binding.native.fp8_qpn8_dispatch_sm70_out(
        out,
        binding.workspace.data_ptr(),
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
    binding = _workspace_binding(layer_name)

    binding.native.fp8_qpn8_dispatch_ba_split_sm70_out(
        qkv_out,
        z_out,
        b_out,
        a_out,
        qkvz_staging,
        ba_staging,
        binding.workspace.data_ptr(),
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
    binding = _workspace_binding(layer_name)

    binding.native.fp8_gemm_sm70_prefill_dispatch_out(
        out,
        binding.workspace.data_ptr(),
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
    binding = _workspace_binding(layer_name)

    binding.native.nvfp4_qpn4_dispatch_sm70_out(
        out,
        binding.workspace.data_ptr(),
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
