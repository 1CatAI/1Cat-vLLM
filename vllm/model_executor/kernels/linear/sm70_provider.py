# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared linear operation binding and shared input/output handling.

The existing kernel selectors admit providers. Weight preparation binds one
provider here; live M dispatch remains inside the existing opaque operators.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch


def _gemm(state, x, out, prefix, *, name, gated=False):
    args = (
        out,
        x,
        state.weight,
        state.scales,
        state.group_size,
        state.k_ld,
        state.q_ld,
    )
    if gated:
        getattr(state.native_ops, name)(*args, True)
    else:
        getattr(state.native_ops, name)(*args)
    return out


def _compact(state, x, out, prefix, *, gated=False):
    state.native_ops.nvfp4_qpn2_compact_tm_gemm_sm70_out(
        out, x, state.weight, state.scales, state.global_scale, state.k_ld, state.q_ld
    )
    return out


def _qpn4(state, x, out, prefix, *, gated=False):
    torch.ops.vllm.sm70_nvfp4_qpn4_dispatch(
        out,
        prefix,
        x,
        state.weight,
        state.scales,
        state.global_scale,
        state.use_scale_code,
        gated,
    )
    return out


def _qpn2_dense(state, x, out, prefix, *, gated=False):
    from vllm.model_executor.kernels.linear.qpn import nvfp4_dequant

    return nvfp4_dequant.nvfp4_qpn2_dispatch_linear(
        x,
        state.weight,
        state.scales,
        state.global_scale,
        out.shape[1],
        x.shape[1],
        state.split_k,
        state.accumulator_chains,
        state.native_ops.arguments,
    )


@dataclass(frozen=True)
class PreparedLinearProvider:
    """Weight-layout-specific binding, with no environment or selector access."""

    name: str
    run: Callable[..., torch.Tensor]
    fused: Callable[..., torch.Tensor] | None = None
    contiguous_input: bool = False
    activation_error: str | None = None
    restore_interleaved: bool = False


def bind_prepared_provider(state: Any) -> PreparedLinearProvider:
    """Resolve once after the existing preparation selected the layout."""
    kind = state.op_kind
    if kind == "nvfp4_qpn4":
        return PreparedLinearProvider(
            "nvfp4_qpn4",
            _qpn4,
            _qpn4,
            True,
            "SM70 NVFP4 QPN4 requires float16 activations, got {dtype}.",
        )
    if kind == "nvfp4_qpn2_dense":
        return PreparedLinearProvider(
            "nvfp4_qpn2_dense",
            _qpn2_dense,
            contiguous_input=True,
            activation_error=(
                "The pre-Ampere NVFP4 QPN2 path requires float16 "
                "activations, got {dtype}."
            ),
        )
    if kind == "nvfp4" and state.use_scale_code:
        # The old fused entry uses the regular/prescaled NVFP4 binding even
        # when its non-fused entry has the compact scale-code layout.
        name = (
            "nvfp4_gemm_sm70_prescaled_out"
            if state.prescaled_scales
            else "nvfp4_gemm_sm70_out"
        )
        return PreparedLinearProvider(
            "nvfp4_compact",
            _compact,
            partial(_gemm, name=name),
            restore_interleaved=state.gated_silu,
        )
    names = {
        "uint4": "awq_gemm_sm70_out",
        "fp8": "fp8_gemm_sm70_out",
        "mxfp4": "mxfp4_gemm_sm70_out",
        "nvfp4": "nvfp4_gemm_sm70_prescaled_out"
        if state.prescaled_scales
        else "nvfp4_gemm_sm70_out",
    }
    if kind not in names:
        raise AssertionError(f"unknown SM70 TurboMind op kind: {kind}")
    run = partial(_gemm, name=names[kind])
    return PreparedLinearProvider(
        names[kind],
        run,
        run if kind == "nvfp4" else None,
        restore_interleaved=state.gated_silu and kind in ("uint4", "nvfp4"),
    )


def apply_prepared(state, x, bias, prefix, *, gated=False):
    """Shared flatten, dispatch, crop, layout restoration, bias and reshape."""
    provider = state.provider
    if gated:
        if not state.gated_silu or provider.fused is None:
            return None
        if x.dtype != torch.float16:
            raise RuntimeError(
                "SM70 TurboMind NVFP4 gated-SiLU requires float16 activations, "
                f"got {x.dtype}."
            )
    elif provider.activation_error and x.dtype != torch.float16:
        raise RuntimeError(provider.activation_error.format(dtype=x.dtype))
    x_2d = x.reshape(-1, x.shape[-1])
    if (gated or provider.contiguous_input) and x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    logical_n = state.output_size // 2 if gated else state.output_size
    # Fused preparation has its own exact layout; preserve the old allocation.
    physical_n = logical_n if gated else (state.padded_output_size or logical_n)
    out = torch.empty((x_2d.shape[0], physical_n), dtype=x.dtype, device=x.device)
    if gated and x_2d.shape[0] == 0:
        return out.reshape(*x.shape[:-1], logical_n)
    run = provider.fused if gated else provider.run
    out = run(state, x_2d, out, prefix, gated=gated)
    if physical_n != logical_n:
        out = out[:, :logical_n]
    if not gated and provider.restore_interleaved:
        out = (
            out.reshape(x_2d.shape[0], logical_n // 2, 2)
            .transpose(1, 2)
            .reshape(x_2d.shape[0], logical_n)
        )
    if bias is not None:
        out.add_(bias)
    return out.reshape(*x.shape[:-1], logical_n)
