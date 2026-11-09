# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import kernels
import vllm.kernels  # noqa: F401
from vllm import envs, ir
from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.config import get_current_vllm_config
from vllm.config.execution_policy import graph_policy, layer_policy
from vllm.config.sm70_dflash2 import (
    resolved_sm70_dflash2_config,
)
from vllm.config.vllm import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.kernels.norm import sm70 as _sm70_provider
from vllm.model_executor.kernels.norm.sm70 import (
    _sm70_gated_norm_device_supported,
    _sm70_gated_norm_shape_supported,
    maybe_gemma_norm,
    use_long_prefill_fused,
)
from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant
from vllm.platforms import current_platform

logger = init_logger(__name__)


def poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    out = torch.empty_like(x)
    ops.poly_norm(  # type: ignore[attr-defined]
        out,
        x,
        weight,
        bias,
        variance_epsilon,
    )
    return out


# --8<-- [start:rms_norm]
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # --8<-- [end:rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

        # Do not pass identity weight to native implementation (causes issue on TPU).
        # Other implementations require weight to be passed even if all ones.
        # Cheat and predict if native will be dispatched to:
        #  1) if native is first in priority list
        #  2) if variance_size_override is given (only supported by native impl)
        # TODO(luka): address weight passing inconsistency:
        # https://github.com/vllm-project/vllm/issues/39370
        priority = get_current_vllm_config().kernel_config.ir_op_priority
        var_override = self.variance_size_override is not None
        native_rms_norm = priority.rms_norm[0] == "native" or var_override
        native_add_rms_norm = priority.fused_add_rms_norm[0] == "native" or var_override
        self.pass_weight = self.has_weight or not native_rms_norm
        self.pass_weight_add = self.has_weight or not native_add_rms_norm

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if residual is None:
            return ir.ops.rms_norm(
                x,
                self.weight.data if self.pass_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        else:
            return ir.ops.fused_add_rms_norm.maybe_inplace(
                x,
                residual,
                self.weight.data if self.pass_weight_add else None,
                self.variance_epsilon,
                self.variance_size_override,
            )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            envs.VLLM_BATCH_INVARIANT
            and residual is None
            and self.variance_size_override is None
        ):
            return rms_norm_batch_invariant(x, self.weight.data, self.variance_epsilon)

        return self.forward_native(x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# --8<-- [start:gemma_rms_norm]
@CustomOp.register("gemma_rms_norm")
class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    # --8<-- [end:gemma_rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self._sm70_dflash2_policy = resolved_sm70_dflash2_config()
        self._execution_graph_policy = graph_policy()
        self._layer_policy = layer_policy()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def _forward_static_no_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        return x.to(orig_dtype)

    @staticmethod
    def _forward_static_with_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = (
            x.float() + residual.float()
            if orig_dtype == torch.float16
            else x + residual
        )
        residual = x
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        return x.to(orig_dtype), residual

    def _use_sm70_compile_native(self, x: torch.Tensor) -> bool:
        return (
            bool(self._layer_policy.gemma_compile_native)
            and bool(self._execution_graph_policy.compile_graph)
            and use_sm70_decode_graph_semantics(self._execution_graph_policy)
            and torch.compiler.is_compiling()
            and x.is_cuda
        )

    def _use_sm70_long_prefill_fused(self, x, residual):
        return use_long_prefill_fused(self._layer_policy, self.weight, x, residual)

    def _maybe_sm70_forward(self, x, residual):
        return maybe_gemma_norm(
            x,
            residual,
            self.weight,
            self.variance_epsilon,
            policy=self._layer_policy,
            dflash=self._sm70_dflash2_policy,
            graph=self._execution_graph_policy,
        )

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        result = self._maybe_sm70_forward(x, residual)
        if result is not None:
            return result
        return self._forward_native_tail(x, residual)

    def _forward_native_tail(self, x, residual):
        if self._use_sm70_compile_native(x):
            if residual is None:
                return self._forward_static_no_residual(
                    self.weight.data,
                    self.variance_epsilon,
                    x,
                )
            return self._forward_static_with_residual(
                self.weight.data,
                self.variance_epsilon,
                x,
                residual,
            )

        orig_dtype = x.dtype
        weight = self.weight.data.float() + 1.0
        if residual is not None:
            x = (
                x.float() + residual.float()
                if orig_dtype == torch.float16
                else x + residual
            )
            residual = x
        # ir.ops.rms_norm handles fp32 upcast internally
        out = ir.ops.rms_norm(x, weight, self.variance_epsilon)
        return (
            out.to(orig_dtype) if residual is None else (out.to(orig_dtype), residual)
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        result = self._maybe_sm70_forward(x, residual)
        if result is not None:
            return result
        if (
            self._layer_policy.gemma_eager
            and bool(self._execution_graph_policy.compile_graph)
            and x.is_cuda
            and current_platform.is_device_capability(70)
        ):
            if residual is None:
                return torch.ops.vllm.sm70_gemma_rms_norm_eager(
                    x,
                    self.weight,
                    self.variance_epsilon,
                )
            return torch.ops.vllm.sm70_gemma_fused_add_rms_norm_eager(
                x,
                residual,
                self.weight,
                self.variance_epsilon,
            )
        return self._forward_native_tail(x, residual)


# --8<-- [start:rms_norm_gated]
@CustomOp.register("rms_norm_gated")
class RMSNormGated(CustomOp):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

    # --8<-- [end:rms_norm_gated]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        activation: str = "swish",
    ):
        """Initialize RMSNormGated.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            group_size: If not None, do GroupNorm with each group
                        having group_size elements.
                        group_size=None is equivalent to group_size=hidden_size
                        (i.e. there's only 1 group).
            norm_before_gate: If True and z is provided: out = norm(x) * silu(z)
                              If False and z is provided: out = norm(x * silu(z))
            device: Device to create parameters on
            dtype: Data type for parameters
            activation: Activation function name for gating
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        cfg = get_current_vllm_config_or_none()
        resolved = cfg.kernel_config.sm70_rmsnorm_gated_exact if cfg else None
        self._sm70_rmsnorm_gated_exact = (
            bool(resolved)
            if resolved is not None
            else bool(envs.VLLM_SM70_RMSNORM_GATED_EXACT)
        )
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    @staticmethod
    def forward_static(
        x: torch.Tensor,
        z: torch.Tensor | None,
        weight: torch.Tensor,
        epsilon: float,
        orig_dtype: torch.dtype,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        activation: str = "swish",
    ) -> torch.Tensor:
        """Pure-PyTorch RMS normalization with optional gating.

        This static method contains the full native logic so that both
        ``forward_native`` and ``MatcherRMSNormGated`` (used by the
        compilation pattern matcher) can share the same implementation.

        If *z* is not None and *norm_before_gate* is True:
            ``out = rms_norm(x) * act(z)``
        If *z* is not None and *norm_before_gate* is False:
            ``out = rms_norm(x * act(z))``
        """
        x = x.float()
        weight = weight.float()
        if z is not None:
            z = z.float()

        assert activation in ["silu", "sigmoid", "swish"]
        act_fn = F.sigmoid if activation == "sigmoid" else F.silu

        if z is not None and not norm_before_gate:
            x = x * act_fn(z)

        if group_size is None:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + epsilon)
            out = x_normed * weight
        else:
            from einops import rearrange

            x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + epsilon)
            out = rearrange(x_normed, "... g d -> ... (g d)") * weight

        if z is not None and norm_before_gate:
            out = out * act_fn(z)

        return out.to(orig_dtype)

    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        if (
            self._sm70_rmsnorm_gated_exact
            and not envs.VLLM_BATCH_INVARIANT
            and x.is_cuda
            and _sm70_gated_norm_device_supported(x.device.index)
            and self.group_size is None
            and self.norm_before_gate
            and self.activation in ("sigmoid", "silu", "swish")
            and _sm70_gated_norm_shape_supported(x, z, self.weight)
        ):
            return torch.ops.vllm.sm70_rmsnorm_gated_exact(
                x, z, self.weight, self.eps, self.activation != "sigmoid"
            )
        return self.forward_static(
            x,
            z,
            self.weight,
            self.eps,
            x.dtype,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

        return rmsnorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_xpu(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_cuda(x, z)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)


def __getattr__(name):
    if name.startswith(("_sm70_", "_use_sm70_")):
        return getattr(_sm70_provider, name)
    raise AttributeError(name)
