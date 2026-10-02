# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 weight-only NVFP4 kernels using the common linear kernel lifecycle."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.parameter import Parameter

from vllm import _sm70_ops as sm70_ops
from vllm.config.kernel import Sm70NvFp4Config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


@dataclass
class Sm70NvFp4LinearLayerConfig(NvFp4LinearLayerConfig):
    input_size: int
    output_size: int
    weight_shape: tuple[int, ...]
    policy: Sm70NvFp4Config
    qpn2_qualified: bool
    qpn4_qualified: bool
    gated_silu: bool
    dense_gated_silu: bool = False


def compact_scales_allowed(policy: Sm70NvFp4Config) -> bool:
    if sm70_tm.use_batched_gemm_layouts() or not policy.shared_scales:
        return False
    version = getattr(torch.ops._C, "nvfp4_qpn2_compact_scales_version_sm70", None)
    return bool(
        hasattr(torch.ops._C, "nvfp4_qpn2_compact_tm_gemm_sm70_out")
        and version is not None
        and version() >= 1
        and policy.qualified
    )


_SM70_NVFP4_QPN4_REQUIRED_OPS = (
    "nvfp4_qpn4_prepare_sm70",
    "nvfp4_qpn4_prepare_scale_code_sm70",
    "nvfp4_qpn4_dequantize_sm70_out",
    "nvfp4_qpn4_prefill_sm70_out",
    "nvfp4_qpn4_dispatch_sm70_out",
)


_SM70_NVFP4_QPN2_CONFIGS = {
    # (K, N, fused gated-SiLU): (split-K, independent accumulator chains)
    (1536, 5120, False): (8, 2),
    (5120, 3584, False): (16, 2),
    # Qwen3.8 GDN qkvzba is logically N=4120 on TP4.  QPN2 consumes the
    # zero-padded physical N=4128 layout and the caller crops the result.
    (5120, 4128, False): (16, 2),
    (5120, 8704, False): (8, 2),
    (5120, 8704, True): (8, 2),
    (4352, 5120, False): (16, 2),
}
_SM70_NVFP4_QPN2_REQUIRED_OPS = (
    "nvfp4_qpn2_prepare_sm70",
    "nvfp4_qpn2_gemm_sm70_out",
    "nvfp4_qpn2_gated_sm70_out",
    "nvfp4_qpn2_dispatch_sm70_out",
)
_SM70_NVFP4_QPN2_PREFILL_REQUIRED_OPS = ("nvfp4_qpn2_prefill_dispatch_sm70_out",)


def _missing_sm70_nvfp4_qpn4_ops() -> list[str]:
    return [
        name
        for name in _SM70_NVFP4_QPN4_REQUIRED_OPS
        if not hasattr(torch.ops._C, name)
    ]


def _qpn2_config(k: int, n: int, gated: bool) -> tuple[int, int]:
    # Keep existing tuned configurations; other aligned local projections use
    # the same native kernels with a split count that divides K/16.
    return _SM70_NVFP4_QPN2_CONFIGS.get(
        (k, n, gated), (8 if gated or k % 256 else 16, 2)
    )


def _pad_qpn2_output_rows(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad checkpoint-native output rows to QPN2's 32-column contract."""
    logical_n = weight.shape[0]
    physical_n = (logical_n + 31) // 32 * 32
    if physical_n == logical_n:
        return weight, scales, physical_n
    padded_weight = weight.new_zeros((physical_n, weight.shape[1]))
    padded_scales = scales.new_zeros((physical_n, scales.shape[1]))
    padded_weight[:logical_n].copy_(weight)
    padded_scales[:logical_n].copy_(scales)
    return padded_weight, padded_scales, physical_n


def _missing_qpn2_ops() -> list[str]:
    return [
        name
        for name in _SM70_NVFP4_QPN2_REQUIRED_OPS
        if not hasattr(torch.ops._C, name)
    ]


def _missing_qpn2_shared_ops() -> list[str]:
    return [
        name
        for name in (
            "nvfp4_qpn2_prepare_scales_sm70",
            "nvfp4_qpn2_tm_dispatch_sm70_out",
        )
        if not hasattr(torch.ops._C, name)
    ]


def _missing_qpn2_prefill_ops() -> list[str]:
    return [
        name
        for name in _SM70_NVFP4_QPN2_PREFILL_REQUIRED_OPS
        if not hasattr(torch.ops._C, name)
    ]


class TurboMindNvFp4LinearKernel(NvFp4LinearKernel):
    config: Sm70NvFp4LinearLayerConfig

    @classmethod
    def is_supported(cls, compute_capability=None):
        if compute_capability is None:
            capability = current_platform.get_device_capability()
            compute_capability = capability.to_int() if capability is not None else None
        if compute_capability not in (70, 72):
            return False, "requires SM70 or SM72 (Volta)"
        return True, None

    @classmethod
    def can_implement(cls, config):
        if not isinstance(config, Sm70NvFp4LinearLayerConfig):
            return False, "requires weight-only compressed-tensors NVFP4 layout"
        return True, None

    def process_weights_after_loading(self, layer):
        sm70_tm.prepare_nvfp4_linear(
            layer, interleave_gated_silu=self.config.dense_gated_silu
        )
        self._release_checkpoint_parameters(layer)

    @staticmethod
    def _release_checkpoint_parameters(layer):
        layer.weight = Parameter(
            torch.empty(0, dtype=torch.uint8, device=layer.weight.device),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            torch.empty(0, dtype=torch.float8_e4m3fn, device=layer.weight_scale.device),
            requires_grad=False,
        )

    def apply_weights(self, layer, x, bias=None):
        return sm70_tm.apply_prepared_linear(layer, x, bias)

    def apply_fused_silu_and_mul(self, layer, x):
        return sm70_tm.apply_prepared_fused_silu_and_mul(layer, x)


class Qpn2NvFp4LinearKernel(TurboMindNvFp4LinearKernel):
    @classmethod
    def can_implement(cls, config):
        supported, reason = super().can_implement(config)
        if not supported:
            return supported, reason
        if len(config.weight_shape) != 2:
            return False, "packed weight must be rank two"
        n, packed_k = config.weight_shape
        k = packed_k * 2
        if k <= 0 or k % 128 or n <= 0:
            return False, "requires K > 0, K % 128 == 0 and N > 0"
        if (config.input_size, config.output_size) != (k, n):
            return False, "local input/output dimensions disagree with packed weight"
        if config.gated_silu and n % 64:
            return False, "fused gate/up requires N % 64 == 0"
        missing = _missing_qpn2_ops()
        if missing:
            return False, f"missing native QPN2 operators: {missing}"
        return True, None

    def process_weights_after_loading(self, layer):
        compact_scales = compact_scales_allowed(self.config.policy)
        qpn2_shared = bool(self.config.policy.shared_weight)
        if qpn2_shared and (missing_shared_ops := _missing_qpn2_shared_ops()):
            logger.warning_once(
                "SM70 NVFP4 shared QPN2 weights are unavailable; "
                "retaining separate layouts. Missing ops: %s.",
                str(missing_shared_ops),
            )
            qpn2_shared = False
        if qpn2_shared:
            qpn2_output_size = (layer.weight.shape[0] + 31) // 32 * 32
            qpn2_scales = sm70_ops.nvfp4_qpn2_prepare_scales_sm70(
                layer.weight_scale.data
            )
        else:
            qpn2_weight, qpn2_weight_scale, qpn2_output_size = _pad_qpn2_output_rows(
                layer.weight.data, layer.weight_scale.data
            )
            qpn2_codes, qpn2_scales = sm70_ops.nvfp4_qpn2_prepare_sm70(
                qpn2_weight, qpn2_weight_scale
            )
        qpn2_global_scale = float(layer.weight_global_scale.item())
        qpn2_prefill_enabled = False
        if self.config.policy.prefill:
            missing_prefill_ops = [] if qpn2_shared else _missing_qpn2_prefill_ops()
            if missing_prefill_ops:
                logger.warning_once(
                    "The requested SM70 NVFP4 QPN2-packed prefill "
                    "route is unavailable; retaining TurboMind for "
                    f"large M. Missing ops: {missing_prefill_ops}."
                )
            else:
                qpn2_prefill_enabled = True
        sm70_tm.prepare_nvfp4_linear(
            layer,
            interleave_gated_silu=False,
            prescale_for_batch=(
                qpn2_shared
                and sm70_tm.use_batched_gemm_layouts()
                and not (qpn2_shared and compact_scales)
            ),
        )
        gated = self.config.gated_silu
        k = layer.input_size_per_partition
        n = qpn2_output_size
        split_k, nacc = _qpn2_config(k, n, False)
        if not qpn2_shared:
            layer.register_buffer("sm70_nvfp4_qpn2_codes", qpn2_codes, persistent=False)
        layer.register_buffer("sm70_nvfp4_qpn2_scales", qpn2_scales, persistent=False)
        layer.sm70_nvfp4_qpn2 = True
        layer.sm70_nvfp4_qpn2_shared_weight = qpn2_shared
        layer.sm70_nvfp4_qpn2_global_scale = qpn2_global_scale
        layer.sm70_nvfp4_qpn2_output_size = qpn2_output_size
        layer.sm70_nvfp4_qpn2_split_k = split_k
        layer.sm70_nvfp4_qpn2_nacc = nacc
        layer.sm70_nvfp4_qpn2_gated_silu = gated
        layer.sm70_nvfp4_qpn2_prefill_enabled = qpn2_prefill_enabled
        if qpn2_shared and compact_scales:
            state = getattr(layer, sm70_tm.STATE_ATTR)
            state.scales = qpn2_scales
            state.global_scale = qpn2_global_scale
            state.use_scale_code = True
            logger.info_once(
                "SM70 QPN2 retains E4M3 scales only; TurboMind restores "
                "shared FP16 scratch for fallback shapes."
            )
        elif qpn2_shared and sm70_tm.use_batched_gemm_layouts():
            logger.info_once(
                "SM70 batched NVFP4 keeps load-time FP16 TurboMind "
                "scales for M>32 while QPN2 retains compact E4M3 "
                "scales for small M."
            )
        logger.info_once(
            "SM70 NVFP4 QPN2 M<=32 route enabled for a compatible "
            "local projection layout contract."
        )
        if qpn2_shared:
            logger.info_once(
                "SM70 NVFP4 QPN2 shares TurboMind 4-bit weights; "
                "only QPN2 E4M3 scales are stored separately."
            )
        if qpn2_prefill_enabled:
            logger.info_once(
                "SM70 NVFP4 opaque QPN2 decode plus QPN2-packed "
                "ephemeral FP16 prefill dispatch enabled for M>=%d.",
                self.config.policy.prefill_min_m,
            )
        layer.sm70_nvfp4_qpn2_prefill_min_m = self.config.policy.prefill_min_m
        self._release_checkpoint_parameters(layer)

    def apply_weights(self, layer, x, bias=None):
        return self.apply_qpn2(layer, x, bias, gated_silu=False)

    def apply_fused_silu_and_mul(self, layer, x):
        if not layer.sm70_nvfp4_qpn2_gated_silu:
            return sm70_tm.apply_prepared_fused_silu_and_mul(layer, x)
        return self.apply_qpn2(layer, x, None, gated_silu=True)

    @staticmethod
    def apply_qpn2(
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        gated_silu: bool,
    ) -> torch.Tensor:
        # Linear layers attach their prepared layout and tuning metadata during
        # loading; torch.nn.Module does not describe these dynamic fields.
        if x.dtype != torch.float16:
            raise RuntimeError(
                f"SM70 NVFP4 QPN2 requires float16 activations, got {x.dtype}."
            )
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        logical_output_size = layer.output_size_per_partition
        kernel_output_size = int(
            getattr(layer, "sm70_nvfp4_qpn2_output_size", logical_output_size)
        )
        if gated_silu:
            logical_output_size //= 2
            kernel_output_size //= 2
        out_2d = torch.empty(
            (x_2d.shape[0], kernel_output_size), dtype=x.dtype, device=x.device
        )
        if x_2d.shape[0] == 0:
            return out_2d[:, :logical_output_size].reshape(
                *x.shape[:-1], logical_output_size
            )
        state = getattr(layer, sm70_tm.STATE_ATTR)
        split_k = int(layer.sm70_nvfp4_qpn2_split_k)
        nacc = int(layer.sm70_nvfp4_qpn2_nacc)
        if gated_silu:
            split_k, nacc = _qpn2_config(x_2d.shape[1], kernel_output_size * 2, True)
        if getattr(layer, "sm70_nvfp4_qpn2_shared_weight", False):
            min_prefill_m = (
                layer.sm70_nvfp4_qpn2_prefill_min_m
                if layer.sm70_nvfp4_qpn2_prefill_enabled
                else 0
            )
            sm70_ops.nvfp4_qpn2_tm_dispatch_sm70_out(
                out_2d,
                x_2d,
                state.weight,
                layer.sm70_nvfp4_qpn2_scales,
                float(layer.sm70_nvfp4_qpn2_global_scale),
                split_k,
                nacc,
                state.scales,
                state.group_size,
                state.k_ld,
                state.q_ld,
                gated_silu,
                min_prefill_m,
                state.prescaled_scales,
            )
        elif getattr(layer, "sm70_nvfp4_qpn2_prefill_enabled", False):
            sm70_ops.nvfp4_qpn2_prefill_dispatch_sm70_out(
                out_2d,
                x_2d,
                layer.sm70_nvfp4_qpn2_codes,
                layer.sm70_nvfp4_qpn2_scales,
                float(layer.sm70_nvfp4_qpn2_global_scale),
                split_k,
                nacc,
                state.weight,
                state.scales,
                state.group_size,
                state.k_ld,
                state.q_ld,
                gated_silu,
                layer.sm70_nvfp4_qpn2_prefill_min_m,
            )
        else:
            sm70_ops.nvfp4_qpn2_dispatch_sm70_out(
                out_2d,
                x_2d,
                layer.sm70_nvfp4_qpn2_codes,
                layer.sm70_nvfp4_qpn2_scales,
                float(layer.sm70_nvfp4_qpn2_global_scale),
                split_k,
                nacc,
                state.weight,
                state.scales,
                state.group_size,
                state.k_ld,
                state.q_ld,
                gated_silu,
            )
        if kernel_output_size != logical_output_size:
            out_2d = out_2d[:, :logical_output_size]
        if bias is not None:
            out_2d.add_(bias)
        return out_2d.reshape(*x.shape[:-1], logical_output_size)


class Qpn4NvFp4LinearKernel(TurboMindNvFp4LinearKernel):
    @classmethod
    def can_implement(cls, config):
        supported, reason = super().can_implement(config)
        if not supported:
            return supported, reason
        missing = _missing_sm70_nvfp4_qpn4_ops()
        if missing:
            return False, f"missing native QPN4 operators: {missing}"
        return True, None

    def try_process_weights(self, layer):
        # Capability and qualification were checked by the common selector.
        workspace = sm70_tm.get_nvfp4_qpn4_dense_workspace(layer.weight)
        if workspace is None:
            logger.warning_once(
                "Insufficient memory for the bounded SM70 NVFP4 QPN4 "
                "prefill workspace; retaining TurboMind."
            )
            return False
        sm70_tm.prepare_nvfp4_qpn4_linear(
            layer, workspace, gated_silu=self.config.dense_gated_silu
        )
        self._release_checkpoint_parameters(layer)
        logger.info_once(
            "Memory-neutral SM70 NVFP4 QPN4 M=1 decode "
            "path enabled with bounded FP16 prefill workspace."
        )
        return True
