# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Weight-only block-scaled FP8 through the native QPN8 operators.

The kernel admits static E4M3 weights with 128 x 128 scales, aligned positive
N/K, and FP16 activations/output on Volta and Turing. KernelConfig owns the
policy; no model identity or parallel layout limits this admission.

M=0 returns an empty output. M=1..8 uses packed GEMM. Volta retains its
TurboMind layout for larger M, avoiding a full dense reconstruction on short
prefills. Turing uses invocation-private dense scratch for its existing FP16
prefill operator. Native arithmetic is unchanged.
"""

from collections.abc import Sequence

import torch
from torch.library import custom_op

from vllm import _sm70_ops as sm70_ops
from vllm.config import get_current_vllm_config_or_none
from vllm.config.kernel import Sm70Fp8Config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.sm70_fp8 import (
    TurboMindFp8LinearKernel,
    _sm70_fp8_qpn8_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform

# Block-scale QPN8 GEMM supports M up to this bound; larger M uses a fallback.
_QPN8_MAX_M = 8
logger = init_logger(__name__)


@custom_op("sm70_fp8::qpn8_native_linear", mutates_args=())
def _qpn8_native_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    split_k: int,
    accumulator_chains: int,
    prefetch_codes: bool,
    fallback_weight: torch.Tensor | None = None,
    fallback_scales: torch.Tensor | None = None,
    fallback_k_ld: int = 0,
    fallback_q_ld: int = 0,
) -> torch.Tensor:
    # The M decision stays inside the opaque op: a compiled graph covers a
    # dynamic M range, so a Python branch traced at small M would be reused
    # for prefill.
    k_dim, n_dim = codes.shape
    out = x.new_empty((x.shape[0], n_dim))
    if x.shape[0] == 0:
        return out
    if x.shape[0] <= _QPN8_MAX_M:
        sm70_ops.fp8_qpn8_gemm_sm70_out(
            out,
            x,
            codes,
            group_scales,
            split_k,
            accumulator_chains,
            True,
            prefetch_codes,
        )
        return out
    if fallback_weight is not None:
        sm70_ops.fp8_gemm_sm70_out(
            out,
            x,
            fallback_weight,
            fallback_scales,
            128,
            fallback_k_ld,
            fallback_q_ld,
            False,
        )
        return out
    dense = torch.empty((k_dim * n_dim,), dtype=torch.float16, device=x.device)
    sm70_ops.fp8_qpn8_prefill_sm70_out(
        out, dense.data_ptr(), x, codes, group_scales, False
    )
    return out


@_qpn8_native_linear.register_fake
def _qpn8_native_linear_fake(
    x,
    codes,
    group_scales,
    split_k,
    accumulator_chains,
    prefetch_codes,
    fallback_weight=None,
    fallback_scales=None,
    fallback_k_ld=0,
    fallback_q_ld=0,
):
    return x.new_empty((x.shape[0], codes.shape[1]))


class QPN8Fp8BlockScaledMMLinearKernel(TurboMindFp8LinearKernel):
    """Block-scaled FP8 on SM70/SM75 via 1Cat's native QPN8 operators."""

    # fp16 activations go straight into the GEMM; no input quantization.
    apply_input_quant = False

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda():
            return False, "requires CUDA"
        if compute_capability is None:
            capability = current_platform.get_device_capability()
            compute_capability = capability.to_int() if capability else None
        if compute_capability not in (70, 72, 75):
            return False, "requires a Volta or Turing CUDA device"
        return True, None

    @staticmethod
    def _policy(config):
        if hasattr(config, "policy"):
            return config.policy
        engine = get_current_vllm_config_or_none()
        return engine.kernel_config.sm70_fp8 if engine else Sm70Fp8Config()

    @classmethod
    def can_implement(cls, config):
        policy = cls._policy(config)
        policy.resolve()
        if not policy.block_qpn8:
            return False, "disabled by kernel_config.sm70_fp8.block_qpn8"
        if set(policy.explicit_enables).intersection(
            {"gated_silu", "prefill_prescaled", "prescaled_decode"}
        ):
            return False, "retains the explicitly configured fused or prescaled variant"
        if policy.force_marlin:
            return False, "Marlin selected by legacy backend compatibility policy"
        if getattr(config, "is_bmm", False):
            return False, "grouped BMM retains its existing grouped implementation"
        if config.input_dtype != torch.float16 or config.out_dtype != torch.float16:
            return False, "requires FP16 activations and output"
        if config.weight_quant_key != kFp8Static128BlockSym:
            return False, "requires static E4M3 weights with 128 x 128 block scales"
        if len(config.weight_shape) != 2:
            return False, "requires a rank-two weight matrix"
        n, k = config.weight_shape
        if n <= 0 or k <= 0 or n % 128 or k % 128:
            return False, f"requires positive N,K divisible by 128, got ({n},{k})"
        missing = [
            name
            for name in (
                "fp8_qpn8_prepare_sm70",
                "fp8_qpn8_gemm_sm70_out",
                "fp8_qpn8_prefill_sm70_out",
            )
            if not hasattr(torch.ops._C, name)
        ]
        if missing:
            return False, f"missing native operators: {missing}"
        return True, None

    def __init__(self, config, layer_param_names: Sequence[str] = ()):
        ScaledMMLinearKernel.__init__(self, config, layer_param_names)
        self.policy = self._policy(config)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        weight = layer.weight.data
        n, k = weight.shape
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None:
            scale = getattr(layer, "weight_scale", None)
        assert scale is not None, "block-FP8 layer without a weight scale"
        block_scales = scale.data.detach().float().contiguous()
        if tuple(block_scales.shape) != (n // 128, k // 128):
            raise ValueError(
                f"QPN8: scale raster {tuple(block_scales.shape)} does not match "
                f"weights {n}x{k} at block [128,128]"
            )
        if weight.dtype != torch.float8_e4m3fn:
            weight = weight.view(torch.float8_e4m3fn)

        codes, group_scales = sm70_ops.fp8_qpn8_prepare_sm70(weight, block_scales)
        k_dim, n_dim = (int(dim) for dim in codes.shape)
        layer.register_buffer(
            "_sm70_block_fp8_qpn8_packed_codes", codes, persistent=False
        )
        layer.register_buffer(
            "_sm70_block_fp8_qpn8_packed_scales", group_scales, persistent=False
        )
        capability = current_platform.get_device_capability()
        if capability is not None and capability.to_int() in (70, 72):
            packed, scales, metadata = sm70_ops.fp8_sm70_prepare(
                weight, block_scales, 128, False
            )
            layer.register_buffer(
                "_sm70_block_fp8_turbomind_packed_weight", packed, persistent=False
            )
            layer.register_buffer(
                "_sm70_block_fp8_turbomind_packed_scales", scales, persistent=False
            )
            layer._qpn8_fallback_k_ld = int(metadata[0].item())
            layer._qpn8_fallback_q_ld = int(metadata[1].item())
        logger.info_once(
            "Block FP8 QPN8 native GEMM supports M=1..8; larger M uses %s "
            "because the native block-scale GEMM does not support those rows.",
            "TurboMind" if hasattr(layer, "_qpn8_fallback_k_ld") else "FP16 prefill",
        )
        layer._qpn8_out_features = n
        layer._qpn8_cfg = _sm70_fp8_qpn8_config(k_dim, n_dim, False)
        layer.sm70_fp8_turbomind = True
        # The startup memory profile accounts for both packed layouts on
        # Volta. Raw checkpoint weights are no longer resident.
        layer.weight = torch.nn.Parameter(
            torch.empty(0, dtype=torch.uint8, device=codes.device), requires_grad=False
        )

    def apply_block_scaled_mm(self, A, B, As, Bs):
        # Satisfies the ABC; apply_weights below bypasses the base-class
        # A/As machinery entirely (weight-only path, fp16 activations).
        raise RuntimeError("unreachable: QPN8 overrides apply_weights")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        codes = layer._sm70_block_fp8_qpn8_packed_codes
        split_k, accumulator_chains, prefetch_codes = layer._qpn8_cfg
        y = torch.ops.sm70_fp8.qpn8_native_linear(
            x.reshape(-1, codes.shape[0]).contiguous(),
            codes,
            layer._sm70_block_fp8_qpn8_packed_scales,
            split_k,
            accumulator_chains,
            prefetch_codes,
            getattr(layer, "_sm70_block_fp8_turbomind_packed_weight", None),
            getattr(layer, "_sm70_block_fp8_turbomind_packed_scales", None),
            getattr(layer, "_qpn8_fallback_k_ld", 0),
            getattr(layer, "_qpn8_fallback_q_ld", 0),
        )
        if bias is not None:
            y = y + bias
        return y.reshape(x.shape[:-1] + (layer._qpn8_out_features,))
