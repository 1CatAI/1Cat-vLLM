# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QPN linears that do not require the Volta TurboMind GEMM registry."""

from dataclasses import dataclass

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization import sm70_turbomind as tm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _turing_supported(compute_capability=None):
    if not current_platform.is_cuda():
        return False, "requires CUDA"
    if compute_capability is None:
        capability = current_platform.get_device_capability(
            torch.accelerator.current_device_index()
        )
        compute_capability = capability.to_int() if capability else None
    if compute_capability != 75:
        return False, "requires Turing SM75; Volta retains its existing kernel"
    return True, None


@dataclass
class TuringNvFp4LinearLayerConfig(NvFp4LinearLayerConfig):
    input_dtype: torch.dtype
    weight_shape: tuple[int, ...]
    scale_shape: tuple[int, ...]
    weight_dtype: torch.dtype
    scale_dtype: torch.dtype


class TuringQpn2NvFp4LinearKernel(NvFp4LinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return _turing_supported(compute_capability)

    @classmethod
    def can_implement(cls, c):
        if not isinstance(c, TuringNvFp4LinearLayerConfig):
            return False, "requires checkpoint-native NVFP4 metadata"
        policy = get_current_vllm_config().kernel_config.sm70_nvfp4
        policy.active = True
        if not policy.dense_qpn2:
            return False, "dense QPN2 disabled by KernelConfig"
        if not tm.use_turbomind(envs.VLLM_SM70_NVFP4_TURBOMIND):
            return False, "disabled by the legacy quantization backend override"
        if c.input_dtype != torch.float16:
            return False, "requires FP16 activations"
        if len(c.weight_shape) != 2:
            return False, "requires rank-two packed weights"
        n, packed_k = c.weight_shape
        k = packed_k * 2
        if n <= 0 or k <= 0 or k % 128:
            return False, "requires N > 0 and positive K divisible by 128"
        if c.weight_dtype != torch.uint8 or c.scale_dtype != torch.float8_e4m3fn:
            return False, "requires NVFP4 bytes and E4M3 block scales"
        if c.scale_shape != (n, k // 16):
            return False, "requires one E4M3 scale per 16 weight values"
        missing = [
            name
            for name in ("nvfp4_qpn2_prepare_sm70", "nvfp4_qpn2_gemm_sm70_out")
            if not hasattr(torch.ops._C, name)
        ]
        if missing:
            return False, f"missing native operators: {missing}"
        return True, None

    def process_weights_after_loading(self, layer):
        tm.prepare_nvfp4_qpn2_dense_linear(layer)
        layer.weight = Parameter(layer.weight.new_empty(0), requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.new_empty(0), requires_grad=False
        )

    def apply_weights(self, layer, x, bias=None):
        return tm.apply_prepared_linear(layer, x, bias)


class TuringQpn8Fp8LinearKernel(FP8ScaledMMLinearKernel):
    def __init__(self, c, layer_param_names):
        # Activations remain FP16; no hardware FP8 activation quantizer is used.
        ScaledMMLinearKernel.__init__(self, c, layer_param_names)

    @classmethod
    def is_supported(cls, compute_capability=None):
        return _turing_supported(compute_capability)

    @classmethod
    def can_implement(cls, c):
        policy = get_current_vllm_config().kernel_config.sm70_fp8
        policy.resolve()
        if not policy.enabled or policy.force_marlin:
            return False, "QPN8 disabled by KernelConfig or legacy backend override"
        if c.input_dtype != torch.float16 or c.out_dtype != torch.float16:
            return False, "requires FP16 input and output"
        if c.weight_quant_key != kFp8StaticTensorSym:
            return False, "requires static per-tensor E4M3 weights"
        if len(c.weight_shape) != 2:
            return False, "requires rank-two weights"
        n, k = c.weight_shape
        if n <= 0 or k <= 0 or n % 32 or k % 128:
            return False, "requires positive N divisible by 32 and K divisible by 128"
        missing = [
            name
            for name in (
                "fp8_qpn8_prepare_sm70",
                "fp8_qpn8_dispatch_sm70_out",
                "fp8_qpn8_prefill_sm70_out",
            )
            if not hasattr(torch.ops._C, name)
        ]
        if missing:
            return False, f"missing native operators: {missing}"
        return True, None

    def process_weights_after_loading(self, layer):
        # The scaled-MM lifecycle supplies [K, N], while native QPN8 packs [N, K].
        tm.prepare_fp8_qpn8_dense_linear(
            layer, layer.weight.t().contiguous(), layer.weight_scale
        )
        layer.weight = Parameter(layer.weight.new_empty(0), requires_grad=False)

    def apply_weights(self, layer, x, bias=None):
        return tm.apply_prepared_fp8_qpn8_linear(layer, x, bias)

    def apply_scaled_mm(self, **kwargs):
        raise NotImplementedError("QPN8 consumes prepared weights and FP16 input")


def _turing_fp8_qpn8_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    n: int,
    split_k: int,
    chains: int,
    prefetch: bool,
) -> torch.Tensor:
    from vllm import _sm70_ops as ops

    out = x.new_empty((x.shape[0], n))
    if x.shape[0] == 0:
        return out
    # The allocation and its address are resolved inside the opaque operation.
    # Concurrent streams own separate storage; AOT artifacts retain no pointer.
    workspace = x.new_empty((x.shape[1], n)) if x.shape[0] > 8 else None
    ops.fp8_qpn8_dispatch_sm70_out(
        out,
        0 if workspace is None else workspace.data_ptr(),
        x,
        codes,
        scales,
        split_k,
        chains,
        prefetch,
        False,
    )
    return out


def _turing_fp8_qpn8_linear_fake(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    n: int,
    split_k: int,
    chains: int,
    prefetch: bool,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="turing_fp8_qpn8_linear",
    op_func=_turing_fp8_qpn8_linear,
    mutates_args=[],
    fake_impl=_turing_fp8_qpn8_linear_fake,
)
