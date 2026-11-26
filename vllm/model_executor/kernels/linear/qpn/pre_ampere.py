# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QPN linears that do not require the Volta TurboMind GEMM registry."""

from dataclasses import dataclass

import torch
from torch.nn.parameter import Parameter

from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.qpn.fp8_block import (
    _QPN8_MAX_M,
    _SM75_MMA_MAX_M,
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


def _device_capability(compute_capability=None):
    """The given capability, or that of this process's current device."""
    if compute_capability is None:
        capability = current_platform.get_device_capability(
            torch.accelerator.current_device_index()
        )
        compute_capability = capability.to_int() if capability else None
    return compute_capability


def _turing_supported(compute_capability=None):
    if not current_platform.is_cuda():
        return False, "requires CUDA"
    if _device_capability(compute_capability) != 75:
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
        if not tm.format_enabled("nvfp4"):
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


_QPN8_FP8_OPERATORS = (
    "fp8_qpn8_prepare_sm70",
    "fp8_qpn8_dispatch_sm70_out",
    "fp8_qpn8_prefill_sm70_out",
)


def _qpn8_fp8_available():
    policy = get_current_vllm_config().kernel_config.sm70_fp8
    policy.resolve()
    if not policy.enabled or policy.force_marlin:
        return False, "QPN8 disabled by KernelConfig or legacy backend override"
    missing = [name for name in _QPN8_FP8_OPERATORS if not hasattr(torch.ops._C, name)]
    if missing:
        return False, f"missing native operators: {missing}"
    return True, None


class TuringQpn8Fp8LinearKernel(FP8ScaledMMLinearKernel):
    def __init__(self, c, layer_param_names):
        # Activations remain FP16; no hardware FP8 activation quantizer is used.
        ScaledMMLinearKernel.__init__(self, c, layer_param_names)

    @classmethod
    def is_supported(cls, compute_capability=None):
        return _turing_supported(compute_capability)

    @classmethod
    def can_implement(cls, c):
        available, reason = _qpn8_fp8_available()
        if not available:
            return False, reason
        if c.input_dtype != torch.float16 or c.out_dtype != torch.float16:
            return False, "requires FP16 input and output"
        if c.weight_quant_key != kFp8StaticTensorSym:
            return False, "requires static per-tensor E4M3 weights"
        if len(c.weight_shape) != 2:
            return False, "requires rank-two weights"
        n, k = c.weight_shape
        if n <= 0 or k <= 0 or n % 32 or k % 128:
            return False, "requires positive N divisible by 32 and K divisible by 128"
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


class Qpn8Mxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 weights as 8-bit QPN8 codes with row-group scales.

    Volta and Turing have no MXFP8 GEMM; TurboMind's FP8 kernel has no 1x32
    E8M0 layout, so both architectures use the QPN8 GEMV for decode and its
    dequantizing cuBLAS prefill."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda():
            return False, "requires CUDA"
        if _device_capability(compute_capability) not in (70, 75):
            return False, "requires Volta SM70 or Turing SM75"
        return True, None

    @classmethod
    def can_implement(cls, c):
        available = _qpn8_fp8_available()
        if not available[0]:
            return available
        # The E8M0 exponent range is checked when the weights load.
        reason = tm.fp8_qpn8_shape_reason(*c.partition_weight_shape)
        return (False, reason) if reason is not None else (True, None)

    def process_weights_after_loading(self, layer):
        tm.prepare_mxfp8_qpn8_linear(layer, layer.weight, layer.weight_scale)
        layer.weight = Parameter(layer.weight.new_empty(0), requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.new_empty(0), requires_grad=False
        )

    def apply_weights(self, layer, x, bias=None):
        return tm.apply_prepared_fp8_qpn8_linear(layer, x, bias)


def _turing_fp8_qpn8_linear(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    n: int,
    split_k: int,
    decode_split_k: int,
    chains: int,
    prefetch: bool,
    sm75_mma: bool = False,
) -> torch.Tensor:
    from vllm import _sm70_ops as ops

    out = x.new_empty((x.shape[0], n))
    if x.shape[0] == 0:
        return out
    # Same M bounds as the block-FP8 route (qpn/fp8_block.py).
    if sm75_mma and _QPN8_MAX_M < x.shape[0] <= _SM75_MMA_MAX_M:
        ops.fp8_qpn8_mma_sm75_out(out, x, codes, scales)
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
        # Chosen here, inside the opaque operation: a shape test in the caller
        # would be traced once under torch.compile and kept for every M. The
        # occupancy-fitted split was measured for the M <= 8 launch only; the
        # M16 and M32 bodies take the table split (the native M=17..32
        # candidate admits split-K 12 or 16).
        decode_split_k if x.shape[0] <= tm.QPN8_OCCUPANCY_MAX_M else split_k,
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
    decode_split_k: int,
    chains: int,
    prefetch: bool,
    sm75_mma: bool = False,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], n))


direct_register_custom_op(
    op_name="turing_fp8_qpn8_linear",
    op_func=_turing_fp8_qpn8_linear,
    mutates_args=[],
    fake_impl=_turing_fp8_qpn8_linear_fake,
)
