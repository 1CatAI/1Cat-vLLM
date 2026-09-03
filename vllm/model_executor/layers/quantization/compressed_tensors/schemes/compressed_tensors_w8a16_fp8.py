# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from vllm import _sm70_ops as sm70_ops
from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_wfp8_a16_linear_kernel,
)
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    STRATEGY_TO_PARAMETER_TYPE,
    STRATEGY_TO_WEIGHT_QUANT_KEY,
)
from vllm.model_executor.layers.quantization.fp8 import (
    _get_sm70_fp8_prefill_exact_dense_workspace,
    _is_sm70_fp8_qpn8_runtime_contract,
    _missing_sm70_fp8_qpn8_ops,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import replace_parameter

__all__ = ["CompressedTensorsW8A16Fp8"]

logger = init_logger(__name__)

_SM70_CHANNEL_FP8_QPN8_SHAPES = {
    "in_proj_qkvz": (4096, 5120),
    "qkv_proj": (3584, 5120),
    "out_proj": (5120, 1536),
    "o_proj": (5120, 1536),
    "gate_up_proj": (8704, 5120),
    "down_proj": (5120, 4352),
}
_SM70_CHANNEL_FP8_QPN8_CONFIGS = {
    # (K, N): (split-K, accumulator chains, prefetch codes)
    (5120, 4096): (16, 2, False),
    (5120, 3584): (16, 2, False),
    (1536, 5120): (12, 2, False),
    (5120, 8704): (16, 2, False),
    (4352, 5120): (16, 2, False),
}
_SM70_CHANNEL_FP8_QPN8_GATED_CONFIG = (8, 2, False)


def _sm70_fp8_qpn8_enabled(enable_by_default: bool) -> bool:
    """Resolve QPN8 while preserving the validated mixed-NVFP4 default."""
    if os.getenv("VLLM_SM70_FP8_QPN8") is None:
        return enable_by_default
    return envs.VLLM_SM70_FP8_QPN8


def _sm70_channel_fp8_qpn8_config(
    layer: torch.nn.Module,
) -> tuple[int, int, bool] | None:
    if getattr(layer, "tp_size", 1) != 4:
        return None
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    if tuple(layer.weight.shape) != _SM70_CHANNEL_FP8_QPN8_SHAPES.get(suffix):
        return None
    n_dim, k_dim = (int(dim) for dim in layer.weight.shape)
    return _SM70_CHANNEL_FP8_QPN8_CONFIGS.get((k_dim, n_dim))


def _sm70_channel_fp8_shape_is_validated(layer: torch.nn.Module) -> bool:
    """Whether an SM70 channel-FP8 GEMM has been validated for this shape.

    ``_SM70_CHANNEL_FP8_QPN8_SHAPES`` is the only record of which dense shapes
    have been exercised on SM70. ``fp8_gemm_sm70_out`` does not raise on the
    others -- it spins without returning -- so the layout conversion below is
    gated on this instead of on the QPN8 route being selected.
    """
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    return tuple(layer.weight.shape) == _SM70_CHANNEL_FP8_QPN8_SHAPES.get(suffix)


def _sm70_unpack_channel_fp8(layer: torch.nn.Module) -> None:
    """Dequantize channel-FP8 weights to the model dtype, in place.

    FP8 E4M3 to FP16 is exact, so this costs one extra byte per element on the
    affected projections and nothing in accuracy.
    """
    scale = layer.weight_scale.data.to(torch.float32)
    if scale.ndim == 1:
        scale = scale.view(-1, 1)
    dequantized = (layer.weight.data.to(torch.float32) * scale).to(layer.orig_dtype)
    replace_parameter(layer, "weight", dequantized)
    for stale in ("weight_scale", "weight_scale_inv", "input_scale"):
        if stale in layer._parameters:
            del layer._parameters[stale]
    layer.sm70_fp8_fp16_dequant = True


class CompressedTensorsW8A16Fp8(CompressedTensorsScheme):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        is_static_input_scheme: bool,
        enable_sm70_qpn8_by_default: bool = False,
    ):
        self.weight_quant = weight_quant
        self.strategy = weight_quant.strategy
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype
        self.is_static_input_scheme = is_static_input_scheme
        self.weight_block_size = self.weight_quant.block_structure

        self.weight_quant_key = STRATEGY_TO_WEIGHT_QUANT_KEY[self.strategy]
        self.activation_quant_key = (
            kFp8StaticTensorSym if is_static_input_scheme else kFp8DynamicTensorSym
        )
        self.use_sm70_fp8_turbomind = (
            self.strategy == QuantizationStrategy.CHANNEL
            and not self.is_static_input_scheme
            and sm70_tm.is_exact_sm70_cuda_platform()
            and sm70_tm.use_turbomind(envs.VLLM_SM70_FP8_TURBOMIND)
        )
        self.use_sm70_fp8_qpn8 = self.use_sm70_fp8_turbomind and _sm70_fp8_qpn8_enabled(
            enable_sm70_qpn8_by_default
        )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            # Validate block quantization shapes
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        # WEIGHT
        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = create_fp8_scale_parameter(
            STRATEGY_TO_PARAMETER_TYPE[self.strategy],
            output_partition_sizes,
            input_size_per_partition,
            layer.weight_block_size,
            weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE (to deal with converted checkpoints)
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)

        if self.use_sm70_fp8_turbomind:
            return

        self.linear_kernel = init_wfp8_a16_linear_kernel(
            weight_quant_key=self.weight_quant_key,
            activation_quant_key=self.activation_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_sm70_fp8_turbomind:
            if layer.orig_dtype != torch.float16:
                raise RuntimeError(
                    "SM70 TurboMind channel-FP8 requires fp16 model weights, "
                    f"got {layer.orig_dtype}."
                )
            if not hasattr(torch.ops._C, "fp8_sm70_prepare"):
                raise RuntimeError(
                    "VLLM_SM70_FP8_TURBOMIND=1 requires a build with CUDA "
                    "arch 7.0 and the SM70 TurboMind extension."
                )
            weight_scale = layer.weight_scale.to(torch.float32).contiguous()
            expected_scale_shape = (layer.weight.shape[0], 1)
            if tuple(weight_scale.shape) != expected_scale_shape:
                raise RuntimeError(
                    "SM70 TurboMind channel-FP8 expected weight scales with "
                    f"shape {expected_scale_shape}, got {tuple(weight_scale.shape)}."
                )
            qpn8_config = (
                _sm70_channel_fp8_qpn8_config(layer)
                if getattr(self, "use_sm70_fp8_qpn8", False)
                else None
            )
            qpn8_concurrency = (
                _is_sm70_fp8_qpn8_runtime_contract()
                if qpn8_config is not None
                else False
            )
            if qpn8_config is not None and not qpn8_concurrency:
                logger.info_once(
                    "The SM70 channel-FP8 QPN8 route retains TurboMind when "
                    "MTP is enabled or max_num_seqs exceeds 8."
                )
            if qpn8_config is not None and qpn8_concurrency:
                missing_ops = _missing_sm70_fp8_qpn8_ops()
                if missing_ops and os.getenv("VLLM_SM70_FP8_QPN8") is not None:
                    raise RuntimeError(
                        "VLLM_SM70_FP8_QPN8=1 requires the source-built SM70 "
                        f"QPN8 extension; missing ops: {missing_ops}."
                    )
                workspace = (
                    None
                    if missing_ops
                    else _get_sm70_fp8_prefill_exact_dense_workspace(layer.weight)
                )
                if not missing_ops and workspace is not None:
                    qpn8_codes, qpn8_scales = sm70_ops.fp8_qpn8_prepare_sm70(
                        layer.weight, weight_scale
                    )
                    replace_parameter(layer, "weight", qpn8_codes)
                    del layer._parameters["weight_scale"]
                    replace_parameter(layer, "weight_scale_inv", qpn8_scales)
                    split_k, nacc, prefetch = qpn8_config
                    layer.input_scale = None
                    layer.sm70_fp8_turbomind = True
                    layer.sm70_fp8_qpn8 = True
                    layer.sm70_fp8_channel_scale = True
                    layer.sm70_fp8_qpn8_split_k = split_k
                    layer.sm70_fp8_qpn8_nacc = nacc
                    layer.sm70_fp8_qpn8_prefetch = prefetch
                    layer.sm70_fp8_prefill_exact_dense_workspace_ptr = (
                        workspace.data_ptr()
                    )
                    if getattr(layer, "prefix", "").rsplit(".", 1)[-1] == (
                        "gate_up_proj"
                    ):
                        gated_split_k, gated_nacc, gated_prefetch = (
                            _SM70_CHANNEL_FP8_QPN8_GATED_CONFIG
                        )
                        layer.sm70_fp8_gated_silu = True
                        layer.sm70_fp8_qpn8_gated_split_k = gated_split_k
                        layer.sm70_fp8_qpn8_gated_nacc = gated_nacc
                        layer.sm70_fp8_qpn8_gated_prefetch = gated_prefetch
                    logger.info_once(
                        "Memory-neutral SM70 channel-FP8 QPN8 path enabled "
                        "for accepted Qwen3.8-27B TP4 dense shapes."
                    )
                    return
                if missing_ops:
                    logger.warning_once(
                        "The SM70 channel-FP8 QPN8 operators are unavailable; "
                        "retaining the TurboMind layout."
                    )
                else:
                    logger.warning_once(
                        "Insufficient memory for the SM70 channel-FP8 QPN8 "
                        "prefill workspace; retaining the TurboMind layout."
                    )
            if not _sm70_channel_fp8_shape_is_validated(layer):
                # No validated SM70 tile covers this shape. fp8_gemm_sm70_out
                # neither falls back nor raises on such shapes: it spins,
                # leaving the GPU at 100% utilization near idle power with no
                # progress and no error. Unpack instead, which is exact.
                logger.warning_once(
                    "No validated SM70 channel-FP8 GEMM covers %s with shape "
                    "%s; unpacking these weights to %s instead.",
                    getattr(layer, "prefix", "<unknown>"),
                    tuple(layer.weight.shape),
                    layer.orig_dtype,
                )
                _sm70_unpack_channel_fp8(layer)
                return
            tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(
                layer.weight, weight_scale, 128, False
            )
            replace_parameter(layer, "weight", tm_weight)
            del layer._parameters["weight_scale"]
            replace_parameter(layer, "weight_scale_inv", tm_scales)
            layer.input_scale = None
            layer.sm70_fp8_turbomind = True
            layer.sm70_fp8_channel_scale = True
            layer.register_buffer("sm70_fp8_meta", meta, persistent=False)
            layer.sm70_fp8_k_ld = int(meta[0].item())
            layer.sm70_fp8_q_ld = int(meta[1].item())
            logger.info_once(
                "SM70 compressed-tensors channel-FP8 TurboMind W8A16 dense "
                "path enabled."
            )
            return

        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            # MarlinFP8ScaledMMLinearKernel uses "weight_scale_inv" for block
            # quant, while CT registers the scale as "weight_scale".
            # Rename by deleting the old parameter and adding the new one so
            # that prepare_fp8_layer_for_marlin (which prefers "weight_scale"
            # over "weight_scale_inv") picks up "weight_scale_inv" correctly.
            weight_scale_data = layer.weight_scale.data
            del layer._parameters["weight_scale"]
            replace_parameter(layer, "weight_scale_inv", weight_scale_data)
        else:
            if self.strategy == QuantizationStrategy.TENSOR:
                # For fused modules with per-tensor scales, expand each scale
                # to its shard's channels.
                replace_parameter(
                    layer,
                    "weight_scale",
                    convert_to_channelwise(layer.weight_scale, layer.logical_widths),
                )

        self.linear_kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "sm70_fp8_fp16_dequant", False):
            return torch.nn.functional.linear(x, layer.weight, bias)
        if getattr(layer, "sm70_fp8_turbomind", False):
            if x.dtype != torch.float16:
                raise RuntimeError(
                    "SM70 TurboMind channel-FP8 requires float16 activations, "
                    f"got {x.dtype}."
                )
            out_shape = (*x.shape[:-1], layer.output_size_per_partition)
            x_2d = x.reshape(-1, x.shape[-1])
            if x_2d.stride(-1) != 1:
                x_2d = x_2d.contiguous()
            out_2d = torch.empty(
                (x_2d.shape[0], layer.output_size_per_partition),
                device=x.device,
                dtype=x.dtype,
            )
            if x_2d.shape[0] == 0:
                return out_2d.reshape(out_shape)
            if getattr(layer, "sm70_fp8_qpn8", False):
                sm70_ops.fp8_qpn8_dispatch_sm70_out(
                    out_2d,
                    int(layer.sm70_fp8_prefill_exact_dense_workspace_ptr),
                    x_2d,
                    layer.weight,
                    layer.weight_scale_inv,
                    int(layer.sm70_fp8_qpn8_split_k),
                    int(layer.sm70_fp8_qpn8_nacc),
                    bool(layer.sm70_fp8_qpn8_prefetch),
                    False,
                )
            else:
                sm70_ops.fp8_gemm_sm70_out(
                    out_2d,
                    x_2d,
                    layer.weight,
                    layer.weight_scale_inv,
                    128,
                    layer.sm70_fp8_k_ld,
                    layer.sm70_fp8_q_ld,
                    False,
                )
            if bias is not None:
                out_2d.add_(bias)
            return out_2d.reshape(out_shape)
        return self.linear_kernel.apply_weights(layer, x, bias)

    def apply_fused_silu_and_mul(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        if not getattr(layer, "sm70_fp8_qpn8", False):
            return None
        if not getattr(layer, "sm70_fp8_gated_silu", False):
            return None
        if x.dtype != torch.float16:
            raise RuntimeError(
                "SM70 channel-FP8 QPN8 gated-SiLU requires float16 "
                f"activations, got {x.dtype}."
            )
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        out_features = layer.output_size_per_partition // 2
        out_2d = torch.empty(
            (x_2d.shape[0], out_features), device=x.device, dtype=x.dtype
        )
        if x_2d.shape[0] == 0:
            return out_2d.reshape(*x.shape[:-1], out_features)
        sm70_ops.fp8_qpn8_dispatch_sm70_out(
            out_2d,
            int(layer.sm70_fp8_prefill_exact_dense_workspace_ptr),
            x_2d,
            layer.weight,
            layer.weight_scale_inv,
            int(layer.sm70_fp8_qpn8_gated_split_k),
            int(layer.sm70_fp8_qpn8_gated_nacc),
            bool(layer.sm70_fp8_qpn8_gated_prefetch),
            True,
        )
        return out_2d.reshape(*x.shape[:-1], out_features)
