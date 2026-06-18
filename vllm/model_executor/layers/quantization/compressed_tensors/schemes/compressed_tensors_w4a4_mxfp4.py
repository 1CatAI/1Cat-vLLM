# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

__all__ = ["CompressedTensorsW4A4Mxfp4"]

logger = init_logger(__name__)


class CompressedTensorsW4A4Mxfp4(CompressedTensorsScheme):
    """
    Compressed tensors scheme for MXFP4.

    Supports models quantized with the compressed-tensors mxfp4-pack-quantized
    format.

    MXFP4 format:
    - 4-bit float weights (E2M1) packed into uint8
    - Per-group E8M0 scales with group_size=32
    - No global scale (unlike NVFP4)

    On SM100+ with FlashInfer: true W4A4 (activations dynamically quantized).
    Otherwise: W4A16 weight-only via Marlin.
    """

    def __init__(self):
        self.group_size = 32
        self.kernel = (
            None if envs.VLLM_SM70_MXFP4_TURBOMIND else init_mxfp4_linear_kernel()
        )

    @classmethod
    def get_min_capability(cls) -> int:
        if envs.VLLM_SM70_MXFP4_TURBOMIND:
            return 70
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        # Packed FP4 weights (2 values per byte)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Per-group E8M0 scales
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def _fallback_kernel(self):
        if self.kernel is None:
            self.kernel = init_mxfp4_linear_kernel()
        return self.kernel

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if sm70_tm.is_exact_sm70_cuda(
            layer.weight_packed, envs.VLLM_SM70_MXFP4_TURBOMIND
        ):
            logger.info_once(
                "SM70 compressed-tensors MXFP4 TurboMind dense path enabled."
            )
            sm70_tm.prepare_mxfp4_linear(layer)
            layer.weight_packed = Parameter(
                torch.empty(
                    0, dtype=torch.uint8, device=layer.weight_packed.device
                ),
                requires_grad=False,
            )
            layer.weight_scale = Parameter(
                torch.empty(
                    0, dtype=torch.uint8, device=layer.weight_scale.device
                ),
                requires_grad=False,
            )
            return

        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        self._fallback_kernel().process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sm70_tm.has_prepared_linear(layer):
            return sm70_tm.apply_prepared_linear(layer, x, bias)
        return self._fallback_kernel().apply_weights(layer, x, bias)
