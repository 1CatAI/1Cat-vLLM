# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    _sm70_moe_backend_requested_explicitly,
)
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    NvFp4SM70MoEMethod,
    validate_nvfp4_sm70_moe_contract,
)
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    should_use_nvfp4_moe_turbomind,
)

from .compressed_tensors_moe_w4a4_nvfp4 import CompressedTensorsW4A4Nvfp4MoEMethod


class CompressedTensorsNvfp4SM70MoEMethod(
    NvFp4SM70MoEMethod, CompressedTensorsW4A4Nvfp4MoEMethod
):
    """Load CT NVFP4 expert tensors and run the native SM70 W4A16 path."""

    _checkpoint_format = "compressed-tensors"
    group_size = 16


def make_nvfp4_moe_method(
    layer: torch.nn.Module, layer_name: str, *, use_a16: bool
) -> FusedMoEMethodBase:
    if should_use_nvfp4_moe_turbomind() and not _sm70_moe_backend_requested_explicitly(
        layer
    ):
        try:
            validate_nvfp4_sm70_moe_contract(layer.moe_config)
        except NotImplementedError:
            pass
        else:
            return CompressedTensorsNvfp4SM70MoEMethod(layer.moe_config)
    return CompressedTensorsW4A4Nvfp4MoEMethod(
        layer.moe_config, layer_name, use_a16=use_a16
    )
