# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which quantization config GLM-5.3's attention projections follow."""

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)


def declared_quant_config(
    quant_config: QuantizationConfig | None,
) -> ModelOptMixedPrecisionConfig | None:
    """The config when the checkpoint names the layers it quantizes (ModelOpt
    mixed precision), so a projection it does not name stays BF16. Checkpoints
    that name none (their fp8 attention weights carry no usable scales) keep
    BF16 projections."""
    if isinstance(quant_config, ModelOptMixedPrecisionConfig):
        return quant_config
    return None
