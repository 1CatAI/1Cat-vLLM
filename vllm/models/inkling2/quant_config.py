# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tensor precision metadata of Inkling2 checkpoints (hf_quant_config.json).

Only the routed experts are quantized (NVFP4 or MXFP8); they are handled by the
model's own MoE layer. Every other layer runs unquantized.

SM70 port: the experts are converted at load time (NVFP4 to the SM70 NVFP4
MoE kernel's layout, MXFP8 dequantized to fp16, which is exact), so the method
is available from compute capability 7.0 and runs fp16 activations.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.models.inkling2.configs import INKLING2_ARCHITECTURES

_COMPUTE_DTYPES = ("float32", "bfloat16", "mxfp8", "nvfp4")


class Inkling2QuantConfig(QuantizationConfig):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.mxfp8_weights: set[str] = set()
        for group in config["model"]["parameter_groups"]:
            compute = group["compute_dtype"]
            if compute not in _COMPUTE_DTYPES:
                raise ValueError(f"Unsupported Inkling2 compute dtype: {compute}")
            if compute == "mxfp8":
                self.mxfp8_weights.update(group["tensors"])

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inkling2"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Inkling2QuantConfig:
        return cls(config)

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> QuantizationMethods | None:
        # Released checkpoints also declare a ModelOpt `quantization_config`
        # in config.json; their per-tensor precisions are in
        # hf_quant_config.json.
        architectures = getattr(hf_config, "architectures", None) or []
        if INKLING2_ARCHITECTURES & set(architectures):
            return "inkling2"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # The NVFP4 experts run the fork's ModelOpt NVFP4 MoE method (the
        # SM70 TurboMind W4A16 path); the model converts the checkpoint's
        # scales to ModelOpt's layout at load. Everything else is unquantized.
        from vllm.model_executor.layers.fused_moe import FusedMoE

        if isinstance(layer, FusedMoE):
            # Always the TurboMind method: the skinny-QPN oracle route (which
            # ModelOpt picks for some shapes, e.g. TP1) has no hook for the
            # softcapped SwiGLU and would silently run plain SiLU.
            from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
                ModelOptNvFp4SM70MoEMethod,
            )

            return ModelOptNvFp4SM70MoEMethod(
                quant_config=self._nvfp4_config(), moe_config=layer.moe_config
            )
        return None

    def _nvfp4_config(self):
        if getattr(self, "_nvfp4", None) is None:
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptNvFp4Config,
            )

            self._nvfp4 = ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
                group_size=16,
            )
        return self._nvfp4

    def is_mxfp8_layer(self, layer_id: int) -> bool:
        name = f"model.llm.layers.{layer_id}.mlp.experts.w13_weight"
        return name in self.mxfp8_weights
