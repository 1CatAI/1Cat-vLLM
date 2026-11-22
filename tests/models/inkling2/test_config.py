# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.warmup.sm70_runtime import model_shape_warmup_task
from vllm.models.inkling2.configs import (
    Inkling2LoopMMConfig,
    Inkling2LoopTextConfig,
    Inkling2MMConfig,
    Inkling2ModelConfig,
)
from vllm.models.inkling2.quant_config import Inkling2QuantConfig
from vllm.transformers_utils.config import _CONFIG_REGISTRY


def test_inkling2_registry_and_configs():
    assert "Inkling2ForConditionalGeneration" in ModelRegistry.get_supported_archs()
    assert "Inkling2LoopForConditionalGeneration" in ModelRegistry.get_supported_archs()
    assert _CONFIG_REGISTRY["inkling2"] is Inkling2MMConfig
    assert _CONFIG_REGISTRY["inkling2_loop"] is Inkling2LoopMMConfig
    assert _CONFIG_REGISTRY["inkling2_model"] is Inkling2ModelConfig
    assert _CONFIG_REGISTRY["inkling2_loop_text"] is Inkling2LoopTextConfig


def test_inkling2_quant_override_is_scoped():
    assert get_quantization_config("inkling2") is Inkling2QuantConfig
    assert Inkling2QuantConfig.get_min_capability() == 70

    inkling_cfg = SimpleNamespace(
        architectures=["Inkling2LoopForConditionalGeneration"]
    )
    assert (
        Inkling2QuantConfig.override_quantization_method({}, None, inkling_cfg)
        == "inkling2"
    )

    generic_modelopt_cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
    assert (
        Inkling2QuantConfig.override_quantization_method(
            {"quant_method": "modelopt"}, None, generic_modelopt_cfg
        )
        is None
    )


def test_model_shape_warmup_task_contract():
    class _PlainModel:
        pass

    assert model_shape_warmup_task(_PlainModel(), runner=None) is None

    called: list[str] = []

    class _WarmupModel:
        def warmup_sm70_shapes(self, runner) -> bool:
            called.append(str(runner))
            return True

    task = model_shape_warmup_task(_WarmupModel(), runner="dummy_runner")
    assert task is not None
    assert task.run() == ("model_sm70_shapes",)
    assert called == ["dummy_runner"]


def test_inkling2_moe_lazy_backend_imports():
    # Remove backend-specific modules if already cached and verify importing
    # nvidia.moe does not pull in SM90/SM120/Humming modules eagerly.
    for mod in (
        "vllm.models.inkling2.nvidia.moe",
        "vllm.models.inkling2.nvidia.ops.moe_humming",
        "vllm.models.inkling2.nvidia.ops.moe_sm120",
    ):
        sys.modules.pop(mod, None)

    import vllm.models.inkling2.nvidia.moe  # noqa: F401

    assert "vllm.models.inkling2.nvidia.ops.moe_humming" not in sys.modules
    assert "vllm.models.inkling2.nvidia.ops.moe_sm120" not in sys.modules
