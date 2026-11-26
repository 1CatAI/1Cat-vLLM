# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An unhandled declared algorithm must not become an unquantized linear."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


def _config(algo: str) -> modelopt.ModelOptMixedPrecisionConfig:
    return modelopt.ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "exclude_modules": [],
                "group_size": 16,
                "quantized_layers": {"model.proj": {"quant_algo": algo}},
            }
        }
    )


@pytest.mark.parametrize("layer_type", [LinearBase, ParallelLMHead])
@pytest.mark.parametrize("algo", ["UNKNOWN", "FP8_PER_CHANNEL_PER_TOKEN"])
def test_rejects_unhandled_linear_algo(
    layer_type: type[torch.nn.Module], algo: str
) -> None:
    config = _config(algo)
    with pytest.raises(ValueError, match=algo + ".*model.proj"):
        config.get_quant_method(MagicMock(spec=layer_type), "model.proj")


@pytest.mark.parametrize("layer_type", [LinearBase, ParallelLMHead])
@pytest.mark.parametrize("mode", ["missing", "empty", "excluded"])
def test_preserves_unquantized_controls(
    layer_type: type[torch.nn.Module], mode: str
) -> None:
    config = _config("" if mode == "empty" else "UNKNOWN")
    if mode == "excluded":
        config.exclude_modules = ["model.proj"]
    prefix = "model.absent" if mode == "missing" else "model.proj"
    method = config.get_quant_method(MagicMock(spec=layer_type), prefix)
    if layer_type is ParallelLMHead:
        # An LM head keeps its own UnquantizedEmbeddingMethod: the linear
        # method's weight preparation expects LinearBase's per-layer policies.
        assert method is None
    else:
        assert isinstance(method, UnquantizedLinearMethod)


@pytest.mark.parametrize("layer_type", [LinearBase, ParallelLMHead])
@pytest.mark.parametrize(
    ("algo", "method_type"),
    [
        ("FP8", modelopt.ModelOptFp8LinearMethod),
        ("NVFP4", modelopt.ModelOptNvFp4LinearMethod),
        ("W4A16_NVFP4", modelopt.ModelOptNvFp4W4A16LinearMethod),
    ],
)
def test_preserves_supported_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    layer_type: type[torch.nn.Module],
    algo: str,
    method_type: type,
) -> None:
    # Keep real dispatch/config/method constructors; replace native factories.
    monkeypatch.setattr(
        modelopt,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    monkeypatch.setattr(modelopt, "init_nvfp4_linear_kernel", lambda: object())
    monkeypatch.setattr(modelopt, "MarlinNvFp4LinearKernel", lambda *args: object())
    config = _config(algo)
    assert isinstance(
        config.get_quant_method(MagicMock(spec=layer_type), "model.proj"),
        method_type,
    )
