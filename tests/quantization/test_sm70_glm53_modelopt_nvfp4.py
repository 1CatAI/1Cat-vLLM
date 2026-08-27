# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow SM70 admission and shape gates for GLM-5.3 ModelOpt NVFP4."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    ModelOptNvFp4SM70MoEMethod,
    validate_nvfp4_sm70_moe_contract,
)


def _glm53_moe_contract(**overrides):
    values = {
        "num_experts": 288,
        "experts_per_token": 8,
        "hidden_dim": 4096,
        "intermediate_size_per_partition": 512,
        "tp_size": 4,
        "moe_parallel_config": SimpleNamespace(use_all2all_kernels=False),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_nvfp4_moe_contract_accepts_glm53_tp4():
    validate_nvfp4_sm70_moe_contract(_glm53_moe_contract())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_experts", 256),
        ("experts_per_token", 6),
        ("hidden_dim", 2048),
        ("intermediate_size_per_partition", 256),
        ("tp_size", 2),
    ],
)
def test_nvfp4_moe_contract_rejects_unvalidated_glm53_shapes(field, value):
    with pytest.raises(NotImplementedError):
        validate_nvfp4_sm70_moe_contract(_glm53_moe_contract(**{field: value}))


def test_nvfp4_moe_contract_rejects_glm53_tp8():
    with pytest.raises(NotImplementedError, match="tensor parallel"):
        validate_nvfp4_sm70_moe_contract(
            _glm53_moe_contract(tp_size=8, intermediate_size_per_partition=256)
        )


def test_pure_nvfp4_glm53_moe_uses_turbomind_w4a16_on_sm70():
    config = ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
    )

    class FakeRoutedExperts:
        moe_config = _glm53_moe_contract()

    with (
        patch.object(modelopt, "RoutedExperts", FakeRoutedExperts),
        patch.object(sm70_tm, "is_exact_sm70_cuda_platform", return_value=True),
        patch.object(sm70_tm, "should_use_nvfp4_moe_turbomind", return_value=True),
    ):
        method = config.get_quant_method(
            FakeRoutedExperts(), "model.layers.3.mlp.experts"
        )

    assert isinstance(method, ModelOptNvFp4SM70MoEMethod)
    assert method.use_a16
