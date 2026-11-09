# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow SM70 admission and shape gates for GLM-5.3 ModelOpt NVFP4."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    ModelOptNvFp4SM70MoEMethod,
    _use_glm53_fused_permute_q8,
    _use_glm53_qpn_w13_q8,
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


def test_nvfp4_moe_contract_accepts_glm53_tp8():
    validate_nvfp4_sm70_moe_contract(
        _glm53_moe_contract(tp_size=8, intermediate_size_per_partition=256)
    )


def test_glm53_fused_permute_q8_defaults_on_and_can_be_disabled(monkeypatch):
    name = "VLLM_SM70_GLM53_MOE_FUSED_PERMUTE_Q8"
    monkeypatch.delenv(name, raising=False)
    assert envs.VLLM_SM70_GLM53_MOE_FUSED_PERMUTE_Q8

    monkeypatch.setenv(name, "0")
    assert not envs.VLLM_SM70_GLM53_MOE_FUSED_PERMUTE_Q8


def test_glm53_fused_permute_q8_is_exact_shape_only(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS", "1")
    layer = SimpleNamespace(
        moe_config=_glm53_moe_contract(tp_size=8, intermediate_size_per_partition=256),
        sm70_nvfp4_num_experts=288,
        sm70_nvfp4_hidden_size=4096,
        sm70_nvfp4_intermediate_size=256,
        sm70_nvfp4_top_k=8,
        sm70_glm53_fused_permute_q8=True,
    )
    x = torch.empty(8, 4096, dtype=torch.float16)
    topk_ids = torch.empty(8, 8, dtype=torch.int32)

    assert _use_glm53_fused_permute_q8(layer, x, topk_ids)
    assert not _use_glm53_fused_permute_q8(layer, x[:7], topk_ids[:7])
    assert not _use_glm53_fused_permute_q8(layer, x, topk_ids.long())


def test_glm53_qpn_w13_q8_is_tp8_fused_permute_only():
    layer = SimpleNamespace(
        moe_config=_glm53_moe_contract(tp_size=8, intermediate_size_per_partition=256),
        sm70_nvfp4_num_experts=288,
        sm70_nvfp4_hidden_size=4096,
        sm70_nvfp4_intermediate_size=256,
        sm70_nvfp4_top_k=8,
        sm70_glm53_qpn_w13_q8=True,
    )
    x = torch.empty(8, 4096, dtype=torch.float16)
    topk_ids = torch.empty(8, 8, dtype=torch.int32)

    assert _use_glm53_qpn_w13_q8(layer, x, topk_ids, fused_permute=True)
    assert not _use_glm53_qpn_w13_q8(layer, x, topk_ids, fused_permute=False)
    layer.moe_config.tp_size = 4
    assert not _use_glm53_qpn_w13_q8(layer, x, topk_ids, fused_permute=True)


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
