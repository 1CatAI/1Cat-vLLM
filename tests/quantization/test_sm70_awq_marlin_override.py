# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM70 TurboMind override must not relabel non-AWQ checkpoints."""

from types import SimpleNamespace

import pytest

from vllm import envs
from vllm.model_executor.layers.quantization import awq_marlin
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig

pytestmark = pytest.mark.cpu_test

AWQ_CFG = {"quant_method": "awq", "bits": 4, "group_size": 32, "zero_point": True}


@pytest.fixture(autouse=True)
def sm70_platform(monkeypatch):
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        has_device_capability=lambda capability: capability <= 70,
    )
    monkeypatch.setattr(awq_marlin, "current_platform", platform)
    monkeypatch.setenv("VLLM_SM70_QUANT_BACKEND", "turbomind")
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


@pytest.mark.parametrize("user_quant", ["marlin", "awq_marlin"])
def test_turbomind_backend_routes_awq_marlin_request_to_awq(user_quant):
    assert AWQMarlinConfig.override_quantization_method(AWQ_CFG, user_quant) == "awq"
    upper = dict(AWQ_CFG, quant_method="AWQ")
    assert AWQMarlinConfig.override_quantization_method(upper, user_quant) == "awq"


@pytest.mark.parametrize("user_quant", ["marlin", "awq_marlin"])
@pytest.mark.parametrize("ckpt_method", ["gptq", "compressed-tensors", "awq_f2c", ""])
def test_turbomind_backend_does_not_relabel_other_checkpoints(user_quant, ckpt_method):
    cfg = dict(AWQ_CFG, quant_method=ckpt_method)
    assert AWQMarlinConfig.override_quantization_method(cfg, user_quant) is None


@pytest.mark.parametrize("user_quant", [None, "awq"])
def test_turbomind_backend_without_marlin_request_keeps_checkpoint_method(user_quant):
    assert AWQMarlinConfig.override_quantization_method(AWQ_CFG, user_quant) is None
