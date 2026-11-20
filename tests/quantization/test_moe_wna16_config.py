# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration-key compatibility for the existing WNA16 MoE fallback."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def supported_capability(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(to_int=lambda: 80),
    )


@pytest.mark.parametrize("quant_method", ["awq", "awq_marlin"])
@pytest.mark.parametrize("group_size", [32, 128])
def test_awq_key_names(quant_method, group_size):
    values = {
        "quant_method": quant_method,
        "w_bit": 4,
        "q_group_size": group_size,
        "zero_point": True,
    }
    config = MoeWNA16Config.from_config(values)
    assert config.weight_bits == 4
    assert config.group_size == group_size
    assert config.has_zp is True
    assert config.full_config == values
    # Explicit awq_marlin can be parsed, but is not automatically overridden.
    assert MoeWNA16Config.is_moe_wna16_compatible(values) == (quant_method == "awq")


@pytest.mark.parametrize("bits", [4, 8])
def test_gptq_key_names_and_precedence(bits):
    values = {
        "quant_method": "gptq",
        "bits": bits,
        "group_size": 64,
        "w_bit": 3,
        "q_group_size": 128,
        "sym": True,
    }
    config = MoeWNA16Config.from_config(values)
    assert config.weight_bits == bits
    assert config.group_size == 64
    assert config.has_zp is False
    assert MoeWNA16Config.is_moe_wna16_compatible(values)


@pytest.mark.parametrize("bits", [3, 16])
def test_unsupported_awq_bits_remain_incompatible(bits):
    assert not MoeWNA16Config.is_moe_wna16_compatible(
        {"quant_method": "awq", "w_bit": bits, "q_group_size": 128}
    )


def test_incompatible_gptq_bits_are_not_overridden_by_awq_alias():
    assert not MoeWNA16Config.is_moe_wna16_compatible(
        {"quant_method": "gptq", "bits": 3, "w_bit": 4}
    )
