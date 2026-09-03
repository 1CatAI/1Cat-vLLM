# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_module
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _get_sm70_mtp_moe_decode_config,
    force_sm70_mtp_moe_legacy_config,
    fused_moe_kernel,
)


@pytest.fixture(autouse=True)
def _enable_tuned_mtp_config(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SM70_MTP_MOE_TUNED_CONFIG", True)


def test_mtp_sm70_decode_config_keeps_legacy_tile_at_m1():
    assert _get_sm70_mtp_moe_decode_config(1, 256, 128, 2048, 8) is None


def test_fused_moe_does_not_specialize_on_routing_dependent_em_alignment():
    assert "EM" in fused_moe_kernel.do_not_specialize_on_alignment


@pytest.mark.parametrize("m", range(2, 17))
def test_mtp_sm70_decode_config_uses_exact_local_tile(m):
    config = _get_sm70_mtp_moe_decode_config(m, 256, 128, 2048, 8)

    assert config is not None
    assert config["BLOCK_SIZE_M"] == 8
    assert config["BLOCK_SIZE_N"] == 128
    assert config["BLOCK_SIZE_K"] == 32


@pytest.mark.parametrize("m", [1, 5])
def test_qwen38_mtp_sm70_decode_config_uses_exact_local_tile(m):
    config = _get_sm70_mtp_moe_decode_config(m, 512, 160, 2560, 10)

    assert config is not None
    assert config["BLOCK_SIZE_M"] == 2
    assert config["BLOCK_SIZE_N"] == 128
    assert config["BLOCK_SIZE_K"] == 64


@pytest.mark.parametrize(
    "shape",
    [
        (17, 256, 128, 2048, 8),
        (2, 128, 128, 2048, 8),
        (2, 256, 512, 2048, 8),
        (2, 256, 128, 4096, 8),
        (2, 256, 128, 2048, 4),
        (2, 512, 160, 2560, 10),
        (5, 256, 160, 2560, 10),
        (5, 512, 128, 2560, 10),
        (5, 512, 160, 2048, 10),
        (5, 512, 160, 2560, 8),
    ],
)
def test_mtp_sm70_decode_config_is_shape_bounded(shape):
    assert _get_sm70_mtp_moe_decode_config(*shape) is None


def test_mtp_sm70_decode_config_can_be_disabled(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_SM70_MTP_MOE_TUNED_CONFIG", False)

    assert _get_sm70_mtp_moe_decode_config(2, 256, 128, 2048, 8) is None


def test_mtp_sm70_decode_config_can_be_forced_to_legacy_for_warmup():
    with force_sm70_mtp_moe_legacy_config():
        config = _get_sm70_mtp_moe_decode_config(2, 256, 128, 2048, 8)

    assert config is None
    assert _get_sm70_mtp_moe_decode_config(2, 256, 128, 2048, 8) is not None


class _FakeSM70Platform:
    @staticmethod
    def is_cuda():
        return True

    @staticmethod
    def is_rocm():
        return False

    @staticmethod
    def has_device_capability(capability):
        return capability == 70


def test_mtp_sm70_decode_config_is_selected_when_opted_in(monkeypatch):
    monkeypatch.setattr(fused_moe_module, "current_platform", _FakeSM70Platform())

    config = fused_moe_module.get_default_config(2, 256, 128, 2048, 8, None)

    assert config["BLOCK_SIZE_M"] == 8
    assert config["BLOCK_SIZE_N"] == 128
    assert config["BLOCK_SIZE_K"] == 32


def test_mtp_sm70_decode_config_rolls_back_to_0dot3(monkeypatch):
    monkeypatch.setattr(fused_moe_module, "current_platform", _FakeSM70Platform())
    monkeypatch.setattr(envs, "VLLM_SM70_MTP_MOE_TUNED_CONFIG", False)

    config = fused_moe_module.get_default_config(1, 256, 128, 2048, 8, None)

    assert config["BLOCK_SIZE_N"] == 32
    assert config["BLOCK_SIZE_K"] == 64
    assert "num_warps" not in config
