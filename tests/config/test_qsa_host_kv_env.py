# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_SM70_QSA_HOST_KV gates host-KV history before QSA resolution."""

import pytest

import vllm.envs as envs
from vllm.config import DeviceConfig, VllmConfig


@pytest.fixture(autouse=True)
def fresh_host_kv_env(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.delenv("VLLM_SM70_QSA_HOST_KV", raising=False)


def test_env_gate_admits_host_kv(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QSA_HOST_KV", "1")
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    assert cfg.kernel_config.qsa_host_kv is True
    assert cfg.kernel_config.qsa_host_kv_device_reference is False


def test_unset_leaves_kernel_config_default():
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    assert cfg.kernel_config.qsa_host_kv is False
    assert cfg.kernel_config.qsa_host_kv_device_reference is False
