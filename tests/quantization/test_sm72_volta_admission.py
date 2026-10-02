# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.kernels.linear.nvfp4.sm70 import TurboMindNvFp4LinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.sm70_fp8 import (
    TurboMindFp8LinearKernel,
)
from vllm.model_executor.layers.quantization import sm70_turbomind as tm
from vllm.models.glm5next.sm70.sparse import Glm5NextSM70SparseBackend
from vllm.platforms.cuda import _get_backend_priorities
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.mark.parametrize("minor,expected", [(0, True), (2, True), (5, False)])
def test_volta_kernel_hardware_admission(monkeypatch, minor, expected):
    from vllm.model_executor.kernels.linear.scaled_mm import sm70_fp8

    monkeypatch.setattr(sm70_fp8.current_platform, "is_cuda", lambda: True)
    assert TurboMindFp8LinearKernel.is_supported(70 + minor)[0] is expected
    assert TurboMindNvFp4LinearKernel.is_supported(70 + minor)[0] is expected
    assert (
        Glm5NextSM70SparseBackend.supports_compute_capability(
            DeviceCapability(7, minor)
        )
        is expected
    )


@pytest.mark.parametrize("minor,expected", [(0, True), (2, True), (5, False)])
def test_volta_selection_uses_worker_device(monkeypatch, minor, expected):
    asked = []

    def capability(value, device_id):
        asked.append(device_id)
        return value == (7, minor)

    monkeypatch.setattr(tm.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(tm.current_platform, "is_device_capability", capability)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)
    assert tm.is_exact_sm70_cuda_platform() is expected
    assert asked and set(asked) == {2}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, minor))
    assert (
        tm.is_exact_sm70_cuda(SimpleNamespace(is_cuda=True, device="cuda:2"), True)
        is expected
    )


def test_sm72_backend_priorities_preserve_volta_and_turing(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLASH_ATTN_V100", "1")
    envs.disable_envs_cache()
    try:
        for capability in (DeviceCapability(7, 0), DeviceCapability(7, 2)):
            assert (
                _get_backend_priorities(False, capability)[0]
                is AttentionBackendEnum.FLASH_ATTN_V100
            )
        assert (
            _get_backend_priorities(False, DeviceCapability(7, 5))[0]
            is AttentionBackendEnum.FLASH_ATTN
        )
    finally:
        envs.disable_envs_cache()
