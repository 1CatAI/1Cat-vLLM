# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import DeviceConfig, KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import (
    RMSNormGated,
    _sm70_prefill_gated_norm_shape_supported,
)


@pytest.mark.parametrize("rows", [0, 192, 4095, 4096, 196608])
def test_prefill_storage_admission(rows):
    x = torch.empty(rows, 128, dtype=torch.float16)
    w = torch.empty(128, dtype=torch.float16)
    assert _sm70_prefill_gated_norm_shape_supported(x, x, w) == (rows >= 4096)
    assert not _sm70_prefill_gated_norm_shape_supported(x, None, w)
    assert not _sm70_prefill_gated_norm_shape_supported(x.float(), x, w)
    assert not _sm70_prefill_gated_norm_shape_supported(x[:, :127], x[:, :127], w)
    assert not _sm70_prefill_gated_norm_shape_supported(x, x, w.repeat(2)[::2])


def test_prefill_policy_survives_constructor_context():
    layers = []
    for enabled in (False, True):
        with set_current_vllm_config(
            VllmConfig(
                device_config=DeviceConfig(device="cpu"),
                kernel_config=KernelConfig(prefill_rmsnorm_gated=enabled),
            )
        ):
            layers.append(RMSNormGated(128, norm_before_gate=True))
    assert [layer._sm70_prefill_rmsnorm_gated for layer in layers] == [False, True]


def require_sm70():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Requires SM70")


@pytest.fixture
def sm70_norm_config():
    require_sm70()
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
        yield


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("rows", [4096, 196608])
@torch.inference_mode()
def test_prefill_norm_fp32_reference_and_peak(rows, activation, sm70_norm_config):
    torch.manual_seed(1023)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
    z = torch.randn_like(x)
    norm = RMSNormGated(
        128,
        eps=1e-6,
        norm_before_gate=True,
        activation=activation,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    norm.weight.data.uniform_(0.5, 1.5)
    expected = RMSNormGated.forward_static(
        x,
        z,
        norm.weight,
        norm.eps,
        x.dtype,
        norm_before_gate=True,
        activation=activation,
    )
    norm.forward_native(x, z)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = norm.forward_native(x, z)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    assert peak < actual.numel() * actual.element_size() * 1.1
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-5)
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative < 1e-3


@torch.inference_mode()
def test_prefill_norm_fullgraph_and_changed_input_replay(sm70_norm_config):
    norm = RMSNormGated(
        128,
        eps=1e-6,
        norm_before_gate=True,
        activation="sigmoid",
        dtype=torch.float16,
        device=torch.device("cuda"),
    )
    compiled = torch.compile(norm.forward_native, fullgraph=True, dynamic=True)
    x = torch.randn(4096, 128, device="cuda", dtype=torch.float16)
    z = torch.randn_like(x)
    compiled(x, z)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled(x, z)
    for scale in (0.001, 1.0, 30.0):
        x.normal_(0, scale)
        z.normal_(0, scale)
        graph.replay()
        expected = RMSNormGated.forward_static(
            x,
            z,
            norm.weight,
            norm.eps,
            x.dtype,
            norm_before_gate=True,
            activation="sigmoid",
        )
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-5)
