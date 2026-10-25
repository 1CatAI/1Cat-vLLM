# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in actual CUDA dispatch; CPU route tests need no GPU initialization."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.test_pp7_fusion_candidate import assert_bits, require_gpu_window
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.norm import sm70
from vllm.model_executor.layers import layernorm
from vllm.model_executor.layers.layernorm import RMSNormGated


@pytest.fixture
def actual_config():
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=0, custom_ops=["all"])
    )
    config.kernel_config.sm70_rmsnorm_gated_exact = True
    with set_current_vllm_config(config):
        yield config


@pytest.mark.parametrize(
    "case,expected",
    [
        ("admitted", "exact"),
        ("old_bound_missing_library", "exact"),
        ("disabled", "triton"),
        ("batch_invariant", "triton"),
        ("cpu", "triton"),
        ("other_arch", "triton"),
        ("grouped", "triton"),
        ("bias", "triton"),
        ("gate_before_norm", "triton"),
        ("activation", "triton"),
        ("unsupported_shape", "triton"),
        ("missing_library", "triton"),
    ],
)
@pytest.mark.parametrize("method", ["forward_native", "forward_cuda"])
def test_cpu_actual_route(case, expected, method, monkeypatch, actual_config):
    from vllm.model_executor.layers.fla.ops import layernorm_guard

    before = torch.cuda.is_initialized()
    norm = RMSNormGated(
        128, norm_before_gate=True, activation="silu", dtype=torch.float16
    )
    x = SimpleNamespace(is_cuda=True, device=SimpleNamespace(index=0), shape=(240, 128))
    z = object()
    calls = []

    def exact(*args):
        calls.append("exact")
        return "exact"

    def triton(*args, **kwargs):
        calls.append("triton")
        return "triton"

    def reference(*args, **kwargs):
        calls.append("reference")
        return "reference"

    if method == "forward_native":
        expected = "exact" if case in ("bias", "missing_library") else expected
        expected = "reference" if expected == "triton" else expected
        monkeypatch.setattr(norm, "forward_static", reference)

    monkeypatch.setattr(torch.ops.vllm, "sm70_rmsnorm_gated_exact", exact)
    monkeypatch.setattr(layernorm_guard, "rmsnorm_fn", triton)
    monkeypatch.setattr(
        sm70, "_sm70_gated_norm_device_supported", lambda _: case != "other_arch"
    )
    monkeypatch.setattr(
        sm70,
        "_sm70_gated_norm_shape_supported",
        lambda *args: case != "unsupported_shape",
    )
    monkeypatch.setattr(
        layernorm.envs, "VLLM_BATCH_INVARIANT", case == "batch_invariant"
    )
    monkeypatch.setattr(
        sm70,
        "_PP7_EXACT_NORM_AVAILABLE",
        case not in ("missing_library", "old_bound_missing_library"),
    )
    if case == "old_bound_missing_library":
        x.shape = (192, 128)
    if case == "disabled":
        norm._sm70_rmsnorm_gated_exact = False
    if case == "cpu":
        x.is_cuda = False
    if case == "grouped":
        norm.group_size = 128
    if case == "bias":
        norm.bias = torch.nn.Parameter(torch.ones(128))
    if case == "gate_before_norm":
        norm.norm_before_gate = False
    if case == "activation":
        norm.activation = "unsupported"
    assert getattr(norm, method)(x, z) == expected
    assert calls == [expected]
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize("tokens", [1, 4, 5, 10, 15, 40, 64])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_gpu_actual_module_dispatch_and_changed_input_graph(
    tokens, activation, actual_config
):
    require_gpu_window()
    rows = tokens * 48
    torch.manual_seed(20261022 + rows)
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
    norm.weight.normal_(1, 0.25)
    assert norm._forward_method.__name__ == "forward_cuda"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        norm(x, z)
    expected_op = (
        "vllm_pp7::rmsnorm_gated_exact_out"
        if rows > 192
        else "_C::sm70_rmsnorm_gated_exact_out"
    )
    assert expected_op in {event.key for event in profile.key_averages()}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            norm(x, z)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = norm(x, z)
    for scale in (0, 0.001, 0.1, 1, 3, 30):
        x.normal_().mul_(scale)
        z.normal_().mul_(scale)
        result.fill_(float("nan"))
        graph.replay()
        expected = RMSNormGated.forward_static(
            x,
            z,
            norm.weight,
            norm.eps,
            x.dtype,
            norm_before_gate=True,
            activation=activation,
        )
        assert torch.isfinite(result).all()
        assert_bits(result, expected)


@pytest.mark.parametrize(
    "case", ["disabled", "missing_library", "outside_bound", "grouped"]
)
@torch.inference_mode()
def test_gpu_actual_fallback_retains_triton(case, actual_config, monkeypatch):
    require_gpu_window()
    from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

    rows = 3073 if case == "outside_bound" else 240
    x = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
    z = torch.randn_like(x)
    norm = RMSNormGated(
        128,
        eps=1e-6,
        norm_before_gate=True,
        activation="silu",
        group_size=128 if case == "grouped" else None,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    if case == "disabled":
        norm._sm70_rmsnorm_gated_exact = False
    if case == "missing_library":
        monkeypatch.setattr(sm70, "_PP7_EXACT_NORM_AVAILABLE", False)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        actual = norm(x, z)
    assert not any(
        "rmsnorm_gated_exact" in event.key for event in profile.key_averages()
    )
    expected = rmsnorm_fn(
        x,
        norm.weight,
        norm.bias,
        z=z,
        eps=norm.eps,
        group_size=norm.group_size,
        norm_before_gate=True,
        activation="silu",
    )
    assert_bits(actual, expected)
    assert torch.isfinite(actual).all()
