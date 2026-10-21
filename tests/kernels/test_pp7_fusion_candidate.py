# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks by default; GPU checks need PP7_FUSION_GPU_TESTS=1 explicitly."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.kernels.norm import sm70
from vllm.model_executor.layers.layernorm import RMSNormGated


@pytest.fixture
def default_vllm_config():
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    config = VllmConfig(compilation_config=CompilationConfig(mode=0))
    with set_current_vllm_config(config):
        yield config


def assert_bits(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )


@pytest.mark.parametrize("rows", [0, 1, 48, 192, 193, 240, 480, 720, 1920, 3072, 3073])
def test_cpu_extended_shape_admission(rows):
    x = torch.empty(rows, 128, dtype=torch.float16)
    w = torch.empty(128, dtype=torch.float16)
    assert sm70._sm70_gated_norm_shape_supported(x, x, w) == (1 <= rows <= 3072)
    assert not sm70._sm70_gated_norm_shape_supported(x, None, w)
    assert not sm70._sm70_gated_norm_shape_supported(x.float(), x, w)
    assert not sm70._sm70_gated_norm_shape_supported(x[:, :127], x[:, :127], w[:127])
    assert not sm70._sm70_gated_norm_shape_supported(x, x, w.repeat(2)[::2])


def test_cpu_missing_additive_library_keeps_reference(monkeypatch):
    monkeypatch.setattr(sm70, "_PP7_EXACT_NORM_AVAILABLE", False)
    monkeypatch.setattr(sm70, "_PP7_FUSION_LOAD_ERROR", "intentional test")
    generator = torch.Generator().manual_seed(20261021)
    x = torch.randn((240, 128), generator=generator, dtype=torch.float16)
    z = torch.randn((240, 128), generator=generator, dtype=torch.float16)
    weight = torch.randn((128,), generator=generator, dtype=torch.float16)
    before = torch.cuda.is_initialized()
    for activation in ("sigmoid", "silu"):
        actual = sm70._sm70_rmsnorm_gated_exact_impl(
            x, z, weight, 1e-6, activation == "silu"
        )
        expected = RMSNormGated.forward_static(
            x,
            z,
            weight,
            1e-6,
            x.dtype,
            norm_before_gate=True,
            activation=activation,
        )
        assert_bits(actual, expected)
    assert torch.cuda.is_initialized() == before


def test_cpu_mm_postload_delegates_only_to_language_model():
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.qwen4_exp.nvidia.model import (
        Qwen4ExpForConditionalGeneration,
    )

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def prepare_loaded_linear_weights(self):
            self.calls += 1

    class Vision(nn.Module):
        def prepare_loaded_linear_weights(self):
            raise AssertionError("Language preparation must not target vision")

    wrapper = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = Language()
    wrapper.visual = Vision()
    before = torch.cuda.is_initialized()
    process_weights_after_loading(
        wrapper,
        SimpleNamespace(dtype=torch.float16, quantization="gguf"),
        torch.device("cpu"),
    )
    assert wrapper.language_model.calls == 1
    assert torch.cuda.is_initialized() == before


def require_gpu_window():
    if os.environ.get("PP7_FUSION_GPU_TESTS") != "1":
        pytest.skip("GPU test window was not explicitly enabled")
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability(70):
        pytest.skip("SM70 only")
    assert sm70._PP7_EXACT_NORM_AVAILABLE, sm70._PP7_FUSION_LOAD_ERROR


@pytest.mark.parametrize("tokens", [1, 4, 5, 10, 15, 40, 64])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_gpu_changed_inputs_graph_canaries_and_real_dispatch(
    tokens, activation, default_vllm_config
):
    require_gpu_window()
    default_vllm_config.kernel_config.sm70_rmsnorm_gated_exact = True
    rows = tokens * 48
    torch.manual_seed(20261021 + rows)
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
    norm.weight.data.normal_(1, 0.25)
    # CPU profiler operator names verify the real candidate dispatch. This is
    # a tiny correctness probe, never a throughput benchmark.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        norm.forward_native(x, z)
    expected_name = (
        "vllm_pp7::rmsnorm_gated_exact_out"
        if rows > 192
        else "_C::sm70_rmsnorm_gated_exact_out"
    )
    assert expected_name in {event.key for event in profile.key_averages()}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            norm.forward_native(x, z)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = norm.forward_native(x, z)
    storage = torch.full((x.numel() + 32,), 19.0, device="cuda", dtype=x.dtype)
    out = storage[16:-16].view_as(x)
    operation = (
        torch.ops.vllm_pp7.rmsnorm_gated_exact_out
        if rows > 192
        else torch.ops._C.sm70_rmsnorm_gated_exact_out
    )
    for scale in (0.0, 0.001, 0.1, 1.0, 3.0, 30.0):
        x.normal_(0, scale)
        z.normal_(0, scale)
        actual.fill_(float("nan"))
        out.fill_(float("nan"))
        graph.replay()
        expected = RMSNormGated.forward_static(
            x,
            z,
            norm.weight,
            1e-6,
            x.dtype,
            norm_before_gate=True,
            activation=activation,
        )
        assert torch.isfinite(expected).all()
        assert torch.isfinite(actual).all()
        assert_bits(actual, expected)
        operation(out, x, z, norm.weight, 1e-6, activation == "silu")
        assert_bits(out, expected)
        assert torch.all(storage[:16] == 19)
        assert torch.all(storage[-16:] == 19)


@torch.inference_mode()
def test_gpu_native_rejects_outside_extended_bound():
    require_gpu_window()
    x = torch.empty(3073, 128, device="cuda", dtype=torch.float16)
    w = torch.ones(128, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match=r"1\.\.3072"):
        torch.ops.vllm_pp7.rmsnorm_gated_exact_out(x, x, x, w, 1e-6, True)
