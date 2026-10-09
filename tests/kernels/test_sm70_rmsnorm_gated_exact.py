# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.layernorm import (
    RMSNormGated,
    _sm70_gated_norm_shape_supported,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("rows", [1, 12, 60, 120, 192, 193])
def test_shape_admission(rows):
    x = torch.empty(rows, 128, dtype=torch.float16)
    w = torch.empty(128, dtype=torch.float16)
    assert _sm70_gated_norm_shape_supported(x, x, w) == (rows <= 192)
    assert not _sm70_gated_norm_shape_supported(x, None, w)
    assert not _sm70_gated_norm_shape_supported(x.float(), x, w)
    assert not _sm70_gated_norm_shape_supported(x[:, :127], x[:, :127], w[:127])
    assert not _sm70_gated_norm_shape_supported(x, x, w.repeat(2)[::2])


def test_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_RMSNORM_GATED_EXACT", raising=False)
    envs.disable_envs_cache()
    assert not envs.VLLM_SM70_RMSNORM_GATED_EXACT


def require_native():
    if not current_platform.is_device_capability(70):
        pytest.skip("Requires SM70")
    if not hasattr(torch.ops._C, "sm70_rmsnorm_gated_exact_out"):
        pytest.skip("Build the native extension first")


def assert_bits(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )


@pytest.mark.parametrize("rows", [1, 12, 24, 48, 60, 96, 120, 192])
@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_changed_graph_inputs_and_canaries(
    rows, activation, monkeypatch, default_vllm_config
):
    require_native()
    monkeypatch.setenv("VLLM_SM70_RMSNORM_GATED_EXACT", "1")
    default_vllm_config.kernel_config.sm70_rmsnorm_gated_exact = True
    envs.disable_envs_cache()
    torch.manual_seed(20260927)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
    z = torch.randn_like(x)
    norm = RMSNormGated(
        128,
        eps=1e-6,
        norm_before_gate=True,
        activation=activation,
        dtype=torch.float16,
        device="cuda",
    )
    norm.weight.data.normal_()
    norm.forward_native(x, z)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = norm.forward_native(x, z)
    storage = torch.full((x.numel() + 32,), 19.0, device="cuda", dtype=x.dtype)
    out = storage[16:-16].view_as(x)
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
        assert_bits(actual, expected)
        torch.ops._C.sm70_rmsnorm_gated_exact_out(
            out, x, z, norm.weight, 1e-6, activation == "silu"
        )
        assert_bits(out, expected)
        assert torch.all(storage[:16] == 19)
        assert torch.all(storage[-16:] == 19)


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_all_fp16_gate_payloads(activation):
    require_native()
    torch.manual_seed(20260927)
    payloads = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32)
    gates = payloads.short().view(torch.float16).reshape(4, 128, 128)
    x = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    w = torch.randn(128, device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)
    for z in gates:
        torch.ops._C.sm70_rmsnorm_gated_exact_out(
            out, x, z, w, 1e-6, activation == "silu"
        )
        expected = RMSNormGated.forward_static(
            x, z, w, 1e-6, x.dtype, norm_before_gate=True, activation=activation
        )
        assert_bits(out, expected)


@torch.inference_mode()
def test_compiled_norm_keeps_native_bits_across_batch_and_fusion_context(
    monkeypatch, default_vllm_config
):
    require_native()
    monkeypatch.setenv("VLLM_SM70_RMSNORM_GATED_EXACT", "1")
    default_vllm_config.kernel_config.sm70_rmsnorm_gated_exact = True
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    envs.disable_envs_cache()
    torch.manual_seed(20260928)
    norm = RMSNormGated(
        128,
        eps=1e-6,
        norm_before_gate=True,
        activation="sigmoid",
        dtype=torch.float16,
        device="cuda",
    )
    norm.weight.data.normal_()

    def with_neighbor(x, z):
        # A neighboring reduction must not pull the norm's FP32 arithmetic
        # back into context-dependent Inductor fusion or autotuning.
        return norm.forward_native(x, z), z.float().sum(dim=-1)

    compiled = torch.compile(with_neighbor, fullgraph=True, dynamic=True)
    for rows in (12, 24, 48, 96, 192, 12):
        for scale in (0.001, 0.1, 3.0):
            x = torch.randn(rows, 128, device="cuda", dtype=torch.float16) * scale
            z = torch.randn_like(x)
            actual, _ = compiled(x, z)
            expected = RMSNormGated.forward_static(
                x,
                z,
                norm.weight,
                1e-6,
                x.dtype,
                norm_before_gate=True,
                activation="sigmoid",
            )
            assert_bits(actual, expected)
