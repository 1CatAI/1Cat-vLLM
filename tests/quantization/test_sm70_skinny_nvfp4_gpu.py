# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _sm70_ops
from vllm.model_executor.kernels.linear.nvfp4 import (
    NvFp4LinearLayerConfig,
    SkinnyNvFp4LinearKernel,
    skinny,
)


def _is_exact_sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


@pytest.mark.skipif(not _is_exact_sm70(), reason="requires an exact SM70 GPU")
@torch.inference_mode()
def test_sm70_skinny_adapter_routes_and_matches_turbomind(monkeypatch):
    supported, reason = SkinnyNvFp4LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    class FakeLayer(torch.nn.Module):
        pass

    torch.manual_seed(20260812)
    device = torch.device("cuda:0")
    n, k = 5120, 1536
    layer = FakeLayer()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.weight = torch.nn.Parameter(
        torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        (torch.rand(n, k // 16, device=device) * 400 + 8).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor(0.0021, dtype=torch.float32, device=device),
        requires_grad=False,
    )

    kernel = SkinnyNvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    state = skinny.sm70_tm.get_prepared_linear_state(layer)
    assert state.op_kind == "nvfp4"

    route_calls = {"simt": 0, "qpn": 0}
    original_simt = _sm70_ops.skinny_nvfp4_gemm_simt
    original_qpn = _sm70_ops.skinny_nvfp4_gemm_qpn

    def counted_simt(*args, **kwargs):
        route_calls["simt"] += 1
        return original_simt(*args, **kwargs)

    def counted_qpn(*args, **kwargs):
        route_calls["qpn"] += 1
        return original_qpn(*args, **kwargs)

    monkeypatch.setattr(_sm70_ops, "skinny_nvfp4_gemm_simt", counted_simt)
    monkeypatch.setattr(_sm70_ops, "skinny_nvfp4_gemm_qpn", counted_qpn)

    cases = [
        (1, torch.float16, "simt"),
        (3, torch.float16, "simt"),
        (4, torch.float16, "qpn"),
        (8, torch.float16, "qpn"),
        (9, torch.float16, "qpn"),
        (16, torch.float16, "qpn"),
        (17, torch.float16, "turbomind"),
        (2048, torch.float16, "turbomind"),
        (1, torch.bfloat16, "simt"),
        (8, torch.bfloat16, "qpn"),
        (17, torch.bfloat16, "turbomind"),
    ]
    for m, dtype, expected_route in cases:
        before = route_calls.copy()
        x = (torch.randn(m, k, device=device) * 0.1).to(dtype)
        actual = kernel.apply_weights(layer, x)
        reference = skinny._turbomind_fallback(
            x.to(torch.float16),
            state.weight,
            state.scales,
            n,
            state.group_size,
            state.k_ld,
            state.q_ld,
        )
        if dtype == torch.bfloat16:
            reference = reference.to(torch.bfloat16)

        denominator = reference.float().abs().max().clamp(min=1e-6)
        relative_error = (
            (actual.float() - reference.float()).abs().max() / denominator
        ).item()
        assert torch.isfinite(actual).all()
        assert actual.dtype == dtype
        assert relative_error < 3e-2
        if expected_route == "simt":
            assert route_calls["simt"] == before["simt"] + 1
            assert route_calls["qpn"] == before["qpn"]
        elif expected_route == "qpn":
            assert route_calls["qpn"] == before["qpn"] + 1
            assert route_calls["simt"] == before["simt"]
        else:
            assert route_calls == before
