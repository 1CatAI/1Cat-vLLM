# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fla.ops.kda import FusedRMSNormGated


@pytest.mark.parametrize("enabled", [False, True])
def test_norm_captures_explicit_custom_op_dispatch(monkeypatch, enabled):
    calls = []

    def native(*args, **kwargs):
        calls.append("native")

    def cuda(*args, **kwargs):
        calls.append("cuda")

    monkeypatch.setattr(FusedRMSNormGated, "forward_native", native)
    monkeypatch.setattr(FusedRMSNormGated, "forward_cuda", cuda)
    config = VllmConfig(
        compilation_config=CompilationConfig(
            custom_ops=["none", ("+" if enabled else "-") + "fused_rms_norm_gated"]
        )
    )
    with set_current_vllm_config(config):
        norm = FusedRMSNormGated(128, activation="sigmoid", device="cpu")
    norm(None, None, out_dtype=torch.float16)
    assert calls == ["cuda" if enabled else "native"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA norm replay")
@pytest.mark.parametrize("enabled", [False, True])
def test_fp32_output_norm_changed_input_graph(enabled):
    config = VllmConfig(
        compilation_config=CompilationConfig(
            custom_ops=["none", ("+" if enabled else "-") + "fused_rms_norm_gated"]
        )
    )
    with set_current_vllm_config(config):
        norm = FusedRMSNormGated(
            128, activation="sigmoid", device="cuda", dtype=torch.float16
        )
    with torch.no_grad():
        norm.weight.fill_(1)
    x = torch.randn(1, 4, 16, 128, device="cuda", dtype=torch.float32)
    gate = torch.randn_like(x, dtype=torch.float16)
    for _ in range(3):
        norm(x, gate, out_dtype=torch.float16)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = norm(x, gate, out_dtype=torch.float16)
    for scale in (1.0, 1000.0, -10.0):
        x.copy_(torch.randn_like(x) * scale)
        gate.copy_(torch.randn_like(gate))
        graph.replay()
        reference = (
            x.double()
            * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-5)
            * torch.sigmoid(gate.double())
        ).half()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)
