# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch


@pytest.mark.parametrize("boundary", ["core", "projection"])
def test_verify_fusion_reaches_opaque_input_boundaries(monkeypatch, boundary):
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.model_executor.layers.quantization import sm70_gdn_ba_verify as ba

    layer = object()
    monkeypatch.setattr(gdn, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(
        gdn,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"test.layer": layer}),
    )
    monkeypatch.setattr(gdn, "_sm70_gdn_projection_dump_requested", lambda name: False)
    monkeypatch.setattr(
        gdn, "_sm70_dump_gdn_projection_tensor", lambda label, name, value: value
    )
    calls = []

    def fused(actual_layer, x, outputs=None):
        assert actual_layer is layer
        assert x.shape == (8, 5120)
        calls.append(outputs)
        for index, value in enumerate(outputs):
            value.fill_(index + 1)
        return outputs

    monkeypatch.setattr(ba, "apply_gdn_ba_verify", fused)
    x = torch.zeros(8, 5120, dtype=torch.float16)
    z = x.new_empty((8, 12, 128))
    if boundary == "core":
        core = torch.zeros_like(z)

        def recurrent(actual_layer, **kwargs):
            assert actual_layer is layer
            assert torch.all(kwargs["mixed_qkv"] == 1)
            assert torch.all(kwargs["b"] == 3)
            assert torch.all(kwargs["a"] == 4)

        monkeypatch.setattr(gdn, "_qwen_gdn_run_recurrent_core", recurrent)
        result = gdn.qwen_gdn_input_projection_core(
            x, z, core, torch.empty(0), torch.empty(0), "test.layer"
        )
        assert result[0] is z and result[1] is core
    else:
        q, b, a = x.new_empty((8, 2560)), x.new_empty((8, 12)), x.new_empty((8, 12))
        gdn.qwen_gdn_input_projection(x, q, z, b, a, "test.layer")
        assert torch.all(q == 1) and torch.all(b == 3) and torch.all(a == 4)
    assert len(calls) == 1
    assert calls[0][1].data_ptr() == z.data_ptr()
    assert torch.all(z == 2)


def test_verify_fusion_reaches_qwen35_override(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.model_executor.layers.quantization import sm70_gdn_ba_verify as ba
    from vllm.model_executor.models import qwen3_5

    monkeypatch.setattr(gdn, "_sm70_gdn_projection_dump_requested", lambda name: False)
    monkeypatch.setattr(qwen3_5, "_encode_layer_name", lambda name: name)
    monkeypatch.setattr(
        qwen3_5, "_sm70_dump_gdn_projection_tensor", lambda label, name, value: value
    )
    monkeypatch.setattr(
        qwen3_5, "_resolve_qwen_gdn_kv_cache_args", lambda *args: (None, None)
    )
    x = torch.zeros(8, 5120, dtype=torch.float16)
    q, z = x.new_full((8, 2560), 1), x.new_full((8, 1536), 2)
    b, a = x.new_full((8, 12), 3), x.new_full((8, 12), 4)
    layer = SimpleNamespace(
        key_dim=2048,
        value_dim=6144,
        tp_size=4,
        num_v_heads=48,
        head_v_dim=128,
        prefix="test.layer",
    )
    calls = []

    def fused(actual_layer, hidden):
        assert actual_layer is layer and hidden is x
        calls.append(True)
        return q, z, b, a

    def recurrent(actual_layer, **kwargs):
        assert actual_layer is layer
        assert kwargs["mixed_qkv"] is q
        assert kwargs["b"] is b and kwargs["a"] is a
        return kwargs["core_attn_out"]

    def project(core, actual_z, output, rows):
        assert actual_z.data_ptr() == z.data_ptr()
        assert actual_z.shape == (8, 12, 128) and rows == 8
        return core

    layer._output_projection = project
    monkeypatch.setattr(ba, "apply_gdn_ba_verify", fused)
    monkeypatch.setattr(qwen3_5, "_qwen_gdn_run_recurrent_core", recurrent)
    result = qwen3_5.Qwen3_5GatedDeltaNet.forward_cuda(layer, x, None)
    assert result.shape == (8, 12, 128) and calls == [True]


@pytest.mark.parametrize("amplitude", [0.0, 0.125, 1.0])
@torch.inference_mode()
def test_m8_qkvz_is_bitwise_and_ba_matches_dense64(amplitude):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    from vllm import _sm70_ops as ops

    torch.manual_seed(891)
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * amplitude
    weight = (torch.randn(4096, 5120, device="cuda") * 0.01).to(torch.float8_e4m3fn)
    scales = torch.ones(4096, 1, device="cuda")
    codes, packed_scales = ops.fp8_qpn8_prepare_sm70(weight, scales)
    ba = (torch.randn(24, 5120, device="cuda") * 0.01).to(torch.bfloat16).half()
    ordinary = x.new_empty((8, 4096))
    q, z = x.new_empty((8, 2560)), x.new_empty((8, 1536))
    b, a = x.new_empty((8, 12)), x.new_empty((8, 12))
    ops.fp8_qpn8_gemm_sm70_out(ordinary, x, codes, packed_scales, 16, 2, True, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.fp8_qpn8_dispatch_ba_split_sm70_out(
            q,
            z,
            b,
            a,
            x.new_empty((8, 4096)),
            x.new_empty((8, 24)),
            0,
            x,
            codes,
            packed_scales,
            ba,
        )
    for _ in range(4):
        graph.replay()
    assert torch.equal(q, ordinary[:, :2560])
    assert torch.equal(z, ordinary[:, 2560:])
    reference = x.double() @ ba.double().T
    torch.testing.assert_close(
        torch.cat((b, a), dim=1).double(), reference, atol=0.002, rtol=0.001
    )
