# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact PLE gates retain rounding boundaries and unique buffer ownership."""

import math

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm.models.qwen4_exp.nvidia.ops import ple_prefill_gate as gate


def reference(key, query, value, norms, eps=1e-6):
    rows, hidden = value.shape
    groups = key.shape[1] // hidden

    def norm(x, weight):
        x = x.reshape(rows, groups, hidden).float()
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return y * (1 + weight.float().reshape(groups, hidden))

    k, q = norm(key, norms[0]), norm(query, norms[1])
    score = (k * q).sum(-1, keepdim=True) / math.sqrt(hidden)
    gates = torch.sigmoid(score.sign() * score.abs().clamp_min(1e-6).sqrt())
    full = gates * value.float().unsqueeze(1)
    expanded = full.half()
    normalized = expanded.float() * torch.rsqrt(
        full.square().mean(-1, keepdim=True) + eps
    )
    normalized *= 1 + norms[2].float().reshape(groups, hidden)
    return normalized.half().flatten(1), gates.squeeze(-1), expanded.flatten(1)


def fake_cuda(shape, dtype=torch.float16, *, stride=None):
    mode = FakeTensorMode()
    with torch.no_grad():
        tensor = (
            torch.empty(shape, device="meta", dtype=dtype)
            if stride is None
            else torch.empty_strided(shape, stride, device="meta", dtype=dtype)
        )
    return FakeTensor(mode, tensor, torch.device("cuda:0"))


def test_admission_checks_storage_and_precision(monkeypatch):
    monkeypatch.setattr(gate.current_platform, "is_device_capability", lambda _: True)
    key, query, value = (
        fake_cuda((3, 10240)),
        fake_cuda((3, 10240)),
        fake_cuda((3, 2560)),
    )
    norms = tuple(fake_cuda((10240,)) for _ in range(3))
    assert gate.prefill_gate_reason(key, query, value, norms, 4) is None
    assert (
        gate.prefill_gate_reason(key, query, value, norms[:2], 4)
        == "unsupported_norm_layout"
    )
    assert (
        gate.prefill_gate_reason(key, query, value, norms, 2)
        == "unqualified_hc_geometry"
    )
    overlap = fake_cuda((3, 10240), stride=(0, 1))
    assert (
        gate.prefill_gate_reason(overlap, query, value, norms, 4)
        == "unsupported_norm_layout"
    )
    bf16 = fake_cuda((3, 10240), torch.bfloat16)
    assert (
        gate.prefill_gate_reason(bf16, query, value, norms, 4)
        == "requires_fp16_operands"
    )
    empty_key, empty_value = fake_cuda((0, 10240)), fake_cuda((0, 2560))
    assert (
        gate.prefill_gate_reason(empty_key, empty_key, empty_value, norms, 4)
        == "empty_query_rows"
    )


def test_export_keeps_gate_scalar_and_declares_mutation():
    class Harness(nn.Module):
        def forward(self, key, query, value, kw, qw, cw, conv):
            key = key.clone()
            normalized, gates = gate.prepare_prefill_gate(
                key, query, value, (kw, qw, cw), 4, 1e-6
            )
            output = gate.finish_prefill_gate(conv.clone(), value, gates)
            return normalized, gates, output

    args = (
        torch.empty(17, 10240),
        torch.empty(17, 10240),
        torch.empty(17, 2560),
        torch.empty(10240),
        torch.empty(10240),
        torch.empty(10240),
        torch.empty(17, 10240),
    )
    graph = torch.export.export(Harness(), args)
    text = str(graph.graph)
    assert "qwen4_exp_ple_prefill_gate" in text
    assert "qwen4_exp_ple_prefill_finish" in text
    assert "[17, 4]" in text
    assert "aten.mul" not in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "rows,norm_dtype,strided",
    [(1, torch.float16, False), (17, torch.float32, True), (513, torch.float16, False)],
)
def test_gpu_rounding_and_storage(rows, norm_dtype, strided):
    torch.manual_seed(1023)
    base = torch.randn(
        rows, 10496 if strided else 10240, device="cuda", dtype=torch.float16
    )
    key = base[:, :10240]
    padding = base[:, 10240:].clone()
    query = torch.randn_like(key)
    value = torch.randn(rows, 2560, device="cuda", dtype=torch.float16) * 0.25
    norms = tuple(
        torch.randn(10240, device="cuda", dtype=norm_dtype) * 0.05 for _ in range(3)
    )
    expected, expected_gate, expanded = reference(key, query, value, norms)
    normalized, gates = gate.prepare_prefill_gate(key, query, value, norms, 4, 1e-6)
    assert normalized.data_ptr() == key.data_ptr()
    torch.testing.assert_close(normalized, expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(gates, expected_gate, atol=5e-6, rtol=2e-5)
    torch.testing.assert_close(base[:, 10240:], padding, atol=0, rtol=0)
    conv = torch.randn_like(key) * 0.125
    expected_output = expanded + conv
    output = gate.finish_prefill_gate(conv, value, gates)
    assert output.data_ptr() == conv.data_ptr()
    torch.testing.assert_close(output, expected_output, atol=3e-3, rtol=3e-3)
    conv = torch.randn_like(key) * 0.125
    combined = (query.float() + (expanded.float() + conv.float())).half()
    output = gate.finish_prefill_gate(conv, value, gates, query)
    torch.testing.assert_close(output, combined, atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_replay_recomputes_changed_inputs():
    torch.manual_seed(1023)
    source = torch.randn(33, 10240, device="cuda", dtype=torch.float16)
    key = torch.empty_like(source)
    query = torch.randn_like(source)
    value = torch.randn(33, 2560, device="cuda", dtype=torch.float16)
    norms = tuple(
        torch.zeros(10240, device="cuda", dtype=torch.float16) for _ in range(3)
    )
    conv_source = torch.randn_like(source)
    conv = torch.empty_like(source)

    def run():
        key.copy_(source)
        _, gates = gate.prepare_prefill_gate(key, query, value, norms, 4, 1e-6)
        conv.copy_(conv_source)
        gate.finish_prefill_gate(conv, value, gates)

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for sign in (1, -1):
        query.copy_(source * sign)
        value.mul_(0.5)
        expected, _, expanded = reference(source, query, value, norms)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(key, expected, atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(conv, expanded + conv_source, atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("groups", [4, 3])
def test_compiled_ple_entry_and_residual_boundary(monkeypatch, groups):
    from types import MethodType

    import vllm.models.qwen4_exp.nvidia.ple_layer as ple_module
    from vllm.config import VllmConfig
    from vllm.forward_context import create_forward_context, override_forward_context
    from vllm.models.qwen4_exp.nvidia.ple_layer import (
        Qwen4ExpPLEGroupedNorm,
        Qwen4ExpPLELayer,
    )

    monkeypatch.setattr(ple_module, "use_sm70_decode_graph_semantics", lambda: False)
    rows = 513
    torch.manual_seed(1023)

    class Embedding(nn.Module):
        def forward(self, hidden, ids, starts, context):
            return hidden[:, :2560]

    class Projection(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.register_buffer(
                "result", torch.randn(rows, width, device="cuda", dtype=torch.float16)
            )

        def forward(self, embeddings):
            return self.result.clone(), None

    module = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(module)
    module.prefix = "test_ple"
    module._sm70_hcx_diagnostics = False
    width = groups * 2560
    module.hidden_size, module.hc_count = 2560, groups
    module.ple_embedding = Embedding()
    module.key_proj, module.value_proj = Projection(width), Projection(2560)
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = Qwen4ExpPLEGroupedNorm(width, 1e-6, 2560, torch.float16).cuda()
        module.add_module(name, norm)

    def short_conv(self, inputs, output=None):
        output.copy_(inputs * 0.125)
        return output

    module._short_conv = MethodType(short_conv, module)
    config = VllmConfig()
    config.compilation_config.static_forward_context[module.prefix] = module
    context = create_forward_context(None, config)
    args = (
        torch.randn(rows, width, device="cuda", dtype=torch.float16),
        torch.arange(rows, device="cuda"),
        torch.tensor([0, rows], device="cuda"),
        torch.empty(1, 0, device="cuda", dtype=torch.int64),
    )
    with torch.inference_mode(), override_forward_context(context):
        config.kernel_config.prefill_ple_compact_gate = False
        control = torch.compile(module, dynamic=False, fullgraph=True)
        expected = control(*args, add_residual=True)
        config.kernel_config.prefill_ple_compact_gate = True
        exported = torch.export.export(
            module, args, kwargs={"add_residual": True}, strict=True
        )
        assert ("qwen4_exp_ple_prefill_gate" in str(exported.graph)) == (groups == 4)
        assert ("qwen4_exp_ple_prefill_finish" in str(exported.graph)) == (groups == 4)
        candidate = torch.compile(module, dynamic=False, fullgraph=True)
        actual = candidate(*args, add_residual=True)
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
        # Direct eager execution retains its original materialized semantics.
        a = module(*args, add_residual=True)
        config.kernel_config.prefill_ple_compact_gate = False
        b = module(*args, add_residual=True)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
