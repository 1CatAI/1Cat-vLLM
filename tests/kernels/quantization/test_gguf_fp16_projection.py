# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import gguf_fp16_projection as module
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    GGUFPreparedProjection,
)


def descriptor(
    shape=(12, 5120), dtype=torch.float16, *, device="cuda", contiguous=True
):
    return SimpleNamespace(
        ndim=len(shape),
        shape=shape,
        dtype=dtype,
        device=SimpleNamespace(type=device, index=0),
        is_contiguous=lambda: contiguous,
    )


@pytest.mark.parametrize("source_type", [1, 30])
@pytest.mark.parametrize("n", [12, 24])
def test_capability_covers_only_m8_converted_float_shards(monkeypatch, source_type, n):
    monkeypatch.setattr(
        module.current_platform, "get_device_capability", lambda _: (7, 0)
    )
    capability = module.fp16_projection_capabilities(
        source_type, descriptor((n, 5120)), torch.float16
    )[0]
    assert capability.reason is None
    assert capability.graph_safe
    assert capability.source_type == ("F16" if source_type == 1 else "BF16")
    assert [m for m in (1, 4, 8, 16, 32, 512) if capability.supports_m(m)] == [8]


@pytest.mark.parametrize(
    "source_type,weight,act_dtype,enabled,cc,reason",
    [
        (30, descriptor(), torch.float16, False, (7, 0), "disabled_by_kernel_config"),
        (0, descriptor(), torch.float16, True, (7, 0), "requires_f16_or_bf16_source"),
        (30, descriptor(), torch.bfloat16, True, (7, 0), "requires_fp16_activations"),
        (
            30,
            descriptor(dtype=torch.bfloat16),
            torch.float16,
            True,
            (7, 0),
            "requires_converted_fp16_weight",
        ),
        (
            30,
            descriptor((12, 2560)),
            torch.float16,
            True,
            (7, 0),
            "floating_projection_shape_not_admitted",
        ),
        (
            30,
            descriptor((48, 5120)),
            torch.float16,
            True,
            (7, 0),
            "floating_projection_shape_not_admitted",
        ),
        (30, descriptor(device="cpu"), torch.float16, True, (7, 0), "device_not_cuda"),
        (30, descriptor(), torch.float16, True, (8, 0), "requires_sm70"),
        (
            30,
            descriptor(contiguous=False),
            torch.float16,
            True,
            (7, 0),
            "requires_contiguous_weight",
        ),
    ],
)
def test_rejections_are_reported(
    monkeypatch, source_type, weight, act_dtype, enabled, cc, reason
):
    monkeypatch.setattr(module.current_platform, "get_device_capability", lambda _: cc)
    capability = module.fp16_projection_capabilities(
        source_type, weight, act_dtype, enabled
    )
    assert capability[0].reason == reason


def test_missing_registered_wrapper_is_reported(monkeypatch):
    monkeypatch.setattr(
        module.current_platform, "get_device_capability", lambda _: (7, 0)
    )
    monkeypatch.setattr(module, "_OPERATOR", "missing_gguf_fp16_projection")
    capability = module.fp16_projection_capabilities(30, descriptor(), torch.float16)[0]
    assert capability.reason == "operator_missing:missing_gguf_fp16_projection"


def test_runtime_dispatch_changes_after_prefill_and_preserves_projection_boundary(
    monkeypatch,
):
    torch.manual_seed(17)
    weight = torch.randn(12, 5120).half()
    original = weight.clone()
    capability = replace(
        module.fp16_projection_capabilities(30, weight, torch.float16)[0], reason=None
    )
    monkeypatch.setattr(
        module, "fp16_projection_capabilities", lambda *_: (capability,)
    )
    monkeypatch.setattr(module.Sm70Fp16GemvSiluKernel, "can_implement", lambda *_: True)
    calls = []

    def gemv(x, w, output, active_columns, activated_columns):
        calls.append((x.shape[0], active_columns, activated_columns, w.data_ptr()))
        output.copy_(torch.mm(x, w.T))

    monkeypatch.setattr(module.Sm70Fp16GemvSiluKernel, "apply_out", gemv)
    for m in (512, 8, 1, 16, 32, 8, 512):
        x = torch.randn(1, m, 5120).half()
        previous = len(calls)
        actual = module._prepared_gguf_fp16_projection(x, weight, 30)
        assert len(calls) == previous + (m == 8)
        torch.testing.assert_close(actual, x @ weight.T, rtol=0, atol=0)
        assert actual.shape == (1, m, 12)
        assert actual.is_contiguous()
    assert calls == [(8, 12, 0, weight.data_ptr())] * 2
    torch.testing.assert_close(weight, original, rtol=0, atol=0)


def test_operator_capability_rejection_retains_mm(monkeypatch):
    x, weight = torch.randn(8, 5120).half(), torch.randn(24, 5120).half()

    def reject(*_):
        pytest.fail("A CPU descriptor must not enter the CUDA operator gate")

    monkeypatch.setattr(module.Sm70Fp16GemvSiluKernel, "can_implement", reject)
    actual = module._prepared_gguf_fp16_projection(x, weight, 1)
    torch.testing.assert_close(actual, x @ weight.T, rtol=0, atol=0)


def test_low_level_operator_rejection_retains_mm(monkeypatch):
    x, weight = torch.randn(8, 5120).half(), torch.randn(12, 5120).half()
    capability = replace(
        module.fp16_projection_capabilities(30, weight, torch.float16)[0], reason=None
    )
    monkeypatch.setattr(
        module, "fp16_projection_capabilities", lambda *_: (capability,)
    )
    monkeypatch.setattr(
        module.Sm70Fp16GemvSiluKernel, "can_implement", lambda *_: False
    )
    actual = module._prepared_gguf_fp16_projection(x, weight, 30)
    torch.testing.assert_close(actual, x @ weight.T, rtol=0, atol=0)


@pytest.mark.parametrize("source_type,enabled", [(1, True), (30, True), (30, False)])
def test_prepared_float_projection_reports_capability_and_keeps_cpu_storage(
    monkeypatch, source_type, enabled
):
    # The existing GGUF fallback is CUDA-registered. Substitute only its
    # dispatch for this CPU test and prove rejected preparation still delegates.
    calls = []

    def fallback(x, weight, source, policy, prefill):
        calls.append((source, policy, prefill))
        return x @ weight.T

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.gguf.fused_mul_mat_gguf", fallback
    )
    weight = torch.randn(12, 5120).half()
    projection = GGUFPreparedProjection(weight, source_type, torch.float16, enabled, 8)
    assert projection.weight.data_ptr() == weight.data_ptr()
    admission = projection.admission()
    assert admission["operators"][0]["reason"] == (
        "device_not_cuda" if enabled else "disabled_by_kernel_config"
    )
    x = torch.randn(8, 5120).half()
    torch.testing.assert_close(projection(x), x @ weight.T, rtol=0, atol=0)
    assert calls == [(source_type, enabled, 8)]


def test_fake_graph_keeps_actual_m_selection_opaque():
    torch._dynamo.reset()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def projection(x, weight):
        return torch.ops.vllm.prepared_gguf_fp16_projection(x, weight, 30, True)

    compiled = torch.compile(projection, backend=backend, dynamic=True, fullgraph=True)
    weight = torch.empty(12, 5120, dtype=torch.float16, device="meta")
    for m in (512, 8, 16, 512):
        x = torch.empty(m, 5120, dtype=torch.float16, device="meta")
        torch._dynamo.mark_dynamic(x, 0, min=2, max=8192)
        output = compiled(x, weight)
        assert output.shape == (m, 12)
        assert output.stride() == (12, 1)
    assert len(graphs) == 1
    nodes = [node for node in graphs[0].graph.nodes if node.op == "call_function"]
    assert len(nodes) == 1
    assert nodes[0].target in (
        torch.ops.vllm.prepared_gguf_fp16_projection,
        torch.ops.vllm.prepared_gguf_fp16_projection.default,
    )
    torch._dynamo.reset()
