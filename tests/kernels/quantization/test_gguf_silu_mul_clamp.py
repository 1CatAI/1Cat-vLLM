# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the original FP16 activation boundaries used by clamped GGUF experts."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.gguf_moe import (
    _original_expert_activation,
)


def reference(gate, up, limit):
    return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def assert_bits(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )


@pytest.mark.parametrize("limit", [None, 0.0, 0.1, 10.0])
def test_cpu_and_strided_inputs_preserve_fallback(limit):
    gate = torch.arange(-16, 16, dtype=torch.float32).view(4, 8)[:, ::2]
    up = torch.flip(gate, [0])
    expected = F.silu(gate) * up if limit is None else reference(gate, up, limit)
    torch.testing.assert_close(
        _original_expert_activation(gate, up, limit), expected, atol=0, rtol=0
    )


def require_native():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    assert hasattr(torch.ops._C, "gguf_silu_mul_clamp_out"), "Rebuild native extension"


@pytest.mark.parametrize("limit", [0.0, 0.1, 10.0, 65504.0])
def test_all_fp16_gate_and_up_payloads(limit):
    require_native()
    payloads = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32)
    values = payloads.short().view(torch.float16)
    for gate, up in (
        (values, torch.ones_like(values)),
        (torch.full_like(values, 1.25), values),
        (values, values.flip(0)),
    ):
        assert_bits(
            _original_expert_activation(gate, up, limit), reference(gate, up, limit)
        )


@pytest.mark.parametrize("rows", [0, 1, 8, 24, 40, 80, 120, 256, 6144])
@pytest.mark.parametrize("width", [2048, 2051])
def test_changed_graph_inputs_and_canaries(rows, width):
    require_native()
    torch.manual_seed(1701)
    gate = torch.randn(rows, width, dtype=torch.float16, device="cuda")
    up = torch.randn_like(gate)
    storage = torch.full((gate.numel() + 32,), 19, dtype=gate.dtype, device=gate.device)
    output = storage[16:-16].view_as(gate)
    native = torch.ops._C.gguf_silu_mul_clamp_out
    native(output, gate, up, 10.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native(output, gate, up, 10.0)
    for scale in (0.0, 0.001, 1.0, 10.0, 30.0):
        gate.normal_(0, scale)
        up.normal_(0, scale)
        output.fill_(float("nan"))
        graph.replay()
        assert_bits(output, reference(gate, up, 10.0))
        assert_bits(
            torch.ops.vllm.gguf_original_expert_activation(gate, up, 10.0), output
        )
        assert torch.all(storage[:16] == 19) and torch.all(storage[-16:] == 19)


def test_native_rejects_mismatched_shapes_and_invalid_limits():
    require_native()
    gate = torch.zeros((4, 8), device="cuda", dtype=torch.float16)
    native = torch.ops._C.gguf_silu_mul_clamp_out
    with pytest.raises(RuntimeError, match="matching contiguous"):
        native(torch.empty_like(gate), gate, gate[:, :7], 10.0)
    for limit in (-1.0, float("nan"), float("inf"), 65505.0):
        with pytest.raises(RuntimeError, match="finite FP16"):
            native(torch.empty_like(gate), gate, gate, limit)
