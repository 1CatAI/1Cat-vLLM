# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.kernels.gguf import native_gated_pair_capabilities
from vllm.model_executor.layers.quantization.gguf_native_pair import (
    _native_gated_pair,
    apply_native_gated_pair,
)


@pytest.mark.parametrize(
    "types",
    [
        (18, 18),
        (21, 23),
        (23, 21),
        (18, 21),
        (21, 18),
        (12, 23),
        (23, 12),
        (12, 21),
        (21, 12),
        (18, 23),
        (22, 21),
        (21, 22),
        (22, 18),
        (18, 22),
        (17, 18),
        (22, 17),
        (29, 22),
        (10, 21),
        (17, 16),
        (16, 22),
    ],
)
def test_capabilities_admit_only_measured_mixed_pairs(monkeypatch, types):
    monkeypatch.setattr(
        torch.ops._C, "gguf_native_pair_sm70_out", lambda *args: None, raising=False
    )
    caps = native_gated_pair_capabilities(types, 5120, 4352, torch.float16)
    assert len(caps) == 2 and all(c.reason is None for c in caps)
    assert [m for m in (1, 5, 8, 16, 20, 32, 512) if caps[0].supports_m(m)] == [8]
    for arguments in (
        ((21, 21), 5120, 4352, torch.float16),
        ((23, 18), 5120, 4352, torch.float16),
        ((21, 23), 5120, 4096, torch.float16),
        ((99,), 5120, 4352, torch.float16),
        (types, 5120, 4352, torch.bfloat16),
        (types, 5120, 4352, torch.float16, False),
        (types, 5120, 4352, torch.float16, True, 80),
    ):
        rejected = native_gated_pair_capabilities(*arguments)
        assert not rejected or all(c.reason for c in rejected)


@pytest.mark.parametrize(
    "types",
    [
        (18, 18),
        (21, 23),
        (23, 21),
        (18, 21),
        (21, 18),
        (12, 23),
        (23, 12),
        (12, 21),
        (21, 12),
        (18, 23),
        (22, 21),
        (21, 22),
        (22, 18),
        (18, 22),
        (17, 18),
        (22, 17),
        (29, 22),
        (10, 21),
        (17, 16),
        (16, 22),
    ],
)
def test_runtime_m_preserves_mixed_canonical_policy(monkeypatch, types):
    calls: list[tuple[Any, ...]] = []

    def native(output, rows, gate, up, gate_type, up_type):
        calls.append(("native", gate_type, up_type))
        output.fill_(7)

    def canonical(rows, codes, stats, caches, descriptors, cb, bb):
        calls.append(("canonical", rows.shape[0], descriptors, cb, bb))
        return torch.full((rows.shape[0], 8704), 2, dtype=rows.dtype)

    def silu(output, pair):
        gate, up = pair.chunk(2, dim=-1)
        output.copy_(torch.nn.functional.silu(gate) * up)

    monkeypatch.setattr(
        torch.ops._C, "gguf_native_pair_sm70_out", native, raising=False
    )
    monkeypatch.setattr(torch.ops._C, "silu_and_mul", silu, raising=False)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.gguf_native_pair."
        "_prepared_gguf_mixed_projection",
        canonical,
    )
    empty = torch.empty(0)
    for m in (512, 8, 1, 5, 16, 20, 32, 8):
        calls.clear()
        result = _native_gated_pair(
            torch.empty(m, 5120, dtype=torch.float16),
            empty,
            empty,
            *types,
            [empty, empty],
            [empty, empty],
            [None, None],
            [2, 21, 32, 0, 0, 4352, 4352, 0, 2, 1, 0, 32, 0, 0, 4352, 4352, 0, 2],
            [],
            [512, -1, 512, -1],
        )
        assert result.shape == (m, 4352)
        if m == 8:
            assert calls == [("native", *types)]
            assert bool((result == 7).all())
        else:
            assert len(calls) == 1 and calls[0][:2] == ("canonical", m)
            assert calls[0][-1] == [512, -1, 512, -1]


def test_dynamic_compile_keeps_runtime_dispatch_opaque():
    torch._dynamo.reset()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def apply(x, records, codes, stats):
        return torch.ops.vllm.gguf_native_gated_pair(
            x,
            records,
            records,
            21,
            23,
            [codes, codes],
            [stats, stats],
            [None, None],
            [2, 21, 32, 0, 0, 4352, 4352, 0, 0] * 2,
            [],
            [],
        )

    compiled = torch.compile(apply, backend=backend, dynamic=True, fullgraph=True)
    records = torch.empty(0, dtype=torch.uint8, device="meta")
    codes = torch.empty(0, dtype=torch.int32, device="meta")
    stats = torch.empty(0, dtype=torch.int64, device="meta")
    for m in (512, 8, 16, 32, 8):
        x = torch.empty(m, 5120, dtype=torch.float16, device="meta")
        torch._dynamo.mark_dynamic(x, 0, min=2, max=8192)
        assert compiled(x, records, codes, stats).shape == (m, 4352)
    assert len(graphs) == 1
    nodes = [node for node in graphs[0].graph.nodes if node.op == "call_function"]
    assert len(nodes) == 1 and "gguf_native_gated_pair" in str(nodes[0].target)
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "types",
    [
        (18, 18),
        (21, 23),
        (23, 21),
        (18, 21),
        (21, 18),
        (12, 23),
        (23, 12),
        (12, 21),
        (21, 12),
        (18, 23),
        (22, 21),
        (21, 22),
        (22, 18),
        (18, 22),
        (17, 18),
        (22, 17),
        (29, 22),
        (10, 21),
        (17, 16),
        (16, 22),
    ],
)
def test_registered_parameter_lists_support_aot_module_capture(types):
    class Projection(torch.nn.Module):
        def __init__(self, source_type):
            super().__init__()
            self.kernel = SimpleNamespace(
                config=SimpleNamespace(
                    group_size=32, partition_weight_shape=(5120, 4352)
                ),
                source_type=source_type,
                operator_capabilities=(),
            )
            self.codes = torch.nn.Parameter(
                torch.empty(0, device="meta", dtype=torch.int32), False
            )
            self.stats = torch.nn.Parameter(
                torch.empty(0, device="meta", dtype=torch.int64), False
            )
            self.fp16_cache = None
            self.cache_capabilities = ()
            self.gguf_tm_k_ld = self.gguf_tm_q_ld = 0
            self.logical_output_size = 4352

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gguf_native_gated_records = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.empty(0, device="meta", dtype=torch.uint8), False
                    )
                    for _ in range(2)
                ]
            )
            self.gguf_native_gated_types = types
            self.gguf_tm_projections = torch.nn.ModuleList(
                [Projection(types[0]), Projection(types[1])]
            )

        def forward(self, x):
            return apply_native_gated_pair(self, x)

    torch._dynamo.reset()
    layer = Layer()
    compiled = torch.compile(layer, backend="eager", fullgraph=True, dynamic=True)
    for m in (512, 8, 16, 8):
        x = torch.empty(m, 5120, dtype=torch.float16, device="meta")
        assert compiled(x).shape == (m, 4352)
    from torch._dynamo.convert_frame import fullgraph_capture
    from torch._dynamo.utils import get_metrics_context

    with get_metrics_context():
        assert fullgraph_capture(layer, (x,), {}) is not None
    torch._dynamo.reset()


def test_pure_pair_uses_single_merged_canonical_fallback(monkeypatch):
    calls = []

    def canonical(
        x, code, stat, cache, family, decoder, group, kld, qld, n, logical, cb, bb
    ):
        calls.append((x.shape[0], decoder, n, cb, bb))
        return x.new_full((x.shape[0], 8704), 2)

    def silu(output, pair):
        gate, up = pair.chunk(2, dim=-1)
        output.copy_(torch.nn.functional.silu(gate) * up)

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.gguf_native_pair._prepared_gguf_projection",
        canonical,
    )
    monkeypatch.setattr(torch.ops._C, "silu_and_mul", silu, raising=False)
    empty = torch.empty(0)
    for m in (512, 1, 5, 16, 20, 32):
        x = torch.empty(m, 5120, dtype=torch.float16)
        out = _native_gated_pair(
            x,
            empty,
            empty,
            18,
            18,
            [empty],
            [empty],
            [None],
            [2, 18, 32, 0, 0, 8704, 8704, 0, 2],
            [],
            [128, -1],
        )
        expected = torch.nn.functional.silu(torch.full_like(out, 2)) * 2
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert len(calls) == 6
    assert all(c[1:] == (18, 8704, [], [128, -1]) for c in calls)
