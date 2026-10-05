# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.kernels.gguf import iq3_gated_pair_capability
from vllm.model_executor.layers.quantization.gguf_iq3_gated import (
    _iq3_gated_pair,
    apply_iq3_gated_pair,
)
from vllm.model_executor.layers.quantization.gguf_iq3_records import (
    signed_index_records,
)


def test_source_bytes_are_losslessly_permuted():
    rng = np.random.default_rng(1290)
    for n, blocks in ((32, 1), (64, 3), (96, 20)):
        source = rng.integers(0, 256, (n, blocks * 110), dtype=np.uint8)
        before = source.copy()
        records = signed_index_records(source)  # Also independently checks inverse.
        assert records.nbytes == source.nbytes
        np.testing.assert_array_equal(source, before)
        assert records.flags.c_contiguous


def test_only_measured_pair_and_m_are_admitted(monkeypatch):
    monkeypatch.setattr(
        torch.ops._C, "gguf_iq3_gated_sm70_out", lambda *a: None, raising=False
    )
    admitted = iq3_gated_pair_capability((21, 21), 5120, 4352, torch.float16)
    assert admitted.reason is None
    assert [m for m in (1, 2, 4, 8, 16, 32, 512) if admitted.supports_m(m)] == [8]
    for types, k, n in (
        ((18, 21), 5120, 4352),
        ((21, 21), 5120, 4096),
        ((21, 21), 4352, 5120),
    ):
        assert iq3_gated_pair_capability(types, k, n, torch.float16).reason
    assert (
        iq3_gated_pair_capability(
            (21, 21), 5120, 4352, torch.float16, compute_capability=80
        ).reason
        == "requires_sm70_device"
    )
    assert iq3_gated_pair_capability((21, 21), 5120, 4352, torch.bfloat16).reason
    assert iq3_gated_pair_capability((21, 21), 5120, 4352, torch.float16, False).reason


@pytest.mark.parametrize("widths", [(4352, 4352), (8704,)])
def test_actual_m_dispatch_and_canonical_rounding(monkeypatch, widths):
    calls = []

    def pair(output, *args):
        calls.append("pair")
        output.fill_(7)

    def canonical(x, *args):
        calls.append("canonical")
        return torch.full((x.shape[0], args[8]), 2, dtype=x.dtype)

    def silu(output, x):
        gate, up = x.chunk(2, dim=-1)
        output.copy_(torch.nn.functional.silu(gate) * up)

    monkeypatch.setattr(torch.ops._C, "gguf_iq3_gated_sm70_out", pair, raising=False)
    monkeypatch.setattr(torch.ops._C, "silu_and_mul", silu, raising=False)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.gguf_iq3_gated."
        "_prepared_gguf_projection",
        canonical,
    )
    empty = torch.empty(0)
    for m in (512, 8, 1, 16, 32, 8, 512):
        calls.clear()
        output = _iq3_gated_pair(
            torch.empty(m, 5120, dtype=torch.float16),
            empty,
            empty,
            [empty] * len(widths),
            [empty] * len(widths),
            [0] * len(widths),
            [0] * len(widths),
            list(widths),
            [512, -1],
        )
        assert output.shape == (m, 4352)
        assert calls == (["pair"] if m == 8 else ["canonical"] * len(widths))
        if m != 8:
            torch.testing.assert_close(
                output,
                torch.full_like(
                    output,
                    (
                        torch.nn.functional.silu(torch.tensor(2, dtype=torch.float16))
                        * 2
                    ).item(),
                ),
                rtol=0,
                atol=0,
            )


def test_range_compilation_keeps_gated_dispatch_opaque():
    torch._dynamo.reset()

    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def pair(x, records, codes, stats):
        return torch.ops.vllm.gguf_iq3_gated_pair(
            x,
            records,
            records,
            [codes, codes],
            [stats, stats],
            [0, 0],
            [0, 0],
            [4352, 4352],
            [512, -1],
        )

    compiled = torch.compile(pair, backend=backend, dynamic=True, fullgraph=True)
    records = torch.empty(9574400, dtype=torch.uint8, device="meta")
    codes = torch.empty(5120, 272, dtype=torch.int32, device="meta")
    stats = torch.empty(160, 4352, dtype=torch.int64, device="meta")
    for m in (512, 8, 16, 512):
        x = torch.empty(m, 5120, dtype=torch.float16, device="meta")
        torch._dynamo.mark_dynamic(x, 0, min=2, max=8192)
        result = compiled(x, records, codes, stats)
        assert result.shape == (m, 4352) and result.stride() == (4352, 1)
    assert len(graphs) == 1
    nodes = [n for n in graphs[0].graph.nodes if n.op == "call_function"]
    assert len(nodes) == 1
    assert "gguf_iq3_gated_pair" in str(nodes[0].target)
    torch._dynamo.reset()


def test_registered_parameter_list_compiles_through_layer_wrapper():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kernel = SimpleNamespace(operator_capabilities=())
            self.codes = torch.nn.Parameter(
                torch.empty(0, device="meta", dtype=torch.int32), False
            )
            self.stats = torch.nn.Parameter(
                torch.empty(0, device="meta", dtype=torch.int64), False
            )
            self.gguf_tm_k_ld = self.gguf_tm_q_ld = 0
            self.logical_output_size = 8704

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gguf_iq3_gated_records = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.empty(0, device="meta", dtype=torch.uint8), False
                    )
                    for _ in range(2)
                ]
            )
            self.gguf_tm_projections = torch.nn.ModuleList([Projection()])

        def forward(self, x):
            return apply_iq3_gated_pair(self, x)

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
