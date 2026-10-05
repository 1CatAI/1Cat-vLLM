# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.model_executor.kernels.gguf import iq3_gated_pair_capability
from vllm.model_executor.layers.quantization.gguf_iq3_gated import _iq3_gated_pair
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


def test_actual_m_dispatch_and_canonical_rounding(monkeypatch):
    calls = []

    def pair(output, *args):
        calls.append("pair")
        output.fill_(7)

    def canonical(x, *args):
        calls.append("canonical")
        return torch.full((x.shape[0], 4352), 2, dtype=x.dtype)

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
    for m in (512, 8, 1, 16, 8, 512):
        calls.clear()
        output = _iq3_gated_pair(
            torch.empty(m, 5120, dtype=torch.float16),
            empty,
            empty,
            [empty] * 2,
            [empty] * 2,
            [0] * 2,
            [0] * 2,
            [512, -1],
        )
        assert output.shape == (m, 4352)
        assert calls == (["pair"] if m == 8 else ["canonical", "canonical"])
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
