# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.gguf_turbomind import (
    _prepared_gguf_projection,
)


def test_runtime_dispatch_uses_actual_rows(monkeypatch):
    """Exercise both branches without GPU timing or packed-weight arithmetic."""
    calls = []
    workspace = torch.empty(5120 * 4352, dtype=torch.float16)
    module = "vllm.model_executor.layers.quantization.gguf_turbomind"
    monkeypatch.setattr(module + "._get_affine_blas_workspace", lambda _: workspace)

    def fused(output, *args):
        calls.append("fused")
        output.zero_()

    def blas(output, *args):
        calls.append("blas")
        output.zero_()

    monkeypatch.setattr(torch.ops._C, "gguf_lattice_gemm_sm70_out", fused)
    monkeypatch.setattr(torch.ops._C, "gguf_lattice_blas_sm70_out", blas)
    codes = torch.empty(0, dtype=torch.int32)
    stats = torch.empty(0, dtype=torch.int64)
    for m in (512, 8, 1, 16, 512):
        x = torch.empty(m, 5120, dtype=torch.float16)
        output = _prepared_gguf_projection(
            x, codes, stats, None, 2, 21, 32, 0, 0, 4352, 4352, [], [512, -1]
        )
        assert calls[-1] == ("blas" if m >= 512 else "fused")
        assert output.shape == (m, 4352)
        assert output.is_contiguous()


def test_q8_segment_uses_m8_and_preserves_prefill_policy(monkeypatch):
    calls: list[tuple[str, int, list[int]]] = []
    module = "vllm.model_executor.layers.quantization.gguf_dense_hmma"

    def segments(rows, *args, m8_only=False):
        assert m8_only
        calls.append(("segments", rows.shape[0], []))
        args[-2].zero_()

    def restore(rows, *args):
        calls.append(("restore", rows.shape[0], args[-1]))
        args[5].zero_()

    monkeypatch.setattr(module + ".apply_segments", segments)
    monkeypatch.setattr(module + ".restore_and_apply", restore)
    codes = stats = high = torch.empty(0, dtype=torch.uint8)
    for m in (512, 8, 1, 20, 8):
        output = _prepared_gguf_projection(
            torch.empty(m, 1024, dtype=torch.float16),
            codes,
            stats,
            high,
            5,
            4,
            32,
            0,
            0,
            5120,
            5120,
            [],
            [512, -1],
        )
        assert output.shape == (m, 5120)
    assert calls == [
        ("restore", 512, [512, -1]),
        ("segments", 8, []),
        ("restore", 1, [512, -1]),
        ("restore", 20, [512, -1]),
        ("segments", 8, []),
    ]


def test_fake_graph_keeps_runtime_projection_opaque():
    torch._dynamo.reset()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def projection(x, codes, stats):
        return torch.ops.vllm.prepared_gguf_projection(
            x, codes, stats, None, 2, 21, 32, 0, 0, 4352, 4352, [], [512, -1]
        )

    compiled = torch.compile(projection, backend=backend, dynamic=True, fullgraph=True)
    codes = torch.empty(5120, 272, dtype=torch.int32, device="meta")
    stats = torch.empty(160, 4352, dtype=torch.int64, device="meta")
    for m in (512, 8, 16, 512):
        x = torch.empty(m, 5120, dtype=torch.float16, device="meta")
        torch._dynamo.mark_dynamic(x, 0, min=2, max=8192)
        output = compiled(x, codes, stats)
        assert output.shape == (m, 4352)
        assert output.stride() == (4352, 1)
    assert len(graphs) == 1
    nodes = [node for node in graphs[0].graph.nodes if node.op == "call_function"]
    assert len(nodes) == 1
    assert nodes[0].target in (
        torch.ops.vllm.prepared_gguf_projection,
        torch.ops.vllm.prepared_gguf_projection.default,
    )
    torch._dynamo.reset()
