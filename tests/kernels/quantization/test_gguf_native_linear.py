# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch

from vllm.model_executor.kernels.gguf import native_linear_capability
from vllm.model_executor.layers.quantization.gguf_native_linear import _native_linear


@pytest.mark.parametrize("source_type", [10, 17, 18, 21, 22])
def test_admission_is_measured_type_shape_and_m(monkeypatch, source_type):
    for name in (
        "gguf_native_linear_n64_sm70_out",
        "gguf_canonical_linear_n64_sm70_out",
    ):
        monkeypatch.setattr(torch.ops._C, name, lambda *args: None, raising=False)
    cap = native_linear_capability(source_type, 4352, 5120, torch.float16)
    assert cap.reason is None
    assert [m for m in (1, 2, 5, 8, 16, 20, 32, 512) if cap.supports_m(m)] == [8]
    for arguments in (
        (23, 4352, 5120, torch.float16),
        (12, 4352, 5120, torch.float16),
        (source_type, 1536, 5120, torch.float16),
        (source_type, 4352, 4096, torch.float16),
        (source_type, 4352, 5120, torch.bfloat16),
        (source_type, 4352, 5120, torch.float16, False),
        (source_type, 4352, 5120, torch.float16, True, 80),
    ):
        assert native_linear_capability(*arguments).reason


@pytest.mark.parametrize("source_type", [10, 17, 18, 21, 22])
def test_runtime_m_preserves_single_canonical_call(monkeypatch, source_type):
    calls: list[tuple[Any, ...]] = []

    def raw(out, rows, records, partials, counters, kind):
        calls.append(("raw", kind))
        out.fill_(7)

    def affine(out, rows, codes, stats, partials, counters, bits, group):
        calls.append(("affine", bits, group))
        out.fill_(7)

    def canonical(x, c, s, cache, family, decoder, group, kld, qld, n, logical, cb, bb):
        calls.append(("canonical", x.shape[0], decoder, cb, bb))
        return x.new_full((*x.shape[:-1], 5120), 3)

    monkeypatch.setattr(
        torch.ops._C, "gguf_native_linear_n64_sm70_out", raw, raising=False
    )
    monkeypatch.setattr(
        torch.ops._C, "gguf_canonical_linear_n64_sm70_out", affine, raising=False
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.gguf_native_linear._prepared_gguf_projection",
        canonical,
    )
    empty = torch.empty(0)
    descriptor = [2, source_type, 32, 0, 0, 5120, 5120, 0, 2]
    for m in (512, 8, 1, 5, 16, 20, 32, 8):
        output = _native_linear(
            torch.empty(m, 4352, dtype=torch.float16),
            empty,
            empty,
            empty,
            source_type,
            [empty],
            [empty],
            [None],
            descriptor,
            [],
            [128, -1],
        )
        assert output.shape == (m, 5120)
        assert torch.equal(output, torch.full_like(output, 7 if m == 8 else 3))
    assert len(calls) == 8
    assert all(c[0] == "canonical" for i, c in enumerate(calls) if i not in (1, 7))
    assert calls[1] == (
        ("affine", 2, 16) if source_type == 10 else ("raw", source_type)
    )


def test_dynamic_export_keeps_runtime_m_decision_opaque():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("records", torch.empty(0, dtype=torch.uint8))
            self.register_buffer("partials", torch.empty(80, 2, 512))
            self.register_buffer("counters", torch.zeros(80, dtype=torch.int32))
            self.register_buffer("codes", torch.empty(0, dtype=torch.int32))
            self.register_buffer("stats", torch.empty(0, dtype=torch.int64))

        def forward(self, x):
            return torch.ops.vllm.gguf_native_linear(
                x,
                self.records,
                self.partials,
                self.counters,
                21,
                [self.codes],
                [self.stats],
                [None],
                [2, 21, 32, 0, 0, 5120, 5120, 0, 0],
                [],
                [],
            )

    export = torch.export.export(
        Model(),
        (torch.empty(512, 4352, dtype=torch.float16),),
        dynamic_shapes={"x": {0: torch.export.Dim("rows", min=1, max=8192)}},
    )
    nodes = [n for n in export.graph.nodes if n.op == "call_function"]
    assert len(nodes) == 1
    assert nodes[0].target == torch.ops.vllm.gguf_native_linear.default
    value = nodes[0].meta["val"]
    assert value.shape[1] == 5120
    assert value.dtype == torch.float16
    assert len(export.range_constraints) == 1
