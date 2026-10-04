# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
    _expert_gate_up,
)


@pytest.mark.parametrize(
    "source,m,expected",
    [
        (21, 1, "raw"),
        (21, 5, "raw"),
        (21, 20, "raw"),
        (22, 5, "raw"),
        (22, 20, "gemm"),
        (21, 10, "gemm"),
        (21, 512, "gemm"),
    ],
)
def test_original_batch_controls_dispatch(monkeypatch, source, m, expected):
    calls = []

    def raw(gate, up, *args):
        calls.append("raw")
        gate.fill_(3)
        up.fill_(7)

    def canonical(out, x, offsets, weights, *args):
        calls.append("gemm")
        out.fill_(int(weights.item()))

    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(
            gguf_lattice_raw_grouped_gate_up_sm70_out=raw,
            gguf_lattice_grouped_gemm_sm70_out=canonical,
        ),
    )
    top_k, n = 10, 160
    x = torch.zeros(m * top_k, 2560, dtype=torch.float16)
    empty = torch.empty(0)
    gate, up = _expert_gate_up(
        x,
        empty,
        empty,
        empty,
        empty,
        torch.tensor(3),
        empty,
        torch.tensor(7),
        empty,
        source,
        512,
        16,
        n,
        top_k,
        [1, 5, 20] if source == 21 else [1, 5],
        [],
    )
    assert calls == (["raw"] if expected == "raw" else ["gemm", "gemm"])
    assert gate.shape == up.shape == (m * top_k, n)
    assert gate.dtype == up.dtype == torch.float16
    assert torch.all(gate == 3) and torch.all(up == 7)


def test_canonical_vector_fallback_uses_routed_rows(monkeypatch):
    calls = []

    def vector(out, *args):
        calls.append("vector")
        out.zero_()

    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(
            gguf_lattice_grouped_vec_sm70_out=vector,
        ),
    )
    empty = torch.empty(0)
    _expert_gate_up(
        torch.zeros(40, 2560, dtype=torch.float16),
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        21,
        512,
        32,
        160,
        10,
        [1, 5],
        [1, 128],
    )
    assert calls == ["vector", "vector"]
