# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
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


@pytest.mark.parametrize("kind", [21, 22])
@pytest.mark.parametrize("rank", range(4))
def test_expert_bank_retains_complete_tp_rows_and_counts_storage(
    monkeypatch, kind, rank
):
    from vllm.model_executor.kernels.gguf import (
        GGUFDecoderFamily,
        GGUFOperatorCapability,
    )
    from vllm.model_executor.layers.quantization import gguf_turbomind_moe as module
    from vllm.transformers_utils.gguf_tensor_reader import quant_size

    monkeypatch.setattr(
        module,
        "current_platform",
        SimpleNamespace(
            is_device_capability=lambda _: True,
        ),
    )
    monkeypatch.setattr(
        module,
        "raw_grouped_gate_up_capabilities",
        lambda *a, **kw: (
            GGUFOperatorCapability(
                GGUFDecoderFamily.LATTICE, str(kind), "raw", True, min_m=1, max_m=1
            ),
        ),
    )
    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(
            gguf_lattice_sm70_prepare=lambda codes, stats, *args: (
                codes,
                stats,
                torch.tensor([1, 2, 3, 4]),
            ),
            awq_moe_build_strided_ptrs=lambda *args: (
                torch.empty(2, dtype=torch.uint8),
                torch.empty(2, dtype=torch.uint8),
            ),
            gguf_lattice_grouped_gemm_sm70_out=object(),
        ),
    )
    _, block_bytes = quant_size(kind)
    sources = np.random.default_rng(kind).integers(
        0, 256, (2, 128, block_bytes), dtype=np.uint8
    )
    sources[:, :, :2] = np.float16(0.01).tobytes()[0], np.float16(0.01).tobytes()[1]
    bank = module.GGUFExpertBank(kind, 2, "cpu", torch.float16, retain_raw=True)
    for expert in range(2):
        bank.add(expert, torch.from_numpy(sources[expert]), rank, 4, axis=0)
    bank.finalize()
    assert not bank.pending and not bank.raw_pending
    stride = (block_bytes + 7) // 8 * 8
    assert bank.raw_weights.shape == (2, 32, stride)
    assert bank.raw_weights.dtype == torch.uint8
    expected = torch.from_numpy(sources[:, rank * 32 : (rank + 1) * 32])
    assert torch.equal(bank.raw_weights[:, :, :block_bytes], expected)
    assert torch.count_nonzero(bank.raw_weights[:, :, block_bytes:]) == 0
    assert bank.raw_weights.numel() == 2 * 32 * stride
