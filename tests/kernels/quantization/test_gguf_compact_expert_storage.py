# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.model_executor.kernels.gguf import (
    compact_expert_storage_capabilities,
    compact_lattice_grouped_capabilities,
)
from vllm.model_executor.layers.quantization import gguf_turbomind_moe as module
from vllm.transformers_utils.gguf_tensor_reader import dequantize


@pytest.mark.parametrize(
    "kind,k,n,experts,is_sm70,reason",
    [
        (18, 2560, 160, 512, True, None),
        (18, 2560, 160, 512, False, "requires_sm70"),
        (18, 2560, 320, 512, True, "compact_expert_storage_shape_has_no_calibration"),
        (21, 2560, 160, 512, True, "compact_expert_storage_shape_has_no_calibration"),
        (18, 2560, 160, 4, True, "compact_expert_storage_shape_has_no_calibration"),
    ],
)
def test_compact_storage_admission(kind, k, n, experts, is_sm70, reason):
    capability = compact_expert_storage_capabilities(
        kind, k, n, experts, torch.float16, is_sm70=is_sm70
    )[0]
    assert capability.reason == reason


def source(n=640, k=2560):
    rng = np.random.default_rng(573)
    blocks = rng.integers(0, 256, (n, k // 256, 98), dtype=np.uint8)
    blocks[:, :, :2] = np.array([0.01], dtype="<f2").view(np.uint8)
    return blocks.reshape(n, -1)


def test_compact_bank_skips_canonical_and_extra_retention(monkeypatch):
    calls = []

    def reorder(out, raw, kind, k):
        calls.append((raw.shape, out.numel(), kind, k))
        out.zero_()

    monkeypatch.setattr(
        module,
        "current_platform",
        SimpleNamespace(
            is_device_capability=lambda _: True,
        ),
    )
    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(
            gguf_lattice_compact_reorder_sm70_out=reorder,
            gguf_lattice_compact_grouped_sm70_out=object(),
        ),
    )
    monkeypatch.setattr(
        module,
        "transcode_lattice",
        lambda *a: pytest.fail("compact storage must not build canonical weights"),
    )
    weight = torch.from_numpy(source())
    bank = module.GGUFExpertBank(
        18, 512, "cpu", torch.float16, retain_raw=True, prefer_compact=True
    )
    for expert in range(512):
        bank.add(expert, weight, 0, 4, axis=0)
    bank.finalize()
    assert bank.storage_layout == "compact_original"
    assert bank.weights.shape == (512, 156800)
    assert sum(b.numel() * b.element_size() for b in bank.buffers()) == 80281600
    assert not bank.pending and not bank.raw_pending
    assert not any(
        hasattr(bank, name)
        for name in (
            "raw_weights",
            "stats",
            "weight_ptrs",
            "stat_ptrs",
        )
    )
    assert len(calls) == 512 and calls[0] == ((160, 984), 156800, 18, 2560)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_bank_graph_with_changed_routing(monkeypatch):
    # Small bank keeps the integration test cheap. Geometry admission is
    # tested independently; use the production operator capability here.
    monkeypatch.setattr(
        module,
        "compact_expert_storage_capabilities",
        lambda kind, k, n, experts, dtype, **kw: compact_lattice_grouped_capabilities(
            kind, k, n, experts, dtype
        ),
    )
    bank = module.GGUFExpertBank(18, 4, "cuda", torch.float16, prefer_compact=True)
    references = []
    for expert in range(4):
        data = source(n=128, k=768)
        blocks = data.reshape(128, 3, 98)
        blocks[:, :, :2] = np.array([0.01 * (expert + 1)], dtype="<f2").view(np.uint8)
        bank.add(expert, torch.from_numpy(data).cuda(), 0, 4, axis=0)
        references.append(
            torch.from_numpy(dequantize(data[:32], 18)).half().cuda().float()
        )
    bank.finalize()
    x = torch.randn(21, 768, device="cuda", dtype=torch.float16)
    offsets = torch.tensor([0, 5, 5, 17, 21], device="cuda", dtype=torch.int32)
    ids = torch.empty(0, device="cuda", dtype=torch.int32)
    for _ in range(3):
        bank(x, offsets, ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = bank(x, offsets, ids)
    for boundaries in ([0, 5, 5, 17, 21], [0, 0, 16, 16, 21]):
        offsets.copy_(torch.tensor(boundaries, device="cuda", dtype=torch.int32))
        x.normal_()
        graph.replay()
        expected = torch.empty_like(out, dtype=torch.float32)
        for expert, reference in enumerate(references):
            begin, end = boundaries[expert : expert + 2]
            expected[begin:end] = x[begin:end].float() @ reference.T
        torch.testing.assert_close(out.float(), expected, rtol=0.001, atol=0.003)
    assert list(dict(bank.named_buffers())) == ["weights"]
