# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch

from vllm.model_executor.kernels.gguf import native_qkv_capabilities
from vllm.model_executor.layers.quantization.gguf_qkv import _native_qkv


@pytest.mark.parametrize(
    "sources",
    [
        (16, 21, 21),
        (10, 23, 21),
        (10, 22, 18),
        (10, 12, 21),
        (10, 18, 21),
        (21, 12, 12),
        (23, 23, 12),
        (10, 23, 12),
        (21, 23, 12),
        (21, 23, 23),
        (23, 12, 12),
        (18, 23, 12),
        (18, 12, 12),
        (12, 23, 21),
    ],
)
def test_only_calibrated_logical_qkv_order_is_admitted(monkeypatch, sources):
    monkeypatch.setattr(
        torch.ops._C, "gguf_qkv_sm70_out", lambda *args: None, raising=False
    )
    caps = native_qkv_capabilities(sources, 5120, 3584, torch.float16)
    assert len(caps) == 3 and all(c.reason is None for c in caps)
    assert all(
        [m for m in (1, 5, 8, 16, 32, 512) if c.supports_m(m)] == [8] for c in caps
    )
    for args in (
        (sources, 2560, 3584, torch.float16),
        (sources, 5120, 4120, torch.float16),
        (sources, 5120, 3584, torch.bfloat16),
        (sources, 5120, 3584, torch.float16, False),
        (sources, 5120, 3584, torch.float16, True, 80),
        ((21, 21, 21), 5120, 3584, torch.float16),
    ):
        assert all(c.reason for c in native_qkv_capabilities(*args))


@pytest.mark.parametrize("coalesced", [False, True])
def test_other_m_preserves_single_or_mixed_canonical_boundary(monkeypatch, coalesced):
    calls: list[tuple[Any, ...]] = []

    def native(out, rows, weights, scales, types, partials, counters):
        calls.append(("native", rows.shape[0], types))
        out.fill_(7)

    def single(x, c, s, cache, family, decoder, group, kld, qld, n, logical, cb, bb):
        calls.append(("single", x.shape[0], decoder, cb, bb))
        return x.new_full((x.shape[0], logical), 3)

    def mixed(x, codes, stats, caches, descriptors, cb, bb):
        calls.append(("mixed", x.shape[0], descriptors, cb, bb))
        return x.new_full((x.shape[0], 3584), 3)

    monkeypatch.setattr(torch.ops._C, "gguf_qkv_sm70_out", native, raising=False)
    module = "vllm.model_executor.layers.quantization.gguf_qkv"
    monkeypatch.setattr(f"{module}._prepared_gguf_projection", single)
    monkeypatch.setattr(f"{module}._prepared_gguf_mixed_projection", mixed)
    empty = torch.empty(0)
    count = 1 if coalesced else 2
    descriptors = [2, 21, 32, 0, 0, 3584, 3584, 2, 2] * count
    for m in (512, 8, 1, 5, 16, 20, 32, 8):
        output = _native_qkv(
            torch.empty(m, 5120, dtype=torch.float16),
            [empty] * 3,
            [empty] * 3,
            [21, 104, 104],
            empty,
            empty,
            [empty] * count,
            [empty] * count,
            [None] * count,
            descriptors,
            [32, 512],
            [512, -1],
        )
        if m == 8:
            assert calls[-1] == ("native", 8, [21, 104, 104]) and torch.all(output == 7)
        else:
            expected = "single" if coalesced else "mixed"
            assert calls[-1][0:2] == (expected, m)
            assert calls[-1][-2:] == ([32, 512], [512, -1])
            assert torch.all(output == 3)
