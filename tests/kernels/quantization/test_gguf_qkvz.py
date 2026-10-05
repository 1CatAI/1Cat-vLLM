# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.kernels.gguf import native_qkvz_capabilities
from vllm.model_executor.layers.quantization.gguf_qkvz import (
    _native_qkvz,
    prepared_source_views,
)


@pytest.mark.parametrize(
    "qkv,z",
    [
        (23, 23),
        (21, 18),
        (18, 21),
        (21, 21),
        (21, 10),
        (18, 10),
        (17, 18),
        (18, 23),
        (21, 22),
        (10, 18),
        (18, 12),
        (21, 23),
        (18, 18),
        (23, 21),
        (21, 12),
        (23, 18),
        (18, 22),
        (12, 21),
        (16, 23),
    ],
)
def test_qkvz_admission_is_measured_complete_projection(monkeypatch, qkv, z):
    monkeypatch.setattr(
        torch.ops._C, "gguf_qkvz_sm70_out", lambda *a: None, raising=False
    )
    types = (qkv, qkv, qkv, z, 30, 30)
    caps = native_qkvz_capabilities(types, 5120, 4120, torch.float16)
    assert len(caps) == 6 and all(c.reason is None for c in caps)
    assert all(c.min_m == c.max_m == 8 for c in caps)
    for arguments in (
        (types, 2560, 4120, torch.float16),
        (types, 5120, 4096, torch.float16),
        (types, 5120, 4120, torch.bfloat16),
        (types, 5120, 4120, torch.float16, False),
        (types, 5120, 4120, torch.float16, True, 80),
        ((21, 18, 21, z, 30, 30), 5120, 4120, torch.float16),
    ):
        assert all(c.reason for c in native_qkvz_capabilities(*arguments))


@pytest.mark.parametrize("kind,bits", [(10, 2), (12, 4)])
def test_canonical_sources_alias_existing_stream_and_stat_stride(kind, bits):
    k, n = 256, 256
    codes = torch.arange(k * n * bits // 32, dtype=torch.int32).view(k, n * bits // 32)
    group = 16 if kind == 10 else 32
    stats = torch.arange(k // group * n, dtype=torch.int32).view(k // group, n)
    projection = SimpleNamespace(
        source_type=kind,
        source_output_sizes=(64, 64, 128),
        kernel=SimpleNamespace(config=SimpleNamespace(partition_weight_shape=(k, n))),
        codes=codes,
        stats=stats,
    )
    views = prepared_source_views([projection])
    offset = 0
    for width, (code, scale) in zip(projection.source_output_sizes, views):
        assert code.untyped_storage().data_ptr() == codes.untyped_storage().data_ptr()
        assert scale.untyped_storage().data_ptr() == stats.untyped_storage().data_ptr()
        assert code.storage_offset() == offset * k * bits // 32
        assert scale.storage_offset() == offset and scale.stride() == (n, 1)
        torch.testing.assert_close(
            scale, stats[:, offset : offset + width], rtol=0, atol=0
        )
        offset += width


def test_runtime_rows_keep_fallback_and_workspace_explicit(monkeypatch):
    calls: list[tuple[Any, ...]] = []

    def joint(out, x, weights, scales, types, partials, counters):
        calls.append(("joint", x.shape[0], partials, counters))
        out.fill_(7)

    def fallback(x, codes, stats, caches, descriptors, cache_bands, blas_bands):
        calls.append(("canonical", x.shape[0]))
        return x.new_full((*x.shape[:-1], 4120), 3)

    monkeypatch.setattr(torch.ops._C, "gguf_qkvz_sm70_out", joint, raising=False)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.gguf_qkvz._prepared_gguf_mixed_projection",
        fallback,
    )
    partials, counters = torch.empty(65, 2, 512), torch.zeros(65, dtype=torch.int32)
    for m in (512, 8, 1, 5, 16, 20, 32, 8):
        output = _native_qkvz(
            torch.empty(m, 5120, dtype=torch.float16),
            [],
            [],
            [],
            partials,
            counters,
            [],
            [],
            [],
            [],
            [],
            [],
        )
        assert output.shape == (m, 4120)
        assert torch.equal(output, torch.full_like(output, 7 if m == 8 else 3))
    assert len(calls) == 8
    assert calls[1][2] is partials and calls[1][3] is counters


def test_dynamic_export_keeps_joint_projection_opaque():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("partials", torch.empty(65, 2, 512))
            self.register_buffer("counters", torch.zeros(65, dtype=torch.int32))

        def forward(self, x):
            return torch.ops.vllm.gguf_native_qkvz(
                x,
                [],
                [],
                [],
                self.partials,
                self.counters,
                [],
                [],
                [],
                [],
                [],
                [],
            )

    m = torch.export.Dim("m", min=1, max=8192)
    exported = torch.export.export(
        Model(),
        (torch.empty(512, 5120, dtype=torch.float16),),
        dynamic_shapes={"x": {0: m}},
    )
    nodes = [n for n in exported.graph.nodes if n.op == "call_function"]
    assert any(n.target == torch.ops.vllm.gguf_native_qkvz.default for n in nodes)
    assert not any("cat" in str(n.target) or "mm" in str(n.target) for n in nodes)
