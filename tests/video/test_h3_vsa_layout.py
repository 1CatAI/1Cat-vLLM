# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact H3 layout fusion, request ownership and fallback checks."""

import pytest
import torch

from vllm.model_executor.models.minimax_h3 import vsa

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a leased SM70 GPU"
)


@pytest.mark.parametrize(
    "prefix,grid,batch,heads,topk",
    [
        ((3, 5), (1, 3, 7), 2, 3, 1),
        ((65,), (7, 5, 11), 1, 2, 2),
        ((), (1, 1, 1), 1, 1, 1),
        ((1, 65, 2), (5, 6, 7), 1, 2, 100),
    ],
)
def test_fused_layout_preserves_output_and_work(
    prefix, grid, batch, heads, topk, monkeypatch
):
    torch.manual_seed(814)
    rows = sum(prefix) + grid[0] * grid[1] * grid[2]
    q, k, v, gate = [
        torch.randn(batch, rows, heads, 128, device="cuda", dtype=torch.float16)
        for _ in range(4)
    ]
    assert vsa._layout_ops(q, k, v, gate) is not None
    kwargs = dict(
        prefix_segments=prefix,
        video_shape=grid,
        gate_compress=gate,
        topk=topk,
        scale=128**-0.5,
    )
    with monkeypatch.context() as patch:
        patch.setattr(vsa, "_layout_ops", lambda *args: None)
        expected, expected_work = vsa.h3_vsa_attention(q, k, v, **kwargs)
    with vsa.h3_vsa_workspace():
        actual, work = vsa.h3_vsa_attention(q, k, v, **kwargs)
        saved = actual.clone()
        scratch = vsa._layout_scratch(
            q, vsa._get_h3_tile_metadata(prefix, grid, q.device)[1].numel() * 64
        )
        vsa.h3_vsa_attention(q * 0.5, k, v, **kwargs)
        assert torch.equal(actual.view(torch.int16), saved.view(torch.int16))
        assert scratch is next(iter(vsa._layout_buffers.get().values()))
    assert vsa._layout_buffers.get() is None
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert all(work[key] == value for key, value in expected_work.items())


def test_workspace_isolates_streams_nested_requests_and_failure():
    q = torch.empty(1, 8, 2, 128, device="cuda", dtype=torch.float16)
    with vsa.h3_vsa_workspace():
        first = vsa._layout_scratch(q, 64)
        assert first is vsa._layout_scratch(q, 64)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            assert first is not vsa._layout_scratch(q, 64)
        with (
            pytest.raises(RuntimeError, match="request failed"),
            vsa.h3_vsa_workspace(),
        ):
            assert first is not vsa._layout_scratch(q, 64)
            raise RuntimeError("request failed")
        assert first is vsa._layout_scratch(q, 64)
    assert vsa._layout_buffers.get() is None
    with vsa.h3_vsa_workspace():
        assert first is not vsa._layout_scratch(q, 64)


def test_layout_falls_back_for_views_and_older_wheels(monkeypatch):
    q = torch.empty(1, 8, 2, 128, device="cuda", dtype=torch.float16)
    assert vsa._layout_ops(q[:, ::2]) is None
    unaligned = torch.empty(q.numel() + 1, device=q.device, dtype=q.dtype)[1:].view_as(
        q
    )
    assert vsa._layout_ops(unaligned) is None
    assert vsa._layout_ops(q.requires_grad_()) is None
    monkeypatch.setattr(vsa, "sparse_extension", lambda: object())
    assert vsa._layout_ops(q.detach()) is None


def test_private_layout_rejects_invalid_shapes_and_aliasing():
    q = torch.empty(1, 64, 2, 128, device="cuda", dtype=torch.float16)
    ops = vsa.sparse_extension()
    rows = torch.arange(64, device=q.device, dtype=torch.int32)
    out = torch.empty(3, *q.shape, device=q.device, dtype=q.dtype)
    with pytest.raises(RuntimeError, match="source map"):
        ops._h3_tile_qkv_prevalidated(q, q, q, rows[:-1], out)
    with pytest.raises(RuntimeError, match="single memory location|overlap"):
        ops._h3_tile_qkv_prevalidated(out[0], q, q, rows, out)
    with pytest.raises(RuntimeError, match="compressed geometry"):
        ops._h3_gate_untile_prevalidated(q, q, q, rows)


def test_gate_fusion_preserves_two_fp16_roundings():
    torch.manual_seed(819)
    sparse = torch.randn(1, 128, 2, 128, device="cuda", dtype=torch.float16)
    compressed = torch.randn(1, 2, 2, 128, device="cuda", dtype=torch.float16)
    rows = torch.tensor([0, 1, 17, 64, 65, 100, 127], device="cuda", dtype=torch.int32)
    gate = torch.randn(1, 7, 2, 128, device="cuda", dtype=torch.float16)
    # Include large finite products and overflow: the fusion must not clamp.
    gate[:, 0] = 65504
    gate[:, 1] = -0.0
    expected = sparse[:, rows.long()] + compressed[:, (rows // 64).long()] * gate
    actual = vsa.sparse_extension()._h3_gate_untile_prevalidated(
        sparse, compressed, gate, rows
    )
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
