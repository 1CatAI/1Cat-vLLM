# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Codec-independent q16/head-group and one/two-pass native schedules.

The existing grouped relative-L2 gate is fixed at 1.02 times the FP16 rounding
floor plus 2e-6; changed graph inputs must satisfy the same bound.
"""

from dataclasses import replace

import pytest
import torch

from tests.kernels.attention.test_sm70_kv_engine_bindings import cache, reference
from vllm.v1.attention.kv_codecs import KV_CODECS

package = pytest.importorskip("flash_attn_v100")
native = pytest.importorskip(
    "flash_attn_v100.flash_attn_interface"
).flash_attn_v100_cuda
CODECS = tuple(c for c in KV_CODECS if c.native_id is not None)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 native grouped-plan qualification",
)


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize("one_pass", [True, False], ids=["one_pass", "two_pass"])
@pytest.mark.parametrize(
    "batch,rows,page,heads,length",
    [
        (1, 9, 32, 1, 127),
        (1, 16, 1648, 1, 1665),
        (1, 16, 3296, 2, 3327),
        (1, 16, 3296, 1, 3327),
        (2, 8, 32, 1, 257),
    ],
)
def test_shared_grouped_plans_and_changed_graph_inputs(
    codec, one_pass, batch, rows, page, heads, length
):
    assert getattr(native, "grouped_fp32_execution_plan_revision", 0) >= 2
    torch.manual_seed(941)
    _, view = cache(codec, pages=5, page=page, heads=heads)
    table = torch.tensor([[4, 1, 3, 0, 2]] * batch, device="cuda", dtype=torch.int32)
    if page == 32:
        table = table.repeat(1, 4)
    lengths = (
        torch.arange(rows, device="cuda", dtype=torch.int32) + length - rows + 1
    ).repeat(batch)
    q = torch.randn((batch * rows, heads * 6, 256), device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)

    def launch():
        package.flash_attn_grouped_fp32_paged(
            q,
            view.key,
            view.value,
            table,
            lengths,
            out=out,
            softmax_scale=256**-0.5,
            kv_cache_dtype=codec.name,
            one_pass=one_pass,
            **view.scale_kwargs,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for delta in (0, 1, -2):
        lengths.add_(delta)
        lengths[0] = 0
        table.copy_(table.flip(1))
        q.mul_(0.99)
        graph.replay()
        torch.accelerator.synchronize()
        expected = reference(q, view, table, lengths, grouped=True)
        floor = (expected.half().double() - expected).norm() / expected.norm()
        error = (out.cpu().double() - expected).norm() / expected.norm()
        assert error <= 1.02 * floor + 2e-6, (error.item(), floor.item())
        assert torch.isfinite(out).all()
        assert torch.count_nonzero(out[0]) == 0


@pytest.mark.parametrize(
    "codec", [c for c in CODECS if c.quantized], ids=lambda c: c.name
)
def test_grouped_eight_byte_offsets_keep_the_qualified_reader(codec):
    torch.manual_seed(942)
    _, view = cache(codec, pages=5, page=32)

    def offset_copy(data):
        # Legal eight-byte offsets/strides must not be rounded down by a
        # sixteen-byte vector read. The scale planes keep their actual strides.
        strides = (32 * 264 + 8, 264, 264, 1)
        storage = torch.empty(5 * strides[0] + 8, dtype=data.dtype, device="cuda")
        result = storage.as_strided(data.shape, strides, storage_offset=8)
        return result.copy_(data)

    view = replace(view, key=offset_copy(view.key), value=offset_copy(view.value))
    q = torch.randn((16, 6, 256), device="cuda", dtype=torch.float16)
    table = torch.tensor([[4, 1, 3, 0, 2]], device="cuda", dtype=torch.int32)
    lengths = torch.arange(16, device="cuda", dtype=torch.int32) + 128
    out = torch.empty_like(q)
    package.flash_attn_grouped_fp32_paged(
        q,
        view.key,
        view.value,
        table,
        lengths,
        out=out,
        softmax_scale=256**-0.5,
        kv_cache_dtype=codec.name,
        **view.scale_kwargs,
    )
    expected = reference(q, view, table, lengths, grouped=True)
    floor = (expected.half().double() - expected).norm() / expected.norm()
    error = (out.cpu().double() - expected).norm() / expected.norm()
    assert error <= 1.02 * floor + 2e-6


def test_grouped_fixed_int8_plan_executes_native_q16():
    codec = next(c for c in CODECS if c.name == "int8")
    _, view = cache(codec, pages=3, page=1648)
    q = torch.randn((16, 6, 256), device="cuda", dtype=torch.float16)
    table = torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.arange(16, device="cuda", dtype=torch.int32) + 1664

    out = torch.empty_like(q)

    def launch():
        package.flash_attn_grouped_fp32_paged(
            q,
            view.key,
            view.value,
            table,
            lengths,
            out=out,
            softmax_scale=256**-0.5,
            kv_cache_dtype=codec.name,
            **view.scale_kwargs,
        )

    launch()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as trace:
        launch()
    kernels = [event.name for event in trace.events()]
    assert any(
        "flash_attention_grouped_verify_e5m2_partial_kernel" in name and "1648" in name
        for name in kernels
    ), kernels
    assert not any("flash_attention_decode" in name for name in kernels), kernels
