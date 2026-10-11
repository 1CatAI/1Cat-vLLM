# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared native plans, including device-selected threshold neighbors.

The ordinary attention bound is frozen at atol=0.004, rtol=0.015. These tests
inspect the installed native implementation; they do not establish model
quality or end-to-end latency.
"""

import pytest
import torch

from tests.kernels.attention.test_sm70_kv_engine_bindings import cache, reference
from vllm.config.flash_v100 import NATIVE_FIELDS
from vllm.v1.attention.kv_codecs import INT8, KV_CODECS

package = pytest.importorskip("flash_attn_v100")
native = pytest.importorskip(
    "flash_attn_v100.flash_attn_interface"
).flash_attn_v100_cuda
runtime_module = pytest.importorskip("flash_attn_v100.runtime")
AttentionRuntime, PythonPolicy = (
    runtime_module.AttentionRuntime,
    runtime_module.PythonPolicy,
)

CODECS = tuple(c for c in KV_CODECS if c.native_id is not None)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 native XQA qualification",
)


def runtime(*, merged=True):
    values = {
        "xqa_e4m3_g6_p256_begin": "257",
        "xqa_e4m3_g6_dual_cta_begin": "513",
        "xqa_e4m3_g6_p512_begin": "769",
        "xqa_e4m3_g6_p896_begin": "1537",
        "xqa_e4m3_g6_p1664_begin": "3073",
        "xqa_e4m3_g6_merged_wave_launch": "1" if merged else "0",
        "xqa_e5m2_p1024_begin": "1025",
        # An old scalar threshold must not override the qualified XQA plan.
        "decode_fp8_xqa_min_seq_len": "4096",
    }
    return AttentionRuntime(
        PythonPolicy(True, False, True, None, None, False, True, True, True),
        [values.get(name) for name, *_ in NATIVE_FIELDS],
    )


def launch(q, view, table, lengths, out, plan, *, partition=256, capacity=6144):
    assert native.xqa_execution_plan_revision >= 2
    return package.flash_attn_decode_paged_xqa(
        q,
        view.key,
        view.value,
        table,
        lengths,
        out=out,
        kv_cache_dtype=view.codec.name,
        max_seq_len_hint=capacity,
        workspace_seq_capacity_hint=capacity,
        partition_size_hint=partition,
        batch_context_routing=True,
        _runtime=plan,
        **view.scale_kwargs,
    )


@pytest.mark.parametrize("codec", CODECS, ids=lambda c: c.name)
@pytest.mark.parametrize(
    "batch,gqa,heads,page,partition,length",
    [
        (1, 4, 2, 32, 256, 73),
        (2, 8, 2, 32, 512, 89),
        (1, 6, 1, 784, 256, 799),
        (4, 6, 1, 16, 256, 127),
        (4, 6, 1, 800, 256, 807),
        (5, 6, 1, 1616, 256, 1633),
        (5, 6, 2, 1616, 256, 1633),
    ],
)
def test_shared_xqa_geometry_and_changed_graph_inputs(
    codec, batch, gqa, heads, page, partition, length
):
    torch.manual_seed(911)
    pages = max(5, (length + page - 1) // page)
    _, view = cache(codec, pages=pages, page=page, heads=heads)
    width = (6144 + page - 1) // page
    table = torch.arange(width, device="cuda", dtype=torch.int32)[None].repeat(batch, 1)
    table.add_(2).remainder_(pages)
    lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    q = torch.randn((batch, gqa * heads, 256), device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    plan = runtime()
    launch(q, view, table, lengths, out, plan, partition=partition)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(q, view, table, lengths, out, plan, partition=partition)
    for delta in (0, -1, 1):
        lengths.fill_(length + delta)
        table.copy_(table.flip(1))
        q.mul_(0.97)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            out.cpu().double(),
            reference(q, view, table, lengths),
            atol=0.004,
            rtol=0.015,
        )
        assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "codec", [c for c in CODECS if c.quantized], ids=lambda c: c.name
)
@pytest.mark.parametrize("merged", [False, True])
def test_wave_and_merged_wave_keep_scales_through_graph_thresholds(codec, merged):
    torch.manual_seed(913)
    _, view = cache(codec, pages=6, page=800)
    table = torch.tensor([[4, 1, 5, 0, 3, 2, 5, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([256], device="cuda", dtype=torch.int32)
    q = torch.randn((1, 6, 256), device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    plan = runtime(merged=merged)
    launch(q, view, table, lengths, out, plan, partition=64)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(q, view, table, lengths, out, plan, partition=64)
    for threshold in (257, 769, 1537, 3073):
        for neighbor in (threshold - 1, threshold, threshold + 1):
            lengths.fill_(neighbor)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(
                out.cpu().double(),
                reference(q, view, table, lengths),
                atol=0.004,
                rtol=0.015,
            )


def test_int8_short_context_executes_native_xqa_with_high_legacy_scalar_threshold():
    _, view = cache(INT8, pages=3, page=800)
    table = torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([17], device="cuda", dtype=torch.int32)
    q = torch.randn((1, 6, 256), device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    plan = runtime()
    launch(q, view, table, lengths, out, plan)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        launch(q, view, table, lengths, out, plan)
        torch.accelerator.synchronize()
    names = [event.name for event in profile.events()]
    assert any(
        "flash_attention_decode_xqa_tc_partition_kernel" in name for name in names
    )
    assert not any("flash_attention_decode_partition_kernel<" in name for name in names)
