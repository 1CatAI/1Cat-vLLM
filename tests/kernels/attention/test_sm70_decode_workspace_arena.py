# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scratch sharing must preserve graph addresses and attention arithmetic."""

import weakref

import pytest
import torch

fa = pytest.importorskip("flash_attn_v100.flash_attn_interface")


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(fa, "_decode_workspace_cache", {})
    monkeypatch.setenv("VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE", "1")


def workspace(q, rows, partitions=4, dtype=torch.float32):
    return fa._get_decode_workspace_for_plan(
        q,
        batch_capacity=rows,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        plan=fa._DecodePlan(256, partitions, partitions, partitions),
        partial_dtype=dtype,
    )


@pytest.mark.parametrize("heads", [6, 12, 24])
def test_smaller_rows_reuse_storage_with_exact_contiguous_views(heads):
    q = torch.empty((32, heads, 32), dtype=torch.float16)
    largest = workspace(q, 32)
    for rows in [16, 8, 4, 2, 1, 24, 32]:
        actual = workspace(q, rows)
        assert actual[0].shape == (rows, heads, 4, 32)
        for big, small in zip(largest[:3], actual[:3]):
            assert big.data_ptr() == small.data_ptr()
            assert small.is_contiguous()
    assert len(fa._decode_workspace_cache) == 1


def test_rollback_keeps_independent_row_buffers(monkeypatch):
    monkeypatch.setenv("VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE", "0")
    q = torch.empty((8, 6, 32), dtype=torch.float16)
    large = workspace(q, 8)
    small = workspace(q, 4)
    assert large[0].data_ptr() != small[0].data_ptr()


def test_non_captured_growth_releases_old_storage():
    q = torch.empty((8, 6, 32), dtype=torch.float16)
    workspace(q, 2)
    old = weakref.ref(next(iter(fa._decode_workspace_cache.values())).tmp_out)
    workspace(q, 8)
    assert old() is None


def test_long_then_batched_short_does_not_multiply_capacities():
    q = torch.empty((56, 12, 256), dtype=torch.float16)
    long = workspace(q, 1, partitions=1024)
    short = workspace(q, 56, partitions=16)
    assert short[0].shape == (56, 12, 16, 256)
    assert long[0].data_ptr() != short[0].data_ptr()
    assert workspace(q, 24, partitions=9)[0].data_ptr() == short[0].data_ptr()
    assert workspace(q, 1, partitions=1024)[0].data_ptr() == long[0].data_ptr()
    assert len(fa._decode_workspace_cache) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_capture_reuse_then_growth_retains_addresses():
    q = torch.empty((8, 6, 32), device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle()
    results = [torch.empty((), device="cuda") for _ in range(3)]
    graphs = []
    refs = []
    with torch.cuda.stream(stream):
        # The first buffer predates capture, but becomes live graph storage.
        workspace(q, 2)
    for index, (rows, partitions) in enumerate([(2, 4), (8, 4), (1, 4)]):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=pool):
            scratch = workspace(q, rows, partitions)[0]
            scratch.fill_(index + 1)
            results[index].copy_(scratch.sum())
        graphs.append(graph)
        refs.append(
            weakref.ref(next(iter(fa._decode_workspace_cache.values())).tmp_out)
        )
    first, second, third = [ref() for ref in refs]
    assert first is not None and second is not None and third is not None
    assert first.data_ptr() != second.data_ptr()
    assert second.data_ptr() == third.data_ptr()
    for index in [0, 2, 1, 0, 1, 2]:
        graphs[index].replay()
        torch.accelerator.synchronize()
        rows, capacity = [(2, 4), (8, 4), (1, 4)][index]
        assert results[index].item() == rows * 6 * capacity * 32 * (index + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_independent_streams_do_not_share_scratch():
    q = torch.empty((8, 6, 32), device="cuda", dtype=torch.float16)
    buffers = []
    for stream in [torch.cuda.Stream(), torch.cuda.Stream()]:
        with torch.cuda.stream(stream):
            buffers.append(workspace(q, 8)[0])
    assert buffers[0].data_ptr() != buffers[1].data_ptr()


@pytest.mark.parametrize("heads", [6, 12, 24])
@pytest.mark.parametrize("kv_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_attention_matches_independent_workspaces(monkeypatch, heads, kv_dtype):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    torch.manual_seed(31)
    q = torch.randn((32, heads, 256), device="cuda", dtype=torch.float16) / 4
    k = torch.randn((128, 16, heads // 6, 256), device="cuda", dtype=torch.float16) / 4
    v = torch.randn_like(k) / 4
    if kv_dtype == "fp8_e4m3":
        k = k.to(torch.float8_e4m3fn).view(torch.uint8)
        v = v.to(torch.float8_e4m3fn).view(torch.uint8)
    table = torch.arange(128, device="cuda", dtype=torch.int32).repeat(32, 1)
    lengths = torch.linspace(1025, 2048, 32, device="cuda").to(torch.int32)
    modes = []
    for share in ["0", "1"]:
        monkeypatch.setenv("VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE", share)
        stream = torch.cuda.Stream()
        pool = torch.cuda.graph_pool_handle()
        cases = {}
        for rows, context in [
            (1, 2048),
            (32, 128),
            (16, 128),
            (8, 2048),
            (4, 128),
            (2, 2048),
        ]:
            out = torch.empty_like(q[:rows])
            case_lengths = (
                lengths[:rows]
                if context == 2048
                else torch.full_like(lengths[:rows], context)
            )
            args = (q[:rows], k, v, table[:rows], case_lengths)
            kwargs = dict(
                out=out,
                kv_cache_dtype=kv_dtype,
                max_seq_len_hint=context,
                workspace_seq_capacity_hint=context,
                partition_size_hint=256,
            )
            with torch.cuda.stream(stream):
                fa.flash_attn_decode_paged(*args, **kwargs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=pool):
                fa.flash_attn_decode_paged(*args, **kwargs)
            cases[rows] = (graph, out, case_lengths)
        modes.append(cases)
    for rows in [1, 32, 4, 16, 2, 8, 32, 1]:
        for cases in modes:
            cases[rows][0].replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(modes[0][rows][1], modes[1][rows][1], rtol=0, atol=0)
