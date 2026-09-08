# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The E4M3 scalar fallback must retain FP32 split state as well."""

from types import SimpleNamespace

import pytest
import torch


def test_e4m3_scalar_rejects_stale_native(monkeypatch):
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    monkeypatch.setattr(interface, "flash_attn_v100_cuda", SimpleNamespace())
    q = torch.empty((1, 6, 256), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="FP32 scalar decode.*revision 4"):
        interface.flash_attn_decode_paged(q, q, q, q, q, kv_cache_dtype="fp8_e4m3")


@pytest.mark.parametrize(
    "rows,dim,page,length,partition",
    [
        (1, 64, 16, 2048, 256),
        (2, 128, 16, 4096, 512),
        (16, 256, 1728, 8192, 1024),
        (1, 256, 3456, 262144, 1024),
    ],
)
def test_scalar_fp32_workspace_and_live_graph(
    monkeypatch, rows, dim, page, length, partition
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    interface = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    if not interface.flash_attn_grouped_e4m3_fp32_available():
        pytest.skip("rebuild FP32 E4M3 decode")
    torch.manual_seed(20260908)
    pages = (length + page - 1) // page
    q = torch.randn((rows, 6, dim), device="cuda", dtype=torch.float16)
    kv = torch.randn((2, pages, page, 1, dim), device="cuda", dtype=torch.float16)
    kv[1].add_(3.0)
    kv = kv.to(torch.float8_e4m3fn).view(torch.uint8)
    k, v = kv.unbind(0)
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None].repeat(rows, 1)
    lengths = torch.arange(
        length - rows + 1, length + 1, device="cuda", dtype=torch.int32
    )
    original = lengths.clone()
    output = torch.empty_like(q)
    get_workspace = interface._get_decode_workspace_for_plan
    workspaces = []

    def checked_workspace(*args, **kwargs):
        result = get_workspace(*args, **kwargs)
        assert result[0].dtype == torch.float32
        workspaces.append(result[0])
        return result

    monkeypatch.setattr(interface, "_get_decode_workspace_for_plan", checked_workspace)

    def call():
        interface.flash_attn_decode_paged(
            q,
            k,
            v,
            table,
            lengths,
            out=output,
            kv_cache_dtype="fp8_e4m3",
            softmax_scale=dim**-0.5,
            k_scale=0.5,
            v_scale=1.25,
            max_seq_len_hint=length,
            workspace_seq_capacity_hint=pages * page,
            partition_size_hint=partition,
        )

    call()
    call()
    assert workspaces[0].data_ptr() == workspaces[1].data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    rk = k.reshape(-1, dim)[:length].view(torch.float8_e4m3fn).double() * 0.5
    rv = v.reshape(-1, dim)[:length].view(torch.float8_e4m3fn).double() * 1.25
    for zero in (False, True, False):
        lengths.copy_(original)
        if zero:
            lengths[-1] = 0
        graph.replay()
        scores = q.transpose(0, 1).double() @ rk.T * dim**-0.5
        scores.masked_fill_(
            torch.arange(length, device="cuda")[None, None] >= lengths[None, :, None],
            -torch.inf,
        )
        expected = (scores.softmax(-1).nan_to_num(0) @ rv).transpose(0, 1)
        assert bool(torch.isfinite(output).all())
        assert bool((output[lengths == 0] == 0).all())
        if bool((lengths != 0).any()):
            error = (output.double() - expected).norm() / expected.norm()
            floor = (expected.half().double() - expected).norm() / expected.norm()
            assert float(error) <= 1.04 * float(floor) + 2e-6
