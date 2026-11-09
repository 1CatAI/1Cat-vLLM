# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("page_size", [16, 3296])
@pytest.mark.parametrize("length", [1032, 32768])
@torch.inference_mode()
def test_e4m3_grouped_scaled_random_pages_and_graph(page_size, length):
    fa = pytest.importorskip("flash_attn_v100")
    if not getattr(fa.flash_attn_grouped_verify_paged, "supports_e4m3", False):
        pytest.skip("Rebuild Flash-V100 for E4M3 grouped verification")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    torch.manual_seed(length + page_size)
    pages = (length + page_size - 1) // page_size + 2
    raw = torch.randn(pages, 2, page_size, 1, 256, device="cuda").half() * 0.25
    cache = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    k, v = cache.unbind(1)
    table = torch.randperm(pages, device="cuda", dtype=torch.int32)[None]
    q = torch.randn(8, 6, 256, device="cuda").half() * 0.25
    seq = torch.tensor([length], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)

    def run():
        return fa.flash_attn_grouped_verify_paged(
            q,
            k,
            v,
            table,
            seq,
            out=out,
            kv_cache_dtype="fp8_e4m3",
            k_scale=0.5,
            v_scale=2.0,
            one_pass=True,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for current in [length, length - 5]:
        seq.fill_(current)
        graph.replay()
        actual = out.clone()
        run()
        assert torch.equal(actual, out)
        keys = k[table[0].long()].view(torch.float8_e4m3fn)
        values = v[table[0].long()].view(torch.float8_e4m3fn)
        keys = keys.reshape(-1, 256)[:current].float() * 0.5
        values = values.reshape(-1, 256)[:current].float() * 2
        scores = q.transpose(0, 1).float() @ keys.t() / 16
        pos = torch.arange(current - 8, current, device="cuda")
        mask = torch.arange(current, device="cuda")[None] > pos[:, None]
        scores.masked_fill_(mask[None], -torch.inf)
        expected = (scores.softmax(-1) @ values).transpose(0, 1)
        relative = (actual.float() - expected).norm() / expected.norm()
        assert relative < 5e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("invalid", ["query_length", "stride"])
def test_e4m3_grouped_rejects_unsupported_contract(invalid):
    fa = pytest.importorskip("flash_attn_v100")
    if not getattr(fa.flash_attn_grouped_verify_paged, "supports_e4m3", False):
        pytest.skip("Rebuild Flash-V100 for E4M3 grouped verification")
    query_len = 1 if invalid == "query_length" else 8
    width = 257 if invalid == "stride" else 256
    cache = torch.zeros(2, 2, 16, 1, width, device="cuda", dtype=torch.uint8)
    k, v = cache[..., :256].unbind(1)
    q = torch.zeros(query_len, 6, 256, device="cuda", dtype=torch.float16)
    table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    seq = torch.tensor([8], device="cuda", dtype=torch.int32)
    message = "requires q=8" if invalid == "query_length" else "aligned KV strides"
    with pytest.raises(RuntimeError, match=message):
        fa.flash_attn_grouped_verify_paged(
            q, k, v, table, seq, kv_cache_dtype="fp8_e4m3", one_pass=True
        )
