# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Batching draft attention must preserve per-request masking and graph replay."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kv_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("block", [16, 1648])
@pytest.mark.parametrize(
    "batch,q_len,context",
    [(1, 8, 2048), (4, 8, 2048), (8, 8, 32768), (4, 4, 4096), (8, 1, 262144)],
)
@torch.inference_mode()
def test_dflash_batch_matches_serial_with_live_graph_metadata(
    batch, q_len, context, kv_dtype, block, monkeypatch
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 is SM70 only")
    flash = pytest.importorskip("flash_attn_v100")
    import vllm.v1.attention.backends.flash_attn_v100 as backend

    torch.manual_seed(20260925)
    heads, kv_heads, dim = 8, 2, 128
    impl = backend.FlashAttnV100Impl(
        num_heads=heads,
        head_size=dim,
        scale=dim**-0.5,
        num_kv_heads=kv_heads,
        alibi_slopes=None,
        sliding_window=2048,
        kv_cache_dtype=kv_dtype,
    )
    assert impl.use_flash_v100_prefill_paged
    tokens = batch * q_len
    # Exercise the strided Q view produced by a fused projection and ensure
    # output padding outside the active batch is never overwritten.
    storage = torch.randn(tokens, heads * dim + 32, device="cuda").half()
    query = storage[:, : heads * dim].view(tokens, heads, dim)
    output = torch.full((tokens + 3, heads, dim), 0.125, device="cuda").half()
    cache = torch.randn(2, 64, block, kv_heads, dim, device="cuda").half()
    if kv_dtype == "fp8_e4m3":
        cache = cache.to(torch.float8_e4m3fn).view(torch.uint8)
    # Reusing physical pages bounds test memory while covering long logical
    # positions, noncontiguous page mappings, and per-request SWA boundaries.
    table = torch.randint(
        0,
        64,
        (batch, (context + block - 1) // block),
        device="cuda",
        dtype=torch.int32,
    )
    starts = torch.arange(batch + 1, dtype=torch.int32) * q_len
    lengths = torch.full((batch,), q_len, device="cuda", dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        query_start_loc=starts.cuda(),
        query_start_loc_cpu=starts,
        seq_lens=lengths,
        seq_lens_cpu=lengths.cpu(),
        block_table=table,
        causal=False,
    )
    layer = SimpleNamespace(
        is_dflash_draft_attn=True, _k_scale_float=0.75, _v_scale_float=1.25
    )
    routes: list[str] = []
    monkeypatch.setattr(backend, "_record_route", routes.append)

    def run():
        return impl._flash_v100_prefill_with_prefix(
            layer, query, None, None, cache, metadata, output
        )

    assert run() is output
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    route = "prefill_prefix_dflash_noncausal_batch"
    assert (route in routes) == (batch > 1)
    for step in range(3):
        storage.normal_()
        table.random_(0, 64)
        lengths.copy_(
            torch.tensor(
                [max(q_len, context - i * 17 - step) for i in range(batch)],
                device="cuda",
                dtype=torch.int32,
            )
        )
        graph.replay()
        expected = []
        for i in range(batch):
            expected.append(
                flash.flash_attn_prefill_paged(
                    query[i * q_len : (i + 1) * q_len].unsqueeze(0),
                    cache[0],
                    cache[1],
                    table[i : i + 1],
                    lengths[i : i + 1],
                    softmax_scale=impl.scale,
                    kv_cache_dtype=kv_dtype,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    causal=False,
                    window_size=impl._flash_v100_window_size(False),
                ).squeeze(0)
            )
        torch.testing.assert_close(output[:tokens], torch.cat(expected), atol=0, rtol=0)
        assert torch.all(output[tokens:] == 0.125)
