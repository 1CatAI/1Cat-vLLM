# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed prefill+decode batches: small-query rows may take the decode route.

Opt-in via ``VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS``. With the flag off
the per-sequence paged prefill loop is untouched. With it on, rows with
``1 <= q <= VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q`` and prefix context (a
resident decoder, or an MTP/DFlash verify row) are expanded token-wise into
one paged-decode call and must match the prefill kernel within fp16
tolerance, while the chunk row stays bit-identical to the flag-off output
(same kernel, same inputs).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

FLAG = "VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS"
NUM_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 256


def _make_impl(kv_cache_dtype: str):
    from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Impl

    return FlashAttnV100Impl(
        num_heads=NUM_HEADS,
        head_size=HEAD_DIM,
        scale=HEAD_DIM**-0.5,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
    )


def _make_mixed_batch(
    *,
    device: str,
    block_size: int,
    kv_cache_dtype: str,
    chunk_len: int,
    chunk_context: int,
    small_rows: list[tuple[int, int]],
):
    """Row 0 is a chunked-prefill row; the rest are (q_len, seq_len) rows."""
    query_lens = [chunk_len] + [q for q, _ in small_rows]
    seq_lens = [chunk_context + chunk_len] + [s for _, s in small_rows]
    blocks_per_row = [(s + block_size - 1) // block_size for s in seq_lens]
    total_blocks = sum(blocks_per_row) + 1  # +1 keeps the K/V axis unambiguous
    kv_cache = torch.randn(
        2,
        total_blocks,
        block_size,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.float16,
        device=device,
    )
    if kv_cache_dtype == "fp8_e5m2":
        # Real E5M2 values (no NaN/Inf bit patterns), stored as the uint8
        # bytes the paged cache carries.
        kv_cache = kv_cache.to(torch.float8_e5m2).view(torch.uint8)
    max_blocks = max(blocks_per_row)
    block_table = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32)
    next_block = 1
    for row, n in enumerate(blocks_per_row):
        block_table[row, :n] = torch.arange(next_block, next_block + n)
        next_block += n
    query_start_loc_cpu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(query_lens), 0)), dtype=torch.int32
    )
    seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    num_tokens = int(query_start_loc_cpu[-1].item())
    query = torch.randn(
        num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    attn_metadata = SimpleNamespace(
        query_start_loc=query_start_loc_cpu.to(device),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_cpu.to(device),
        seq_lens_cpu=seq_lens_cpu,
        block_table=block_table.to(device),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        causal=True,
        max_model_len=262144,
    )
    return query, kv_cache, attn_metadata, query_start_loc_cpu


def _run(impl, query, kv_cache, attn_metadata) -> torch.Tensor:
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    output = torch.empty_like(query)
    impl._flash_v100_prefill_with_prefix(
        layer, query, None, None, kv_cache, attn_metadata, output
    )
    torch.accelerator.synchronize()
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e5m2"])
@pytest.mark.parametrize("block_size", [16, 784])
@pytest.mark.parametrize(
    "small_rows",
    [
        [(1, 4097)],  # one resident decoder
        [(1, 4097), (1, 65537)],  # two resident decoders
        [(5, 4101), (1, 20000)],  # MTP verify row (K=4) plus a decoder
        [(16, 8208)],  # largest small-q row
    ],
)
@torch.inference_mode()
def test_prefix_prefill_small_query_rows_match_prefill_route(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_dtype: str,
    block_size: int,
    small_rows: list[tuple[int, int]],
) -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("FlashAttention-V100 is SM70/V100 only")
    pytest.importorskip("flash_attn_v100")
    import vllm.v1.attention.backends.flash_attn_v100 as backend

    torch.manual_seed(1234)
    device = "cuda"
    impl = _make_impl(kv_cache_dtype)
    if not (impl.use_flash_v100_prefill_paged and impl.use_flash_v100_decode):
        pytest.skip("paged prefill and decode ops are both required")
    query, kv_cache, attn_metadata, query_start_loc = _make_mixed_batch(
        device=device,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        chunk_len=96,
        chunk_context=1000,
        small_rows=small_rows,
    )
    chunk_end = int(query_start_loc[1].item())
    spans = list(zip(query_start_loc[1:-1].tolist(), query_start_loc[2:].tolist()))

    # Observe route selection without enabling the summary env (which would
    # register an atexit hook in the test process).
    routes: list[str] = []
    monkeypatch.setattr(backend, "_record_route", routes.append)

    monkeypatch.delenv(FLAG, raising=False)
    out_off = _run(impl, query, kv_cache, attn_metadata)
    assert not any(r.startswith("prefill_prefix_decode_rows") for r in routes)

    monkeypatch.setenv(FLAG, "1")
    routes.clear()
    out_on = _run(impl, query, kv_cache, attn_metadata)
    assert any(r.startswith("prefill_prefix_decode_rows") for r in routes), routes

    # Chunk row: same kernel, same inputs -> bit-identical.
    assert torch.equal(out_on[:chunk_end], out_off[:chunk_end])
    # Small-q rows: decode kernel versus paged prefill kernel; both read the
    # same cache bytes, so only accumulation order differs.
    atol = 1e-2 if kv_cache_dtype == "fp8_e5m2" else 5e-3
    for start, end in spans:
        torch.testing.assert_close(
            out_on[start:end], out_off[start:end], atol=atol, rtol=1e-2
        )
    assert torch.isfinite(out_on).all()
