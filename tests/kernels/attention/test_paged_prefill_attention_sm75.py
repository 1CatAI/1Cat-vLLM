# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turing paged prefill attention (head dim 256) against an fp32 reference."""

import pytest
import torch

from vllm.platforms import current_platform

HEAD_DIM = 256


def _turing_device() -> torch.device:
    if current_platform.is_cuda():
        for index in range(torch.accelerator.device_count()):
            if current_platform.is_device_capability((7, 5), device_id=index):
                return torch.device("cuda", index)
    pytest.skip("no SM75 GPU")


def _paged_cache(kv_lens, page_size, kv_heads, device, generator):
    """K/V in the backend's [blocks, 2, page, heads, dim] layout, each
    sequence on shuffled pages."""
    pages_per = [(n + page_size - 1) // page_size for n in kv_lens]
    total = sum(pages_per)
    cache = (
        torch.randn(
            total,
            2,
            page_size,
            kv_heads,
            HEAD_DIM,
            dtype=torch.float16,
            device=device,
            generator=generator,
        )
        * 0.5
    )
    order = torch.randperm(total, device=device, generator=generator).to(torch.int32)
    table = torch.zeros(len(kv_lens), max(pages_per), dtype=torch.int32, device=device)
    start = 0
    for b, n in enumerate(pages_per):
        table[b, :n] = order[start : start + n]
        start += n
    key_cache, value_cache = cache.unbind(1)
    return key_cache, value_cache, table


def _reference(q, key_cache, value_cache, table, q_lens, kv_lens, scale):
    page_size, kv_heads = key_cache.size(1), key_cache.size(2)
    group = q.size(1) // kv_heads
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    start = 0
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        pages = table[b, : (kv_len + page_size - 1) // page_size].long()
        k = key_cache[pages].reshape(-1, kv_heads, HEAD_DIM)[:kv_len].float()
        v = value_cache[pages].reshape(-1, kv_heads, HEAD_DIM)[:kv_len].float()
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[start : start + q_len].float(), k)
        positions = torch.arange(kv_len - q_len, kv_len, device=q.device)
        keys = torch.arange(kv_len, device=q.device)
        scores = scores * scale
        scores.masked_fill_(keys[None, None, :] > positions[None, :, None], -torch.inf)
        out[start : start + q_len] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        start += q_len
    return out


@pytest.mark.parametrize(
    "q_lens,kv_lens,page_size,heads,kv_heads",
    [
        # single tokens, partial pages, a context shorter than one key tile
        ((1, 5, 37, 70, 129), (1, 17, 300, 70, 4000), 16, 12, 2),
        ((300,), (300,), 32, 12, 2),
        ((2048,), (5000,), 64, 12, 2),
        ((200, 64), (1000, 64), 16, 24, 4),
    ],
)
# 0 picks the split count by waves; 2 and 4 force split-KV with its merge,
# including splits without visible keys (short contexts).
@pytest.mark.parametrize("num_kv_splits", [0, 1, 2, 4])
def test_matches_the_fp32_reference(
    q_lens, kv_lens, page_size, heads, kv_heads, num_kv_splits
):
    device = _turing_device()
    generator = torch.Generator(device=device).manual_seed(0)
    key_cache, value_cache, table = _paged_cache(
        kv_lens, page_size, kv_heads, device, generator
    )
    q = (
        torch.randn(
            sum(q_lens),
            heads,
            HEAD_DIM,
            dtype=torch.float16,
            device=device,
            generator=generator,
        )
        * 0.5
    )
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    scale = HEAD_DIM**-0.5
    out = torch.full_like(q, float("nan"))

    torch.ops._C.paged_prefill_attention_sm75(
        out,
        q,
        key_cache,
        value_cache,
        cu_seqlens_q,
        seqused_k,
        table,
        max(q_lens),
        scale,
        num_kv_splits,
    )

    reference = _reference(q, key_cache, value_cache, table, q_lens, kv_lens, scale)
    # fp16 inputs and fp16 probabilities in P.V, as in FlashAttention-2: the
    # largest error measured on these cases is about 1.2e-3.
    torch.testing.assert_close(out.float(), reference, rtol=0, atol=3e-3)


def _selection_impl():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    impl = object.__new__(FlashAttentionImpl)
    impl.turing_prefill_min_query_len = 65
    impl.head_size = HEAD_DIM
    impl.vllm_flash_attn_version = 2
    impl.sliding_window = (-1, -1)
    impl.alibi_slopes = None
    impl.sinks = None
    impl.logits_soft_cap = 0
    impl.kv_cache_dtype = "auto"
    impl.batch_invariant_enabled = False
    return impl


@pytest.mark.parametrize(
    ("blocks", "page_size", "expected"),
    [
        (64, 16, True),
        (64, 64, True),
        (64, 128, False),
        (64, 800, False),
        # The [blocks, 2, page, heads, dim] view spans more than 2^31 elements.
        (2**31 // (2 * 16 * 2 * HEAD_DIM) + 1, 16, False),
    ],
)
def test_selection_keeps_unsupported_caches_on_fa2(
    monkeypatch, blocks, page_size, expected
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=None: capability == (7, 5),
    )
    impl = _selection_impl()
    metadata = SimpleNamespace(max_query_len=128, causal=True)
    query = torch.empty(128, 12, HEAD_DIM, dtype=torch.float16, device="meta")
    cache = torch.empty(
        blocks, 2, page_size, 2, HEAD_DIM, dtype=torch.float16, device="meta"
    )
    key_cache = cache[:, 0]
    assert impl._use_turing_prefill(metadata, query, key_cache) is expected
