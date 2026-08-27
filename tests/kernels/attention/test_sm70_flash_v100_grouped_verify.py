# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch

_EXPANDED_BASELINE_MEAN_ERROR = {
    (16, 8, 120): 5.159488409844926e-6,
    (16, 8, 121): 8.621820597909391e-6,
    (16, 3, 253): 6.551763817697065e-6,
    (16, 4, 252): 5.888879059057217e-6,
    (16, 5, 507): 4.691817139246268e-6,
    (16, 6, 506): 4.563758920994587e-6,
    (16, 8, 1016): 3.1533952551399125e-6,
}


def _require_grouped_verify():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("grouped Flash-V100 verification is SM70-only")
    flash_attn_v100 = pytest.importorskip("flash_attn_v100")
    if not hasattr(flash_attn_v100, "flash_attn_grouped_verify_paged"):
        pytest.skip("Flash-V100 extension lacks grouped verification")
    return flash_attn_v100


def _make_case(
    *,
    page_size: int,
    query_len: int,
    prefix_len: int,
) -> tuple[torch.Tensor, ...]:
    total_len = prefix_len + query_len
    logical_pages = math.ceil(total_len / page_size)
    physical_pages = logical_pages + 3
    source_shape = (physical_pages, page_size, 1, 256)
    key_source = torch.randn(source_shape, dtype=torch.float16, device="cuda").mul_(
        0.25
    )
    value_source = torch.randn_like(key_source).mul_(0.25)
    key_cache = key_source.to(torch.float8_e5m2).view(torch.uint8)
    value_cache = value_source.to(torch.float8_e5m2).view(torch.uint8)
    block_table = torch.randperm(physical_pages, dtype=torch.int32, device="cuda")[
        :logical_pages
    ].view(1, -1)
    query = torch.randn((query_len, 6, 256), dtype=torch.float16, device="cuda").mul_(
        0.25
    )
    seq_lens = torch.tensor([total_len], dtype=torch.int32, device="cuda")
    return query, key_cache, value_cache, block_table, seq_lens


def _make_interleaved_case(
    *,
    page_size: int,
    query_len: int,
    prefix_len: int,
) -> tuple[torch.Tensor, ...]:
    total_len = prefix_len + query_len
    logical_pages = math.ceil(total_len / page_size)
    physical_pages = logical_pages + 3
    source_shape = (physical_pages, 2, page_size, 1, 256)
    source = torch.randn(source_shape, dtype=torch.float16, device="cuda").mul_(0.25)
    cache = source.to(torch.float8_e5m2).view(torch.uint8)
    key_cache, value_cache = cache.unbind(1)
    block_table = torch.randperm(physical_pages, dtype=torch.int32, device="cuda")[
        :logical_pages
    ].view(1, -1)
    query = torch.randn((query_len, 6, 256), dtype=torch.float16, device="cuda").mul_(
        0.25
    )
    seq_lens = torch.tensor([total_len], dtype=torch.int32, device="cuda")
    return query, key_cache, value_cache, block_table, seq_lens


def _reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    *,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    query_len = query.shape[0]
    prefix_len = seq_len - query_len
    logical_pages = math.ceil(seq_len / key_cache.shape[1])
    physical_pages = block_table[0, :logical_pages].to(dtype=torch.long)
    key = (
        key_cache.view(torch.float8_e5m2)[physical_pages]
        .flatten(0, 1)[:seq_len, 0]
        .float()
        .mul_(k_scale)
    )
    value = (
        value_cache.view(torch.float8_e5m2)[physical_pages]
        .flatten(0, 1)[:seq_len, 0]
        .float()
        .mul_(v_scale)
    )
    rows = []
    for token_idx in range(query_len):
        visible = prefix_len + token_idx + 1
        scores = torch.einsum(
            "hd,nd->hn", query[token_idx].float(), key[:visible]
        ).mul_(256**-0.5)
        probabilities = torch.softmax(scores, dim=-1)
        rows.append(torch.einsum("hn,nd->hd", probabilities, value[:visible]))
    return torch.stack(rows).half()


@pytest.mark.parametrize(
    ("page_size", "query_len", "prefix_len", "strict_fp32_gate"),
    [
        (16, 1, 127, True),
        (16, 8, 120, False),
        (16, 8, 121, False),
        (16, 3, 253, False),
        (16, 4, 252, False),
        (16, 5, 507, False),
        (16, 6, 506, False),
        (16, 8, 1016, False),
        (16, 16, 1008, True),
        (16, 3, 1025, True),
        (784, 5, 2049, True),
        (1648, 8, 4097, True),
        (1648, 16, 4089, True),
        (1728, 8, 4097, True),
        (1728, 16, 4089, True),
        (3296, 8, 8193, True),
        (3296, 16, 8185, True),
        (3456, 8, 8193, True),
        (3456, 16, 8185, True),
    ],
)
@torch.inference_mode()
def test_grouped_verify_matches_fp32_reference_with_random_pages(
    page_size: int,
    query_len: int,
    prefix_len: int,
    strict_fp32_gate: bool,
) -> None:
    flash_attn_v100 = _require_grouped_verify()
    torch.manual_seed(20260824 + page_size + query_len)
    query, key_cache, value_cache, block_table, seq_lens = _make_case(
        page_size=page_size,
        query_len=query_len,
        prefix_len=prefix_len,
    )
    k_scale = 0.5
    v_scale = 2.0
    expected = _reference(
        query,
        key_cache,
        value_cache,
        block_table,
        int(seq_lens.item()),
        k_scale=k_scale,
        v_scale=v_scale,
    )
    two_pass = flash_attn_v100.flash_attn_grouped_verify_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        k_scale=k_scale,
        v_scale=v_scale,
        one_pass=False,
    ).clone()
    one_pass = flash_attn_v100.flash_attn_grouped_verify_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        k_scale=k_scale,
        v_scale=v_scale,
        one_pass=True,
    ).clone()
    decode_block_table = block_table.repeat(query_len, 1).contiguous()
    decode_seq_lens = torch.arange(
        prefix_len + 1,
        prefix_len + query_len + 1,
        dtype=torch.int32,
        device=query.device,
    )
    accepted_xqa = flash_attn_v100.flash_attn_decode_paged_xqa(
        query,
        key_cache,
        value_cache,
        decode_block_table,
        decode_seq_lens,
        kv_cache_dtype="fp8_e5m2",
        k_scale=k_scale,
        v_scale=v_scale,
        max_seq_len_hint=prefix_len + query_len,
        workspace_seq_capacity_hint=prefix_len + query_len,
    )
    torch.accelerator.synchronize()

    for actual in (two_pass, one_pass):
        difference = actual.float().sub(expected.float()).abs()
        if strict_fp32_gate:
            assert difference.max().item() <= 6.2e-5
            assert difference.mean().item() <= 6.0e-6
    one_pass_error = one_pass.float().sub(expected.float()).abs()
    accepted_error = accepted_xqa.float().sub(expected.float()).abs()
    assert one_pass_error.max().item() <= accepted_error.max().item()
    if strict_fp32_gate:
        mean_error_limit = accepted_error.mean().item()
    else:
        mean_error_limit = _EXPANDED_BASELINE_MEAN_ERROR[
            (page_size, query_len, prefix_len)
        ]
    assert one_pass_error.mean().item() <= mean_error_limit + 5.0e-7
    torch.testing.assert_close(one_pass, two_pass, atol=6.2e-5, rtol=2.0e-3)


@pytest.mark.parametrize("query_len", [8, 16])
@torch.inference_mode()
def test_grouped_verify_cuda_graph_replay_tracks_runtime_seq_len(
    query_len: int,
) -> None:
    flash_attn_v100 = _require_grouped_verify()
    torch.manual_seed(20260825)
    query, key_cache, value_cache, block_table, seq_lens = _make_case(
        page_size=1648,
        query_len=query_len,
        prefix_len=4097,
    )
    output = torch.empty_like(query)
    flash_attn_v100.flash_attn_grouped_verify_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        out=output,
        one_pass=True,
    )
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_attn_v100.flash_attn_grouped_verify_paged(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            out=output,
            one_pass=True,
        )

    for prefix_len in (127, 2049, 4097):
        seq_lens.fill_(prefix_len + query.shape[0])
        graph.replay()
        torch.accelerator.synchronize()
        expected = _reference(
            query,
            key_cache,
            value_cache,
            block_table,
            prefix_len + query.shape[0],
            k_scale=1.0,
            v_scale=1.0,
        )
        difference = output.float().sub(expected.float()).abs()
        assert difference.max().item() <= 6.2e-5
        assert difference.mean().item() <= 6.0e-6


@pytest.mark.parametrize("page_size", [1648, 3296])
@pytest.mark.parametrize("query_len", [8, 16])
@torch.inference_mode()
def test_grouped_verify_fixed_interleaved_is_bitwise(
    page_size: int, query_len: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    flash_attn_v100 = _require_grouped_verify()
    torch.manual_seed(20260826 + page_size)
    query, key_cache, value_cache, block_table, seq_lens = _make_interleaved_case(
        page_size=page_size,
        query_len=query_len,
        prefix_len=8193,
    )

    monkeypatch.setenv("VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED", "0")
    control = flash_attn_v100.flash_attn_grouped_verify_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        one_pass=True,
    ).clone()

    monkeypatch.setenv("VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED", "1")
    monkeypatch.setenv("VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS", "0")
    fixed = flash_attn_v100.flash_attn_grouped_verify_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        one_pass=True,
    ).clone()

    monkeypatch.setenv("VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS", "1")
    staged = flash_attn_v100.flash_attn_grouped_verify_paged(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        one_pass=True,
    ).clone()
    torch.accelerator.synchronize()

    assert torch.equal(fixed, control)
    assert torch.equal(staged, control)
