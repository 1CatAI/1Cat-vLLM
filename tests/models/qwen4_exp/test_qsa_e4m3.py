# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

qsa_sparse_paged_attention = qsa_ops.qsa_sparse_paged_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_qsa_sparse_paged_attention_rejects_uint8_cache_without_e4m3() -> None:
    query = torch.zeros((1, 6, 256), dtype=torch.float16, device="cuda")
    key_cache = torch.zeros((1, 8, 1, 256), dtype=torch.uint8, device="cuda")
    value_cache = torch.zeros_like(key_cache)
    logical_indices = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    token_to_req = torch.zeros(1, dtype=torch.int32, device="cuda")

    with pytest.raises(
        ValueError,
        match="unquantized K/V caches must match query dtype",
    ):
        qsa_sparse_paged_attention(
            query,
            key_cache,
            value_cache,
            logical_indices,
            block_table,
            token_to_req,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_qsa_cache_write_uses_calibrated_e4m3_scale() -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("QSA E4M3 cache write regression is SM70-only")
    torch.manual_seed(5)
    tokens, block_size, head_dim = 7, 8, 256
    key = torch.randn((tokens, 1, head_dim), dtype=torch.float16, device="cuda")
    value = torch.randn_like(key)
    k_scale = key.abs().max().float() / 448.0
    v_scale = value.abs().max().float() / 448.0
    key_cache = torch.zeros(
        (1, block_size, 1, head_dim), dtype=torch.uint8, device="cuda"
    )
    value_cache = torch.zeros_like(key_cache)
    slots = torch.arange(tokens, dtype=torch.int64, device="cuda")

    ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        "fp8_e4m3",
        k_scale,
        v_scale,
    )

    # The CUDA converter promotes each FP16 input to FP32 before dividing by
    # the FP32 scale.  Keeping the division in FP16 changes tie rounding for a
    # small number of values and does not model reshape_and_cache_flash.
    expected_key = (key.float() / k_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    expected_value = (value.float() / v_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(key_cache[0, :tokens], expected_key)
    assert torch.equal(value_cache[0, :tokens], expected_value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_qsa_sparse_paged_attention_calibrated_e4m3() -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("QSA E4M3 software conversion regression is SM70-only")
    torch.manual_seed(11)
    rows, heads, head_dim = 3, 6, 256
    page_size, pages, topk = 8, 2, 10
    query = torch.randn(
        (rows, heads, head_dim), dtype=torch.float16, device="cuda"
    ).mul_(0.2)
    key = torch.randn(
        (pages, page_size, 1, head_dim), dtype=torch.float16, device="cuda"
    ).mul_(0.35)
    value = torch.randn_like(key).mul_(0.3)
    k_scale = float(key.abs().max().item()) / 448.0
    v_scale = float(value.abs().max().item()) / 448.0
    key_cache = (key / k_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    value_cache = (value / v_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    logical_indices = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(
        rows, 1
    )
    block_table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device="cuda")

    actual = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    decoded_key = key_cache.view(torch.float8_e4m3fn).float() * k_scale
    decoded_value = value_cache.view(torch.float8_e4m3fn).float() * v_scale
    selected_key = decoded_key.view(-1, head_dim)[:topk]
    selected_value = decoded_value.view(-1, head_dim)[:topk]
    scores = torch.einsum("rhd,sd->rhs", query.float(), selected_key) / math.sqrt(
        head_dim
    )
    reference = torch.einsum(
        "rhs,sd->rhd", torch.softmax(scores, dim=-1), selected_value
    )
    torch.testing.assert_close(actual.float(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_qsa_xqa_page4_receives_calibrated_e4m3_scales(monkeypatch) -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("QSA XQA page4 is SM70-only")
    pytest.importorskip("flash_attn_v100")
    torch.manual_seed(19)
    page_size, topk, head_dim = 2052, 2051, 256
    query = torch.randn((1, 6, head_dim), dtype=torch.float16, device="cuda").mul_(0.2)
    key = torch.randn(
        (1, page_size, 1, head_dim), dtype=torch.float16, device="cuda"
    ).mul_(0.35)
    value = torch.randn_like(key).mul_(0.3)
    k_scale = float(key.abs().max().item()) / 448.0
    v_scale = float(value.abs().max().item()) / 448.0
    key_cache = (key / k_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    value_cache = (value / v_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    logical_indices = torch.arange(topk, dtype=torch.int32, device="cuda").view(1, -1)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    token_to_req = torch.zeros(1, dtype=torch.int32, device="cuda")
    query_positions = torch.tensor([topk - 1], dtype=torch.int64, device="cuda")
    sequence_lengths = torch.tensor([topk], dtype=torch.int32, device="cuda")
    kwargs = {
        "query_positions": query_positions,
        "sequence_lengths": sequence_lengths,
        "kv_cache_dtype": "fp8_e4m3",
        "k_scale": k_scale,
        "v_scale": v_scale,
    }

    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", False)
    reference = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        **kwargs,
    )
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", True)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4_MIN_ROWS", 1)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_GROUPED_PAGE4", False)
    actual = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        **kwargs,
    )
    torch.testing.assert_close(actual.float(), reference.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_qsa_xqa_page4_large_e4m3_key_accumulates_in_fp32(monkeypatch) -> None:
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("QSA XQA page4 is SM70-only")
    pytest.importorskip("flash_attn_v100")
    page_size, topk, head_dim = 2052, 2051, 256
    query = torch.ones((1, 6, head_dim), dtype=torch.float16, device="cuda")
    # E4M3FN bit patterns 0x7e/0xfe encode +/-448.  The raw QK dot is
    # 448 * 256 = 114688, above FP16's finite range, before k_scale is folded
    # into softmax_scale by the XQA kernel.
    key_cache = torch.full(
        (1, page_size, 1, head_dim),
        0x7E,
        dtype=torch.uint8,
        device="cuda",
    )
    value_cache = torch.empty_like(key_cache)
    value_cache[:, 0::2].fill_(0x7E)
    value_cache[:, 1::2].fill_(0xFE)
    logical_indices = torch.arange(topk, dtype=torch.int32, device="cuda").view(1, -1)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    token_to_req = torch.zeros(1, dtype=torch.int32, device="cuda")
    query_positions = torch.tensor([topk - 1], dtype=torch.int64, device="cuda")
    sequence_lengths = torch.tensor([topk], dtype=torch.int32, device="cuda")
    kwargs = {
        "query_positions": query_positions,
        "sequence_lengths": sequence_lengths,
        "kv_cache_dtype": "fp8_e4m3",
        "k_scale": 0.05,
        "v_scale": 1.0 / 448.0,
    }

    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", False)
    reference = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        **kwargs,
    )
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4", True)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_XQA_PAGE4_MIN_ROWS", 1)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_GROUPED_PAGE4", False)
    actual = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        **kwargs,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference.float(), atol=3e-2, rtol=3e-2)
