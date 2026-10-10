# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound attention intermediates while retaining dense continuation semantics."""

import math

import pytest
import torch
import torch.nn.functional as functional

from vllm.v1.attention.backends import turboquant_attn as attention


@pytest.mark.parametrize("rows", [129, 257])
@pytest.mark.parametrize("cached_len", [0, 255])
@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bounded_sdpa_matches_dense_current_chunk(
    monkeypatch, rows, cached_len, heads, dtype, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    generator = torch.Generator(device=device).manual_seed(123)
    query = torch.randn(rows, heads, 32, generator=generator, device=device).to(dtype)
    key = torch.randn(rows + cached_len, 2, 32, generator=generator, device=device).to(
        dtype
    )
    value = torch.randn_like(key)
    scale = 1 / math.sqrt(32)
    positions = torch.arange(rows, device=device)[:, None] + cached_len
    mask = torch.arange(key.shape[0], device=device)[None, :] <= positions
    original = functional.scaled_dot_product_attention
    expected = original(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        attn_mask=mask,
        scale=scale,
        enable_gqa=(heads > 2),
    )[0].transpose(0, 1)
    calls = []

    def observe(q, k, v, **kwargs):
        calls.append(q.shape[2])
        assert q.shape[2] <= 128
        assert kwargs["attn_mask"].shape == (q.shape[2], key.shape[0])
        # The current chunk stays in its original precision and storage.
        assert k.data_ptr() == key.data_ptr()
        assert v.data_ptr() == value.data_ptr()
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(functional, "scaled_dot_product_attention", observe)
    actual = attention._continuation_sdpa(query, key, value, cached_len, scale)
    assert calls == ([128, 1] if rows == 129 else [128, 128, 1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
