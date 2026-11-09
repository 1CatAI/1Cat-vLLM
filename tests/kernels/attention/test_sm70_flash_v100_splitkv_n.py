# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def _require_splitkv_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("split-D D256 prefill is SM70-only")
    from vllm.v1.attention.backends.flash_v100 import ops

    op = ops.get_sm70_splitd_d256_splitkv_op()
    if op is None:
        pytest.skip("FA2 library lacks the split-KV operator")
    return op


def _reference(query, key, value, scale):
    query_len, kv_len = query.shape[1], key.shape[1]
    scores = query[0].float().transpose(0, 1) @ key[0, :, 0].float().T * scale
    positions = torch.arange(query_len, device="cuda")[:, None] + kv_len - query_len
    future = torch.arange(kv_len, device="cuda")[None] > positions
    scores = scores.masked_fill(future, float("-inf"))
    return (scores.softmax(-1) @ value[0, :, 0].float()).transpose(0, 1)


@pytest.mark.parametrize("num_splits", [2, 3, 8, 40, 64])
@pytest.mark.parametrize("query_len,kv_len", [(64, 70016), (512, 33024)])
def test_splitkv_matches_unsplit_dense(num_splits, query_len, kv_len):
    op = _require_splitkv_op()
    torch.manual_seed(num_splits)
    query = torch.randn(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    key = torch.randn(1, kv_len, 1, 256, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    scale = 256**-0.5
    rows = query_len * 6
    partial_out = torch.empty(num_splits * rows * 256, device="cuda")
    partial_max = torch.empty(num_splits * rows, device="cuda")
    partial_sum = torch.empty_like(partial_max)
    dense = torch.empty_like(query)
    split = torch.empty_like(query)
    torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_dense_fwd(
        query, key, value, dense, scale, True
    )
    op(
        query,
        key,
        value,
        partial_out,
        partial_max,
        partial_sum,
        split,
        scale,
        True,
        num_splits,
    )
    torch.testing.assert_close(split, dense, atol=1e-4, rtol=0)


def test_splitkv3_entry_is_three_splits():
    op = _require_splitkv_op()
    torch.manual_seed(0)
    query = torch.randn(1, 4096, 6, 256, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 32768, 1, 256, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    rows = 4096 * 6
    workspace = [
        torch.empty(3 * rows * 256, device="cuda"),
        torch.empty(3 * rows, device="cuda"),
        torch.empty(3 * rows, device="cuda"),
    ]
    legacy = torch.empty_like(query)
    generic = torch.empty_like(query)
    torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_dense_splitkv3_fwd(
        query, key, value, *workspace, legacy, 256**-0.5, True
    )
    op(query, key, value, *workspace, generic, 256**-0.5, True, 3)
    assert torch.equal(legacy, generic)


@pytest.mark.parametrize(
    "query_len,seq_len", [(5, 20011), (37, 20011), (100, 30001), (1099, 60013)]
)
def test_padded_short_prefix_chunk_matches_reference(query_len, seq_len):
    """Bridge padding: leading zero queries, trailing zero keys and queries."""
    _require_splitkv_op()
    from vllm.v1.attention.backends.flash_v100 import dense_prefill

    torch.manual_seed(query_len)
    kv_pad = -seq_len % 32
    padded_q = -(-(query_len + kv_pad) // 64) * 64
    lead = padded_q - query_len - kv_pad
    query = torch.randn(1, query_len, 6, 256, device="cuda", dtype=torch.float16)
    key = torch.randn(1, seq_len + 64, 1, 256, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    # Stale cache bytes past the sequence must not leak into the result.
    key[:, seq_len:] = float("nan")
    value[:, seq_len:] = float("nan")
    key[:, seq_len : seq_len + kv_pad] = 0
    value[:, seq_len : seq_len + kv_pad] = 0
    padded = torch.zeros(1, padded_q, 6, 256, device="cuda", dtype=torch.float16)
    padded[:, lead : lead + query_len] = query
    kv_len = seq_len + kv_pad
    cu_q, cu_k = dense_prefill.uniform_cu_seqlens(
        padded, batch_size=1, query_len=padded_q, kv_len=kv_len
    )
    assert dense_prefill._prefill_dense_splitkv_splits(padded, padded_q, kv_len) > 1
    result = dense_prefill.try_sm70_fa2_d256_prefill(
        padded,
        key[:, :kv_len],
        value[:, :kv_len],
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=padded_q,
        max_seqlen_k=kv_len,
        softmax_scale=256**-0.5,
        causal=True,
        window_size=(-1, -1),
        out=torch.empty_like(padded),
    )
    assert result is not None
    expected = _reference(query, key[:, :seq_len], value[:, :seq_len], 256**-0.5)
    torch.testing.assert_close(
        result[0, lead : lead + query_len].float(), expected, atol=2e-3, rtol=0
    )
