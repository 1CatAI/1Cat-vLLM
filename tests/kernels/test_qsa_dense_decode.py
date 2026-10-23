# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0)
    or not hasattr(torch.ops._C, "qsa_dense_decode_sm70_out"),
    reason="requires SM70 and the qsa_dense_decode_sm70_out operator",
)

HEADS, KV_HEADS, HEAD_SIZE, BLOCK = 6, 1, 256, 16
WIDTH = 2051  # indexer budget 2048 + compress_ratio - 1


@pytest.mark.parametrize("requests", [1, 4])
@pytest.mark.parametrize("context", [37, 300, 701, 1500, 2045])
def test_dense_matches_all_token_sparse_selection(requests, context):
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dense import qsa_dense_decode

    torch.manual_seed(context + requests)
    dev = "cuda"
    tokens = 5 * requests
    blocks = (context + 8 + BLOCK - 1) // BLOCK
    key_cache = (
        torch.randn(requests * blocks + 4, BLOCK, KV_HEADS, HEAD_SIZE, device=dev) * 0.5
    ).half()
    value_cache = torch.randn(
        requests * blocks + 4, BLOCK, KV_HEADS, HEAD_SIZE, device=dev
    ).half()
    query = torch.randn(tokens, HEADS, HEAD_SIZE, device=dev).half()
    gate = torch.randn(tokens, HEADS, HEAD_SIZE, device=dev).half()
    block_table = torch.arange(requests * blocks, device=dev, dtype=torch.int32).view(
        requests, blocks
    )
    token_to_req = torch.arange(requests, device=dev, dtype=torch.int32)
    token_to_req = token_to_req.repeat_interleave(5)
    positions = torch.cat(
        [
            torch.arange(context - 5 - 7 * r, context - 7 * r, dtype=torch.int32)
            for r in range(requests)
        ]
    ).to(dev)
    seq_lens = torch.tensor(
        [context - 7 * r for r in range(requests)], device=dev, dtype=torch.int32
    )
    indices = torch.full((tokens, WIDTH), -1, device=dev, dtype=torch.int32)
    for t in range(tokens):
        n = int(positions[t]) + 1
        indices[t, :n] = torch.arange(n, device=dev, dtype=torch.int32)
    expected = torch.empty(tokens, HEADS, HEAD_SIZE, device=dev, dtype=torch.half)
    qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        indices,
        block_table,
        token_to_req,
        expected,
        kv_cache_dtype="auto",
        k_scale=1.0,
        v_scale=1.0,
        query_positions=positions,
        sequence_lengths=seq_lens,
        output_gate=gate,
    )
    actual = torch.empty_like(expected)
    qsa_dense_decode(
        query,
        key_cache,
        value_cache,
        block_table,
        token_to_req,
        positions,
        actual,
        gate,
        requests,
    )
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert float(error) < 1e-3
