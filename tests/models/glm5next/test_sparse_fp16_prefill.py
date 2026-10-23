# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded sparse prefill numerics, graph replay and production dispatch."""

from itertools import product
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.sparse_fp16_gemm import (
    sparse_attention_fp16_gemm_chunked,
)
from vllm.models.deepseek_v4.common.portable_sparse import (
    can_use_sparse_fp16_gemm_prefill,
    use_portable_sparse,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "tokens,requests,query,actual,decode,expected",
    [
        (768, 1, 768, 768, 5, True),
        (17, 1, 17, 17, 5, True),
        (768, 3, 256, 768, 5, True),  # Three balanced prefills.
        (768, 2, 384, 768, 5, True),
        (36, 2, 19, 36, 5, True),
        (30, 3, 11, 30, 5, True),
        (12, 2, 6, 12, 5, True),
        (768, 2, 763, 768, 5, False),  # One five-token decode.
        (768, 3, 758, 768, 5, False),  # Two five-token decodes.
        (768, 3, 756, 768, 5, False),  # Uneven pure prefill: fail closed.
        (17, 2, 16, 17, 5, False),
        (20, 4, 5, 20, 5, False),
        (5, 1, 5, 5, 5, False),
        (17, 1, 17, 17, 33, False),
        (32, 1, 17, 17, 5, False),  # Tensor has padding.
        (32, 1, 17, 32, 5, False),  # Impossible single-query metadata.
        (20, 2, 9, 20, 5, False),  # Total exceeds N times maximum.
        (17, 2, 32, 17, 5, False),  # Maximum exceeds total.
        (17, 2, 17, 17, 5, False),  # One zero-length query.
        (0, 0, 0, 0, 1, False),
        (0, 1, 0, 0, 1, False),
        (17, 0, 17, 17, 5, False),
        (17, -1, 17, 17, 5, False),
        (17, 1, 17, 17, 0, False),
        (17, 1, 17, -1, 5, False),
    ],
)
def test_prefill_admission(tokens, requests, query, actual, decode, expected):
    assert (
        can_use_sparse_fp16_gemm_prefill(
            tokens, requests, query, decode, num_actual_tokens=actual
        )
        == expected
    )


@pytest.mark.parametrize("decode", [1, 5, 7])
def test_prefill_admission_never_accepts_a_decode_or_empty_query(decode):
    # Enumerate actual query lists rather than restating the admission formula.
    for requests in (1, 2, 3, 4):
        for lengths in product(range(9), repeat=requests):
            tokens = sum(lengths)
            admitted = can_use_sparse_fp16_gemm_prefill(
                tokens,
                requests,
                max(lengths),
                decode,
                num_actual_tokens=tokens,
            )
            if admitted:
                assert all(length > decode for length in lengths), lengths


@pytest.mark.parametrize("lengths", [(5, 763), (763, 5)])
def test_prefill_admission_rejects_mixed_queries_in_either_order(lengths):
    assert not can_use_sparse_fp16_gemm_prefill(
        sum(lengths), len(lengths), max(lengths), 5, num_actual_tokens=sum(lengths)
    )


requires_cuda = pytest.mark.skipif(
    not torch.accelerator.is_available() or not use_portable_sparse(),
    reason="requires a portable sparse CUDA device",
)


def _scratch(q, width, capacity=16):
    return (
        torch.empty_like(q),
        torch.empty((capacity, width, 512), device=q.device, dtype=q.dtype),
        torch.empty((capacity, 64, width), device=q.device, dtype=torch.float32),
        torch.empty((capacity, 64, width), device=q.device, dtype=q.dtype),
    )


def _reference(q, kv, indices, lengths):
    position = torch.arange(indices.shape[1], device=q.device)[None, :]
    valid = (position < lengths[:, None]) & (indices >= 0) & (indices < len(kv))
    selected = kv[indices.clamp(0, len(kv) - 1).long()].float()
    scores = torch.bmm(q.float(), selected.transpose(1, 2)) * 512**-0.5
    scores.masked_fill_(~valid[:, None, :], torch.finfo(torch.float32).min)
    probs = torch.softmax(scores, -1).masked_fill(~valid[:, None, :], 0)
    return torch.bmm(probs, selected)


@requires_cuda
@pytest.mark.parametrize("width", [2048, 2176])
@pytest.mark.parametrize("tokens", [16, 17, 24, 31, 33, 64])
def test_chunked_prefill_graph_numerics(width, tokens):
    torch.manual_seed(731)
    q = torch.randn(tokens, 66, 512, device="cuda", dtype=torch.float16)[:, :64]
    kv = torch.randn(4096, 512, device="cuda", dtype=torch.float16)
    indices = torch.randint(
        0, len(kv), (tokens, width + 3), device="cuda", dtype=torch.int32
    )[:, :width]
    lengths = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    buffers = _scratch(q, width)

    def run():
        sparse_attention_fp16_gemm_chunked(q, kv, indices, lengths, 512**-0.5, *buffers)

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for amplitude in (1.0, 4.0):
        q.normal_(std=amplitude)
        kv.normal_()
        indices.random_(0, len(kv))
        indices[:, 3] = -1
        indices[:, 8] = len(kv) + 3
        counts = ([0, 1, 17, 2051, width, width + 1, -1] * tokens)[:tokens]
        lengths.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.accelerator.synchronize()
        actual = buffers[0].clone()
        run()
        assert torch.equal(actual, buffers[0])
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual.float(), _reference(q, kv, indices, lengths), rtol=3e-3, atol=2e-3
        )


@requires_cuda
def test_prefill_chunk_boundaries_and_fp32_overflow():
    q = torch.full((33, 64, 512), 100.0, device="cuda", dtype=torch.float16)
    kv = torch.full((32, 512), 2.0, device="cuda", dtype=torch.float16)
    indices = torch.full((33, 2176), -1, device="cuda", dtype=torch.int32)
    indices[:, :32] = torch.arange(32, device="cuda", dtype=torch.int32)
    lengths = torch.full((33,), 32, device="cuda", dtype=torch.int32)
    for capacity in (8, 16, 32):
        buffers = _scratch(q, 2176, capacity)
        sparse_attention_fp16_gemm_chunked(q, kv, indices, lengths, 512**-0.5, *buffers)
        assert buffers[2][0, 0, 0] > torch.finfo(torch.float16).max
        torch.testing.assert_close(buffers[0], torch.full_like(q, 2), rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize(
    "query_lengths,padding,sm70_route",
    [
        ((17,), 0, "chunked"),
        ((17, 19), 0, "chunked"),
        ((9, 10, 11), 0, "chunked"),
        ((5, 27), 0, "scalar"),
        ((27, 5), 0, "scalar"),
        ((6, 6, 52), 0, "scalar"),
        ((5, 5, 5, 5), 0, "scalar"),
        ((5, 5, 5), 0, "per_token"),
        ((6, 6), 0, "per_token"),  # Existing small-batch path keeps priority.
        ((17,), 15, "scalar"),
    ],
)
def test_backend_dispatch_and_request_isolation(
    monkeypatch, query_lengths, padding, sm70_route
):
    from vllm.models.glm5next.sm70 import sparse
    from vllm.v1.worker.workspace import WorkspaceManager

    torch.manual_seed(739)
    actual_tokens = sum(query_lengths)
    tokens = actual_tokens + padding
    requests = len(query_lengths)
    manager = WorkspaceManager(torch.device("cuda"))
    monkeypatch.setattr(sparse, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(
        sparse,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=max(32, tokens)),
            speculative_config=SimpleNamespace(num_speculative_tokens=4),
        ),
    )
    indices = torch.full((tokens, 2176), -1, device="cuda", dtype=torch.int32)
    indices[:actual_tokens, :32] = torch.arange(
        0, 128, 4, device="cuda", dtype=torch.int32
    )
    impl = sparse.Glm5NextSM70SparseImpl(
        num_heads=64,
        head_size=512,
        scale=512**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="float16",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        indexer=SimpleNamespace(topk_indices_buffer=indices),
        qk_rope_head_dim=0,
        kv_lora_rank=512,
    )
    manager.lock()
    called = []

    def trace_route(name, function):
        def traced(*args, **kwargs):
            called.append(name)
            return function(*args, **kwargs)

        return traced

    for name, attribute in (
        ("chunked", "sparse_attention_fp16_gemm_chunked"),
        ("per_token", "sparse_attention_fp16_gemm"),
        ("scalar", "sm70_sparse_attention_gathered"),
    ):
        monkeypatch.setattr(
            sparse, attribute, trace_route(name, getattr(sparse, attribute))
        )
    q = torch.randn(tokens, 64, 512, device="cuda", dtype=torch.float16)
    # Each request owns two different physical pages, deliberately reversed.
    # The two/three-request cases cross the 16-query scratch-chunk boundary.
    kv = torch.randn(2 * requests, 64, 512, device="cuda", dtype=torch.float16)
    block_table = (
        torch.arange(2 * requests, device="cuda", dtype=torch.int32)
        .view(requests, 2)
        .flip(1)
        .contiguous()
    )
    request_ids = torch.tensor(
        [request for request, length in enumerate(query_lengths) for _ in range(length)]
        + [0] * padding,
        device="cuda",
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        num_reqs=requests,
        num_actual_tokens=actual_tokens,
        max_query_len=max(query_lengths),
        req_id_per_token=request_ids,
        block_table=block_table,
        block_size=64,
    )
    actual, _ = impl.forward_mqa(q, kv, metadata, None)
    actual = actual.clone()
    expected_route = (
        sm70_route if current_platform.is_device_capability((7, 0)) else "scalar"
    )
    assert called == [expected_route]
    lengths = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    lengths[:actual_tokens] = 32
    # Independently resolve each request's logical indices to physical pages.
    # Accidentally sharing request zero's KV must fail this dense reference.
    page = indices.clamp_min(0).long() // 64
    global_indices = (
        block_table[request_ids.long()[:, None], page] * 64 + indices.clamp_min(0) % 64
    )
    global_indices.masked_fill_(indices < 0, -1)
    torch.testing.assert_close(
        actual.float(),
        _reference(q, kv.reshape(-1, 512), global_indices, lengths),
        rtol=3e-3,
        atol=2e-3,
    )
    if len(query_lengths) > 1 and sm70_route == "chunked":
        # Re-run the same inputs one request at a time, changing only grouping
        # and where scratch chunks end. Use each request's own physical pages.
        start = 0
        for request, length in enumerate(query_lengths):
            separate_metadata = SimpleNamespace(
                num_reqs=1,
                num_actual_tokens=length,
                max_query_len=length,
                req_id_per_token=torch.zeros(length, device="cuda", dtype=torch.int32),
                block_table=block_table[request : request + 1],
                block_size=64,
            )
            separate, _ = impl.forward_mqa(
                q[start : start + length], kv, separate_metadata, None
            )
            torch.testing.assert_close(
                actual[start : start + length], separate, rtol=3e-3, atol=2e-3
            )
            start += length
