# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded sparse prefill numerics, graph replay and production dispatch."""

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
    "tokens,requests,query,decode,expected",
    [
        (768, 1, 768, 5, True),
        (17, 1, 17, 5, True),
        (17, 2, 16, 5, False),
        (20, 4, 5, 5, False),
        (5, 1, 5, 5, False),
        (17, 1, 17, 33, False),
        (32, 1, 17, 5, False),
        (0, 0, 0, 1, False),
    ],
)
def test_prefill_admission(tokens, requests, query, decode, expected):
    assert can_use_sparse_fp16_gemm_prefill(tokens, requests, query, decode) == expected


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
@pytest.mark.parametrize("tokens", [16, 17, 31, 64])
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
    "tokens,requests,query", [(17, 1, 17), (17, 2, 16), (20, 4, 5)]
)
def test_backend_uses_bounded_prefill_only_for_single_request(
    monkeypatch, tokens, requests, query
):
    from vllm.models.glm5next.sm70 import sparse
    from vllm.v1.worker.workspace import WorkspaceManager

    torch.manual_seed(739)
    manager = WorkspaceManager(torch.device("cuda"))
    monkeypatch.setattr(sparse, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(
        sparse,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
            speculative_config=SimpleNamespace(num_speculative_tokens=4),
        ),
    )
    indices = torch.full((tokens, 2176), -1, device="cuda", dtype=torch.int32)
    indices[:, :32] = torch.arange(32, device="cuda", dtype=torch.int32)
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
    original = sparse.sparse_attention_fp16_gemm_chunked

    def traced(*args):
        called.append(True)
        return original(*args)

    monkeypatch.setattr(sparse, "sparse_attention_fp16_gemm_chunked", traced)
    q = torch.randn(tokens, 64, 512, device="cuda", dtype=torch.float16)
    kv = torch.randn(1, 64, 512, device="cuda", dtype=torch.float16)
    metadata = SimpleNamespace(
        num_reqs=requests,
        max_query_len=query,
        req_id_per_token=torch.zeros(tokens, device="cuda", dtype=torch.int32),
        block_table=torch.zeros(requests, 1, device="cuda", dtype=torch.int32),
        block_size=64,
    )
    actual, _ = impl.forward_mqa(q, kv, metadata, None)
    expected_dispatch = current_platform.is_device_capability((7, 0)) and requests == 1
    assert bool(called) == expected_dispatch
    lengths = torch.full((tokens,), 32, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(
        actual.float(),
        _reference(q, kv.reshape(-1, 512), indices, lengths),
        rtol=3e-3,
        atol=2e-3,
    )
