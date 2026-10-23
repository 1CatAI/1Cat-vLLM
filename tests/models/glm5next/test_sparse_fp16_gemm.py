# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Numerics, speculation batch invariance and replay of sparse GEMM attention."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.sparse_fp16_gemm import (
    sparse_attention_fp16_gemm,
)
from vllm.models.deepseek_v4.common.portable_sparse import (
    sparse_fp16_gemm_max_tokens,
    use_portable_sparse,
)
from vllm.platforms.interface import DeviceCapability


@pytest.mark.parametrize(
    "capability,heads,dim,width,dtype,expected",
    [
        (DeviceCapability(7, 0), 64, 512, 2048, torch.float16, 15),
        (DeviceCapability(7, 0), 64, 512, 2176, torch.float16, 15),
        (None, 64, 512, 2048, torch.float16, 0),
        (DeviceCapability(7, 2), 64, 512, 2048, torch.float16, 0),
        (DeviceCapability(8, 6), 64, 512, 2048, torch.float16, 0),
        (DeviceCapability(8, 9), 64, 512, 2048, torch.float16, 0),
        (DeviceCapability(7, 0), 8, 512, 2048, torch.float16, 0),
        (DeviceCapability(7, 0), 64, 576, 2048, torch.float16, 0),
        (DeviceCapability(7, 0), 64, 512, 4096, torch.float16, 0),
        (DeviceCapability(7, 0), 64, 512, 2048, torch.bfloat16, 0),
    ],
)
def test_gemm_admission(capability, heads, dim, width, dtype, expected):
    assert sparse_fp16_gemm_max_tokens(capability, heads, dim, width, dtype) == expected


@pytest.mark.parametrize("width", [2048, 2176])
def test_fp16_gemm_workspace_is_reserved_before_capture(monkeypatch, width):
    from vllm.models.glm5next.sm70 import sparse
    from vllm.v1.worker.workspace import WorkspaceManager

    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    manager = WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(sparse, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(
        sparse,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda: DeviceCapability(7, 0),
        ),
    )
    monkeypatch.setattr(
        sparse,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=32)
        ),
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
        indexer=SimpleNamespace(topk_indices_buffer=torch.empty(32, width)),
        qk_rope_head_dim=0,
        kv_lora_rank=512,
    )
    assert impl.fp16_gemm_max_tokens == 15
    assert impl.fp16_gemm_workspace_tokens == 16
    manager.lock()
    buffers = manager.get_simultaneous(
        ((16, 64, 512), torch.float16),
        ((16, width, 512), torch.float16),
        ((16, 64, width), torch.float32),
        ((16, 64, width), torch.float16),
    )
    assert buffers[2].dtype == torch.float32


requires_cuda = pytest.mark.skipif(
    not torch.accelerator.is_available() or not use_portable_sparse(),
    reason="requires a portable sparse CUDA device",
)


def _scratch(q, width):
    tokens, heads, dim = q.shape
    return (
        torch.empty_like(q),
        torch.empty((tokens, width, dim), device=q.device, dtype=q.dtype),
        torch.empty((tokens, heads, width), device=q.device, dtype=torch.float32),
        torch.empty((tokens, heads, width), device=q.device, dtype=q.dtype),
    )


def _reference(q, kv, indices, lengths):
    rows = []
    for row, count in enumerate(lengths.cpu().tolist()):
        ids = indices[row, : max(0, min(count, indices.shape[1]))]
        ids = ids[(ids >= 0) & (ids < len(kv))]
        if not len(ids):
            rows.append(torch.zeros_like(q[row]).float())
            continue
        selected = kv[ids.long()].float()
        scores = q[row].float() @ selected.T * 512**-0.5
        rows.append(torch.softmax(scores, dim=-1) @ selected)
    return torch.stack(rows)


@requires_cuda
@pytest.mark.parametrize("amplitude", [0.2, 1.0, 4.0])
@pytest.mark.parametrize("width", [2048, 2176])
def test_sparse_gemm_graph_lengths_and_numerics(amplitude, width):
    torch.manual_seed(719)
    q = torch.randn(5, 66, 512, device="cuda", dtype=torch.float16)[:, :64]
    kv = torch.randn(4096, 512, device="cuda", dtype=torch.float16)
    indices = torch.randint(
        0, len(kv), (5, width + 7), device="cuda", dtype=torch.int32
    )
    indices = indices[:, :width]
    lengths = torch.zeros(5, device="cuda", dtype=torch.int32)
    buffers = _scratch(q, width)

    def run():
        sparse_attention_fp16_gemm(q, kv, indices, lengths, 512**-0.5, *buffers)

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for counts in (
        [1, 17, 256, width, 0],
        [width - 1, 0, -1, width + 1, 33],
        [2048, 2051, 2049, 2050, 1],
    ):
        q.normal_(std=amplitude)
        kv.normal_()
        indices.random_(0, len(kv))
        indices[:, 3] = -1
        indices[:, 8] = len(kv) + 10
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
@pytest.mark.parametrize("tokens", [2, 3, 5, 8, 10, 15])
@pytest.mark.parametrize("width", [2048, 2176])
def test_sparse_gemm_speculation_batch_invariance(tokens, width):
    torch.manual_seed(723)
    q = torch.randn(tokens, 64, 512, device="cuda", dtype=torch.float16)
    kv = torch.randn(4096, 512, device="cuda", dtype=torch.float16)
    indices = torch.randint(
        0, len(kv), (tokens, width), device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor(
        ([17, width, 257, 0, 2051] * 3)[:tokens], device="cuda", dtype=torch.int32
    )
    buffers = _scratch(q, width)
    sparse_attention_fp16_gemm(q, kv, indices, lengths, 512**-0.5, *buffers)
    batched = buffers[0].clone()
    for row in range(tokens):
        one = _scratch(q[row : row + 1], width)
        sparse_attention_fp16_gemm(
            q[row : row + 1],
            kv,
            indices[row : row + 1],
            lengths[row : row + 1],
            512**-0.5,
            *one,
        )
        assert torch.equal(batched[row], one[0][0])


@requires_cuda
@pytest.mark.parametrize("width", [2048, 2176])
def test_sparse_gemm_fp32_scores_do_not_overflow_fp16(width):
    # Every input is finite but the unscaled QK product exceeds FP16 range.
    q = torch.full((1, 64, 512), 100.0, device="cuda", dtype=torch.float16)
    kv = torch.full((32, 512), 2.0, device="cuda", dtype=torch.float16)
    indices = torch.full((1, width), -1, device="cuda", dtype=torch.int32)
    indices[0, :32] = torch.arange(32, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([32], device="cuda", dtype=torch.int32)
    buffers = _scratch(q, width)
    sparse_attention_fp16_gemm(q, kv, indices, lengths, 512**-0.5, *buffers)
    assert buffers[2][0, 0, 0] > torch.finfo(torch.float16).max
    assert torch.isfinite(buffers[0]).all()
    torch.testing.assert_close(buffers[0], torch.full_like(q, 2.0), rtol=0, atol=0)
