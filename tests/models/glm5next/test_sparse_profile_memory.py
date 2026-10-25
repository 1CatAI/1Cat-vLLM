# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse implementation and indexer keep their own profile reservations."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import sparse_attn_indexer_kpool as indexer
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.models.glm5next.sm70 import sparse
from vllm.v1.worker.workspace import WorkspaceManager

pytestmark = pytest.mark.cpu_test


def _workspace(monkeypatch):
    # WorkspaceManager's allocation is real CPU storage; CUDA allocator cleanup
    # is irrelevant here and must not initialize a device during this test.
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    return WorkspaceManager(torch.device("cpu"))


@pytest.mark.parametrize("cache_dtype", ["float16", "fp8_e4m3"])
def test_sparse_backend_workspace_survives_profile(monkeypatch, cache_dtype):
    manager = _workspace(monkeypatch)
    monkeypatch.setattr(sparse, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(
        sparse,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4)
        ),
    )
    impl = sparse.Glm5NextSM70SparseImpl(
        num_heads=2,
        head_size=512,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=cache_dtype,
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        indexer=SimpleNamespace(topk_indices_buffer=torch.empty((4, 128))),
        qk_rope_head_dim=0,
        kv_lora_rank=512,
    )
    reserved = manager._current_workspaces[0]
    expected_bytes = 4 * 2 * 512 * 2
    if cache_dtype == "fp8_e4m3":
        expected_bytes += 4 * 128 * 512 * 2 + 2 * (4 * 2 * 128 * 2)
    assert reserved is not None and reserved.numel() == expected_bytes
    layer = SimpleNamespace(impl=impl, num_heads=2, v_head_dim=4, qk_nope_head_dim=3)
    q = torch.ones((3, 2, 3), dtype=torch.float16)
    latent = torch.ones((3, 512), dtype=torch.float16)
    pe = torch.empty((3, 1, 0), dtype=torch.float16)
    output = torch.ones((3, 8), dtype=torch.float16)
    MLAAttention.forward_impl(layer, q, latent, pe, latent, None, output)
    assert manager._current_workspaces[0] is reserved
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize("draft_tokens", [0, 7])
def test_indexer_profile_keeps_gather_radix_and_logits(monkeypatch, draft_tokens):
    manager = _workspace(monkeypatch)
    monkeypatch.setattr(indexer, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(
        indexer, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )
    monkeypatch.setattr(
        indexer,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=16),
            speculative_config=SimpleNamespace(num_speculative_tokens=draft_tokens),
        ),
    )
    monkeypatch.setattr(indexer.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
    inputs = torch.ones((2, 128), dtype=torch.float16)
    indices = torch.zeros((2, 8), dtype=torch.int32)
    allocations = []
    original_empty = torch.empty

    def recording_empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        allocations.append(result.numel() * result.element_size())
        return result

    monkeypatch.setattr(torch, "empty", recording_empty)
    result = indexer.sparse_attn_indexer_kpool(
        inputs,
        "indexer.k_cache",
        inputs,
        inputs,
        None,
        inputs,
        inputs,
        128,
        "ue8m0",
        8,
        128,
        65536,
        16,
        indices,
        False,
    )
    assert result is indices
    # Each workspace view is 256-byte aligned by the production manager.
    expected_gather = 16 * 128 + 256 + indexer.RADIX_TOPK_WORKSPACE_SIZE
    expected_logits = max(1024**2, (draft_tokens + 1) * 65536 * 4)
    assert allocations == [expected_gather, expected_logits]
    reserved = manager._current_workspaces[0]
    assert reserved is not None and reserved.numel() == expected_gather
