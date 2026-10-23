# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation and invalidation preserve cache contents and ownership."""

import pytest
import torch

from vllm.v1.attention.backends.flash_v100.workspace import V100Workspace

pytestmark = pytest.mark.cpu_test


def test_decode_cache_capacity_reuse_growth_and_invalidation():
    workspace = V100Workspace()
    cache = workspace.decode_cache
    other = V100Workspace().decode_cache
    args = (2, 16, torch.float16, torch.device("cpu"))
    cache.ensure_capacity(3, *args)
    assert cache.key is not None and cache.value is not None
    cache.key[:3].fill_(2)
    cache.value[:3].fill_(7)
    cache.length = 3
    key, value = cache.key, cache.value
    cache.ensure_capacity(8, *args)
    assert cache.key is key and cache.value is value
    cache.ensure_capacity(17, *args)
    assert cache.capacity == 32
    assert cache.key is not key and cache.value is not value
    torch.testing.assert_close(cache.key[:3], key[:3], rtol=0, atol=0)
    torch.testing.assert_close(cache.value[:3], value[:3], rtol=0, atol=0)
    assert other.key is None and other.length == 0
    cache.invalidate()
    assert cache.key is None and cache.value is None
    assert cache.length == cache.capacity == 0
    assert torch.all(key[:3] == 2) and torch.all(value[:3] == 7)


def test_metadata_capture_capacity_is_fixed_and_copy_refreshes_live_storage():
    from vllm.v1.attention.backends.flash_v100.workspace import MetadataWorkspace

    workspace = MetadataWorkspace()
    device = torch.device("cpu")
    draft, smallq = workspace.draft, workspace.smallq
    assert draft.ensure(2, 3, 1, device)
    table, lengths, starts = draft.block_table, draft.seq_lens, draft.query_start_loc
    assert draft.ensure(4, 3, 1, device)
    assert not draft.ensure(4, 3, 3, device)
    assert not draft.ensure(2, 4, 1, device)
    assert draft.block_table is table and draft.seq_lens is lengths
    assert draft.query_start_loc is starts
    assert table is not None and lengths is not None and starts is not None
    draft.copy_metadata(
        torch.tensor([[7, 8, 9]], dtype=torch.int32),
        torch.tensor([11], dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
    )
    assert table[0].tolist() == [7, 8, 9]
    assert lengths[0] == 11 and starts[:2].tolist() == [0, 3]
    assert smallq.ensure(16, 2, 3, 8, 1, device)
    table, indices = smallq.block_table, smallq.token_indices
    assert smallq.ensure(32, 4, 3, 16, 2, device)
    assert not smallq.ensure(32, 4, 3, 17, 2, device)
    assert not smallq.ensure(32, 4, 3, 16, 3, device)
    assert not smallq.ensure(16, 2, 4, 8, 1, device)
    assert smallq.block_table is table and smallq.token_indices is indices
    assert indices is not None and indices.tolist() == list(range(16))
