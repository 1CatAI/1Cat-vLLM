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
