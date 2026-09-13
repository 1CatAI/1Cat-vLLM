# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for retention under interleaved long aligned prefills."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

# These tests allocate only block metadata; no device/distributed state exists.
pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _request(name, length, hash_size=8):
    return SimpleNamespace(
        request_id=name,
        num_tokens=length,
        block_hashes=[
            hashlib.sha256(f"{name}:{i}".encode()).digest()
            for i in range(length // hash_size)
        ],
    )


def _mamba(pool, block_size, group):
    return MambaManager(
        MambaSpec(
            block_size=block_size,
            shapes=((1,),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
        ),
        pool,
        enable_caching=True,
        kv_cache_group_id=group,
    )


def _prefill(managers, request, block_size, alignment=None):
    for start in range(0, request.num_tokens, block_size):
        end = min(start + block_size, request.num_tokens)
        for manager in managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(request.request_id, start)
            manager.allocate_new_blocks(request.request_id, end, end)
            manager.cache_blocks(request, end, alignment_tokens=alignment or block_size)
    for manager in managers:
        manager.free(request.request_id)


def _hit(manager, request, pool, block_size, eagle):
    hashes = BlockHashListWithBlockSize(request.block_hashes, 8, block_size)
    return (
        len(
            manager.find_longest_cache_hit(
                hashes,
                request.num_tokens - 1,
                [manager.kv_cache_group_id],
                pool,
                manager.kv_cache_spec,
                eagle,
                block_size,
            )[0]
        )
        * block_size
    )


@pytest.mark.parametrize(
    "length", [799, 800, 801, 15999, 16000, 16001, 16799, 16800, 16801]
)
@pytest.mark.parametrize("eagle", [False, True])
def test_sparse_retains_both_prompt_replay_boundaries(monkeypatch, length, eagle):
    monkeypatch.setenv("VLLM_MAMBA_SPARSE_CACHE_INTERVAL", "16000")
    pool = BlockPool(80, True, 8)
    manager = _mamba(pool, 800, 1)
    request = _request("boundary", length)
    _prefill([manager], request, 800)
    assert (
        _hit(manager, request, pool, 800, eagle)
        == max(0, (length - 1) // 800 - int(eagle)) * 800
    )
    assert pool.get_num_free_blocks() == 79


@pytest.mark.parametrize("interval,expected", [(0, 0), (320, 169 * 16)])
def test_long_b_does_not_flush_a_when_sparse_enabled(monkeypatch, interval, expected):
    monkeypatch.setenv("VLLM_MAMBA_SPARSE_CACHE_INTERVAL", str(interval))
    pool = BlockPool(650, True, 8)
    full = FullAttentionManager(
        FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float16
        ),
        pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )
    managers = [full] + [_mamba(pool, 16, group) for group in range(1, 5)]
    a, b = _request("a", 170 * 16 + 3), _request("b", 175 * 16 + 7)
    _prefill(managers, a, 16)
    assert min(_hit(m, a, pool, 16, True) for m in managers) == 169 * 16
    _prefill(managers, b, 16)
    assert min(_hit(m, a, pool, 16, True) for m in managers) == expected
    assert pool.get_num_free_blocks() == 649


@pytest.mark.parametrize("alignment", [800, 1600])
def test_admission_falls_back_for_different_alignment(monkeypatch, alignment):
    def retained(interval):
        monkeypatch.setenv("VLLM_MAMBA_SPARSE_CACHE_INTERVAL", str(interval))
        pool = BlockPool(80, True, 8)
        manager = _mamba(pool, 800, 1)
        request = _request("fallback", 16001)
        _prefill([manager], request, 800, alignment)
        hashes = BlockHashListWithBlockSize(request.block_hashes, 8, 800)
        return [i for i, h in enumerate(hashes) if pool.get_cached_block(h, [1])]

    dense, sparse = retained(0), retained(16000)
    if alignment == 800:
        assert len(sparse) < len(dense)
    else:
        assert sparse == dense


@pytest.mark.parametrize("interval", [-1, 1000])
def test_invalid_interval_fails_early(monkeypatch, interval):
    monkeypatch.setenv("VLLM_MAMBA_SPARSE_CACHE_INTERVAL", str(interval))
    with pytest.raises(ValueError, match="block multiple"):
        _mamba(BlockPool(80, True, 8), 800, 1)


@pytest.mark.parametrize("enable_caching", [False, True])
def test_scratch_reuse_preserves_cached_and_shared_blocks(enable_caching):
    pool = BlockPool(7, enable_caching, 8)
    cached, first, second, shared = pool.get_new_blocks(4)
    untouched = list(pool.free_block_queue.get_all_free_blocks())
    request = _request("cached", 8)
    if enable_caching:
        pool.cache_full_blocks(request, [cached], 0, 1, 8, 0)
    pool.touch([shared])
    pool.free_blocks([cached])
    pool.free_blocks([first, second, shared])
    assert shared.ref_cnt == 1
    reused = pool.get_new_blocks(2)
    assert reused == ([first, second] if enable_caching else untouched)
    if enable_caching:
        assert pool.get_cached_block(request.block_hashes[0], [0]) == [cached]
    pool.free_blocks(reused + [shared])
    assert pool.get_num_free_blocks() == 6
