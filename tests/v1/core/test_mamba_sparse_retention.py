# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for retention under interleaved long aligned prefills."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

# These tests allocate only block metadata; no device/distributed state exists.
pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _request(name, length, hash_size=8):
    return SimpleNamespace(
        request_id=name,
        num_tokens=length,
        shared_prefix_boundary=0,
        skip_reading_prefix_cache=False,
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


def _prefill(managers, request, block_size, alignment=None, interval=None, eagle=False):
    boundaries = KVCacheCoordinator.get_replay_boundaries(
        SimpleNamespace(eagle_group_ids={0} if eagle else set()),
        request,
        alignment or block_size,
    )
    for start in range(0, request.num_tokens, block_size):
        end = min(start + block_size, request.num_tokens)
        for manager in managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(request.request_id, start)
            manager.allocate_new_blocks(request.request_id, end, end)
            kwargs = (
                {"retention_interval": interval, "replay_boundaries": boundaries}
                if isinstance(manager, MambaManager)
                else {}
            )
            manager.cache_blocks(
                request, end, alignment_tokens=alignment or block_size, **kwargs
            )
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
@pytest.mark.parametrize("interval", [0, 16000])
def test_sparse_retains_both_prompt_replay_boundaries(length, eagle, interval):
    pool = BlockPool(80, True, 8)
    manager = _mamba(pool, 800, 1)
    request = _request("boundary", length)
    _prefill([manager], request, 800, interval=interval, eagle=eagle)
    assert (
        _hit(manager, request, pool, 800, eagle)
        == max(0, (length - 1) // 800 - int(eagle)) * 800
    )
    assert pool.get_num_free_blocks() == 79


@pytest.mark.parametrize(
    "interval,expected", [(None, 0), (0, 169 * 16), (320, 169 * 16)]
)
def test_long_b_does_not_flush_a_when_sparse_enabled(interval, expected):
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
    _prefill(managers, a, 16, interval=interval, eagle=True)
    assert min(_hit(m, a, pool, 16, True) for m in managers) == 169 * 16
    _prefill(managers, b, 16, interval=interval, eagle=True)
    assert min(_hit(m, a, pool, 16, True) for m in managers) == expected
    assert pool.get_num_free_blocks() == 649


@pytest.mark.parametrize("alignment", [800, 1600])
def test_admission_falls_back_for_different_alignment(alignment):
    def retained(interval):
        pool = BlockPool(80, True, 8)
        manager = _mamba(pool, 800, 1)
        request = _request("fallback", 16001)
        _prefill([manager], request, 800, alignment, interval=interval)
        hashes = BlockHashListWithBlockSize(request.block_hashes, 8, 800)
        return [i for i, h in enumerate(hashes) if pool.get_cached_block(h, [1])]

    dense, sparse = retained(None), retained(16000)
    if alignment == 800:
        assert len(sparse) < len(dense)
    else:
        assert sparse == dense


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


def _cache_manager(block_size=32, interval=0, eagle=True, capacity=1000):
    full = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        capacity,
        [],
        [KVCacheGroupSpec(["full"], full), KVCacheGroupSpec(["mamba"], mamba)],
    )
    return KVCacheManager(
        config, 100000, 8, use_eagle=eagle, prefix_cache_retention_interval=interval
    )


def _coordinated_prefill(cm, request, block_size=32):
    for start in range(0, request.num_tokens, block_size):
        end = min(start + block_size, request.num_tokens)
        for manager in cm.coordinator.single_type_managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(request.request_id, start)
            manager.allocate_new_blocks(request.request_id, end, end)
        cm.coordinator.cache_blocks(request, end)
    cm.coordinator.free(request.request_id)


@pytest.mark.parametrize("interval", [None, 0, 32, 16000])
def test_upstream_config_semantics(interval):
    assert CacheConfig().prefix_cache_retention_interval == 0
    assert (
        CacheConfig(
            prefix_cache_retention_interval=interval
        ).prefix_cache_retention_interval
        == interval
    )
    manager = _cache_manager(interval=interval)
    assert manager.coordinator.retention_interval == interval
    assert (
        CacheConfig(prefix_cache_retention_interval=interval).compute_hash()
        == CacheConfig().compute_hash()
    )


@pytest.mark.parametrize("interval", [-1, 1000])
def test_invalid_interval_fails_after_geometry_resolution(interval):
    with pytest.raises(ValueError, match="resolved alignment"):
        _cache_manager(block_size=800, interval=interval)


@pytest.mark.parametrize("length", [127, 128, 129, 160, 161])
@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize("interval", [0, 320])
def test_hybrid_identical_resend_and_longer_sibling(length, eagle, interval):
    cm = _cache_manager(interval=interval, eagle=eagle)
    a = _request("same", length)
    _coordinated_prefill(cm, a)
    _, resend = cm.get_computed_blocks(a)
    sibling = _request("same", length + 64)
    _, extension = cm.get_computed_blocks(sibling)
    assert resend == max(0, (length - 1) // 32 - int(eagle)) * 32
    assert extension == (max(0, length // 32 - 1) if eagle else (length - 1) // 32) * 32
    assert cm.block_pool.get_num_free_blocks() == 999


@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize("retain_junction", [False, True])
def test_detected_junction_survives_semantic_only_retention(eagle, retain_junction):
    cm = _cache_manager(interval=0, eagle=eagle)
    a, b, c = [
        _request(name, length * 32 + 3)
        for name, length in [("a", 80), ("b", 60), ("c", 70)]
    ]
    shared_hashes = 30 * 32 // 8
    b.block_hashes[:shared_hashes] = a.block_hashes[:shared_hashes]
    c.block_hashes[:shared_hashes] = a.block_hashes[:shared_hashes]
    _coordinated_prefill(cm, a)
    _, first_hit = cm.get_computed_blocks(b)
    assert first_hit == 0
    assert b.shared_prefix_boundary == (30 - int(eagle)) * 32
    if not retain_junction:
        b.shared_prefix_boundary = 0
    _coordinated_prefill(cm, b)
    _, later_hit = cm.get_computed_blocks(c)
    assert later_hit == ((30 - int(eagle)) * 32 if retain_junction else 0)
    assert cm.block_pool.get_num_free_blocks() == 999


@pytest.mark.parametrize("block_size", [16, 512, 800, 1024])
@pytest.mark.parametrize("interval_blocks", [1, 5, 10, 20, 40])
def test_token_interval_uses_resolved_block_size(block_size, interval_blocks):
    pool = BlockPool(100, True, 8)
    manager = _mamba(pool, block_size, 1)
    request = _request("periodic", 45 * block_size + 3)
    _prefill([manager], request, block_size, interval=interval_blocks * block_size)
    hashes = BlockHashListWithBlockSize(request.block_hashes, 8, block_size)
    retained = [i + 1 for i, h in enumerate(hashes) if pool.get_cached_block(h, [1])]
    assert retained == sorted(set(range(interval_blocks, 46, interval_blocks)) | {45})
    assert pool.get_num_free_blocks() == 99


@pytest.mark.parametrize("eagle_groups", [{0}, {1}, {2}, {0, 2}])
@pytest.mark.parametrize("length", [127, 128, 129])
def test_mixed_main_and_draft_groups_retain_joint_restore_point(eagle_groups, length):
    cm = _cache_manager()
    config = cm.coordinator.kv_cache_config
    config.kv_cache_groups.append(
        KVCacheGroupSpec(["draft_mamba"], config.kv_cache_groups[1].kv_cache_spec)
    )
    for index, group in enumerate(config.kv_cache_groups):
        group.is_eagle_group = index in eagle_groups
    cm = KVCacheManager(
        config, 100000, 8, use_eagle=True, prefix_cache_retention_interval=0
    )
    request = _request("mixed", length)
    _coordinated_prefill(cm, request)
    _, hit = cm.get_computed_blocks(request)
    assert hit == max(0, (length - 1) // 32 - 1) * 32
    _, extension = cm.get_computed_blocks(_request("mixed", length + 64))
    assert extension == max(0, length // 32 - 1) * 32
