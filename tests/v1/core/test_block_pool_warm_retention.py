# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
from typing import Any

import pytest
import torch

import vllm.envs as envs
from vllm.config.cache import CacheConfig, KVRetentionConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)


def _policy(**kwargs: Any) -> KVRetentionConfig:
    return KVRetentionConfig(
        **dict(enabled=True, soft_cap=1.0, hard_cap=1.0, absolute_cap=1.0) | kwargs
    )


def _pool(**kwargs: Any) -> BlockPool:
    return BlockPool(17, True, 16, retention_policy=_policy(), **kwargs)


def _cache(pool: BlockPool, blocks: list[KVCacheBlock], group: int = 0) -> None:
    for block in blocks:
        block.block_hash = make_block_hash_with_group_id(
            BlockHash(str(block.block_id).encode()), group
        )
        pool.cached_block_hash_to_block.insert(block.block_hash, block)


def test_warm_blocks_are_reclaimable_without_capacity_loss():
    pool = _pool()
    blocks = pool.get_new_blocks(16)
    _cache(pool, blocks)
    pool.free_blocks(blocks)
    assert pool.warm_block_count == 16
    assert pool.free_block_queue.num_free_blocks == 0
    assert pool.get_num_free_blocks() == 16
    assert pool.get_usage() == 0.0
    allocated = pool.get_new_blocks(16)
    assert len({block.block_id for block in allocated}) == 16
    assert all(block.ref_cnt == 1 and block.block_hash is None for block in allocated)
    assert pool.warm_block_count == pool.get_num_free_blocks() == 0
    pool.free_blocks(allocated)
    assert pool.get_num_free_blocks() == 16


def test_external_eviction_and_reset_preserve_queue_membership():
    pool = _pool()
    blocks = pool.get_new_blocks(16)
    _cache(pool, blocks)
    pool.free_blocks(blocks)
    victim = blocks[0]
    pool.evict_blocks({victim.block_id})
    assert victim.block_hash is None
    assert pool.warm_block_count == 15
    assert pool.free_block_queue.num_free_blocks == 1
    assert pool.get_new_blocks(1) == [victim]
    assert not pool.reset_prefix_cache()
    pool.free_blocks([victim])
    assert pool.reset_prefix_cache()
    assert pool.warm_block_count == 0
    assert not pool._warm_freed_at and not pool._warm_hits
    assert all(block.block_hash is None for block in pool.blocks)
    assert len({block.block_id for block in pool.get_new_blocks(16)}) == 16


def test_partition_quotas_count_reclaimable_history_and_preserve_state():
    pool = _pool(device_state_blocks=5, device_state_group_ids=(7,))
    state = pool.get_new_blocks(4, 7)
    history = pool.get_new_blocks(12, 0)
    _cache(pool, state, 7)
    _cache(pool, history)
    pool.free_blocks(state)
    pool.free_blocks(history)
    assert pool.warm_block_count == 12
    assert pool.warm_block_cap == 12
    assert pool.get_num_free_blocks(7) == 4
    assert pool.get_num_free_blocks(0) == 12
    assert pool.can_allocate_group_blocks({7: 4, 0: 12})
    assert all(block.block_id >= 5 for block in pool.get_new_blocks(12, 0))
    assert all(0 < block.block_id < 5 for block in pool.get_new_blocks(4, 7))
    assert pool.get_num_free_blocks() == 0


@pytest.mark.parametrize("fraction, expected", [(0.0, 0), (0.5, 8), (1.0, 16)])
def test_absolute_cap_bounds_even_recent_entries(fraction, expected):
    pool = BlockPool(
        17,
        True,
        16,
        retention_policy=_policy(
            soft_cap=0.0,
            hard_cap=fraction,
            absolute_cap=fraction,
            grace_seconds=1e6,
        ),
    )
    blocks = pool.get_new_blocks(16)
    _cache(pool, blocks)
    pool.free_blocks(blocks)
    assert pool.warm_block_count == expected
    assert pool.get_num_free_blocks() == 16


def test_duplicate_hash_keeps_reuse_score_until_last_copy_is_evicted():
    pool = _pool()
    first, second = pool.get_new_blocks(2)
    key = make_block_hash_with_group_id(BlockHash(b"same prefix"), 0)
    for block in (first, second):
        block.block_hash = key
        pool.cached_block_hash_to_block.insert(key, block)
    pool.free_blocks([first, second])
    pool.touch([first])
    pool.free_blocks([first])
    assert pool._warm_hits[key] == 1
    pool.evict_blocks({second.block_id})
    assert pool._warm_hits[key] == 1
    pool.evict_blocks({first.block_id})
    assert key not in pool._warm_hits
    assert pool.warm_block_count == 0
    assert pool.get_num_free_blocks() == 16


def _worker_policy(queue: Any, policy: KVRetentionConfig) -> None:
    queue.put((policy.enabled, policy.hard_cap))


def test_captured_policy_survives_worker_environment_changes(monkeypatch):
    monkeypatch.setenv("VLLM_KV_RETENTION_ENABLED", "0")
    envs.disable_envs_cache()
    captured = CacheConfig().retention
    monkeypatch.setenv("VLLM_KV_RETENTION_ENABLED", "1")
    assert KVRetentionConfig().enabled
    assert not captured.enabled
    assert not KVRetentionConfig(enabled=False).enabled
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_worker_policy, args=(queue, captured))
    try:
        process.start()
        assert queue.get(timeout=30) == (False, captured.hard_cap)
        process.join(30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        queue.close()
        queue.join_thread()


def test_retention_policy_does_not_change_the_model_graph_hash():
    assert (
        CacheConfig(retention=_policy()).compute_hash()
        == CacheConfig(retention=_policy(enabled=False)).compute_hash()
    )


@pytest.mark.parametrize(
    "values",
    [
        dict(soft_cap=-0.1),
        dict(absolute_cap=1.1),
        dict(grace_seconds=-1),
        dict(grace_seconds=float("inf")),
        dict(hard_cap=0.1, soft_cap=0.2),
        dict(hard_cap=0.5, absolute_cap=0.4),
        dict(p25_seconds=-1),
    ],
)
def test_invalid_retention_policy_is_rejected(values):
    with pytest.raises(ValueError):
        KVRetentionConfig(**values)


@pytest.mark.parametrize("enable_caching", [False, True])
@pytest.mark.parametrize("groups", [1, 2])
def test_manager_passes_captured_policy_through_all_coordinators(
    monkeypatch, enable_caching, groups
):
    captured = _policy(hard_cap=0.5, soft_cap=0.25)
    monkeypatch.setenv("VLLM_KV_RETENTION_ENABLED", "0")
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    manager = KVCacheManager(
        kv_cache_config=KVCacheConfig(
            num_blocks=17,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec([f"layer{i}"], spec) for i in range(groups)
            ],
        ),
        max_model_len=256,
        hash_block_size=16,
        enable_caching=enable_caching,
        retention_policy=captured,
    )
    pool = manager.block_pool
    blocks = pool.get_new_blocks(2)
    _cache(pool, blocks)
    pool.free_blocks(blocks)
    assert pool.warm_block_count == (2 if enable_caching else 0)
    assert pool.get_num_free_blocks() == 16
    assert pool.warm_block_cap == (8 if enable_caching else 0)
