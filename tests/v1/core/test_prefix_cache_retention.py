# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse prefix-cache retention for Mamba / sliding-window groups.

Ported from upstream vLLM (#43447, #45845, #37898, #52216, #54713): a
retention interval decides which recurrent-state checkpoints and SWA tails
stay in the prefix cache, replay boundaries and Marconi-style shared-prefix
junctions are always kept, and unhashed blocks are reused before cached ones.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

from .test_prefix_caching import make_request

pytestmark = pytest.mark.cpu_test

BLOCK = 16


@pytest.fixture(autouse=True)
def _init_none_hash():
    kv_cache_utils.init_none_hash(sha256)


def _mamba_spec(block_size: int = BLOCK) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=(1, 1),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def _swa_spec(block_size: int, window: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=window,
    )


def _mamba_mask(end_block, retention_interval, boundaries=(), start_block=0):
    return MambaManager.reachable_block_mask(
        start_block=start_block,
        end_block=end_block,
        alignment_tokens=BLOCK,
        kv_cache_spec=_mamba_spec(),
        use_eagle=False,
        retention_interval=retention_interval,
        reachable_boundaries=boundaries,
    )


def test_mamba_mask_dense_when_retention_unset():
    assert _mamba_mask(10, None, (159,)) is None


def test_mamba_mask_semantic_keeps_only_reachable_boundaries():
    # 160-token prompt: the replay boundary 159 aligns to 144, i.e. the
    # state after block 8; the junction at token 64 is the state after block
    # 3. Everything else is dropped.
    mask = _mamba_mask(10, 0, (159, 64))
    assert mask == [i in (3, 8) for i in range(10)]
    # Windowed ranges keep the same absolute positions.
    assert _mamba_mask(10, 0, (159, 64), start_block=4) == [
        i in (3, 8) for i in range(4, 10)
    ]


def test_mamba_mask_periodic_adds_interval_boundaries():
    # Interval of 4 blocks: states at blocks 3, 7 plus the replay boundary
    # (block 8).
    mask = _mamba_mask(10, 4 * BLOCK, (159,))
    assert mask == [i in (3, 7, 8) for i in range(10)]
    # An interval at or below the block size means every block.
    assert _mamba_mask(10, BLOCK, (159,)) is None


def test_swa_mask_semantic_keeps_replay_tail_only():
    # window 32 = 2 blocks per tail; alignment 64 = 4 blocks per segment.
    spec = _swa_spec(BLOCK, 32)
    dense = SlidingWindowManager.reachable_block_mask(
        0, 8, 64, spec, use_eagle=False, retention_interval=None
    )
    assert dense == [i % 4 >= 2 for i in range(8)]
    sparse = SlidingWindowManager.reachable_block_mask(
        0,
        8,
        64,
        spec,
        use_eagle=False,
        retention_interval=0,
        reachable_boundaries=(127,),
    )
    # Replay boundary 127 -> aligned 64 -> tail is blocks 2, 3.
    assert sparse == [i in (2, 3) for i in range(8)]
    with_eagle = SlidingWindowManager.reachable_block_mask(
        0,
        8,
        64,
        spec,
        use_eagle=True,
        retention_interval=0,
        reachable_boundaries=(127,),
    )
    # EAGLE peeks one block past the boundary: three blocks 2, 3, 4.
    assert with_eagle == [i in (2, 3, 4) for i in range(8)]


class _Boundaries:
    get_replay_boundaries = KVCacheCoordinator.get_replay_boundaries
    reachable_boundaries = KVCacheCoordinator.reachable_boundaries

    def __init__(self, eagle_group_ids):
        self.eagle_group_ids = eagle_group_ids


def test_replay_boundaries_with_and_without_eagle():
    coordinator = _Boundaries(set())
    request = SimpleNamespace(num_prompt_tokens=100, shared_prefix_boundary=0)
    assert KVCacheCoordinator.get_replay_boundaries(coordinator, request, BLOCK) == (
        99,
    )
    coordinator = _Boundaries({0})
    # Unaligned prompt: resend and extension both resume one block lower.
    assert KVCacheCoordinator.get_replay_boundaries(coordinator, request, BLOCK) == (
        80,
    )
    # Block-aligned prompt: the resend is capped at n - 1 -> two positions.
    request = SimpleNamespace(num_prompt_tokens=96, shared_prefix_boundary=0)
    assert KVCacheCoordinator.get_replay_boundaries(coordinator, request, BLOCK) == (
        64,
        80,
    )
    request.shared_prefix_boundary = 32
    assert KVCacheCoordinator.reachable_boundaries(coordinator, request, BLOCK) == (
        64,
        80,
        32,
    )


def test_free_queue_prepend_reuses_uncached_blocks_first():
    pool = BlockPool(num_gpu_blocks=6, enable_caching=True, hash_block_size=BLOCK)
    blocks = pool.get_new_blocks(4)
    pool.free_blocks(blocks[:2])
    pool.free_blocks(blocks[2:], prepend=True)
    order = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]
    # Prepended blocks come first, then the untouched tail, then appended.
    assert order[:2] == [blocks[2].block_id, blocks[3].block_id]
    assert order[-2:] == [blocks[0].block_id, blocks[1].block_id]
    # Block 0 is the reserved null block.
    assert pool.free_block_queue.num_free_blocks == 5


def _hybrid_config(retention_interval, num_blocks=64) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(["mamba"], _mamba_spec()),
        ],
        prefix_cache_retention_interval=retention_interval,
    )


def _cached_mamba_blocks(manager, request, num_blocks) -> list[int]:
    """Indices of the request's blocks whose Mamba state is prefix-cached."""
    return [
        idx
        for idx in range(num_blocks)
        if manager.block_pool.get_cached_block(request.block_hashes[idx], [1])
        is not None
    ]


def _prefill(manager, request, chunk_ends):
    """Run a chunked prefill through allocate_slots, caching full blocks."""
    computed_blocks, num_computed, _ = manager.get_computed_blocks(request)
    position = num_computed
    request.num_computed_tokens = num_computed
    for end in chunk_ends:
        if end <= position:
            continue
        blocks = manager.allocate_slots(
            request,
            end - position,
            num_computed,
            computed_blocks,
        )
        assert blocks is not None
        computed_blocks, num_computed = None, 0
        position = end
        request.num_computed_tokens = end


def test_semantic_retention_keeps_replay_boundary_and_pins_junction():
    manager = KVCacheManager(
        _hybrid_config(retention_interval=0),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK,
    )
    assert isinstance(manager.coordinator.single_type_managers[1], MambaManager)
    prefix = [i for i in range(6) for _ in range(BLOCK)]

    # Request A: 8 blocks + 5 tokens, prefilled in one chunk to its replay
    # boundary (block 8) and then the tail, as the scheduler split would do.
    tokens_a = prefix + [7] * (2 * BLOCK) + [8] * 5
    req_a = make_request("a", tokens_a, BLOCK, sha256)
    _prefill(manager, req_a, [8 * BLOCK, len(tokens_a)])
    # Only the replay boundary state (block 7, ending at token 128) is kept
    # in the Mamba group's prefix cache; earlier boundary states were never
    # hashed. (Align mode nulls retired positions in req_to_blocks, so the
    # hash map is the oracle.)
    assert _cached_mamba_blocks(manager, req_a, 8) == [7]
    manager.free(req_a)

    # Request B shares the first 6 blocks with A and then diverges. Full
    # attention hits 6 blocks, the Mamba group has no state there, so the hit
    # is 0 and the uncached shared prefix (96 tokens) becomes the junction.
    tokens_b = prefix + [9] * (3 * BLOCK) + [10] * 3
    req_b = make_request("b", tokens_b, BLOCK, sha256)
    _, num_computed, junction = manager.get_computed_blocks(req_b)
    assert num_computed == 0
    assert junction == 6 * BLOCK
    req_b.shared_prefix_boundary = junction
    # The scheduler ends a chunk at the junction so its state materializes,
    # then continues to the replay boundary and the tail.
    _prefill(manager, req_b, [junction, 9 * BLOCK, len(tokens_b)])
    # The junction state (after 96 tokens, block 5) and B's own replay
    # boundary (block 8) are cached; nothing in between.
    assert _cached_mamba_blocks(manager, req_b, 9) == [5, 8]
    manager.free(req_b)

    # Request C shares the same 6-block prefix: now both groups hit it.
    tokens_c = prefix + [11] * (2 * BLOCK)
    req_c = make_request("c", tokens_c, BLOCK, sha256)
    _, num_computed, junction = manager.get_computed_blocks(req_c)
    assert num_computed == 6 * BLOCK
    assert junction == 0


def test_dense_retention_still_caches_every_boundary():
    manager = KVCacheManager(
        _hybrid_config(retention_interval=None),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK,
    )
    tokens = [i for i in range(4) for _ in range(BLOCK)] + [5] * 3
    request = make_request("d", tokens, BLOCK, sha256)
    # Dense retention: the scheduler ends a chunk at every block boundary.
    _prefill(manager, request, [BLOCK, 2 * BLOCK, 3 * BLOCK, 4 * BLOCK, len(tokens)])
    assert _cached_mamba_blocks(manager, request, 4) == [0, 1, 2, 3]


def test_retention_interval_must_match_alignment():
    with pytest.raises(ValueError, match="multiple of the cache-hit alignment"):
        KVCacheManager(
            _hybrid_config(retention_interval=BLOCK + 1),
            max_model_len=8192,
            enable_caching=True,
            hash_block_size=BLOCK,
        )
