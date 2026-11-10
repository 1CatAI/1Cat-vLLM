# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from itertools import groupby
from typing import Any

from vllm.config.cache import KVRetentionConfig
from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_request_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class _PartitionedBlockQueue:
    """Keep state IDs in a small GPU range; history uses the remaining IDs.

    Blocks retain globally unique IDs and the ordinary hash/refcount lifetime.
    Each range has its own eviction order, so prefix checkpoints can be evicted
    without growing GPU state tensors to the host-history capacity.
    """

    def __init__(self, blocks, state_blocks, state_group_ids):
        self.state_blocks = state_blocks
        self.state_group_ids = frozenset(state_group_ids)
        self.state = FreeKVCacheBlockQueue(blocks[1:state_blocks])
        self.history = FreeKVCacheBlockQueue(blocks[state_blocks:])

    @property
    def num_free_blocks(self):
        return self.state.num_free_blocks + self.history.num_free_blocks

    def for_group(self, group_id):
        if group_id is None:
            raise ValueError("Partitioned cache allocation requires a group ID")
        return self.state if group_id in self.state_group_ids else self.history

    def popleft_n(self, n, group_id=None):
        return self.for_group(group_id).popleft_n(n)

    def remove(self, block):
        queue = self.state if block.block_id < self.state_blocks else self.history
        queue.remove(block)

    def append_n(self, blocks):
        self.state.append_n([b for b in blocks if b.block_id < self.state_blocks])
        self.history.append_n([b for b in blocks if b.block_id >= self.state_blocks])

    def prepend_n(self, blocks):
        self.state.prepend_n([b for b in blocks if b.block_id < self.state_blocks])
        self.history.prepend_n([b for b in blocks if b.block_id >= self.state_blocks])

    def get_all_free_blocks(self):
        return self.state.get_all_free_blocks() + self.history.get_all_free_blocks()


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
        use_lookahead_block_hashes: Whether block hashes also cover the token
            after each block; KV cache events then carry that token.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
        device_state_blocks: int = 0,
        device_state_group_ids: tuple[int, ...] = (),
        use_lookahead_block_hashes: bool = False,
        retention_policy: KVRetentionConfig | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        self.use_lookahead_block_hashes = use_lookahead_block_hashes
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        initial_queue = FreeKVCacheBlockQueue(self.blocks)
        self.free_block_queue: FreeKVCacheBlockQueue | _PartitionedBlockQueue = (
            initial_queue
        )

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = initial_queue.popleft()
        self.null_block.is_null = True

        if device_state_blocks:
            if not 1 < device_state_blocks < num_gpu_blocks:
                raise ValueError("State partition must leave history capacity")
            self.free_block_queue = _PartitionedBlockQueue(
                self.blocks, device_state_blocks, device_state_group_ids
            )

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

        policy = retention_policy or KVRetentionConfig()
        self._warm_enabled = policy.enabled
        self.warm_blocks: OrderedDict[int, KVCacheBlock] = OrderedDict()
        self._warm_freed_at: dict[int, float] = {}
        self._warm_hits: dict[BlockHashWithGroupId, int] = {}
        self._warm_grace_s = policy.grace_seconds
        self._device_state_blocks = device_state_blocks
        self._retention_tiers = list(
            zip(
                (0.25, 0.35, 0.50, 0.65, 0.80),
                (
                    policy.p25_seconds,
                    policy.p35_seconds,
                    policy.p50_seconds,
                    policy.p65_seconds,
                    policy.p80_seconds,
                ),
            )
        )
        # Block zero is null. In a partitioned pool it is already inside the
        # state partition, so the history count must not subtract it twice.
        history_blocks = num_gpu_blocks - max(device_state_blocks, 1)
        self._warm_soft_cap = int(history_blocks * policy.soft_cap)
        self._warm_hard_cap = int(history_blocks * policy.hard_cap)
        self._warm_abs_cap = int(history_blocks * policy.absolute_cap)

    def _warm_active(self) -> bool:
        """warm 机制是否可用（开关打开且确实处于缓存模式）。"""
        return self._warm_enabled and self.enable_caching

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            block_mask: Optional mask aligned with
                ``blocks[num_cached_blocks:num_full_blocks]``. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks). Used by groups whose ``find_longest_cache_hit`` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        event_block_indices: list[int] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if event_block_indices is not None:
                event_block_indices.append(num_cached_blocks + i)

        if event_block_indices:
            # A BlockStored describes one contiguous logical token range. Sparse
            # retention/null blocks can leave gaps; each run needs its own tokens
            # and logical parent hash, including when that parent is not resident.
            curr_mm_idx = 0
            for _, run in groupby(
                enumerate(event_block_indices), lambda item: item[1] - item[0]
            ):
                indices = [block_idx for _, block_idx in run]
                start, end = indices[0], indices[-1] + 1
                parent_block_hash = (
                    maybe_convert_block_hash(block_hashes[start - 1])
                    if start > 0
                    else None
                )
                extra_keys_list: list[tuple[Any, ...] | None] = []
                for block_idx in indices:
                    extra_keys, curr_mm_idx = generate_request_block_hash_extra_keys(
                        request,
                        block_idx * block_size,
                        (block_idx + 1) * block_size,
                        curr_mm_idx,
                        self.use_lookahead_block_hashes,
                    )
                    extra_keys_list.append(extra_keys)

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[
                            maybe_convert_block_hash(block_hashes[i]) for i in indices
                        ],
                        parent_block_hash=parent_block_hash,
                        token_ids=request.all_token_ids[
                            start * block_size : end * block_size
                        ],
                        block_size=block_size,
                        lora_id=request.lora_request.adapter_id
                        if request.lora_request
                        else None,
                        medium=MEDIUM_GPU,
                        lora_name=request.lora_request.name
                        if request.lora_request
                        else None,
                        extra_keys=extra_keys_list,
                        group_idx=kv_cache_group_id,
                    )
                )

    def get_new_blocks(
        self, num_blocks: int, kv_cache_group_id: int | None = None
    ) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks(kv_cache_group_id):
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        # [本地 2026-11-09 重放 P-05] 空闲队列不够时，先从 warm 表按
        # 「压力门控的保留时长」逐出最老的块（cold→hot）补进来。
        # `get_num_free_blocks` 已把 warm 计入总量，所以上面的预检通过并不代表
        # free 队列本身够用 —— 这里必须补足，否则 popleft_n 会拿到少于请求的块。
        if self._warm_active() and self.warm_blocks:
            _queue_free = (
                self.free_block_queue.for_group(kv_cache_group_id).num_free_blocks
                if isinstance(self.free_block_queue, _PartitionedBlockQueue)
                and kv_cache_group_id is not None
                else self.free_block_queue.num_free_blocks
            )
            _shortage = num_blocks - _queue_free
            if _shortage > 0:
                self._promote_warm_blocks(_shortage, kv_cache_group_id)

        ret: list[KVCacheBlock] = (
            self.free_block_queue.popleft_n(num_blocks, kv_cache_group_id)
            if isinstance(self.free_block_queue, _PartitionedBlockQueue)
            else self.free_block_queue.popleft_n(num_blocks)
        )

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Connector invalidation must detach warm entries as well. Promotion
        # and trimming remove them first and therefore enqueue exactly once.
        if self.warm_blocks.pop(block.block_id, None) is not None:
            assert block.ref_cnt == 0
            self._warm_freed_at.pop(block.block_id, None)
            self.free_block_queue.append_n([block])

        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        # The content is going away for good, so its reuse count would only
        # leak. Keyed exactly as `touch` records it, group id included.
        if self.cached_block_hash_to_block.get_one_block(block_hash) is None:
            self._warm_hits.pop(block_hash, None)
        block.reset_hash()

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                    group_idx=get_group_id(block_hash),
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # [本地 2026-11-09 重放 P-05] 命中 warm 块时把它从 warm 表摘掉。
            # 它不在 free 队列里（warm 块不是 free 候选），故不能走下面的
            # `free_block_queue.remove`。摘掉后它被正常引用；下次释放时会拿到
            # **新的**时间戳并排到队尾 —— 这就是「热块自然存活更久」的机制。
            _in_warm = False
            if self._warm_active() and block.ref_cnt == 0 and not block.is_null:
                _in_warm = self.warm_blocks.pop(block.block_id, None) is not None
                if _in_warm:
                    self._warm_freed_at.pop(block.block_id, None)
            # Count the reuse against the content, not the block: this is what
            # lets a repeatedly-requested prefix outlive a one-shot one in the
            # trim below.
            if self._warm_active() and not block.is_null and block.block_hash:
                self._warm_hits[block.block_hash] = (
                    self._warm_hits.get(block.block_hash, 0) + 1
                )
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null and not _in_warm:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(
        self, ordered_blocks: Iterable[KVCacheBlock], reuse_first: bool = False
    ) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
            reuse_first: Hand these blocks out again before any other free
                block, cached or not. A sliding window that has moved past its
                blocks releases them with this: their KV can never serve a
                future hit at any alignment boundary the window still covers,
                so keeping them cache-valuable would only let the running
                prefill rotate the whole pool and evict other requests.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        freed_blocks = [
            block for block in blocks_list if block.ref_cnt == 0 and not block.is_null
        ]
        if self.enable_caching:
            # Uncached scratch has no reusable prefix to evict, and blocks a
            # sliding window has moved past (``reuse_first``) can never serve a
            # future hit. Recycle both before consuming the queue's older
            # cached entries, so a long prefill does not rotate the whole pool
            # and evict other requests' prefixes.
            self.free_block_queue.prepend_n(
                [
                    block
                    for block in freed_blocks
                    if reuse_first or block.block_hash is None
                ]
            )
            # [本地 2026-11-09 重放 P-05] 带 hash 的块不再直接进 free 队列，
            # 而是进 warm 表：它们仍挂在 prefix-cache hash 表上，后续同前缀请求
            # 可以直接命中，省掉一次重算。等空闲块真的不够时再由
            # `_promote_warm_blocks` 逐出。
            #
            # `reuse_first`（滑动窗口已划过的块）**不进 warm** —— 它们在任何
            # 对齐边界上都不可能再命中，留着只会占位。
            if self._warm_active():
                _now = time.monotonic()
                _to_queue: list[KVCacheBlock] = []
                for block in freed_blocks:
                    if reuse_first or block.block_hash is None:
                        continue
                    if block.block_id < self._device_state_blocks:
                        _to_queue.append(block)
                        continue
                    self.warm_blocks[block.block_id] = block
                    self._warm_freed_at[block.block_id] = _now
                if len(self.warm_blocks) > self._warm_soft_cap:
                    _to_queue.extend(self._trim_warm(_now))
                if _to_queue:
                    self.free_block_queue.append_n(_to_queue)
            else:
                self.free_block_queue.append_n(
                    [
                        block
                        for block in freed_blocks
                        if not reuse_first and block.block_hash is not None
                    ]
                )
        else:
            self.free_block_queue.append_n(freed_blocks)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Reset invalidates all hashes together. Return every reclaimable
        # warm entry before clearing metadata; active blocks were rejected above.
        self.free_block_queue.append_n(list(self.warm_blocks.values()))
        self.warm_blocks.clear()
        self._warm_freed_at.clear()
        self._warm_hits.clear()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def _belongs_to_group(self, block_id: int, kv_cache_group_id: int | None) -> bool:
        """[本地 2026-11-09 重放 P-05] 该块是否属于指定 kv-cache 组。

        未分区（`device_state_blocks == 0`）时恒为 True —— G30 即此形态。
        分区时按 block_id 数值范围判定：state 组 = `block_id < device_state_blocks`。
        """
        if kv_cache_group_id is None or not self._device_state_blocks:
            return True
        if not isinstance(self.free_block_queue, _PartitionedBlockQueue):
            return True
        in_state = kv_cache_group_id in self.free_block_queue.state_group_ids
        return (block_id < self._device_state_blocks) == in_state

    @property
    def warm_block_count(self) -> int:
        """Blocks currently held in the warm table."""
        return len(self.warm_blocks)

    @property
    def warm_block_cap(self) -> int:
        """High watermark; only the absolute cap overrides the grace window."""
        return self._warm_hard_cap if self._warm_active() else 0

    @property
    def warm_block_soft_cap(self) -> int:
        """Target size; above it only expired entries are trimmed."""
        return self._warm_soft_cap

    def _trim_order(self, block_id: int) -> tuple[float, int]:
        """Eviction rank: release time first, hit count second.

        Time is the primary axis because age is what "nobody is using this
        anymore" means. Two entries from the same moment are then separated by
        how often their content was reused: within equally old entries, the
        one hit by every session outlives the one that was used once.

        Ascending on both keys, so the result reads least-wanted-first.
        """
        block_hash = self.warm_blocks[block_id].block_hash
        return (
            self._warm_freed_at.get(block_id, 0.0),
            self._warm_hits.get(block_hash, 0) if block_hash is not None else 0,
        )

    def _trim_warm(self, now: float) -> list[KVCacheBlock]:
        """Trim the warm table under two watermarks and two ranking signals.

        The warm table is not a cache of everything a session touched -- most
        of a long context is never read again. What earns a place is a record
        of reuse: the system prompt and tool preamble that every session hits,
        and the parts of earlier sessions that new requests actually re-read.
        Everything else should age out.

        Two signals rank an entry, time first and reuse second (see
        `_trim_order`):

          * release time -- content nobody has asked for in a while is not
            worth keeping, whatever else is true of it;
          * hit count, accumulated per block hash in `touch` -- among entries
            of comparable age, the content that keeps getting reused wins.

        Combining them this way keeps a frequently reused block ahead of a
        one-shot one from the same era, while still letting genuinely stale
        content age out even if it was popular once.

        A grace window overrides both: a block touched within
        `_warm_grace_s` is never trimmed. Without it a block used one step ago
        could lose to a colder but older-looking entry purely because it was
        freshly inserted, and the request that just used it would recompute.

        Watermarks:

          * below soft, nothing is trimmed;
          * above soft, entries past the retention tier and out of the grace
            window are dropped, coldest-and-least-reused first;
          * above hard, the retention tier stops gating, but the grace window
            still holds. Only if the table is *still* over hard once every
            remaining entry is recent does age alone decide -- size is a hard
            bound because `get_num_free_blocks` counts these blocks as usable.

        Returns:
            Blocks moved back to the free queue.
        """
        returned: list[KVCacheBlock] = []

        def _release(block_id: int) -> None:
            block = self.warm_blocks.pop(block_id)
            self._warm_freed_at.pop(block_id, None)
            self._maybe_evict_cached_block(block)
            returned.append(block)

        retention = self._get_retention_seconds()

        def _evictable(block_id: int) -> bool:
            """Recently touched blocks are off-limits.

            A block used moments ago must not lose to a cold one: the request
            that just referenced it would have to recompute the prefix, so a
            low hit count on a brand-new block is not evidence of low value.
            """
            freed_at = self._warm_freed_at.get(block_id, 0.0)
            return freed_at > 0.0 and (now - freed_at) >= self._warm_grace_s

        # Phase 1: drop entries that are both past the retention tier and out
        # of the grace window, stopping at the soft cap.
        if len(self.warm_blocks) > self._warm_soft_cap:
            surplus = len(self.warm_blocks) - self._warm_soft_cap
            candidates = [
                block_id
                for block_id in self.warm_blocks
                if _evictable(block_id)
                and (now - self._warm_freed_at.get(block_id, 0.0)) >= retention
            ]
            candidates.sort(key=self._trim_order)
            for block_id in candidates[:surplus]:
                _release(block_id)

        # Phase 2: still above the hard cap, so the retention tier stops
        # gating. The grace window still holds -- eviction here is about
        # making room, not about discarding what is in active use.
        if len(self.warm_blocks) > self._warm_hard_cap:
            excess = len(self.warm_blocks) - self._warm_soft_cap
            candidates = [b for b in self.warm_blocks if _evictable(b)]
            candidates.sort(key=self._trim_order)
            for block_id in candidates[:excess]:
                _release(block_id)

            # Entries inside the grace window stay past the hard cap -- that is
            # the point of the window. They do not get to stay forever: the
            # absolute cap below bounds how much of the pool retention may
            # hold, so a sustained burst cannot starve the free queue.
            if len(self.warm_blocks) > self._warm_abs_cap:
                victims = sorted(self.warm_blocks.keys(), key=self._trim_order)[
                    : len(self.warm_blocks) - self._warm_abs_cap
                ]
                for block_id in victims:
                    _release(block_id)

        if returned:
            logger.debug(
                "warm trim: released %d (now %d, soft=%d, hard=%d, "
                "retention=%.0fs, grace=%.0fs)",
                len(returned),
                len(self.warm_blocks),
                self._warm_soft_cap,
                self._warm_hard_cap,
                retention,
                self._warm_grace_s,
            )
        return returned

    def _get_retention_seconds(self) -> float:
        """[本地 2026-11-09 重放 P-05] 当前压力下 warm 块的保留时长（秒）。

        压力越低保留越久；≥80% 返回 0（立即 cold→hot 回收）。
        此压力口径包括已分配和 warm；`get_usage` 只包括已分配块。
        """
        total = self.num_gpu_blocks - 1  # 排除 null block
        free = self.free_block_queue.num_free_blocks
        warm = len(self.warm_blocks)
        allocated = max(total - free - warm, 0)
        usage = (allocated + warm) / max(total, 1)

        for threshold, retention in self._retention_tiers:
            if usage < threshold:
                return float(retention)
        return 0.0

    def _promote_warm_blocks(
        self, num_blocks: int, kv_cache_group_id: int | None = None
    ) -> int:
        """[本地 2026-11-09 重放 P-05] 把 warm 块降级回 free 队列。

        始终 oldest-first（cold→hot）。两阶段：
          Phase 1 —— 逐出**已超过**当前保留时长的块；
          Phase 2 —— 若仍不足，无论年龄继续从队头逐出（保证调用方拿到块）。

        频繁命中的块天然受保护：每次命中都会被 `touch()` 移出 warm，重新释放时
        拿到新时间戳并排到队尾，因此能熬过很多轮回收；冷块从队头先老死。

        逐出时一并清掉 hash（`_maybe_evict_cached_block`），此后该块不再可能
        被前缀命中 —— 这正是「降级」的含义。

        Returns:
            实际降级的块数。
        """
        if not self.warm_blocks:
            return 0
        now = time.monotonic()
        retention = self._get_retention_seconds()
        promoted = 0

        def _promote(block_id: int) -> None:
            block = self.warm_blocks.pop(block_id)
            self._warm_freed_at.pop(block_id, None)
            self._maybe_evict_cached_block(block)
            self.free_block_queue.append_n([block])

        # Phase 1：按 FIFO 序逐出已超时的块。
        expired: list[int] = []
        for block_id in self.warm_blocks:
            if promoted >= num_blocks:
                break
            if not self._belongs_to_group(block_id, kv_cache_group_id):
                continue
            freed_at = self._warm_freed_at.get(block_id, 0.0)
            if freed_at <= 0.0 or (now - freed_at) >= retention:
                expired.append(block_id)
                promoted += 1
        for block_id in expired:
            _promote(block_id)

        # Phase 2：仍不足 ⇒ 无视年龄继续从队头取。
        if promoted < num_blocks:
            for block_id in list(self.warm_blocks.keys()):
                if promoted >= num_blocks:
                    break
                if not self._belongs_to_group(block_id, kv_cache_group_id):
                    continue
                _promote(block_id)
                promoted += 1

        if promoted:
            logger.debug(
                "Promoted %d warm blocks (retention=%.0fs, %d remain).",
                promoted,
                retention,
                len(self.warm_blocks),
            )
        return promoted

    def get_num_free_blocks(self, kv_cache_group_id: int | None = None) -> int:
        """Get the number of free blocks in the pool.

        [本地 2026-11-09 重放 P-05] Includes warm blocks: they are freed (can be
        promoted back into the free queue on demand), so excluding them makes
        callers believe the pool is full while warm blocks sit unused — the
        scheduler then raises "Cannot get N free blocks" before
        `_promote_warm_blocks` ever runs, and the engine stalls with
        `waiting > 0` and an idle GPU. Both return paths must add them.

        Returns:
            The number of free blocks.
        """
        if kv_cache_group_id is not None and isinstance(
            self.free_block_queue, _PartitionedBlockQueue
        ):
            in_state = kv_cache_group_id in self.free_block_queue.state_group_ids
            # Only history blocks enter warm retention; group accounting is O(1).
            warm_n = 0 if in_state else len(self.warm_blocks)
            return (
                self.free_block_queue.for_group(kv_cache_group_id).num_free_blocks
                + warm_n
            )
        return self.free_block_queue.num_free_blocks + len(self.warm_blocks)

    def can_allocate_group_blocks(self, demand: dict[int, int]) -> bool:
        """Check partition quotas; callers still check aggregate free capacity."""
        if not isinstance(self.free_block_queue, _PartitionedBlockQueue):
            return True
        state = sum(
            n for i, n in demand.items() if i in self.free_block_queue.state_group_ids
        )
        history = sum(demand.values()) - state
        return (
            state <= self.free_block_queue.state.num_free_blocks
            and history
            <= self.free_block_queue.history.num_free_blocks + len(self.warm_blocks)
        )

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
