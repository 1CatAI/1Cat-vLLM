# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

import vllm.envs as envs
from vllm.distributed.kv_events import BlockStored, KVCacheEvent
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize, KVCacheBlock
from vllm.v1.core.partial_prefix_cache import PartialEntry, PartialPrefixCache
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


_V41_SESSION_PREFIX = envs.VLLM_V41_SESSION_PREFIX == "1"
# A hit at P needs window rows [P - 128, P) still in the 256-row rings.
_V41_RING_REACH = 128


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        prefix_cache_retention_interval: int | None = None,
        partial_prefix_cache: bool = False,
    ) -> None:
        self.max_model_len = max_model_len
        # When unset, fall back to `max_model_len` so the recycling-aware cap
        # collapses to the prior (uncapped) admission behavior. The scheduler
        # always supplies the real value at runtime.
        if max_in_flight_tokens is None:
            max_in_flight_tokens = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.coordinator.configure_prefix_cache_retention(
            prefix_cache_retention_interval
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config
        self._step = 0
        self._block_copies: list[tuple[int, int]] = []
        # (release step, blocks): copy endpoints kept until the copy has run.
        self._deferred_release: list[tuple[int, list[KVCacheBlock]]] = []
        # (request, boundary, num_preemptions) whose step ends at a boundary.
        self._pending_snapshots: list[tuple[Request, int, int]] = []
        self.partial_cache: PartialPrefixCache | None = None
        if partial_prefix_cache and enable_caching:
            self._init_partial_prefix_cache(hash_block_size)
        self.kv_cache_event_metadata = tuple(
            (
                get_kv_cache_spec_kind(group.kv_cache_spec).value,
                get_kv_cache_spec_sliding_window(group.kv_cache_spec),
            )
            for group in kv_cache_config.kv_cache_groups
        )

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    # V4.1 window rings hold only the last executed sequence's recent rows.
    _v41_last: tuple[list[int], int] | None = None

    def _v41_session_hit(self, request: Request, hit: int) -> bool:
        last = self._v41_last
        if last is None or hit % 2:
            return False
        tokens, processed = last
        if not processed - _V41_RING_REACH <= hit <= processed:
            return False
        return list(request.all_token_ids)[:hit] == tokens[:hit]

    def _init_partial_prefix_cache(self, hash_block_size: int) -> None:
        managers = self.coordinator.single_type_managers
        supported = any(isinstance(m, MambaManager) for m in managers) and all(
            type(m) in (FullAttentionManager, SlidingWindowManager)
            or (isinstance(m, MambaManager) and m.mamba_cache_mode == "align")
            for m in managers
        )
        if not supported:
            logger.warning(
                "Partial prefix cache supports full-attention, sliding-window "
                "and aligned Mamba groups; keeping block-aligned hits "
                "(groups: %s).",
                [type(m).__name__ for m in managers],
            )
            return
        self.hash_block_size = hash_block_size
        # Entries are keyed under the last block boundary shared by all groups.
        self.partial_block_size = max(m.block_size for m in managers)
        if any(self.partial_block_size % m.block_size for m in managers):
            logger.warning("Partial prefix cache needs nested block sizes.")
            return
        self.partial_cache = PartialPrefixCache(
            block_pool=self.block_pool,
            # Allocation reclaims entries under pressure; the bound keeps a
            # burst of new turns from displacing the full-block cache.
            max_blocks=max(2 * len(managers), self.block_pool.num_gpu_blocks // 4),
        )
        logger.info(
            "Partial prefix cache enabled: %d-token boundary blocks, %d groups, "
            "up to %d entry blocks.",
            self.partial_block_size,
            len(managers),
            self.partial_cache.max_blocks,
        )

    def _partial_parent(self, request: Request, boundary: int):
        """(parent hash, start of the tail) for a boundary."""
        block_start = boundary // self.partial_block_size * self.partial_block_size
        index = block_start // self.hash_block_size - 1
        return (request.block_hashes[index] if index >= 0 else None), block_start

    def _partial_key(self, request: Request, boundary: int):
        parent, block_start = self._partial_parent(request, boundary)
        lora_id = request.lora_request.lora_int_id if request.lora_request else 0
        return (
            parent,
            tuple(request.all_token_ids[block_start:boundary]),
            request.cache_salt,
            lora_id,
        )

    def _partial_block_range(self, manager, boundary: int) -> tuple[int, int]:
        """Block indices ``[first, last]`` an entry keeps for one group."""
        block_size = manager.block_size
        last = (boundary - 1) // block_size
        if isinstance(manager, SlidingWindowManager):
            first = max(0, boundary - (manager.sliding_window - 1)) // block_size
            return min(first, last), last
        return last, last

    def partial_prefix_stops(
        self, request: Request, start: int, end: int
    ) -> tuple[int, ...]:
        """Boundaries inside ``(start, end)`` whose state must be kept."""
        if self.partial_cache is None or not request.prefix_cache_boundaries:
            return ()
        hashed = len(request.block_hashes) * self.hash_block_size
        return tuple(
            b
            for b in request.prefix_cache_boundaries
            if start < b < end
            and b % self.partial_block_size
            and b // self.partial_block_size * self.partial_block_size <= hashed
            and not self.partial_cache.contains(self._partial_key(request, b))
        )

    def _partial_prefix_hit(
        self, request: Request, hit_length: int, max_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int] | None:
        cache = self.partial_cache
        if cache is None or not cache.entries:
            return None
        hashes = request.block_hashes
        lora_id = request.lora_request.lora_int_id if request.lora_request else 0
        tokens = request.all_token_ids
        ratio = self.partial_block_size // self.hash_block_size
        entry = None
        for num_blocks in range(
            min(len(hashes) // ratio, max_length // self.partial_block_size),
            hit_length // self.partial_block_size - 1,
            -1,
        ):
            parent = hashes[num_blocks * ratio - 1] if num_blocks else None
            if parent not in cache.by_parent:
                continue
            entry = cache.lookup(
                parent,
                tokens,
                num_blocks * self.partial_block_size,
                max_length,
                request.cache_salt,
                lora_id,
                self._step,
            )
            if entry is not None and entry.length > hit_length:
                break
            entry = None
        if entry is None:
            return None
        boundary = entry.length
        blocks: list[list[KVCacheBlock]] = []
        for group_id, manager in enumerate(self.coordinator.single_type_managers):
            first, _ = self._partial_block_range(manager, boundary)
            if isinstance(manager, FullAttentionManager):
                group_hashes = (
                    hashes
                    if manager.block_size == self.hash_block_size
                    else BlockHashListWithBlockSize(
                        hashes, self.hash_block_size, manager.block_size
                    )
                )
                full = []
                for i in range(first):
                    cached = self.block_pool.get_cached_block(
                        group_hashes[i], [group_id]
                    )
                    if cached is None:
                        return None
                    full.append(cached[0])
                blocks.append(full + entry.blocks[group_id])
            else:
                blocks.append(
                    [self.block_pool.null_block] * first + entry.blocks[group_id]
                )
        demand = {group_id: 1 for group_id, copy in enumerate(entry.copy_last) if copy}
        if not self.block_pool.can_allocate_group_blocks(demand):
            return None
        request.partial_prefix_entry = entry
        return tuple(blocks), boundary

    def _defer_release(self, blocks: list[KVCacheBlock]) -> None:
        # Async scheduling runs up to two steps ahead of the worker; the copy
        # emitted in this step has executed once three more steps started.
        self._deferred_release.append((self._step + 3, blocks))

    def _redirect_partial_hit(self, request: Request, entry: PartialEntry) -> None:
        """Copy-on-write: continue the request on private copies."""
        sources = []
        for group_id, manager in enumerate(self.coordinator.single_type_managers):
            req_blocks = manager.req_to_blocks[request.request_id]
            last = len(req_blocks) - 1
            assert req_blocks[last] is entry.blocks[group_id][-1]
            # Blocks before the last one are full and only read.
            manager.num_cached_block[request.request_id] = (
                last if entry.copy_last[group_id] else last + 1
            )
            if not entry.copy_last[group_id]:
                continue
            source = req_blocks[last]
            target = self.block_pool.get_new_blocks(1, group_id)[0]
            req_blocks[last] = target
            self._block_copies.append((source.block_id, target.block_id))
            sources.append(source)
        # Keep the reference taken by the cache hit until the copy has run.
        if sources:
            self._defer_release(sources)

    def _materialize_snapshots(self) -> None:
        cache = self.partial_cache
        assert cache is not None
        pending, self._pending_snapshots = self._pending_snapshots, []
        managers = self.coordinator.single_type_managers
        for request, boundary, num_preemptions in pending:
            if request.num_preemptions != num_preemptions or request.is_finished():
                continue
            key = self._partial_key(request, boundary)
            if cache.contains(key):
                continue
            sources: list[list[KVCacheBlock]] = []
            for manager in managers:
                req_blocks = manager.req_to_blocks.get(request.request_id)
                first, last = self._partial_block_range(manager, boundary)
                if not req_blocks or len(req_blocks) <= last:
                    break
                group_sources = req_blocks[first : last + 1]
                if any(block.is_null for block in group_sources):
                    break
                sources.append(group_sources)
            if len(sources) != len(managers):
                continue
            demand = {g: len(blocks) for g, blocks in enumerate(sources)}
            while not self.block_pool.can_allocate_group_blocks(demand):
                if not cache.evict_one():
                    break
            if not self.block_pool.can_allocate_group_blocks(demand):
                continue
            targets: list[list[KVCacheBlock]] = []
            for group_id, group_sources in enumerate(sources):
                group_targets = self.block_pool.get_new_blocks(
                    len(group_sources), group_id
                )
                for source, target in zip(group_sources, group_targets):
                    self._block_copies.append((source.block_id, target.block_id))
                targets.append(group_targets)
            flat = [block for blocks in sources for block in blocks]
            # The request may release these blocks this step.
            self.block_pool.touch(flat)
            self._defer_release(flat)
            cache.insert(
                PartialEntry(
                    key=key,
                    length=boundary,
                    blocks=targets,
                    copy_last=[boundary % m.block_size != 0 for m in managers],
                    ready_step=self._step,
                )
            )

    def _reclaim_partial_blocks(self, num_blocks: int) -> bool:
        """Whether ``num_blocks`` are free, releasing partial entries if needed.

        Partial entries only save recomputation, so running requests take
        precedence over them.
        """
        if num_blocks <= self.block_pool.get_num_free_blocks():
            return True
        cache = self.partial_cache
        while cache is not None and cache.evict_one():
            if num_blocks <= self.block_pool.get_num_free_blocks():
                return True
        return False

    def take_block_copies(self) -> list[tuple[int, int]]:
        copies, self._block_copies = self._block_copies, []
        return copies

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        request.shared_prefix_boundary = 0
        request.partial_prefix_entry = None
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        # Keep the existing two-value lookup API used by 1Cat connectors. The
        # synchronous coordinator exposes the junction from this lookup only.
        request.shared_prefix_boundary = self.coordinator.shared_prefix_boundary
        if (
            _V41_SESSION_PREFIX
            and num_new_computed_tokens
            and not self._v41_session_hit(request, num_new_computed_tokens)
        ):
            request.shared_prefix_boundary = 0
            return self.empty_kv_cache_blocks, 0

        partial = self._partial_prefix_hit(
            request, num_new_computed_tokens, max_cache_hit_length
        )
        if partial is not None:
            computed_blocks, num_new_computed_tokens = partial

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.
            full_sequence_must_fit: Only allocate blocks if the KV cache has enough
                free blocks to hold the full sequence, accounting for prefix cache hits
                and sliding window. Used as an admission gate to prevent over-admitting
                requests when chunked prefill would otherwise only check the first chunk

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)

            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
            )
            if not self._reclaim_partial_blocks(num_blocks_to_allocate):
                return None

        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        # Free on the processed-token basis: in-flight steps' attention windows
        # still read blocks below the optimistic boundary, and rejected spec
        # tokens can roll it back.
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )
        partial_entry = (
            request.partial_prefix_entry
            if new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            else None
        )
        if partial_entry is not None:
            # Private copy-on-write blocks.
            num_blocks_to_allocate += sum(partial_entry.copy_last)

        if not self._reclaim_partial_blocks(num_blocks_to_allocate):
            # Cannot allocate new blocks
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )
            if partial_entry is not None:
                self._redirect_partial_hit(request, partial_entry)
                request.partial_prefix_entry = None

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens
            if request.cache_output_tokens
            else request.num_prompt_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        if self.partial_cache is not None and request.prefix_cache_boundaries:
            step_end = total_computed_tokens + num_new_tokens
            if (
                step_end in request.prefix_cache_boundaries
                and step_end % self.partial_block_size
                and step_end < request.num_prompt_tokens
            ):
                # The state at ``step_end`` exists once this step has run;
                # snapshot it before the next step advances it.
                self._pending_snapshots.append(
                    (request, step_end, request.num_preemptions)
                )

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        if _V41_SESSION_PREFIX:
            self._v41_last = (list(request.all_token_ids), request.num_computed_tokens)
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for prefix-anchored SWA
                gap eviction.
        """
        self.coordinator.remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if self.partial_cache is not None:
            self.partial_cache.clear()
            for _, blocks in self._deferred_release:
                self.block_pool.free_blocks(blocks)
            self._deferred_release = []
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        events = self.block_pool.take_events()
        for event in events:
            if not isinstance(event, BlockStored):
                continue
            if event.group_idx is None:
                continue
            if event.group_idx < 0 or event.group_idx >= len(
                self.kv_cache_event_metadata
            ):
                logger.warning(
                    "Group index `%s` not in KV cache metadata", event.group_idx
                )
                continue
            # Annotate here so BlockPool can keep emitting structural cache
            # events without owning semantic KV cache spec metadata.
            kind, sliding_window = self.kv_cache_event_metadata[event.group_idx]
            event.kv_cache_spec_kind = kind
            event.kv_cache_spec_sliding_window = sliding_window
        return events

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def get_block_ids_for_computed_tokens(
        self,
        request_id: str,
        num_computed_tokens: int,
    ) -> tuple[list[int], ...]:
        """Get block ids covering the request's computed tokens."""
        block_ids = self.get_block_ids(request_id)
        clipped_block_ids: list[list[int]] = []
        for group, ids in zip(self.kv_cache_config.kv_cache_groups, block_ids):
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec) or isinstance(
                spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            ):
                clipped_block_ids.append(ids)
                continue

            num_valid_blocks = cdiv(num_computed_tokens, spec.block_size)
            clipped_block_ids.append(ids[:num_valid_blocks])
        return tuple(clipped_block_ids)

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    def take_boundary_state_offloads(
        self,
    ) -> dict[str, list[tuple[int, int, int]]]:
        """Drain exact Mamba ``align`` state blocks for KV connectors.

        Returns ``{request_id: [(group_id, block_id, boundary_tokens), ...]}``.
        These scheduler-local handoffs avoid reconstructing mutable Mamba block
        tables from a connector's append-only block-ID mirror.
        """
        offloads: dict[str, list[tuple[int, int, int]]] = {}
        for mgr in self.coordinator.single_type_managers:
            for (
                request_id,
                group_id,
                block,
                boundary_tokens,
            ) in mgr.take_pending_boundary_state_offloads():
                offloads.setdefault(request_id, []).append(
                    (group_id, block.block_id, boundary_tokens)
                )
        return offloads

    def new_step_starts(self) -> None:
        """Called when a new step is started."""
        self._step += 1
        if self._deferred_release:
            keep = []
            for release_step, blocks in self._deferred_release:
                if release_step <= self._step:
                    self.block_pool.free_blocks(blocks)
                else:
                    keep.append((release_step, blocks))
            self._deferred_release = keep
        if self._pending_snapshots:
            self._materialize_snapshots()
        self.coordinator.new_step_starts()
