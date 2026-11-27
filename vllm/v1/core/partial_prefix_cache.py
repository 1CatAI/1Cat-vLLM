# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache entries that end inside a physical block.

Full-block prefix caching can resume a request only at multiples of the block
size. Hybrid models with recurrent (Mamba/GDN) layers also need the recurrent
state at the resume position, which ``align`` mode keeps only at block
boundaries. With large blocks a conversation turn then recomputes up to a whole
block of history, and in thinking mode the next turn diverges a few tokens
before the end of the previous prompt, so its last block boundary is the best
possible hit.

A partial entry stores, for one exact token position ``P``, cache-owned copies
of the blocks a request needs to resume at ``P`` that the ordinary prefix cache
does not hold:

* full attention: the partially filled block that contains ``P``;
* aligned Mamba: the block holding the recurrent state at ``P``;
* sliding window: every block of the window before ``P`` (interior window
  blocks are only cached at aligned boundaries).

It is keyed by the hash of the last full ``block_size`` block before ``P`` and
the tokens after it. The full blocks before ``P`` stay in the ordinary prefix
cache. A later request that matches them and the tail resumes at ``P``; the
block it continues writing is a private copy (copy-on-write), so entries are
never modified.

Entries hold one reference on each of their blocks and are released in LRU
order, or when the block pool runs short.

[本地 2026-11-27] Entries also expire by age (``ttl_seconds``, default 600s).
The intended life cycle is: a follow-up turn lands within the window, the entry
is hit once, and from then on the ordinary prefix cache owns that history --
the entry has no further value and must yield its blocks. Age is measured from
insertion and is not refreshed by hits, so a hot entry cannot pin blocks
indefinitely.
"""

import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock

# (parent full-block hash or None, tail tokens, cache salt, LoRA id, tail
# multimodal items)
PartialKey = tuple[
    BlockHash | None,
    tuple[int, ...],
    str | None,
    int,
    tuple[tuple[str, int], ...],
]


def tail_mm_items(
    mm_features: Sequence[Any] | None, start: int, end: int
) -> tuple[tuple[str, int], ...]:
    """``(identifier, offset)`` of the multimodal items overlapping ``[start, end)``.

    Placeholder tokens are the same for every image of a given size, so the
    tail tokens alone do not identify what a request's tail contains.
    """
    return tuple(
        (feature.identifier, feature.mm_position.offset - start)
        for feature in mm_features or ()
        if feature.mm_position.offset < end
        and start < feature.mm_position.offset + feature.mm_position.length
    )


@dataclass
class PartialEntry:
    key: PartialKey
    # Exact number of tokens the entry resumes.
    length: int
    # Cache-owned blocks per KV cache group, in block-table order.
    blocks: list[list[KVCacheBlock]]
    # Per group: whether the last block is partially filled and must be copied
    # before a request appends to it.
    copy_last: list[bool]
    # Scheduler step whose worker pass fills ``blocks``; usable after it.
    ready_step: int
    # Referenced full-block parents kept resident while this entry is cached.
    pinned_blocks: list[KVCacheBlock] = field(default_factory=list)
    # Number of prefix-cache lookups served by this entry.
    hit_count: int = 0
    # ``time.monotonic()`` at insertion; 0 means "stamp on insert".
    created_at: float = 0.0

    @property
    def owned_blocks(self) -> int:
        return sum(len(blocks) for blocks in self.blocks)

    @property
    def num_blocks(self) -> int:
        return self.owned_blocks + len(self.pinned_blocks)


@dataclass
class PartialPrefixCache:
    block_pool: BlockPool
    # Upper bound on the physical blocks held by entries.
    max_blocks: int
    # Entry lifetime from insertion, in seconds. <=0 disables expiry.
    ttl_seconds: float = 600.0
    entries: "OrderedDict[PartialKey, PartialEntry]" = field(
        default_factory=OrderedDict
    )
    by_parent: dict[BlockHash | None, set[PartialKey]] = field(default_factory=dict)
    pinned_ref_counts: dict[int, int] = field(default_factory=dict)
    num_blocks: int = 0

    def contains(self, key: PartialKey) -> bool:
        return key in self.entries

    def insert(self, entry: PartialEntry) -> None:
        if entry.key in self.entries:
            self._remove(entry.key)
        if entry.created_at <= 0.0:
            entry.created_at = time.monotonic()
        # Reclaim aged-out entries before adding a new one, so the new entry
        # competes with live entries only.
        self.evict_expired()
        self.entries[entry.key] = entry
        self.by_parent.setdefault(entry.key[0], set()).add(entry.key)
        self.num_blocks += entry.owned_blocks
        for blk in entry.pinned_blocks:
            prev = self.pinned_ref_counts.get(blk.block_id, 0)
            if prev == 0:
                self.num_blocks += 1
            self.pinned_ref_counts[blk.block_id] = prev + 1
        while self.num_blocks > self.max_blocks and len(self.entries) > 1:
            if not self.evict_one(protected=entry):
                break

    def _remove(self, key: PartialKey) -> None:
        entry = self.entries.pop(key)
        siblings = self.by_parent.get(key[0])
        if siblings is not None:
            siblings.discard(key)
            if not siblings:
                del self.by_parent[key[0]]
        self.num_blocks -= entry.owned_blocks
        for blk in entry.pinned_blocks:
            rem = self.pinned_ref_counts.get(blk.block_id, 1) - 1
            if rem <= 0:
                self.pinned_ref_counts.pop(blk.block_id, None)
                self.num_blocks -= 1
            else:
                self.pinned_ref_counts[blk.block_id] = rem
        self.block_pool.free_blocks(
            [b for blocks in entry.blocks for b in blocks] + list(entry.pinned_blocks)
        )

    def remove(self, key: PartialKey) -> bool:
        """Drop one entry and free its blocks; False if it is already gone."""
        if key not in self.entries:
            return False
        self._remove(key)
        return True

    def has_cold_evictable(self, protected: PartialEntry | None = None) -> bool:
        """True if at least one non-protected unhit entry can be evicted."""
        return any(
            e is not protected and e.hit_count == 0 for e in self.entries.values()
        )

    def _expired(self, entry: PartialEntry, now: float) -> bool:
        return self.ttl_seconds > 0 and now - entry.created_at >= self.ttl_seconds

    def evict_expired(self, now: float | None = None) -> int:
        """Drop every entry older than ``ttl_seconds``; return how many went."""
        if self.ttl_seconds <= 0 or not self.entries:
            return 0
        now = time.monotonic() if now is None else now
        stale = [k for k, e in self.entries.items() if self._expired(e, now)]
        for key in stale:
            self._remove(key)
        return len(stale)

    def evict_one(self, protected: PartialEntry | None = None) -> bool:
        """Release the least wanted entry (expired, then unhit LRU, then hit LRU).

        Never evicts ``protected`` (e.g. an entry already matched for the
        in-progress ``allocate_slots`` call).
        """
        if not self.entries:
            return False
        now = time.monotonic()
        for key, entry in self.entries.items():
            if entry is not protected and self._expired(entry, now):
                self._remove(key)
                return True
        for key, entry in self.entries.items():
            if entry is not protected and entry.hit_count == 0:
                self._remove(key)
                return True
        for key, entry in self.entries.items():
            if entry is not protected:
                self._remove(key)
                return True
        return False

    def lookup(
        self,
        parent: BlockHash | None,
        tokens: Sequence[int],
        block_start: int,
        max_length: int,
        cache_salt: str | None,
        lora_id: int,
        current_step: int,
        mm_features: Sequence[Any] | None = None,
    ) -> PartialEntry | None:
        """Longest ready entry under ``parent`` whose tail matches ``tokens``.

        The tail starts at ``block_start``, the start of the block after
        ``parent``.
        """
        # Aged-out entries are dead weight; drop them before matching so a
        # stale entry never shadows a live one.
        self.evict_expired()
        best: PartialEntry | None = None
        for key in tuple(self.by_parent.get(parent, ())):
            entry = self.entries.get(key)
            if entry is None:
                continue
            if (
                entry.ready_step >= current_step
                or entry.length > max_length
                or key[2] != cache_salt
                or key[3] != lora_id
                or (best is not None and entry.length <= best.length)
            ):
                continue
            tail = key[1]
            if tuple(tokens[block_start : block_start + len(tail)]) == tail and (
                key[4]
                == tail_mm_items(mm_features, block_start, block_start + len(tail))
            ):
                best = entry
        if best is not None:
            best.hit_count += 1
            self.entries.move_to_end(best.key)
        return best

    def clear(self) -> None:
        while self.evict_one():
            pass
