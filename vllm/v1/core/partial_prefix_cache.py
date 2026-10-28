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
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock

# (parent full-block hash or None, tail tokens, cache salt, LoRA id)
PartialKey = tuple[BlockHash | None, tuple[int, ...], str | None, int]


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

    @property
    def num_blocks(self) -> int:
        return sum(len(blocks) for blocks in self.blocks)


@dataclass
class PartialPrefixCache:
    block_pool: BlockPool
    # Upper bound on the blocks held by entries.
    max_blocks: int
    entries: "OrderedDict[PartialKey, PartialEntry]" = field(
        default_factory=OrderedDict
    )
    by_parent: dict[BlockHash | None, set[PartialKey]] = field(default_factory=dict)
    num_blocks: int = 0

    def contains(self, key: PartialKey) -> bool:
        return key in self.entries

    def insert(self, entry: PartialEntry) -> None:
        if entry.key in self.entries:
            self._remove(entry.key)
        self.entries[entry.key] = entry
        self.by_parent.setdefault(entry.key[0], set()).add(entry.key)
        self.num_blocks += entry.num_blocks
        while self.num_blocks > self.max_blocks and len(self.entries) > 1:
            self.evict_one()

    def _remove(self, key: PartialKey) -> None:
        entry = self.entries.pop(key)
        siblings = self.by_parent.get(key[0])
        if siblings is not None:
            siblings.discard(key)
            if not siblings:
                del self.by_parent[key[0]]
        self.num_blocks -= entry.num_blocks
        self.block_pool.free_blocks([b for blocks in entry.blocks for b in blocks])

    def evict_one(self) -> bool:
        """Release the least recently used entry. Returns False if empty."""
        if not self.entries:
            return False
        self._remove(next(iter(self.entries)))
        return True

    def lookup(
        self,
        parent: BlockHash | None,
        tokens: Sequence[int],
        block_start: int,
        max_length: int,
        cache_salt: str | None,
        lora_id: int,
        current_step: int,
    ) -> PartialEntry | None:
        """Longest ready entry under ``parent`` whose tail matches ``tokens``.

        The tail starts at ``block_start``, the start of the block after
        ``parent``.
        """
        best: PartialEntry | None = None
        for key in self.by_parent.get(parent, ()):
            entry = self.entries[key]
            if (
                entry.ready_step >= current_step
                or entry.length > max_length
                or key[2] != cache_salt
                or key[3] != lora_id
                or (best is not None and entry.length <= best.length)
            ):
                continue
            tail = key[1]
            if tuple(tokens[block_start : block_start + len(tail)]) == tail:
                best = entry
        if best is not None:
            self.entries.move_to_end(best.key)
        return best

    def clear(self) -> None:
        while self.evict_one():
            pass
