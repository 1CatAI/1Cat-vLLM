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

import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock

logger = init_logger(__name__)


def _partial_value_evict_enabled() -> bool:
    """[本地 2026-11-11 P-06] Pick the sacrifice by value instead of LRU.

    Default off so P5 can be rolled back with a single env var. Callers must
    also check that the block pool's value metrics are live
    (``_tier_enabled``): with tiers disabled ``hits_recent`` never advances
    and every value degenerates to zero.
    """
    return (
        os.environ.get("VLLM_PARTIAL_VALUE_EVICT", "0").strip().lower()
        not in ("", "0", "false", "no", "off")
    )


def _partial_max_frac() -> float:
    """[本地 2026-11-11] partial entry 预算占总池的比例。

    ``KVCacheManager`` 用它算 ``max_blocks``；提出来是为了让热重载
    (``reload_partial_settings``) 能重算同一个值，不必复制解析逻辑。

    默认 0.05（原为硬编码的 0.25）。错值回落到默认而不是抛异常 ——
    一个手滑的 env 不该让引擎起不来。
    """
    raw = os.environ.get("VLLM_PARTIAL_MAX_FRAC", "0.05")
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Ignoring malformed VLLM_PARTIAL_MAX_FRAC=%r; keeping 0.05", raw
        )
        return 0.05


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
    # --- [本地 2026-11-11 P-06] 价值度量 ---
    # Cumulative resume count and a sliding-window variant (decayed lazily by
    # PartialPrefixCache._decay). Used to pick which entry to sacrifice.
    hits: int = 0
    created_ts: float = 0.0
    hits_recent: float = 0.0

    @property
    def num_blocks(self) -> int:
        return sum(len(blocks) for blocks in self.blocks)

    def gain_tokens(self, block_size: int) -> int:
        """Tokens one hit saves beyond what the ordinary prefix cache covers.

        When ``length`` sits exactly on a block boundary the full-block cache
        already resumes there, so this entry's marginal value is zero.
        """
        return self.length % block_size

    def value(self, block_size: int) -> float:
        """Sacrifice ranking key: total saved tokens, NOT per-block.

        [本地 2026-11-11 P-06] Deliberately not divided by ``num_blocks``.
        Entries are released whole, so the cost of sacrificing one is its
        total value; dividing would bias toward entries that happen to span
        more groups. Cross-layer comparison against ordinary blocks was
        considered and rejected: ``get_num_free_blocks()`` already counts
        every evictable ordinary block, so ordinary blocks are not an
        alternative source for the shortfall.
        """
        return self.hits_recent * self.gain_tokens(block_size)

    def is_settling(self, now: float, grace_s: float) -> bool:
        """Still inside the window where it has not had a chance to be hit.

        An entry is born with ``hits_recent == 0`` and only earns value when a
        later request resumes from it. Ranking purely by value therefore hands
        every newcomer a score of zero, so the next insert evicts it before it
        can ever prove itself -- the cache would keep only entries that were
        already hit and slowly starve. Ordinary blocks do not have this
        problem because ``_classify`` gives a fresh block an observation
        period; this is the partial-cache equivalent.

        Mirrors the "leaf protection" idea in FlashStates: do not let a
        just-created node lose to a node that simply happens to have a history.
        """
        if self.created_ts <= 0.0:
            return False
        return (now - self.created_ts) < grace_s


@dataclass
class PartialPrefixCache:
    block_pool: BlockPool
    # Upper bound on the blocks held by entries.
    max_blocks: int
    # [本地 2026-11-11] KV cache group 数。热重载重算 ``max_blocks`` 时要用
    # 同一个 ``2 × groups`` 下限，否则下调预算会把某个 group 挤到放不下
    # 一个 entry。构造时记下，避免运行期再去反查 manager 列表。
    _num_groups: int = 1
    # [本地 2026-11-11 P-06] Block size used by ``gain_tokens`` / ``value``.
    # Comes from KVCacheManager.partial_block_size at construction.
    block_size: int = 1
    entries: "OrderedDict[PartialKey, PartialEntry]" = field(
        default_factory=OrderedDict
    )
    by_parent: dict[BlockHash | None, set[PartialKey]] = field(default_factory=dict)
    num_blocks: int = 0
    # [本地 2026-11-11 P-06] Lazy decay bookkeeping for hits_recent.
    _last_decay_ts: float = 0.0
    _value_window_s: float = 3600.0
    # [本地 2026-11-11 P0] Diagnostic throttle. 0.0 ⇒ first call logs
    # immediately, which is what we want for the startup baseline.
    _next_diag_ts: float = 0.0
    _diag_s: float = 120.0
    # [本地 2026-11-11 P-06b] A fresh entry has hits_recent == 0, so ranking
    # it by value evicts it before any request can resume from it. Entries
    # younger than this are not sacrificial. Sized well above a typical agent
    # turn gap: too short and newcomers still starve, too long and a burst of
    # worthless entries pins the cache.
    #
    # ``None`` means "not chosen by the caller": __post_init__ then resolves
    # it from ``VLLM_PARTIAL_GRACE_S``. Keeping the 300.0 literal here instead
    # would make the env var registered in envs.py silently inert, which is
    # exactly the bug this sentinel avoids.
    _grace_s: float | None = None
    # Upper bound on the blocks settling entries may hold collectively. Half
    # the cache: enough for a normal burst of new turns, not enough for a
    # stream of worthless entries to evict a hot one. Computed in
    # __post_init__ because it depends on max_blocks.
    _grace_budget_blocks: int = 0
    # --- [本地 2026-11-11 观测 P1] 依赖失效计量 ---
    # ``lookup`` 找到 entry 之后，还要 boundary 之前的普通缓存整块全都还在
    # 才能真正恢复。那些整块被驱逐时 entry 就永久变废，但它仍占着 blocks。
    # 这里累计「找到了 entry 却因为整块缺失而放弃」的次数，用来量化这个缺口。
    # 纯只读计数，不参与任何决策。
    dep_miss_count: int = 0
    # 上一次输出诊断时的累计值，用差值算周期增量。
    _dep_miss_seen: int = 0
    # 上一次真正输出诊断的时刻，用来算 dep_miss 速率（0 = 尚未输出过）。
    _last_diag_ts: float = 0.0

    def __post_init__(self) -> None:
        self._grace_budget_blocks = max(1, self.max_blocks // 2)
        if self._grace_s is None:
            try:
                self._grace_s = float(
                    os.environ.get("VLLM_PARTIAL_GRACE_S", "300").strip()
                )
            except ValueError:
                logger.warning(
                    "Ignoring malformed VLLM_PARTIAL_GRACE_S; using 300s"
                )
                self._grace_s = 300.0

    def contains(self, key: PartialKey) -> bool:
        return key in self.entries

    def insert(self, entry: PartialEntry) -> None:
        if entry.key in self.entries:
            self._remove(entry.key)
        self.entries[entry.key] = entry
        self.by_parent.setdefault(entry.key[0], set()).add(entry.key)
        self.num_blocks += entry.num_blocks
        self.enforce_budget()

    def enforce_budget(self) -> int:
        """[本地 2026-11-11] 把占用压回 ``max_blocks``，返回淘汰的 entry 数。

        抽出来是因为有三个调用点：insert 之后、``_reclaim_partial_blocks``
        里、以及热重载调小预算之后。最后那个必须立刻收敛 —— 否则一个下调
        的预算要等到下一次插入才被遵守，运维会以为没生效。

        [本地 2026-11-11 P-06] Sacrifice the least valuable entry rather
        than the oldest. Ranking happens inside the cache only; ordinary
        blocks are never an alternative because ``get_num_free_blocks()``
        already counts every evictable one.
        """
        evicted = 0
        if _partial_value_evict_enabled():
            while self.num_blocks > self.max_blocks and len(self.entries) > 1:
                victim = self.lowest_value_entry()
                if victim is None:
                    break
                self._remove(victim.key)
                evicted += 1
        else:
            while self.num_blocks > self.max_blocks and len(self.entries) > 1:
                self.evict_one()
                evicted += 1
        return evicted

    def _decay(self, now: float) -> None:
        """Lazily decay every entry's ``hits_recent``.

        [本地 2026-11-11 P-06] Driven from here rather than from the block
        pool's sweep: ``PartialPrefixCache`` is owned by ``KVCacheManager``,
        which owns ``BlockPool``, so the pool cannot reach this object without
        a cycle. Decaying on the read path keeps the comparison accurate at
        the moment it is used, which is all that matters.
        """
        if self._last_decay_ts <= 0.0:
            self._last_decay_ts = now
            return
        elapsed = now - self._last_decay_ts
        if elapsed <= 0.0 or self._value_window_s <= 0.0:
            self._last_decay_ts = now
            return
        factor = 1.0 - elapsed / self._value_window_s
        if factor >= 0.9999:
            return
        if factor < 0.0:
            factor = 0.0
        for entry in self.entries.values():
            if entry.hits_recent > 0.0:
                entry.hits_recent *= factor
        self._last_decay_ts = now

    def lowest_value_entry(self) -> "PartialEntry | None":
        """Entry with the lowest total value, or None when empty.

        Fresh entries are not sacrificial: they cannot have been hit yet, so
        ranking them by value would evict every newcomer before any request
        could resume from it, and the cache would slowly starve down to
        entries that already had a history.

        That protection is *budgeted*, not unconditional. Without a cap a
        stream of worthless new entries would push out a high-value entry
        holding the same slots -- the newcomers have to be ranked against the
        mature entries once they occupy more than ``_grace_budget_blocks``.
        """
        if not self.entries:
            return None
        now = time.monotonic()
        self._decay(now)
        grace = self._grace_s

        settling: list[PartialEntry] = []
        mature: list[PartialEntry] = []
        for entry in self.entries.values():
            (settling if entry.is_settling(now, grace) else mature).append(entry)

        if not mature:
            # Nothing has had a chance to prove itself: fall back to plain
            # LRU. The block budget is a hard cap, so something must go.
            return min(self.entries.values(), key=lambda e: e.created_ts)

        worst_mature = min(mature, key=lambda e: e.value(self.block_size))
        if not settling:
            return worst_mature

        protected_blocks = sum(e.num_blocks for e in settling)
        if protected_blocks <= self._grace_budget_blocks:
            # Within budget: the newcomers keep their chance, and the least
            # valuable mature entry pays for it.
            return worst_mature

        # Over budget: newcomers compete like everyone else.
        worst_settling = min(settling, key=lambda e: e.value(self.block_size))
        if worst_settling.value(self.block_size) <= worst_mature.value(
            self.block_size
        ):
            return worst_settling
        return worst_mature

    def remove_entry(self, key: PartialKey) -> None:
        """Public wrapper around ``_remove`` for callers outside this class."""
        if key in self.entries:
            self._remove(key)

    def log_diagnostics(self) -> None:
        """[本地 2026-11-11 P0] Value/reach distribution of cached entries.

        ``aligned`` counts entries whose ``length`` sits on a block boundary:
        their ``gain_tokens`` is zero, so ordinary full-block caching already
        covers them and value-based eviction has no signal for them.

        Throttled here rather than at the call site so any caller can invoke it
        on the stats tick without knowing the period. Returns early when the
        cache is empty -- an empty cache is the steady state of a quiet engine
        and logging it every tick would bury the signal.
        """
        now = time.monotonic()
        if now < self._next_diag_ts:
            return
        # Time since the previous emitted line, for the dep_miss rate. Measured
        # rather than inferred from the throttle, and _last_diag_ts <= 0 means
        # "first emission": the accumulated count is then a baseline, not a
        # delta (reporting 1234 new misses on the first line would be a lie).
        first_diag = self._last_diag_ts <= 0.0
        elapsed = 0.0 if first_diag else now - self._last_diag_ts
        self._last_diag_ts = now
        self._next_diag_ts = now + self._diag_s
        n = len(self.entries)
        if n == 0:
            return
        gains = sorted(e.length % self.block_size for e in self.entries.values())
        blocks = sorted(e.num_blocks for e in self.entries.values())
        recents = sorted(e.hits_recent for e in self.entries.values())
        # [本地 2026-11-11 观测 P1] 累计 hits：hits_recent 会被衰减，分不清
        # 「曾经热过现在冷」和「从建立起一次都没命中」——后者才是真死条目。
        cum = sorted(e.hits for e in self.entries.values())
        never = sum(1 for h in cum if h == 0)
        aligned = sum(1 for g in gains if g == 0)
        # [本地 2026-11-11 观测 P1] 依赖失效速率：找到了 entry 却因 boundary
        # 之前的普通缓存整块缺失而放弃。这个数非零就说明 entry 在变成废块。
        dep_miss = self.dep_miss_count
        dep_delta = 0 if first_diag else dep_miss - self._dep_miss_seen
        self._dep_miss_seen = dep_miss
        dep_rate = (dep_delta * 60.0 / elapsed) if elapsed > 0.0 else 0.0
        logger.info(
            "Partial diag: n=%d blocks=%d/%d (%.1f%%) "
            "| gain p50=%d p90=%d max=%d aligned=%d (%.0f%%) "
            "| blocks/entry p50=%d max=%d "
            "| hits_recent p50=%.2f p90=%.2f max=%.2f "
            "| hits_cum p50=%d p90=%d max=%d never_hit=%d (%.0f%%) "
            "| dep_miss=%d (+%d, %.2f/min)",
            n,
            self.num_blocks,
            self.max_blocks,
            100.0 * self.num_blocks / max(1, self.max_blocks),
            gains[n // 2],
            gains[int(n * 0.9)],
            gains[-1],
            aligned,
            100.0 * aligned / n,
            blocks[n // 2],
            blocks[-1],
            recents[n // 2],
            recents[int(n * 0.9)],
            recents[-1],
            cum[n // 2],
            cum[int(n * 0.9)],
            cum[-1],
            never,
            100.0 * never / n,
            dep_miss,
            dep_delta,
            dep_rate,
        )

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
        mm_features: Sequence[Any] | None = None,
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
            if tuple(tokens[block_start : block_start + len(tail)]) == tail and (
                key[4]
                == tail_mm_items(mm_features, block_start, block_start + len(tail))
            ):
                best = entry
        if best is not None:
            self.entries.move_to_end(best.key)
            # [本地 2026-11-11 P-06] Only an actual resume proves this entry
            # still saves work; insert() alone proves nothing.
            best.hits += 1
            best.hits_recent += 1.0
        return best

    def clear(self) -> None:
        while self.evict_one():
            pass
