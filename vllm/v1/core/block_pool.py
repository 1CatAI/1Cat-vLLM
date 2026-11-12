# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math  # [本地 2026-11-10 P-05]
import os  # [本地 2026-11-10 P-05]
import time  # [本地 2026-11-10 P-05]
from collections.abc import Iterable, Sequence
from itertools import groupby
from typing import Any

import vllm.envs as envs  # [本地 2026-11-11 配额]
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
    TieredBlockQueue,  # [本地 2026-11-11 P-06]
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


def _tiered_queue_enabled() -> bool:
    """[本地 2026-11-11 P-06] Value-ordered free queue master switch.

    Read from the environment directly (same convention as the other
    ``VLLM_KV_TIER_*`` knobs) so it is independent of when ``envs`` finishes
    importing. Registered in ``envs.py`` anyway, otherwise
    ``validate_environ`` warns about an unknown ``VLLM_`` variable.
    """
    return os.environ.get("VLLM_KV_TIERED_QUEUE", "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


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

        self._device_state_blocks = device_state_blocks
        if device_state_blocks:
            if not 1 < device_state_blocks < num_gpu_blocks:
                raise ValueError("State partition must leave history capacity")
            self.free_block_queue = _PartitionedBlockQueue(
                self.blocks, device_state_blocks, device_state_group_ids
            )

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

        self._init_value_tiers()

    def _init_value_tiers(self) -> None:
        """[本地 2026-11-10 P-05] 价值分级参数。

        分级只影响「块作为缓存身份能活多久」，不影响分配：所有块始终在
        free 队列里，调度器看到的口径与 upstream 完全一致。这是刻意选择
        ——把高价值块移出队列会让池子被切开，长请求撞上「有块但拿不到」，
        并重新引入 get_num_free_blocks 漏算导致的 KV 卡死。
        """
        self._tier_enabled: bool = (
            os.environ.get("VLLM_KV_TIER_ENABLED", "1").strip().lower()
            not in ("", "0", "false", "no", "off")
        ) and self.enable_caching

        def _f(name: str, default: float) -> float:
            return float(os.environ.get(name, str(default)))

        def _i(name: str, default: int) -> int:
            return int(os.environ.get(name, str(default)))

        # [本地 2026-11-11 P-06] 这两个**无条件**初始化，不受 _tier_enabled
        # 控制：partial 的价值淘汰可以独立配置，参数缺失会导致
        # AttributeError，或者更糟——hits_recent 恒为 0 让价值比较静默失效。
        self._tier_value_window_s: float = max(
            _f("VLLM_KV_TIER_VALUE_WINDOW_S", 3600.0), 1.0
        )
        self._last_decay_ts: float = 0.0
        self._tier_diag_s: float = max(_f("VLLM_KV_TIER_DIAG_S", 60.0), 1.0)
        # 首次调用立即输出基线（与 sweep 不同：sweep 首次应延后）。
        self._tier_next_diag: float = 0.0

        if not self._tier_enabled:
            if _tiered_queue_enabled():
                logger.warning(
                    "VLLM_KV_TIERED_QUEUE=1 is ignored because "
                    "VLLM_KV_TIER_ENABLED=0 (the tiered queue needs tier "
                    "classification to place blocks)."
                )
            return

        # 归一化尺度。age 用绝对跨度而非池内分位数：分位数需要周期扫全池
        # 排序，会与相位分离的 max_wait 抢时长，且单块变化引起全体重排。
        # [本地 2026-11-11 P-06] 加下限：_tier_scores 用它做除数，
        # 0 会 ZeroDivisionError。原本只在 sweep 里调用（60s 一次），
        # 接入 append_n 后每次释放都会调用，首次满池释放就会崩。
        self._tier_age_span: float = max(_f("VLLM_KV_TIER_AGE_SPAN", 21600.0), 1.0)
        self._tier_freq_sat: float = max(_f("VLLM_KV_TIER_FREQ_SAT", 64.0), 1.0)
        # [本地 2026-11-11 P0-α] L1 判据改为**池内分位数**，这个值降级为
        # 「绝对值地板」：分位数再低，也要至少命中这么多次才进 L1。
        #
        # 起因：实测生产 220 分钟 L1/L2A 从未出现任何块。512 是按「系统提示
        # 词被上千个会话读」设的，但真实负载下没有块能达到，于是 L1 层根本
        # 不存在，「保护系统提示词」这个核心收益为零。
        #
        # 地板而不是门槛，是因为分位数是相对量：只剩一对块、其中一个命中 1
        # 次时，p90 就是 1，没有地板会让「刚被读过一次」的块拿到永不因年龄
        # 降级的待遇。
        self._tier_l1_min_hits: int = max(
            _i("VLLM_KV_TIER_L1_MIN_HITS", 3), 1
        )
        # 分位点。0.9 ⇒ 有命中的块里，前 10% 进 L1。
        self._tier_l1_frac: float = min(
            max(_f("VLLM_KV_TIER_L1_FRAC", 0.9), 0.5), 0.999
        )
        # 上一轮 sweep 算出的动态阈值。None ⇒ 还没有样本，用静态地板。
        # 用**上一轮**的值而不是本轮：_classify 在遍历途中就要用，
        # 而分位数要遍历完才知道。滞后一轮（60s）不影响判断。
        self._tier_l1_min_hits_dyn: int | None = None
        self._tier_hot_freq: float = _f("VLLM_KV_TIER_HOT_FREQ", 0.6)
        self._tier_hot_age: float = _f("VLLM_KV_TIER_HOT_AGE", 0.5)

        # 各层 TTL（秒）。L1~L2 到期只降级（仍可命中），L3 到期清 hash。
        self._tier_ttl: tuple[float, ...] = (
            _f("VLLM_KV_TIER_L3_TTL", 600.0),
            _f("VLLM_KV_TIER_L2C_TTL", 1800.0),
            _f("VLLM_KV_TIER_L2B_TTL", 3600.0),
            _f("VLLM_KV_TIER_L2A_TTL", 7200.0),
            _f("VLLM_KV_TIER_L1_TTL", 21600.0),
        )
        self._tier_sweep_s: float = max(_f("VLLM_KV_TIER_SWEEP_S", 60.0), 1.0)
        self._tier_next_sweep: float = time.monotonic() + self._tier_sweep_s

        # [本地 2026-11-11 观测 P2] L3 是【流量】不是【存量】。
        #
        # ``tier_block_counts`` 只数有 hash 的块，而 L3 的终局就是被清掉
        # hash 退场 —— 所以 ``KV tiers: L3`` 必然是 0，且这个 0 不代表
        # 「没有东西在过期」。真正要看的是【回收速率】：一个 L3 块从进层
        # 到被清最多待 L3_TTL(600s)，统计时刻永远看不到存量。
        #
        # 这两个字段就是那个速率：累计退役数 + 上次采样值（算增量）。
        self._retired_blocks: int = 0
        # 其中因 TTL 到期（而非被新分配挤掉）而退役的部分。
        self._retired_by_ttl: int = 0
        self._retired_seen: int = 0
        self._retired_by_ttl_seen: int = 0
        # 退役时刻的闲置时长直方图（秒）。桶边界对齐现有 TTL，直接可判
        # 「各层 TTL 是否过长」：绝大多数落在第一个桶 ⇒ 块刚进 L3 就被清，
        # 上层 TTL 形同虚设；落在最后几桶 ⇒ 上层确实留住了有用的内容。
        self._retire_idle_hist: list[int] = [0] * 6
        self._retire_idle_boundaries: tuple[float, ...] = (
            600.0, 1800.0, 3600.0, 7200.0, 21600.0
        )

        # [本地 2026-11-11 配额] 保护层容量上限。
        #
        # 在此之前 L1+L2 的规模只是分类的副产品：命中多的块进高等级，
        # 高等级 TTL 长，于是自然堆到 26% 左右。没有任何机制保证它不
        # 会继续膨胀，也没有机制把它压到某个目标值。
        #
        # 这里加一道真正的配额：L1+L2 的块数不得超过总池的一个比例。
        # 超出部分从**最低保护层**开始驱逐（先 L2C，最后 L1），因为
        # 保护强度本就该随等级递减。
        #
        # 默认关闭：配额会改变驱逐时机，属于行为变更，必须显式开启。
        self._tier_quota_enabled: bool = bool(
            envs.VLLM_KV_TIER_QUOTA_ENABLED
        )
        # 总池的 L1+L2 上限比例，夹到 (0, 1] 防止错值把保护层清零或放大。
        self._tier_protect_frac: float = min(
            max(float(envs.VLLM_KV_TIER_PROTECT_FRAC), 0.0), 1.0
        )
        # L1 在保护预算内的份额，夹到 [0, 1]；0.2 ⇒ L1:L2 = 1:4。
        self._tier_l1_share: float = min(
            max(float(envs.VLLM_KV_TIER_L1_SHARE), 0.0), 1.0
        )
        # 配额驱逐计数，用于观测是否真的在生效。
        self._quota_evicted: int = 0
        self._quota_seen: int = 0

        # [本地 2026-11-11 P-06] 最后一步：按开关把队列换成价值分层版。
        # 放在这里而不是 __init__ 中部，是因为此时 null_block 已取出、
        # 分区队列（若有）已构造，执行时机天然正确。
        self._maybe_install_tiered_queue()

    # ------------------------------------------------------------------
    # [本地 2026-11-11 P-06] P0 诊断 + 价值度量
    # ------------------------------------------------------------------

    def _maybe_install_tiered_queue(self) -> None:
        """Replace the free queue with a value-ordered one, if enabled.

        The null block is already popped by the time this runs, so
        ``get_all_free_blocks()`` (not ``self.blocks``) is the correct input:
        it contains exactly the blocks currently sitting in the queue.
        Passing ``self.blocks`` would put the null block back in.
        """
        if not _tiered_queue_enabled():
            return
        # Takes ``now`` from the caller: a bulk return then pays for one clock
        # read instead of one per block.
        reclassify = lambda b, now: self._classify(b, now)  # noqa: E731
        q = self.free_block_queue
        if isinstance(q, FreeKVCacheBlockQueue):
            # Current production shape: no device state partition.
            self.free_block_queue = TieredBlockQueue(
                q.get_all_free_blocks(), num_tiers=5, reclassify=reclassify
            )
        elif isinstance(q, _PartitionedBlockQueue) and isinstance(
            q.history, FreeKVCacheBlockQueue
        ):
            # Mamba state blocks stay pure LRU: they cannot be rebuilt, so
            # demoting one by value would strand a request's recurrent state.
            q.history = TieredBlockQueue(
                q.history.get_all_free_blocks(), num_tiers=5, reclassify=reclassify
            )
        else:
            return
        logger.info(
            "Tiered free queue enabled: %d tiers, %d free blocks.",
            5,
            self.free_block_queue.num_free_blocks,
        )

    def reload_tier_settings(self) -> dict:
        """[本地 2026-11-11] Re-read the tunable tier knobs, in place.

        Exists so an operator can retune the cache without paying a cold start.
        That cost (minutes) is weight loading and CUDA-graph capture, none of
        which depends on these numbers -- the values are plain floats and ints
        read fresh on every sweep, classify and diag pass, so replacing the
        attributes is enough.

        Structural switches are deliberately excluded. ``VLLM_KV_TIER_ENABLED``
        and ``VLLM_KV_TIERED_QUEUE`` decide whether these fields exist at all
        and which queue class is installed, so flipping them still needs a
        restart; ``reload_kv_config`` reports them back instead of pretending.

        The dynamic L1 threshold is dropped rather than kept: it was computed
        under the previous ``l1_frac``, and serving a stale cut for up to a
        full sweep period would make a retune look like it did nothing.
        """
        if not self._tier_enabled:
            return {}

        def _f(name: str, default: float) -> float:
            return float(os.environ.get(name, str(default)))

        def _i(name: str, default: int) -> int:
            return int(os.environ.get(name, str(default)))

        self._tier_value_window_s = max(
            _f("VLLM_KV_TIER_VALUE_WINDOW_S", 3600.0), 1.0
        )
        self._tier_diag_s = max(_f("VLLM_KV_TIER_DIAG_S", 60.0), 1.0)
        self._tier_age_span = max(_f("VLLM_KV_TIER_AGE_SPAN", 21600.0), 1.0)
        self._tier_freq_sat = max(_f("VLLM_KV_TIER_FREQ_SAT", 64.0), 1.0)
        self._tier_l1_min_hits = max(_i("VLLM_KV_TIER_L1_MIN_HITS", 3), 1)
        self._tier_l1_frac = min(max(_f("VLLM_KV_TIER_L1_FRAC", 0.9), 0.5), 0.999)
        self._tier_hot_freq = _f("VLLM_KV_TIER_HOT_FREQ", 0.6)
        self._tier_hot_age = _f("VLLM_KV_TIER_HOT_AGE", 0.5)
        self._tier_ttl = (
            _f("VLLM_KV_TIER_L3_TTL", 600.0),
            _f("VLLM_KV_TIER_L2C_TTL", 1800.0),
            _f("VLLM_KV_TIER_L2B_TTL", 3600.0),
            _f("VLLM_KV_TIER_L2A_TTL", 7200.0),
            _f("VLLM_KV_TIER_L1_TTL", 21600.0),
        )
        self._tier_sweep_s = max(_f("VLLM_KV_TIER_SWEEP_S", 60.0), 1.0)
        # [本地 2026-11-11 配额] 两个比例可热改 —— 它们只改「下一轮砍多少」，
        # 不改变数据结构。总开关 ``VLLM_KV_TIER_QUOTA_ENABLED`` 不进这里：
        # 它决定配额阶段是否执行，但既然字段始终存在，热改也无害；之所以
        # 仍列为结构开关，是为了让运维对「刚才那次重启是否必要」有确定答案。
        self._tier_protect_frac = min(
            max(_f("VLLM_KV_TIER_PROTECT_FRAC", 0.15), 0.0), 1.0
        )
        self._tier_l1_share = min(
            max(_f("VLLM_KV_TIER_L1_SHARE", 0.2), 0.0), 1.0
        )
        self._tier_l1_min_hits_dyn = None

        # Pull the next passes forward so a newly set period applies now
        # instead of once the old one expires.
        now = time.monotonic()
        self._tier_next_diag = now
        self._tier_next_sweep = now + self._tier_sweep_s
        return {
            "diag_s": self._tier_diag_s,
            "sweep_s": self._tier_sweep_s,
            "l1_frac": self._tier_l1_frac,
            "l1_min_hits": self._tier_l1_min_hits,
            "value_window_s": self._tier_value_window_s,
            "ttl": list(self._tier_ttl),
            # [本地 2026-11-11 配额] 返回当前配额设置，供运维核对。
            "quota_enabled": self._tier_quota_enabled,
            "protect_frac": self._tier_protect_frac,
            "l1_share": self._tier_l1_share,
        }

    def _sync_block_tier(self, block: KVCacheBlock, new_tier: int) -> None:
        """The only place that may change ``block.tier``.

        Must not early-return on ``block.tier == new_tier``: the callers that
        matter have already reset the field. ``reset_value_state()`` sets tier
        to 0, and ``_maybe_evict_cached_block`` calls it right before asking
        for a sync -- so the block still sitting in layer 4 looks like it is
        already tier 0 and would never move, leaving a hashless block camped
        in the highest tier. Whether a move is needed is decided inside the
        queue, from where the block actually is.
        """
        if not (0 <= new_tier < 5):
            # Same clamp the queue applies on return: keep the field and the
            # queue position consistent instead of writing a bad tier and
            # having repromote silently reject the move.
            new_tier = 0
        block.tier = new_tier
        q = self.free_block_queue
        if isinstance(q, TieredBlockQueue):
            q.repromote(block, new_tier)
        elif isinstance(q, _PartitionedBlockQueue):
            # State blocks stay pure LRU (Mamba state cannot be rebuilt, so
            # demoting one by value would strand the request). Only the
            # history partition is tiered, and it holds the higher ids.
            if (
                block.block_id >= q.state_blocks
                and isinstance(q.history, TieredBlockQueue)
            ):
                q.history.repromote(block, new_tier)

    def _rebuild_free_queue(self) -> None:
        """Rebuild the tiered queue from the blocks currently in it.

        Needed after a whole-pool tier reset, otherwise every block keeps its
        old layer position while claiming tier 0, and a benchmark that starts
        with ``reset_prefix_cache`` gets a skewed pool.

        Caller must guarantee all non-null blocks are free;
        ``reset_prefix_cache`` validates ``num_used_blocks == 1`` beforehand.
        """
        q = self.free_block_queue
        if isinstance(q, TieredBlockQueue):
            assert q._n_allocated == 0, (
                f"rebuild requires all blocks free, got {q._n_allocated} allocated"
            )
            # Rebuild in place: swapping the object would strand any external
            # reference on a detached queue while the pool allocates from the
            # new one.
            q.rebuild(q.get_all_free_blocks())
        elif isinstance(q, _PartitionedBlockQueue) and isinstance(
            q.history, TieredBlockQueue
        ):
            assert q.history._n_allocated == 0, (
                "rebuild requires the history partition to be all free"
            )
            q.history.rebuild(q.history.get_all_free_blocks())
        self._last_decay_ts = 0.0

    def _decay_hits_recent(self, now: float) -> None:
        """Lazily decay every block's ``hits_recent`` toward the window.

        Called from ``block_value`` only. Not driven by the sweep on purpose:
        the sweep stops running when the pool is idle, which is exactly when
        stale hits most need to decay.
        """
        if self._last_decay_ts <= 0.0:
            self._last_decay_ts = now
            return
        elapsed = now - self._last_decay_ts
        if elapsed <= 0.0:
            self._last_decay_ts = now
            return
        factor = 1.0 - elapsed / self._tier_value_window_s
        if factor >= 0.9999:
            return
        if factor < 0.0:
            factor = 0.0
        for block in self.blocks:
            if block.hits_recent > 0.0:
                block.hits_recent *= factor
        self._last_decay_ts = now

    def block_value(self, block: KVCacheBlock) -> float:
        """Saved tokens per hit for an ordinary block (same unit as
        ``PartialEntry.value``): one hit saves ``hash_block_size`` tokens and
        costs one block.
        """
        if block.block_hash is None:
            return 0.0
        self._decay_hits_recent(time.monotonic())
        return block.hits_recent * self.hash_block_size

    def _log_value_diagnostics(self, now: float) -> None:
        """[本地 2026-11-11 P0] Block-value distribution, throttled.

        Read-only: touches no queue state and never evicts. Driven from
        ``tier_block_counts`` (scheduler calls it on the stats tick) so it
        also runs when the pool sits idle.

        The point is to answer one question before building anything on top:
        do block values actually differ? If every block has the same hit
        count, value-ordered eviction degenerates to LRU and the whole
        design has no signal to work with.
        """
        if not self._tier_enabled:
            return
        if now < self._tier_next_diag:
            return
        self._tier_next_diag = now + self._tier_diag_s

        hits: list[int] = []
        recents: list[float] = []
        for block in self.blocks:
            if block.is_null or block.block_hash is None:
                continue
            hits.append(block.hit_count)
            recents.append(block.hits_recent)
        if not hits:
            logger.info("KV value diag: pool has no hashed blocks.")
            return

        hits.sort()
        recents.sort()

        def quantile(values: list, p: float):
            return values[min(len(values) - 1, int(len(values) * p))]

        l1_threshold = (
            self._tier_l1_min_hits_dyn
            if self._tier_l1_min_hits_dyn is not None
            else self._tier_l1_min_hits
        )
        l1_eligible = sum(1 for h in hits if h >= l1_threshold)
        # ``tier_block_counts`` counts every block (allocated ones included),
        # while this queue figure counts only blocks sitting in the free
        # queue. Expect the queue number to be LOWER by exactly the blocks
        # currently lent out to requests -- e.g. "L1 6" with tier4=0 in the
        # queue means all six L1 blocks are in use. A disagreement is only
        # evidence that _sync_block_tier failed when it is in the other
        # direction (queue holds blocks a level claims to have none of).
        # Only show it when the queue is actually tiered, otherwise it is all
        # zeros and just noise.
        #
        # Do not name this ``q``: that shadows the quantile helper above,
        # and the shadowing only bites when the pool actually has hashed
        # blocks (the helper is skipped otherwise), so an empty-pool test
        # passes and production blows up with "'TieredBlockQueue' object is
        # not callable".
        queue = self.free_block_queue
        if isinstance(queue, TieredBlockQueue):
            q_occ = str(queue.tier_occupancy())
        elif isinstance(queue, _PartitionedBlockQueue) and isinstance(
            queue.history, TieredBlockQueue
        ):
            q_occ = str(queue.history.tier_occupancy())
        else:
            q_occ = "-"
        logger.info(
            "KV value diag: n=%d | hit_count p50=%d p90=%d p99=%d max=%d "
            "| hits_recent p50=%.2f p90=%.2f p99=%.2f max=%.2f "
            "| l1_eligible=%d (thr=%d dyn=%s floor=%d) | distinct_hits=%d "
            "| queue=%s free_by_tier=%s",
            len(hits),
            quantile(hits, 0.50),
            quantile(hits, 0.90),
            quantile(hits, 0.99),
            hits[-1],
            quantile(recents, 0.50),
            quantile(recents, 0.90),
            quantile(recents, 0.99),
            recents[-1],
            l1_eligible,
            l1_threshold,
            self._tier_l1_min_hits_dyn,
            self._tier_l1_min_hits,
            len(set(hits)),
            type(self.free_block_queue).__name__,
            q_occ,
        )

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
        # [本地 2026-11-10 P-05 修复] 写入即「活跃」。此前只有 touch() 记时间，
        # 导致写入后从未被复用的块 last_hit_ts 恒为 0 —— 它们被判为 L3，却又
        # 因为「没有空闲起点」被 sweep 永久跳过，于是永久占着 hash。
        # 实测后果：L3 常驻 356 块、两小时不退场。
        # 语义澄清：last_hit_ts 实际是「最后一次活跃」（写入或命中），
        # hit_count 才只统计「被复用」，两者不能混用。
        now = time.monotonic() if self._tier_enabled else 0.0
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
            if self._tier_enabled:
                blk.last_hit_ts = now
                blk.created_ts = now  # [本地 2026-11-11 P-06] 诊断用
                # [本地 2026-11-10 修复] 新写入的块立即标成 L2C 观察期，
                # 而不是留在默认的 0（=L3）。否则在 sweep 下次重算之前
                # （最长 60s，引擎空闲时更久），所有刚写入的块都在指标里
                # 显示为 L3 —— 看起来像「10 分钟 TTL 不生效」，实际只是
                # 尚未定级。分级本身仍由 _classify 在 sweep 中决定。
                blk.tier = 1
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

        ret: list[KVCacheBlock] = (
            self.free_block_queue.popleft_n(num_blocks, kv_cache_group_id)
            if isinstance(self.free_block_queue, _PartitionedBlockQueue)
            else self.free_block_queue.popleft_n(num_blocks)
        )

        # [本地 2026-11-10 P-05] 分级只决定「什么时候放弃缓存身份」，
        # 不决定「分配给谁」。扫一次队列实现的就是「什么都不做」——它只让
        # 失去价值的块提前放掉 hash，从而把队头真正让给新内容。主动扫描
        # 而不是靠每次分配顺带检查：池子空闲时没人分配，僵尸缓存就永远
        # 不会退场。
        if self._tier_enabled:
            self._maybe_sweep_tiers()

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

    def _maybe_evict_cached_block(self, block: KVCacheBlock, ttl_expiry: bool = False) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.
            ttl_expiry: [本地 2026-11-11 观测 P2] True only when the sweep is
                evicting because the block sat idle past L3_TTL. The other two
                callers evict to make room for a new allocation, which is a
                different event: mixing them would make the idle histogram
                describe "how long content survived before being pushed out"
                instead of "how long it sat unreused before expiring", and
                only the latter answers whether the tier TTLs are too long.

        Returns:
            True if the block is evicted, False otherwise.
        """
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

        # [本地 2026-11-11 观测 P2] 退役计量。必须在 reset_value_state()
        # 之前读 last_hit_ts —— 它会被清零，之后就再也算不出闲置时长了。
        # 纯计数，不改变任何决策。
        self._retired_blocks += 1
        if ttl_expiry:
            self._retired_by_ttl += 1
            if block.last_hit_ts > 0.0:
                idle = time.monotonic() - block.last_hit_ts
                for i, bound in enumerate(self._retire_idle_boundaries):
                    if idle < bound:
                        self._retire_idle_hist[i] += 1
                        break
                else:
                    # 超过最长 TTL 才退场：说明上层 TTL 确实留住了内容。
                    self._retire_idle_hist[-1] += 1

        # [本地 2026-11-10 P-05] 丢弃 hash 意味着这块内容再也无法命中，
        # 它的复用历史到此为止。等级必须一起清掉，否则回收的旧记录会让
        # 同一 block_id 上新写入的内容继承前任的「价值」。
        block.reset_hash()
        block.reset_value_state()
        # [本地 2026-11-11 P-06] Content is gone, so the block belongs in the
        # lowest tier. Without this it stays parked in whatever layer it was
        # in (often L1) until the next sweep, wasting the highest slot.
        self._sync_block_tier(block, 0)

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
        now = time.monotonic() if self._tier_enabled else 0.0
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            # [本地 2026-11-10 P-05] 命中即价值证据。累加放在这里而不是
            # cache_full_blocks：一个块被写入只说明有人产生过这段内容，
            # 被 touch 才说明有人复用了它。
            if self._tier_enabled and block.block_hash is not None:
                block.hit_count += 1
                block.last_hit_ts = now
                # [本地 2026-11-11 P-06] Sliding-window counter. Cumulative
                # hit_count cannot tell a block that was hot hours ago from
                # one that is hot now; decay happens lazily in block_value().
                block.hits_recent += 1.0
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

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()
            block.reset_value_state()

        # [本地 2026-11-11 P-06] Every block now claims tier 0, but the tiered
        # queue still has them spread across layers. Rebuild it, otherwise a
        # benchmark that resets before measuring starts with blocks stuck
        # behind empty low tiers.
        self._rebuild_free_queue()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self, kv_cache_group_id: int | None = None) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        if kv_cache_group_id is not None and isinstance(
            self.free_block_queue, _PartitionedBlockQueue
        ):
            return self.free_block_queue.for_group(kv_cache_group_id).num_free_blocks
        return self.free_block_queue.num_free_blocks

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
            and history <= self.free_block_queue.history.num_free_blocks
        )

    def _tier_scores(self, block: KVCacheBlock, now: float) -> tuple[float, float]:
        """[本地 2026-11-10 P-05] (age_score, freq_score)，都是越高越该留。

        age_score —— 最近被复用的时刻离现在多近，1 表示刚刚。
        freq_score —— 累计被复用了多少次，取 log 是为了压住长尾：命中次数是
        幂律分布（系统提示词上万次、普通块个位数），线性归一化会让除 L1 之外
        的所有块挤在 0.01 以下，分级失去区分度。

        两个都用固定尺度，不用池内分位数：分位数要求周期扫全池排序，会和
        相位分离的 max_wait 抢时长，而且单个块的状态变化会引起全体重排，
        分数会抖。
        """
        if block.last_hit_ts <= 0.0:
            age_score = 0.0
        else:
            age_score = 1.0 - (now - block.last_hit_ts) / self._tier_age_span
            age_score = min(max(age_score, 0.0), 1.0)
        freq_score = math.log1p(block.hit_count) / math.log1p(self._tier_freq_sat)
        return age_score, min(freq_score, 1.0)

    def _classify(self, block: KVCacheBlock, now: float) -> int:
        """[本地 2026-11-10 P-05] 状态块定级。0=L3 … 4=L1。

        注意：L1 之外的分级全部依赖 age_score，而 age_score 只在「命中过」
        时有意义。一个被写入却从未被复用的块 ages 到 0，会稳定落在 L3——
        这正是想要的：没被读过的新内容本来就该最先退场。
        """
        if block.block_hash is None:
            return 0
        idle = now - block.last_hit_ts if block.last_hit_ts > 0.0 else 0.0

        # [本地 2026-11-11 P0-α] 门槛取「池内分位数」与「绝对地板」的较高者。
        # 分位数让判据随负载自适应（高频场景门槛自然抬高，低频自然降低），
        # 地板防止样本极少时把「刚读过一次」的块当成系统提示词。
        #
        # 已知特性：判据是 ``>=``，所以命中数恰好等于门槛的块全都会进 L1。
        # 重复值多时（例如一批块都命中 50 次）实际入选比例会超过 1-frac。
        # 这是可接受的：那些块的价值确实相同，而 L1 并不是永久锁定——
        # 闲置超过 _tier_ttl[4]（6 h）仍会一路降到 L3。
        l1_threshold = (
            self._tier_l1_min_hits_dyn
            if self._tier_l1_min_hits_dyn is not None
            else self._tier_l1_min_hits
        )

        # 基础等级只由复用证据决定，往下再按各层 TTL 逐级衰减。
        if block.hit_count >= l1_threshold:
            # L1 判定刻意不看年龄。固定提示词在深夜可能几小时没人用，
            # 但它仍然是每个会话都要读的那段内容，所以只看累计命中量。
            base = 4
        elif block.hit_count == 0:
            # [本地 2026-11-10 修复] 从未被复用的块：没有价值证据，只给一个
            # L3_TTL 的观察期。此前依赖 age_score 衰减，要等
            # _tier_age_span×0.5（默认 3 小时）才降到 L3，实测两小时仍常驻。
            # 现在语义是「写入后 10 分钟没人再读 ⇒ 让位」。
            return 1 if idle < self._tier_ttl[0] else 0
        else:
            age_score, freq_score = self._tier_scores(block, now)
            if freq_score >= self._tier_hot_freq:
                base = 3 if age_score >= self._tier_hot_age else 2
            else:
                base = 2 if age_score >= self._tier_hot_age else 1

        # [本地 2026-11-10 修复] 让 _tier_ttl 真正生效。此前这个元组里
        # L2C/L2B/L2A/L1 四条 TTL 一次都没被读过，降级完全交给 age_score
        # 连续衰减，"该层到期就掉一级"的设计从未实现——复用过的块要等
        # 3 小时才降 L3。
        while base > 0 and idle >= self._tier_ttl[base]:
            base -= 1
        return base

    def _apply_tier_quota(
        self, protected: list[tuple[int, KVCacheBlock]]
    ) -> tuple[int, int, int]:
        """[本地 2026-11-11 配额] 把 L1+L2 压回预算内。

        Args:
            protected: 本轮 sweep 里 tier != 0 的块，(tier, block) 序。
                不含被引用、null、无 hash 的块 —— 那些不能回收。

        Returns:
            (L1 块数, L2 块数, 本轮配额驱逐数)。调用方仅在开关打开时使用。

        设计要点：

        1. **预算 = 总池 × protect_frac**，其中 L1 占 l1_share，L2 占其余。
           两个上限分开算，而不是只限总量 —— 否则 L1 可以吃掉整个预算，
           把 L2 挤空，1:4 的比例就名存实亡。

        2. **先驱逐 L2C，最后 L1**。保护强度随等级递减，超配额时先牺牲
           保护最弱的。同层内按 last_hit_ts 升序（最久未命中先走），
           与队列的 LRU 语义一致。

        3. **只清 hash，不碰引用**。候选块 ref_cnt 必然为 0（sweep 只收集
           这类），所以 ``_maybe_evict_cached_block`` 走的是安全的退役
           路径：内容和 host 副本不受影响，只是不再能命中本地缓存。

        4. **超额为 0 时一个块都不动**。这条很重要：配额是上限而非目标，
           池子小的时候不应主动把保护层压到 0。
        """
        total = len(self.blocks)
        budget_total = int(total * self._tier_protect_frac)
        budget_l1 = int(budget_total * self._tier_l1_share)
        # L2 拿剩下的全部，避免两次取整后 L1+L2 仍超总量。
        budget_l2 = max(budget_total - budget_l1, 0)

        l1_blocks = [b for t, b in protected if t == 4]
        l2_blocks = [b for t, b in protected if t in (1, 2, 3)]

        evicted = 0
        # 采集时按 tier 升序即可让 L2C(1) 排在 L2B(2)/L2A(3) 前面，
        # 但 L1 是 4 —— 两类分开处理，天然保证了「先 L2 后 L1」。
        for group, budget in ((l2_blocks, budget_l2), (l1_blocks, budget_l1)):
            excess = len(group) - budget
            if excess <= 0:
                continue
            # 同层内最久未命中的先走。
            group.sort(key=lambda b: b.last_hit_ts)
            for block in group[:excess]:
                if block.ref_cnt > 0 or block.block_hash is None:
                    continue
                self._maybe_evict_cached_block(block)
                evicted += 1

        self._quota_evicted += evicted
        return len(l1_blocks), len(l2_blocks), evicted

    def _maybe_sweep_tiers(self) -> None:
        """[本地 2026-11-10 P-05] 周期重新定级，并让过期内容退场。

        只对「已经没有引用」的块干活：还挂在请求上的块随时会被用到，此刻
        重新定级除了制造抖动没有别的作用。

        定级只决定「这个块值不值得留」，所以它必须跑在过期判定之前——
        反过来会让一个刚刚掉到 L3 的块按旧等级的长 TTL 被放行。

        只有 L3 会清 hash。L1~L2 的 TTL 到期不做任何事：等级每轮重算，
        一个真的没用的块自然会掉到 L3，再由 L3 那条规则清理。把「降级」
        单独实现一遍只会让两条规则互相打架。
        """
        now = time.monotonic()
        if now < self._tier_next_sweep:
            return
        self._tier_next_sweep = now + self._tier_sweep_s

        demoted = 0
        dropped = 0
        # [本地 2026-11-11 配额] (tier, block) 保护层候选，供配额阶段使用。
        protected: list[tuple[int, KVCacheBlock]] = []
        # [本地 2026-11-10 诊断] 为什么块留在池里不走——分类计数。
        # 只有 ref>0 / null / 无 hash 三种跳过原因，加上「tier 非 L3」
        # 和「已清理」，五类互斥，加起来必等于 len(self.blocks)。
        n_ref = 0
        n_null = 0
        n_nohash = 0
        n_kept = 0
        n_ttl_wait = 0  # tier==0 但 idle < TTL（正常观察期）
        l3_with_ref = 0  # tier 仍标 0 且 ref>0（嫌疑：引用未释放）
        oldest_l3_idle = 0.0
        # [本地 2026-11-11 P0-α] 收集被复用过的块的命中数，本轮末尾算出
        # 下一轮用的 L1 门槛。sweep 本来就在遍历全池，收集只是往 list 里
        # append，实测增量约 22 µs / 60 s。
        reused_hits: list[int] = []
        for block in self.blocks:
            # 收集放在所有 continue 之前：还挂在请求上的块同样有复用证据，
            # 漏掉它们会让分布偏向「当前空闲」的那部分。
            if (
                not block.is_null
                and block.block_hash is not None
                and block.hit_count > 0
            ):
                reused_hits.append(block.hit_count)
            if block.ref_cnt > 0:
                n_ref += 1
                if block.tier == 0 and not block.is_null and block.block_hash:
                    l3_with_ref += 1
                continue
            if block.is_null:
                n_null += 1
                continue
            if block.block_hash is None:
                n_nohash += 1
                continue
            tier = self._classify(block, now)
            if tier != block.tier:
                demoted += block.tier > tier
                # [本地 2026-11-11 P-06] Move it in the queue too, not just
                # the field; otherwise a demoted block keeps its old slot.
                self._sync_block_tier(block, tier)
            if tier != 0:
                n_kept += 1
                # [本地 2026-11-11 配额] 保护层的候选块。此刻它的 tier 已
                # 同步到队列，ref_cnt==0（上面已 continue 掉被引用的），
                # 是唯一可以安全回收的一类。
                #
                # ⚠️ 只有 hit_count > 0 才算「受保护」。``_classify`` 把
                # 从未被复用的块放进 L2C 只是给它一个 600s 观察期，那**不是**
                # 价值证据。若把它们也纳入配额，每个新写入的块刚落地就被
                # 驱逐，永远等不到复用 —— 配额会变成「禁用前缀缓存观察期」，
                # 实测每轮都在追着新块跑，L2 数压不住。
                # hit_count==0 的块交给 L3 的 TTL 自然淘汰即可。
                if self._tier_quota_enabled and block.hit_count > 0:
                    protected.append((tier, block))
                continue
            # [本地 2026-11-10 修复] last_hit_ts 现在由 cache_full_blocks
            # 写入时设置，正常路径不会出现 0。若仍为 0（异常路径/机制启用
            # 前写入），从此刻开始计时，而不是永久跳过——此前正是这个
            # `continue` 让从未复用的块永不过期。
            if block.last_hit_ts <= 0.0:
                block.last_hit_ts = now
                continue
            idle = now - block.last_hit_ts
            if idle < self._tier_ttl[0]:
                n_ttl_wait += 1
                continue
            if idle > oldest_l3_idle:
                oldest_l3_idle = idle
            # [本地 2026-11-11 观测 P2] 这一处才是"TTL 到期回收"，
            # 另两个调用点是"腾地方给新分配"，计量时必须区分。
            self._maybe_evict_cached_block(block, ttl_expiry=True)
            dropped += 1

        # [本地 2026-11-11 配额] 保护层容量上限。必须在遍历之后：protected
        # 是本轮 sweep 刚定完级的候选，而配额要按最新定级来砍。
        if self._tier_quota_enabled:
            q_l1, q_l2, q_ev = self._apply_tier_quota(protected)
        else:
            q_l1 = q_l2 = q_ev = -1

        # [本地 2026-11-11 P0-α] 更新下一轮用的 L1 门槛。
        # 用「上一轮的值」是必然的：_classify 在遍历途中就要用门槛，而分位
        # 数要遍历完才知道。滞后一轮（60 s）对慢变量没有影响。
        if reused_hits:
            reused_hits.sort()
            idx = min(
                len(reused_hits) - 1,
                int(len(reused_hits) * self._tier_l1_frac),
            )
            self._tier_l1_min_hits_dyn = max(
                reused_hits[idx], self._tier_l1_min_hits
            )
        else:
            # 没有任何块被复用 ⇒ 没有证据支持 L1；回落到绝对地板的下方，
            # 让 L1 保持为空而不是凭空造出资格。
            self._tier_l1_min_hits_dyn = None

        # [本地 2026-11-11 观测 P2] 周期回收量。dropped 是【本轮】TTL 到期
        # 清掉的块数；跨轮累计用来算速率。retired_all 另外含"被新分配挤掉"
        # 的块，两者都报，差值本身就是「池子压力」的信号。
        retired_all = self._retired_blocks
        retired_ttl = self._retired_by_ttl
        d_all = retired_all - self._retired_seen
        d_ttl = retired_ttl - self._retired_by_ttl_seen
        self._retired_seen = retired_all
        self._retired_by_ttl_seen = retired_ttl
        hist = self._retire_idle_hist

        # [本地 2026-11-11 配额] 配额段拼进同一行，但只在开关打开时出现，
        # 未启用时日志与之前逐字一致（便于和既有基线对比）。
        quota_str = ""
        quota_args: list[object] = []
        if self._tier_quota_enabled:
            _q_total = int(len(self.blocks) * self._tier_protect_frac)
            _q_l1 = int(_q_total * self._tier_l1_share)
            quota_str = (
                " | quota L1+L2=%d/%d (L1 %d/%d, L2 %d/%d) evicted=%d (+%d)"
            )
            quota_args = [
                q_l1 + q_l2,
                _q_total,
                q_l1,
                _q_l1,
                q_l2,
                max(_q_total - _q_l1, 0),
                self._quota_evicted,
                self._quota_evicted - self._quota_seen,
            ]
            self._quota_seen = self._quota_evicted

        # 只有在真的发生清理或降级时才占一行 info。空转的 sweep 每 60s
        # 来一次，若也打 info，一天就是 1440 行无信息量的日志。
        log = logger.info if (dropped or demoted or q_ev > 0) else logger.debug
        log(
            "KV tier sweep: total=%d ref=%d null=%d nohash=%d keptT=%d "
            "ttl_wait=%d dropped=%d demoted=%d | L3_with_ref=%d "
            "| reused=%d l1_thr=%s "
            "| retired ttl_expiry=%d (+%d) all=%d (+%d) "
            "| retire_idle <600s=%d <1800=%d <3600=%d <7200=%d "
            "<21600=%d >=21600=%d" + quota_str,
            len(self.blocks),
            n_ref,
            n_null,
            n_nohash,
            n_kept,
            n_ttl_wait,
            dropped,
            demoted,
            l3_with_ref,
            len(reused_hits),
            self._tier_l1_min_hits_dyn,
            retired_ttl,
            d_ttl,
            retired_all,
            d_all,
            hist[0],
            hist[1],
            hist[2],
            hist[3],
            hist[4],
            hist[5],
            *quota_args,
        )

    def tier_block_counts(self) -> list[int]:
        """[本地 2026-11-10 P-05] 各层当前块数，下标即 tier（0=L3 … 4=L1）。

        [本地 2026-11-10 修复] 这里同时充当 sweep 的周期触发点。
        此前 sweep 只挂在 get_new_blocks() 上，池子闲置时没有分配就永远
        不扫描，僵尸缓存不退场。本方法由 scheduler 的 make_stats() 按日志
        周期调用（约 10s 一次），是不依赖分配的天然节拍。
        sweep 内部仍有 _tier_sweep_s（默认 60s）节流，不会真的每 10s 扫全池。
        """
        if self._tier_enabled:
            self._maybe_sweep_tiers()
            # [本地 2026-11-11 P-06] P0 diagnostics ride the same tick: it is
            # the only periodic hook that also fires while the pool is idle.
            self._log_value_diagnostics(time.monotonic())
        counts = [0] * 5
        for block in self.blocks:
            if block.is_null or block.block_hash is None:
                continue
            counts[block.tier] += 1
        return counts

    def tier_free_occupancy(self) -> list[int]:
        """[本地 2026-11-11 P-06] Actual free-block distribution per tier.

        Different question from ``tier_block_counts``, which reports the
        *intended* tier of every hashed block (allocated ones included). Here
        we report how many blocks are really sitting in each level right now.

        A large gap between the two is itself the signal: blocks claiming
        tier 4 while queued in layer 0 means the tier sync is broken.
        ``tier_block_counts`` keeps its original meaning so existing stats and
        the long-run accounting keep working.
        """
        q = self.free_block_queue
        if isinstance(q, TieredBlockQueue):
            return q.tier_occupancy()
        if isinstance(q, _PartitionedBlockQueue) and isinstance(
            q.history, TieredBlockQueue
        ):
            return q.history.tier_occupancy()
        return [0] * 5

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
