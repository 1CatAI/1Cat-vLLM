# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.serving import _PrefixCacheBoundaries
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.partial_prefix_cache import (
    PartialEntry,
    PartialPrefixCache,
    tail_mm_items,
)
from vllm.v1.request import PREFIX_CACHE_BOUNDARY_ALIGNMENT, Request

pytestmark = pytest.mark.cpu_test


def _entry(pool, key, length, ready_step, num_blocks=2):
    blocks = pool.get_new_blocks(num_blocks)
    return PartialEntry(
        key=key,
        length=length,
        blocks=[[block] for block in blocks],
        copy_last=[True] * num_blocks,
        ready_step=ready_step,
    )


def test_lookup_returns_longest_ready_matching_tail():
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    cache = PartialPrefixCache(block_pool=pool, max_blocks=16)
    parent = BlockHash(b"parent")
    cache.insert(_entry(pool, (parent, (1, 2), None, 0, ()), 6, ready_step=1))
    cache.insert(_entry(pool, (parent, (1, 2, 3), None, 0, ()), 7, ready_step=1))
    cache.insert(_entry(pool, (parent, (1, 9), None, 0, ()), 6, ready_step=1))
    tokens = [0, 0, 0, 0, 1, 2, 3, 4]
    hit = cache.lookup(parent, tokens, 4, 7, None, 0, current_step=2)
    assert hit is not None and hit.length == 7
    # The longer tail is past the lookup limit (prompt length - 1).
    hit = cache.lookup(parent, tokens, 4, 6, None, 0, current_step=2)
    assert hit is not None and hit.length == 6
    # Salt and LoRA are part of the key.
    assert cache.lookup(parent, tokens, 4, 7, "salt", 0, current_step=2) is None
    assert cache.lookup(parent, tokens, 4, 7, None, 3, current_step=2) is None


def test_entry_is_not_used_in_the_step_whose_copy_fills_it():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=4)
    cache = PartialPrefixCache(block_pool=pool, max_blocks=8)
    cache.insert(_entry(pool, (None, (5,), None, 0, ()), 1, ready_step=3))
    assert cache.lookup(None, [5, 6], 0, 1, None, 0, current_step=3) is None
    assert cache.lookup(None, [5, 6], 0, 1, None, 0, current_step=4) is not None


def test_eviction_releases_block_references():
    pool = BlockPool(num_gpu_blocks=9, enable_caching=True, hash_block_size=4)
    free_before = pool.get_num_free_blocks()
    cache = PartialPrefixCache(block_pool=pool, max_blocks=4)
    cache.insert(_entry(pool, (None, (1,), None, 0, ()), 1, ready_step=0))
    cache.insert(_entry(pool, (None, (2,), None, 0, ()), 1, ready_step=0))
    cache.insert(_entry(pool, (None, (3,), None, 0, ()), 1, ready_step=0))
    # Bounded by max_blocks: the oldest entry is gone and its blocks are free.
    assert not cache.contains((None, (1,), None, 0, ()))
    assert cache.num_blocks == 4
    assert pool.get_num_free_blocks() == free_before - 4
    cache.clear()
    assert cache.num_blocks == 0 and not cache.by_parent
    assert pool.get_num_free_blocks() == free_before


def _request(boundaries):
    return Request(
        request_id="r",
        prompt_token_ids=list(range(1000)),
        sampling_params=SamplingParams(
            max_tokens=4, extra_args={"prefix_cache_boundaries": boundaries}
        ),
        pooling_params=None,
    )


def test_request_boundaries_align_to_recurrent_chunks():
    align = PREFIX_CACHE_BOUNDARY_ALIGNMENT
    request = _request([align - 1, align + 5, 3 * align, 999, 1000, 5000])
    assert request.prefix_cache_boundaries == (align, 3 * align, 999 // align * align)


class _Tokenizer:
    """ChatML-like template: one token per role marker and content word."""

    def apply_chat_template(
        self,
        messages,
        tools=None,
        chat_template=None,
        add_generation_prompt=False,
        tokenize=True,
        **kwargs,
    ):
        ids = []
        if tools:
            ids += [900] * len(tools)
        for message in messages:
            ids += [100, len(message["role"])]
            ids += [len(word) for word in message["content"].split()]
            ids += [101]
        if add_generation_prompt:
            ids += [100, 9] + ([7, 7] if kwargs.get("enable_thinking") else [8])
        return ids


@pytest.mark.parametrize("thinking", [False, True])
def test_chat_boundaries_end_messages_and_system_prompt(thinking):
    tokenizer = _Tokenizer()
    messages = [
        {"role": "system", "content": "be brief please"},
        {"role": "user", "content": "hello there"},
    ]
    kwargs = {"enable_thinking": thinking, "add_generation_prompt": True}
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=thinking
    )
    request = SimpleNamespace(messages=messages, tools=None)
    boundaries = _PrefixCacheBoundaries()(tokenizer, request, prompt, None, kwargs)
    system_end = len(tokenizer.apply_chat_template(messages[:1]))
    last_message_end = len(tokenizer.apply_chat_template(messages))
    assert boundaries == [system_end, last_message_end]


def test_chat_boundaries_skip_prompts_that_do_not_match():
    tokenizer = _Tokenizer()
    request = SimpleNamespace(messages=[{"role": "user", "content": "hi"}], tools=None)
    assert _PrefixCacheBoundaries()(tokenizer, request, [1, 2, 3], None, {}) == []


def _image(identifier, offset, length):
    return SimpleNamespace(
        identifier=identifier,
        mm_position=SimpleNamespace(offset=offset, length=length),
    )


def test_tail_mm_items_only_reports_items_overlapping_the_tail():
    features = [_image("a", 0, 4), _image("b", 6, 4), _image("c", 20, 4)]
    assert tail_mm_items(features, 4, 12) == (("b", 2),)
    assert tail_mm_items(features, 2, 8) == (("a", -2), ("b", 4))
    assert tail_mm_items(None, 0, 8) == ()


def test_same_placeholder_tokens_with_another_image_do_not_hit():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=4)
    cache = PartialPrefixCache(pool, max_blocks=8)
    tokens = [5, 9, 9, 9, 6]
    tail = tuple(tokens)
    first = [_image("first", 1, 3)]
    key = (None, tail, None, 0, tail_mm_items(first, 0, len(tail)))
    cache.insert(_entry(pool, key, len(tail), 1))
    assert cache.lookup(None, tokens, 0, 5, None, 0, 2, first) is not None
    assert (
        cache.lookup(None, tokens, 0, 5, None, 0, 2, [_image("second", 1, 3)]) is None
    )
    assert cache.lookup(None, tokens, 0, 5, None, 0, 2) is None


def test_snapshot_is_skipped_when_the_pool_is_exhausted():
    # #1483: an unpartitioned pool passes the partition-quota check, so the
    # snapshot copy must also check aggregate free blocks instead of raising.
    import torch

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        12, [], [KVCacheGroupSpec(["full"], full), KVCacheGroupSpec(["m"], mamba)]
    )
    cm = KVCacheManager(config, 1024, 16, partial_prefix_cache=True)
    assert cm.partial_cache is not None
    request = SimpleNamespace(
        request_id="r",
        num_preemptions=0,
        is_finished=lambda: False,
        all_token_ids=list(range(40)),
        block_hashes=[BlockHash(bytes([i])) for i in range(2)],
        cache_salt=None,
        lora_request=None,
        mm_features=[],
    )
    for manager in cm.coordinator.single_type_managers:
        manager.allocate_new_blocks("r", 40, 40)
    leftover = cm.block_pool.get_new_blocks(cm.block_pool.get_num_free_blocks())
    cm._pending_snapshots.append((request, 40, 0))

    cm._materialize_snapshots()

    assert not cm.partial_cache.entries
    assert not cm.take_block_copies()
    cm.block_pool.free_blocks(leftover)


class _StrictUserQueryTokenizer(_Tokenizer):
    """Simulates Qwen3.5/Qwen3-Coder templates that reject query-less renders."""

    def apply_chat_template(
        self,
        messages,
        tools=None,
        chat_template=None,
        add_generation_prompt=False,
        tokenize=True,
        **kwargs,
    ):
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("No user query found in messages.")
        return super().apply_chat_template(
            messages,
            tools=tools,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )


def test_chat_boundaries_probe_system_prompt_when_single_system_raises():
    tokenizer = _StrictUserQueryTokenizer()
    tool_obj = SimpleNamespace(
        model_dump=lambda exclude_none=True: {
            "type": "function",
            "function": {"name": "read_file"},
        }
    )
    messages_a = [
        {"role": "system", "content": "you are a coding agent"},
        {"role": "user", "content": "inspect main file"},
    ]
    messages_b = [
        {"role": "system", "content": "you are a coding agent"},
        {"role": "user", "content": "run unit suite"},
    ]
    kwargs = {"add_generation_prompt": True}
    prompt_a = tokenizer.apply_chat_template(
        messages_a, tools=[{"name": "read_file"}], add_generation_prompt=True
    )
    prompt_b = tokenizer.apply_chat_template(
        messages_b, tools=[{"name": "read_file"}], add_generation_prompt=True
    )
    req_a = SimpleNamespace(messages=messages_a, tools=[tool_obj])
    req_b = SimpleNamespace(messages=messages_b, tools=[tool_obj])
    finder = _PrefixCacheBoundaries()
    bounds_a = finder(tokenizer, req_a, prompt_a, None, kwargs)
    bounds_b = finder(tokenizer, req_b, prompt_b, None, kwargs)
    assert len(bounds_a) == 2 and len(bounds_b) == 2
    sys_bound = bounds_a[0]
    assert bounds_b[0] == sys_bound
    assert prompt_a[:sys_bound] == prompt_b[:sys_bound]


def test_circular_buffer_group_enables_partial_prefix_cache_and_cow():
    import torch

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    full = FullAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=4, dtype=torch.float16
    )
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=4, head_size_v=0, dtype=torch.float16
    )
    swa = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float16,
        sliding_window=128,
    )
    mamba = MambaSpec(
        block_size=128,
        shapes=((4,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        64,
        [],
        [
            KVCacheGroupSpec(["qsa"], full),
            KVCacheGroupSpec(["ring"], ring),
            KVCacheGroupSpec(["swa"], swa),
            KVCacheGroupSpec(["gdn"], mamba),
        ],
    )
    cm = KVCacheManager(config, 1024, 128, partial_prefix_cache=True)
    assert cm.partial_cache is not None
    free_initial = cm.block_pool.get_num_free_blocks()

    # Boundary 64 is 64-aligned and divisible by ring.block_size=8, but NOT by
    # partial_block_size=128. CircularBufferManager MUST still snapshot slot 0
    # and copy-on-write on hit!
    def _hasher(req):
        return [
            BlockHash(bytes(x % 256 for x in req.all_token_ids[i : i + 128]))
            for i in range(0, len(req.all_token_ids) // 128 * 128, 128)
        ]

    r1 = Request(
        request_id="r1",
        prompt_token_ids=list(range(100, 180)),
        sampling_params=SamplingParams(
            max_tokens=4, extra_args={"prefix_cache_boundaries": [64]}
        ),
        pooling_params=None,
        block_hasher=_hasher,
    )
    cm.new_step_starts()
    stops = cm.partial_prefix_stops(r1, 0, 80)
    assert stops == (64,)
    assert cm.allocate_slots(r1, 64) is not None
    r1.num_computed_tokens = 64

    # Next step materializes the snapshot at 64 across all 4 groups (0, 1, 2, 3).
    cm.new_step_starts()
    snap_copies, snap_groups = cm.take_block_copies_with_groups()
    assert len(snap_copies) == 4
    assert snap_groups == [0, 1, 2, 3]
    entry = next(iter(cm.partial_cache.entries.values()))
    assert entry.length == 64
    assert entry.copy_last == [True, True, True, True]

    assert cm.allocate_slots(r1, 16) is not None
    r1.num_computed_tokens = 80
    cm.free(r1)

    # Advance step so entry is ready, then hit with r2 sharing first 64 tokens.
    cm.new_step_starts()
    r2 = Request(
        request_id="r2",
        prompt_token_ids=list(range(100, 164)) + [200, 201, 202, 203],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=_hasher,
    )
    comp_blocks, num_hit = cm.get_computed_blocks(r2)
    assert num_hit == 64
    assert cm.allocate_slots(r2, 4, num_hit, comp_blocks) is not None
    cow_copies, cow_groups = cm.take_block_copies_with_groups()
    assert len(cow_copies) == 4
    assert cow_groups == [0, 1, 2, 3]
    # Ring group's active block is a private CoW copy, not the cache entry's block.
    ring_mgr = cm.coordinator.single_type_managers[1]
    assert len(ring_mgr.req_to_blocks["r2"]) == 1
    assert ring_mgr.req_to_blocks["r2"][0] is not entry.blocks[1][0]

    cm.free(r2)
    for _ in range(4):
        cm.new_step_starts()
    cm.reset_prefix_cache()
    assert cm.block_pool.get_num_free_blocks() == free_initial


def test_chat_boundaries_include_multi_turn_user_fork_boundary():
    tokenizer = _StrictUserQueryTokenizer()
    shared_history = [
        {"role": "system", "content": "you are a coding agent"},
        {"role": "user", "content": "read server module"},
        {"role": "assistant", "content": "calling read tool now"},
    ]
    msg_a = shared_history + [{"role": "user", "content": "audit scheduler loop"}]
    msg_b = shared_history + [{"role": "user", "content": "audit cache eviction"}]
    kwargs = {"add_generation_prompt": True}
    prompt_a = tokenizer.apply_chat_template(msg_a, add_generation_prompt=True)
    prompt_b = tokenizer.apply_chat_template(msg_b, add_generation_prompt=True)
    finder = _PrefixCacheBoundaries()
    bounds_a = finder(
        tokenizer, SimpleNamespace(messages=msg_a, tools=None), prompt_a, None, kwargs
    )
    bounds_b = finder(
        tokenizer, SimpleNamespace(messages=msg_b, tools=None), prompt_b, None, kwargs
    )
    fork_end = len(tokenizer.apply_chat_template(shared_history))
    assert fork_end in bounds_a
    assert fork_end in bounds_b
    assert prompt_a[:fork_end] == prompt_b[:fork_end]


def test_partial_prefix_hit_reclaims_older_lru_entry_under_tight_pool():
    import torch

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    full = FullAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=4, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=128,
        shapes=((4,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    # 12 blocks total (1 null + 11 usable): two 2-block partial entries + 6 held
    # blocks leave only 1 free block, so CoW hit needs to evict the older LRU!
    config = KVCacheConfig(
        12, [], [KVCacheGroupSpec(["qsa"], full), KVCacheGroupSpec(["gdn"], mamba)]
    )
    cm = KVCacheManager(config, 1024, 128, partial_prefix_cache=True)
    assert cm.partial_cache is not None

    def _hasher(_req):
        return []

    for idx, base in enumerate((100, 300)):
        req = Request(
            request_id=f"seed_{idx}",
            prompt_token_ids=list(range(base, base + 80)),
            sampling_params=SamplingParams(
                max_tokens=4, extra_args={"prefix_cache_boundaries": [64]}
            ),
            pooling_params=None,
            block_hasher=_hasher,
        )
        cm.new_step_starts()
        assert cm.allocate_slots(req, 64) is not None
        req.num_computed_tokens = 64
        cm.new_step_starts()
        cm.take_block_copies_with_groups()
        assert cm.allocate_slots(req, 16) is not None
        req.num_computed_tokens = 80
        cm.free(req)

    for _ in range(4):
        cm.new_step_starts()
    assert len(cm.partial_cache.entries) == 2

    # Leave only 1 free block (not enough for 2-group CoW without evicting seed_0).
    held = cm.block_pool.get_new_blocks(cm.block_pool.get_num_free_blocks() - 1)
    probe = Request(
        request_id="probe",
        prompt_token_ids=list(range(300, 364)) + [901, 902],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=_hasher,
    )
    _, hit_tokens = cm.get_computed_blocks(probe)
    assert hit_tokens == 64
    assert len(cm.partial_cache.entries) == 1
    cm.block_pool.free_blocks(held)
