# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.serving import (
    _AssistantReasoningCache,
    _PrefixCacheBoundaries,
)
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


def test_assistant_reasoning_cache_restores_omitted_reasoning_across_turns():
    cache = _AssistantReasoningCache(max_entries=16)
    prior_msgs = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "inspect src/wal.py"},
    ]
    # 1. Tool-calling turn recorded with compact JSON arguments, queried with
    # re-formatted JSON arguments and omitted reasoning_content.
    cache.record(
        prior_msgs,
        "",
        [("read_file", '{"path":"src/wal.py","start":1}')],
        "Need to inspect WAL header invariants first.",
        cache_salt="session-a",
        reasoning_ended=True,
    )
    restored = cache.lookup(
        prior_msgs,
        "",
        [
            {
                "id": "call_new_id",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{ "start": 1, "path": "src/wal.py" }',
                },
            }
        ],
        cache_salt="session-a",
    )
    assert restored == "Need to inspect WAL header invariants first."

    # 2. Identical tool call in a different conversation prefix or different
    # cache_salt MUST NOT leak reasoning across turns or sessions (no any_key).
    other_prior_same_tail = [
        {"role": "system", "content": "different repo"},
        {"role": "user", "content": "fix auth bug"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "ls", "arguments": "{}"}}],
        },
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "inspect src/wal.py"},
    ]
    same_tc = [
        {
            "id": "call_other",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"src/wal.py","start":1}',
            },
        }
    ]
    assert (
        cache.lookup(other_prior_same_tail, "", same_tc, cache_salt="session-a") is None
    )
    assert cache.lookup(prior_msgs, "", same_tc, cache_salt="session-b") is None

    # 3. Prior assistant tool_calls participate in context digest so distinct
    # tool trajectories with identical tool outputs do not collide.
    traj_a = prior_msgs + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "not found"},
    ]
    traj_b = prior_msgs + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path":"b.py"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "not found"},
    ]
    cache.record(
        traj_a, "File a.py is missing.", [], "Checked a.py.", cache_salt="session-a"
    )
    assert (
        cache.lookup(traj_a, "File a.py is missing.", [], cache_salt="session-a")
        == "Checked a.py."
    )
    assert (
        cache.lookup(traj_b, "File a.py is missing.", [], cache_salt="session-a")
        is None
    )

    # 4. Length-truncated unfinished thinking turns (reasoning_ended=False or
    # empty content + empty tool_calls) are never cached or fabricated as closed
    # </think> blocks on subsequent turns.
    cache.record(
        prior_msgs,
        "",
        [],
        "Partial truncated reasoning cut off mid-",
        cache_salt="session-a",
        reasoning_ended=False,
    )
    assert cache.lookup(prior_msgs, "", [], cache_salt="session-a") is None
    cache.record(
        prior_msgs,
        "",
        [],
        "Empty output turn",
        cache_salt="session-a",
        reasoning_ended=True,
    )
    assert cache.lookup(prior_msgs, "", [], cache_salt="session-a") is None


def test_decode_ckpt_promotes_64_aligned_output_and_coalesces_warm_tail():
    import torch

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.request import RequestStatus

    full = FullAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=4, dtype=torch.float16
    )
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=4, head_size_v=0, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=256,
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
            KVCacheGroupSpec(["gdn"], mamba),
        ],
    )
    cm = KVCacheManager(
        config,
        2048,
        256,
        partial_prefix_cache=True,
        enable_decode_ckpt=True,
    )
    assert cm.partial_cache is not None
    assert cm._enable_decode_ckpt is True
    free_initial = cm.block_pool.get_num_free_blocks()

    def _hasher(req):
        return [
            BlockHash(bytes(x % 256 for x in req.all_token_ids[i : i + 256]))
            for i in range(0, len(req.all_token_ids) // 256 * 256, 256)
        ]

    # Turn 1: prompt=90 tokens, generates 60 decode tokens -> total=150 -> B64=128.
    t1 = Request(
        request_id="t1",
        prompt_token_ids=list(range(10, 100)),
        sampling_params=SamplingParams(max_tokens=128),
        pooling_params=None,
        block_hasher=_hasher,
    )
    cm.new_step_starts()
    assert cm.allocate_slots(t1, 90) is not None
    assert cm.take_decode_ckpt_mounts() == []
    t1.num_computed_tokens = 90
    t1.append_output_token_ids([500])

    # First decode step mounts ping-pong Mamba checkpoint blocks.
    cm.new_step_starts()
    assert cm.allocate_slots(t1, 1) is not None
    mounts = cm.take_decode_ckpt_mounts()
    assert len(mounts) == 1
    req_id, even_ids, odd_ids = mounts[0]
    assert req_id == "t1" and len(even_ids) == 1 and len(odd_ids) == 1
    even_mamba_id = even_ids[0]  # 128 // 64 == 2 -> parity 0 (even)

    # Simulate decoding past 128 up to 150 tokens and finishing cleanly.
    t1.append_output_token_ids(list(range(501, 560)))
    t1.num_computed_tokens = 149
    t1.status = RequestStatus.FINISHED_STOPPED
    cm.free(t1)

    # Immediate next step (c=1 sequential multi-turn agent): Turn 2 arrives and
    # MUST hit 128 (including 38 decode-generated tokens!) in the SAME step with
    # zero (A->B, B->C) copy chains, and coalesce its near-end gen-prompt tail!
    cm.new_step_starts()
    t2_tokens = list(t1.all_token_ids[:149]) + list(range(700, 740))
    t2 = Request(
        request_id="t2",
        prompt_token_ids=t2_tokens,
        sampling_params=SamplingParams(
            max_tokens=128,
            extra_args={"prefix_cache_boundaries": [len(t2_tokens) - 10]},
        ),
        pooling_params=None,
        block_hasher=_hasher,
    )
    comp_blocks, hit = cm.get_computed_blocks(t2)
    assert hit == 128
    # Warm continuation skips the 10-token generation-prompt split -> 1 prefill step!
    assert cm.partial_prefix_stops(t2, hit, len(t2_tokens)) == ()
    assert cm.allocate_slots(t2, len(t2_tokens) - hit, hit, comp_blocks) is not None
    copies, _ = cm.take_block_copies_with_groups()
    dsts = {dst for _, dst in copies}
    srcs = {src for src, _ in copies}
    assert srcs.isdisjoint(dsts), (
        f"Unexpected chained block copy in same step: {copies}"
    )
    entry = next(iter(cm.partial_cache.entries.values()))
    assert entry.length == 128
    assert entry.blocks[2][0].block_id == even_mamba_id

    # Also verify client abort (FINISHED_ABORTED) promotes odd parity B64=192,
    # while preemption (RUNNING) does NOT promote and leaks zero blocks.
    t2.num_computed_tokens = len(t2_tokens)
    t2.append_output_token_ids([800])
    cm.new_step_starts()
    assert cm.allocate_slots(t2, 1) is not None
    _, _, odd_ids_2 = cm.take_decode_ckpt_mounts()[0]
    t2.append_output_token_ids(list(range(801, 830)))
    t2.num_computed_tokens = 205
    t2.status = RequestStatus.FINISHED_ABORTED
    cm.free(t2)
    cm.new_step_starts()
    cm.take_block_copies_with_groups()
    lengths = sorted(e.length for e in cm.partial_cache.entries.values())
    assert lengths == [128, 192]
    entry_192 = [e for e in cm.partial_cache.entries.values() if e.length == 192][0]
    assert entry_192.blocks[2][0].block_id == odd_ids_2[0]

    for _ in range(4):
        cm.new_step_starts()
    cm.reset_prefix_cache()
    assert cm.block_pool.get_num_free_blocks() == free_initial


def test_partial_prefix_pins_full_parent_blocks_and_deduplicates_swarm_leaders():
    import torch

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.request import RequestStatus

    full = FullAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=4, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=256,
        shapes=((4,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        32,
        [],
        [
            KVCacheGroupSpec(["qsa"], full),
            KVCacheGroupSpec(["gdn"], mamba),
        ],
    )
    cm = KVCacheManager(
        config,
        2048,
        256,
        partial_prefix_cache=True,
        enable_decode_ckpt=True,
    )
    free_initial = cm.block_pool.get_num_free_blocks()

    def _hasher(req):
        return [
            BlockHash(
                bytes(((x * 31 + i) % 256) for x in req.all_token_ids[i : i + 256])
            )
            for i in range(0, len(req.all_token_ids) // 256 * 256, 256)
        ]

    # Swarm burst: leader is prefilling [0..600) with shared system boundary at 384.
    leader_tokens = list(range(100, 700))
    follower_tokens = list(range(100, 484)) + list(range(900, 1020))
    leader = Request(
        request_id="leader",
        prompt_token_ids=leader_tokens,
        sampling_params=SamplingParams(
            max_tokens=32,
            extra_args={"prefix_cache_boundaries": [384]},
        ),
        pooling_params=None,
        block_hasher=_hasher,
    )
    follower = Request(
        request_id="follower",
        prompt_token_ids=follower_tokens,
        sampling_params=SamplingParams(
            max_tokens=32,
            extra_args={"prefix_cache_boundaries": [384]},
        ),
        pooling_params=None,
        block_hasher=_hasher,
    )

    cm.new_step_starts()
    comp_l, hit_l = cm.get_computed_blocks(leader)
    assert hit_l == 0
    stops = cm.partial_prefix_stops(leader, 0, len(leader_tokens))
    assert stops == (384,)
    assert cm.allocate_slots(leader, 384, 0, comp_l) is not None
    # Follower shares both full block [0..256) and partial boundary 384 -> waits!
    assert cm.should_wait_for_inflight_snapshot(follower, [(leader, 0, 384)]) is True

    # Finish leader's 384-token boundary step -> snapshot is usable immediately
    # in the next step AND pins its FullAttention parent block [0..256)!
    leader.num_computed_tokens = 384
    cm.new_step_starts()
    assert cm.allocate_slots(leader, len(leader_tokens) - 384) is not None
    cm.take_block_copies_with_groups()
    leader.num_computed_tokens = len(leader_tokens)
    leader.status = RequestStatus.FINISHED_STOPPED
    cm.free(leader)

    entry_384 = next(iter(cm.partial_cache.entries.values()))
    assert entry_384.length == 384
    assert len(entry_384.pinned_blocks) == 1
    assert entry_384.pinned_blocks[0].ref_cnt >= 1

    # Allocate and free unrelated traffic to churn free_block_queue, then verify
    # follower STILL hits 384 (not degraded or invalidated due to parent eviction)!
    for k in range(6):
        noise = Request(
            request_id=f"noise-{k}",
            prompt_token_ids=[5000 + k * 500 + j for j in range(300)],
            sampling_params=SamplingParams(max_tokens=8),
            pooling_params=None,
            block_hasher=_hasher,
        )
        cm.new_step_starts()
        assert cm.allocate_slots(noise, 300) is not None
        noise.num_computed_tokens = 300
        noise.status = RequestStatus.FINISHED_STOPPED
        cm.free(noise)

    cm.new_step_starts()
    _, hit_f = cm.get_computed_blocks(follower)
    assert hit_f == 384

    cm.reset_prefix_cache()
    assert cm.block_pool.get_num_free_blocks() == free_initial


def test_short_conv_blocks_zeroed_on_alloc_and_decode_ckpt_mount():
    import torch

    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.request import RequestStatus

    for short_conv_type in (MambaAttentionBackendEnum.SHORT_CONV, "short_conv"):
        full = FullAttentionSpec(
            block_size=256, num_kv_heads=1, head_size=4, dtype=torch.float16
        )
        gdn = MambaSpec(
            block_size=256,
            shapes=((4,), (4, 4)),
            dtypes=(torch.float16, torch.float32),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            mamba_cache_mode="align",
        )
        ple = MambaSpec(
            block_size=256,
            shapes=((13, 16),),
            dtypes=(torch.float16,),
            mamba_type=short_conv_type,
            mamba_cache_mode="align",
        )
        config = KVCacheConfig(
            32,
            [],
            [
                KVCacheGroupSpec(["qsa"], full),
                KVCacheGroupSpec(["gdn"], gdn),
                KVCacheGroupSpec(["ple"], ple),
            ],
        )
        cm = KVCacheManager(
            config,
            2048,
            256,
            partial_prefix_cache=True,
            enable_decode_ckpt=True,
        )
        free_initial = cm.block_pool.get_num_free_blocks()

        req = Request(
            request_id="sc-zero-1",
            prompt_token_ids=list(range(80)),
            sampling_params=SamplingParams(max_tokens=32),
            pooling_params=None,
            block_hasher=lambda _r: [],
        )
        cm.new_step_starts()
        allocated = cm.allocate_slots(req, 80)
        assert allocated is not None
        zero_ids_prefill = set(cm.take_new_block_ids())
        assert allocated.blocks[0][0].block_id in zero_ids_prefill
        assert allocated.blocks[2][0].block_id in zero_ids_prefill
        assert allocated.blocks[1][0].block_id not in zero_ids_prefill

        req.num_computed_tokens = 80
        req.append_output_token_ids([501])
        cm.new_step_starts()
        assert cm.allocate_slots(req, 1) is not None
        mounts = cm.take_decode_ckpt_mounts()
        assert len(mounts) == 1
        _, even_ids, odd_ids = mounts[0]
        zero_ids_decode = set(cm.take_new_block_ids())
        # Group 2 (ple short_conv) is index 1 in _mamba_group_ids=(1, 2).
        assert even_ids[1] in zero_ids_decode
        assert odd_ids[1] in zero_ids_decode
        assert even_ids[0] not in zero_ids_decode

        req.status = RequestStatus.FINISHED_STOPPED
        cm.free(req)
        cm.reset_prefix_cache()
        assert cm.block_pool.get_num_free_blocks() == free_initial


def test_decode_ckpt_duplicate_snapshot_defers_chosen_release():
    import torch

    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.request import RequestStatus

    full = FullAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=4, dtype=torch.float16
    )
    gdn = MambaSpec(
        block_size=256,
        shapes=((4,),),
        dtypes=(torch.float16,),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        32,
        [],
        [
            KVCacheGroupSpec(["qsa"], full),
            KVCacheGroupSpec(["gdn"], gdn),
        ],
    )
    cm = KVCacheManager(
        config,
        2048,
        256,
        partial_prefix_cache=True,
        enable_decode_ckpt=True,
    )
    free_initial = cm.block_pool.get_num_free_blocks()

    prompt = list(range(90))
    out_tail = list(range(900, 950))
    r1 = Request(
        request_id="dup-1",
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=64),
        pooling_params=None,
        block_hasher=lambda _r: [],
    )
    r2 = Request(
        request_id="dup-2",
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=64),
        pooling_params=None,
        block_hasher=lambda _r: [],
    )
    cm.new_step_starts()
    assert cm.allocate_slots(r1, 90) is not None
    assert cm.allocate_slots(r2, 90) is not None
    r1.num_computed_tokens = 90
    r2.num_computed_tokens = 90
    r1.append_output_token_ids([899])
    r2.append_output_token_ids([899])

    cm.new_step_starts()
    assert cm.allocate_slots(r1, 1) is not None
    assert cm.allocate_slots(r2, 1) is not None
    mounts = dict((rid, (ev, od)) for rid, ev, od in cm.take_decode_ckpt_mounts())
    r2_even_blk_id = mounts["dup-2"][0][0]

    r1.append_output_token_ids(out_tail)
    r2.append_output_token_ids(out_tail)
    r1.num_computed_tokens = 140
    r2.num_computed_tokens = 140
    r1.status = RequestStatus.FINISHED_STOPPED
    r2.status = RequestStatus.FINISHED_STOPPED
    cm.free(r1)
    cm.free(r2)

    # Next step materializes dup-1 at boundary=128 and skips dup-2 via cache.contains;
    # dup-2's chosen_mamba block must stay held in _deferred_release for 3 steps.
    cm.new_step_starts()
    r2_even_blk = cm.block_pool.blocks[r2_even_blk_id]
    assert r2_even_blk.ref_cnt >= 1
    cm.new_step_starts()
    assert r2_even_blk.ref_cnt >= 1
    cm.new_step_starts()
    assert r2_even_blk.ref_cnt >= 1
    cm.new_step_starts()
    assert r2_even_blk.ref_cnt == 0

    cm.reset_prefix_cache()
    assert cm.block_pool.get_num_free_blocks() == free_initial


def test_spec_decode_crosses_2048_mamba_block_and_hits_full_and_partial_on_turn2():
    import torch

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.request import RequestStatus

    full = FullAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=4, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=2048,
        shapes=((4,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        64,
        [],
        [
            KVCacheGroupSpec(["qsa"], full),
            KVCacheGroupSpec(["gdn"], mamba),
        ],
    )
    cm = KVCacheManager(
        config,
        8192,
        256,
        prefix_cache_retention_interval=8192,
        partial_prefix_cache=True,
        enable_decode_ckpt=True,
    )
    free_initial = cm.block_pool.get_num_free_blocks()

    def _hasher(req):
        start = len(req.block_hashes) * 256
        return [
            BlockHash(
                bytes(((x * 131 + i) % 256) for x in req.all_token_ids[i : i + 256])
            )
            for i in range(start, len(req.all_token_ids) // 256 * 256, 256)
        ]

    # Turn 1: prompt=1980 (< 2048), speculative decode steps cross 2048 up to 2120.
    prompt_ids = list(range(1000, 2980))
    t1 = Request(
        request_id="spec-t1",
        prompt_token_ids=prompt_ids,
        sampling_params=SamplingParams(
            max_tokens=256,
            extra_args={"prefix_cache_boundaries": [1975]},
        ),
        pooling_params=None,
        block_hasher=_hasher,
    )
    cm.new_step_starts()
    assert cm.allocate_slots(t1, 1980) is not None
    t1.num_computed_tokens = 1980
    t1.append_output_token_ids([5000])

    cur = 1980
    next_tok = 5001
    while cur < 2120:
        cm.new_step_starts()
        draft = list(range(next_tok, next_tok + 7))
        assert cm.allocate_slots(t1, 1 + len(draft), num_lookahead_tokens=len(draft))
        cm.take_decode_ckpt_mounts()
        t1.num_computed_tokens = cur + 8
        t1.append_output_token_ids(list(range(next_tok, next_tok + 8)))
        cur += 8
        next_tok += 8

    t1.status = RequestStatus.FINISHED_STOPPED
    cm.free(t1)

    cm.new_step_starts()
    t2_tokens = list(t1.all_token_ids[:2120]) + list(range(9000, 9160))
    t2 = Request(
        request_id="spec-t2",
        prompt_token_ids=t2_tokens,
        sampling_params=SamplingParams(
            max_tokens=64,
            extra_args={"prefix_cache_boundaries": [len(t2_tokens) - 5]},
        ),
        pooling_params=None,
        block_hasher=_hasher,
    )
    comp_blocks, hit = cm.get_computed_blocks(t2)
    assert hit == 2112
    assert cm.partial_prefix_stops(t2, hit, len(t2_tokens)) == (
        (len(t2_tokens) - 5) // 64 * 64,
    )
    assert cm.allocate_slots(t2, len(t2_tokens) - hit, hit, comp_blocks) is not None
    t2.status = RequestStatus.FINISHED_ABORTED
    cm.free(t2)

    for _ in range(4):
        cm.new_step_starts()
    cm.reset_prefix_cache()
    assert cm.block_pool.get_num_free_blocks() == free_initial


def test_partial_cache_dedups_pinned_blocks_and_protects_active_hit_on_eviction():
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.partial_prefix_cache import PartialEntry, PartialPrefixCache

    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=256)
    cache = PartialPrefixCache(block_pool=pool, max_blocks=20)
    parent_a, parent_b, sys_blk, t0, t1_blk, t2_blk = pool.get_new_blocks(6)

    sys_entry = PartialEntry(
        key=(None, (1, 2, 3), None, 0, ()),
        length=3,
        blocks=((sys_blk,),),
        copy_last=(True,),
        ready_step=0,
        pinned_blocks=(),
    )
    cache.insert(sys_entry)
    sys_entry.hit_count = 1

    turn_entries = []
    for i, tail in enumerate((t0, t1_blk, t2_blk)):
        for p in (parent_a, parent_b):
            p.ref_cnt += 1
        ent = PartialEntry(
            key=(None, tuple(range(100 + i)), None, 0, ()),
            length=100 + i,
            blocks=((tail,),),
            copy_last=(True,),
            ready_step=0,
            pinned_blocks=(parent_a, parent_b),
        )
        cache.insert(ent)
        turn_entries.append(ent)
    # 1 (sys_blk) + 3 (tails) + 2 (deduped parent_a, parent_b) == 6 blocks!
    assert cache.num_blocks == 6

    assert cache.evict_one(protected=turn_entries[0]) is True
    assert sys_entry.key in cache.entries
    assert turn_entries[0].key in cache.entries
    assert turn_entries[1].key not in cache.entries
    assert parent_a.ref_cnt == 3 and parent_b.ref_cnt == 3


def test_agent_prefix_boundaries_and_decoded_span_splicing():
    from vllm.entrypoints.openai.chat_completion.serving import (
        _AssistantReasoningCache,
        _PrefixCacheBoundaries,
    )

    class _MockChatTokenizer:
        chat_template = "Supported types are xhigh, high, medium, low"

        def get_vocab(self):
            return {
                "<|im_start|>": 1,
                "<|im_end|>": 2,
                "<think>": 3,
                "</think>": 4,
            }

        def encode(self, text: str, add_special_tokens: bool = False):
            if text == "\n":
                return [10]
            return [100 + (ord(ch) % 90) for ch in text]

        def apply_chat_template(
            self,
            messages,
            tools=None,
            chat_template=None,
            add_generation_prompt=False,
            tokenize=True,
            **kwargs,
        ):
            if kwargs.get("reasoning_effort") in ("max", "minimal", "none"):
                raise ValueError(f"Unsupported effort: {kwargs['reasoning_effort']}")
            ids: list[int] = []
            if tools:
                ids.extend([50] * 140)
            for msg in messages:
                ids.append(1)
                role = msg.get("role", "")
                ids.extend(self.encode(role))
                ids.append(10)
                if role == "assistant":
                    ids.append(3)
                    ids.append(10)
                    rc = (
                        msg.get("reasoning_content") or msg.get("reasoning") or ""
                    ).strip()
                    if rc:
                        ids.extend(self.encode(rc))
                    ids.append(10)
                    ids.append(4)
                    ids.append(10)
                    ids.append(10)
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            ids.extend(self.encode(part.get("text", "")))
                elif isinstance(content, str):
                    ids.extend(
                        self.encode(content.strip() if role == "assistant" else content)
                    )
                if role == "tool":
                    ids.extend(self.encode(msg.get("tool_call_id", "")))
                ids.append(2)
                ids.append(10)
            if add_generation_prompt:
                ids.extend([1, 101, 10, 3, 10])
            return ids

    tok = _MockChatTokenizer()
    bounds = _PrefixCacheBoundaries()
    rcache = _AssistantReasoningCache()

    sys_msg = {
        "role": "system",
        "content": [
            {"type": "text", "text": "S" * 80, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "D" * 40},
        ],
    }
    user_1 = {"role": "user", "content": [{"type": "text", "text": "Inspect repo"}]}
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    kwargs = {"reasoning_effort": "max"}
    t1_msgs = [sys_msg, user_1]
    t1_prompt = tok.apply_chat_template(
        t1_msgs, tools=tools, add_generation_prompt=True, reasoning_effort="xhigh"
    )
    gen_suffix = bounds._generation_prompt(tok, tok.chat_template, kwargs)
    assert gen_suffix == (1, 101, 10, 3, 10)
    b1 = bounds(
        tok,
        None,
        t1_prompt,
        tok.chat_template,
        kwargs,
        messages=t1_msgs,
        tools=tools,
    )
    assert 141 in b1
    assert 228 in b1
    assert len(t1_prompt) - len(gen_suffix) in b1

    decoded_out = [10] + tok.encode("plan") + [10, 4, 10, 10] + tok.encode("OK") + [2]
    rcache.record(
        t1_msgs,
        "OK",
        [],
        "\nplan",
        prompt_token_ids=t1_prompt,
        output_token_ids=decoded_out,
        gen_suffix=gen_suffix,
        im_end_id=2,
        nl_ids=(10,),
    )

    t2_msgs = [
        sys_msg,
        user_1,
        {"role": "assistant", "content": "OK"},
        {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
    ]
    rerendered_t2 = tok.apply_chat_template(
        [
            sys_msg,
            user_1,
            {"role": "assistant", "content": "OK", "reasoning_content": "\nplan"},
            t2_msgs[3],
        ],
        tools=tools,
        add_generation_prompt=True,
        reasoning_effort="xhigh",
    )
    expected_decode_prefix = list(t1_prompt) + decoded_out + [10]
    assert rerendered_t2[: len(expected_decode_prefix)] != expected_decode_prefix

    spliced_t2 = rcache.splice_prompt_tokens(t2_msgs, rerendered_t2, tok)
    assert spliced_t2[: len(expected_decode_prefix)] == expected_decode_prefix
    b2 = bounds(
        tok,
        None,
        spliced_t2,
        tok.chat_template,
        kwargs,
        messages=t2_msgs,
        tools=tools,
    )
    assert len(expected_decode_prefix) in b2
