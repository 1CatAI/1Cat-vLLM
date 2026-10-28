# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.serving import _PrefixCacheBoundaries
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.partial_prefix_cache import PartialEntry, PartialPrefixCache
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
    cache.insert(_entry(pool, (parent, (1, 2), None, 0), 6, ready_step=1))
    cache.insert(_entry(pool, (parent, (1, 2, 3), None, 0), 7, ready_step=1))
    cache.insert(_entry(pool, (parent, (1, 9), None, 0), 6, ready_step=1))
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
    cache.insert(_entry(pool, (None, (5,), None, 0), 1, ready_step=3))
    assert cache.lookup(None, [5, 6], 0, 1, None, 0, current_step=3) is None
    assert cache.lookup(None, [5, 6], 0, 1, None, 0, current_step=4) is not None


def test_eviction_releases_block_references():
    pool = BlockPool(num_gpu_blocks=9, enable_caching=True, hash_block_size=4)
    free_before = pool.get_num_free_blocks()
    cache = PartialPrefixCache(block_pool=pool, max_blocks=4)
    cache.insert(_entry(pool, (None, (1,), None, 0), 1, ready_step=0))
    cache.insert(_entry(pool, (None, (2,), None, 0), 1, ready_step=0))
    cache.insert(_entry(pool, (None, (3,), None, 0), 1, ready_step=0))
    # Bounded by max_blocks: the oldest entry is gone and its blocks are free.
    assert not cache.contains((None, (1,), None, 0))
    assert cache.num_blocks == 4
    assert pool.get_num_free_blocks() == free_before - 4
    cache.clear()
    assert cache.num_blocks == 0 and not cache.by_parent
    assert pool.get_num_free_blocks() == free_before


def _request(boundaries, reasoning_ended=None):
    return Request(
        request_id="r",
        prompt_token_ids=list(range(1000)),
        sampling_params=SamplingParams(
            max_tokens=4, extra_args={"prefix_cache_boundaries": boundaries}
        ),
        pooling_params=None,
        reasoning_ended=reasoning_ended,
    )


def test_request_boundaries_align_to_recurrent_chunks():
    align = PREFIX_CACHE_BOUNDARY_ALIGNMENT
    request = _request([align - 1, align + 5, 3 * align, 999, 1000, 5000])
    assert request.prefix_cache_boundaries == (align, 3 * align, 999 // align * align)


def test_reasoning_requests_do_not_cache_output_tokens():
    assert not _request([], reasoning_ended=False).cache_output_tokens
    assert _request([], reasoning_ended=True).cache_output_tokens
    assert _request([]).cache_output_tokens


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
