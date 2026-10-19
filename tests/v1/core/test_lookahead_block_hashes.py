# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookahead block hashes for EAGLE-like drafters (port of vllm#61589).

The EAGLE/MTP draft KV at a block's last position depends on the token after
the block. Instead of dropping the last matched block on every hit, each hash
also covers that token, so a block is reused only when it is proven equal.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import VllmConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.distributed.kv_events import BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256, sha256_cbor
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    is_lookahead_block_hashing_enabled,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .utils import create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    use_lookahead_hashes: bool,
    hash_fn=sha256,
    max_tokens: int = 17,
) -> Request:
    sampling_params = SamplingParams(max_tokens=max_tokens)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(
            block_size, hash_fn, use_lookahead_hashes
        ),
    )


def full_attention_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def hybrid_mamba_config(
    hash_block_size: int,
    mamba_block_size: int,
    num_blocks: int,
    num_speculative_blocks: int = 0,
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_attention"],
                FullAttentionSpec(
                    block_size=hash_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=num_speculative_blocks,
                ),
            ),
        ],
    )


@pytest.mark.parametrize(
    ("lookahead", "num_prompt_tokens", "expected_blocks"),
    [
        # Legacy drop: pop the logits block, then drop one more.
        (False, 48, 1),
        (False, 37, 1),
        # Lookahead hashes prove the boundary, so no extra block is dropped.
        (True, 48, 2),
        (True, 37, 2),
    ],
)
def test_eagle_identical_prompt_hit(
    lookahead: bool, num_prompt_tokens: int, expected_blocks: int
):
    block_size = 16
    manager = KVCacheManager(
        full_attention_config(block_size, num_blocks=10),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=not lookahead,
        use_lookahead_block_hashes=lookahead,
        hash_block_size=block_size,
    )
    token_ids = [0] * num_prompt_tokens

    prime = make_request("prime", token_ids, block_size, lookahead)
    computed_blocks, _ = manager.get_computed_blocks(prime)
    manager.allocate_slots(
        prime,
        len(token_ids),
        len(computed_blocks.blocks[0]) * block_size,
        computed_blocks,
    )
    manager.free(prime)

    resend = make_request("resend", token_ids, block_size, lookahead)
    computed_blocks, num_tokens = manager.get_computed_blocks(resend)
    assert len(computed_blocks.blocks[0]) == expected_blocks
    assert num_tokens == expected_blocks * block_size


def test_lookahead_token_controls_last_block_hit():
    block_size = 2
    manager = KVCacheManager(
        full_attention_config(block_size, num_blocks=20),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=False,
        use_lookahead_block_hashes=True,
        hash_block_size=block_size,
    )

    first = make_request("first", [0, 1, 2, 3, 4, 5], block_size, True)
    computed_blocks, _ = manager.get_computed_blocks(first)
    manager.allocate_slots(first, first.num_tokens, 0, computed_blocks)
    manager.free(first)

    # Block [2, 4) was cached with lookahead token 4.
    same_lookahead = make_request("same", [0, 1, 2, 3, 4, 6], block_size, True)
    _, num_tokens = manager.get_computed_blocks(same_lookahead)
    assert num_tokens == 2 * block_size

    different_lookahead = make_request(
        "different", [0, 1, 2, 3, 7, 6], block_size, True
    )
    _, num_tokens = manager.get_computed_blocks(different_lookahead)
    assert num_tokens == block_size


def test_kv_events_reconstruct_lookahead_block_hashes():
    block_size = 2
    manager = KVCacheManager(
        full_attention_config(block_size, num_blocks=20),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=False,
        use_lookahead_block_hashes=True,
        hash_block_size=block_size,
        enable_kv_cache_events=True,
    )
    request = make_request("request", [0, 1, 2, 3, 4], block_size, True)
    computed_blocks, _ = manager.get_computed_blocks(request)
    manager.allocate_slots(request, request.num_tokens, 0, computed_blocks)

    events = manager.take_events()

    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].block_hashes == [
        kv_cache_utils.maybe_convert_block_hash(block_hash)
        for block_hash in request.block_hashes
    ]
    assert events[0].extra_keys is not None
    reconstructed_hashes = []
    parent_hash = events[0].parent_block_hash
    for block_start, extra_keys in zip(
        range(0, len(events[0].token_ids), block_size),
        events[0].extra_keys,
        strict=True,
    ):
        parent_hash = kv_cache_utils.hash_block_tokens(
            sha256,
            parent_hash,
            events[0].token_ids[block_start : block_start + block_size],
            extra_keys,
        )
        reconstructed_hashes.append(parent_hash)
    assert reconstructed_hashes == request.block_hashes


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_lookahead_hasher_waits_for_the_next_token(hash_fn):
    """A lookahead hash is emitted once the token after its block exists, it
    covers that token, and appending tokens one by one yields the same hashes
    as hashing the full sequence at once."""
    init_none_hash(hash_fn)
    block_size = 4
    incremental = make_request(
        "incremental", list(range(8)), block_size, True, hash_fn=hash_fn
    )
    # Tokens 0..7: block [4, 8) waits for token 8.
    assert len(incremental.block_hashes) == 1
    incremental.append_output_token_ids(8)
    assert len(incremental.block_hashes) == 2

    expected_second_hash = hash_fn(
        (incremental.block_hashes[0], (4, 5, 6, 7), (None, 8, None))
    )
    assert incremental.block_hashes[1] == expected_second_hash

    fresh = make_request("fresh", list(range(9)), block_size, True, hash_fn=hash_fn)
    assert incremental.block_hashes == fresh.block_hashes

    plain = make_request("plain", list(range(9)), block_size, False, hash_fn=hash_fn)
    assert plain.block_hashes[0] != fresh.block_hashes[0]


@pytest.mark.parametrize(
    ("use_eagle", "lookahead", "expected_hit"),
    [
        # No speculative decoding: the last aligned boundary below the final token.
        (False, False, 1088),
        # Legacy EAGLE drop: one alignment lower.
        (True, False, 544),
        # Lookahead hashes: as without speculative decoding.
        (False, True, 1088),
    ],
)
def test_eagle_hybrid_mamba_hits_partial_prompt_boundary(
    use_eagle: bool, lookahead: bool, expected_hit: int
):
    """A resend of a 1249-token prompt on a hybrid model with 544-token Mamba
    alignment. The prefill is chunked at the aligned boundaries, as the
    scheduler does in Mamba ``align`` mode, so a state exists at 544 and 1088.
    Upstream reaches 1248 through intra-block prefill checkpoints this tree
    does not have; here hits are multiples of the alignment."""
    hash_block_size = 16
    mamba_block_size = 544
    token_ids = list(range(1249))
    manager = KVCacheManager(
        hybrid_mamba_config(hash_block_size, mamba_block_size, num_blocks=200),
        max_model_len=8192,
        enable_caching=True,
        use_eagle=use_eagle,
        use_lookahead_block_hashes=lookahead,
        hash_block_size=hash_block_size,
    )

    first = make_request("first", token_ids, hash_block_size, lookahead)
    computed_blocks, num_computed = manager.get_computed_blocks(first)
    manager.allocate_slots(first, 544, num_computed, computed_blocks)
    first.num_computed_tokens = 544
    manager.new_step_starts()
    for chunk_end in (1088, 1248, 1249):
        manager.allocate_slots(first, chunk_end - first.num_computed_tokens)
        first.num_computed_tokens = chunk_end
        manager.new_step_starts()
    manager.free(first)

    second = make_request("second", token_ids, hash_block_size, lookahead)
    _, num_computed = manager.get_computed_blocks(second)

    assert num_computed == expected_hit


def test_hybrid_mamba_retention_lookahead_keeps_last_prompt_boundary():
    """Lookahead block hashes need no EAGLE drop, so sparse retention keeps the
    Mamba state at the prompt's last boundary below ``num_tokens - 1`` rather
    than one alignment unit lower; otherwise a resend can only hit the lower
    state."""
    block_size = 32
    manager = KVCacheManager(
        hybrid_mamba_config(
            block_size, block_size, num_blocks=100, num_speculative_blocks=1
        ),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        prefix_cache_retention_interval=0,
        use_eagle=False,
        use_lookahead_block_hashes=True,
    )

    # 129 tokens: the last boundary below the logits token is 128.
    token_ids = [i for i in range(4) for _ in range(block_size)] + [4]
    req0 = make_request("0", token_ids, block_size, True)
    for chunk_end in (32, 64, 96, 128, 129):
        blocks = manager.allocate_slots(
            req0,
            chunk_end - req0.num_computed_tokens,
            num_lookahead_tokens=1,
        )
        assert blocks is not None
        req0.num_computed_tokens = chunk_end

    pool = manager.block_pool
    for i in range(4):
        cached = pool.get_cached_block(req0.block_hashes[i], kv_cache_group_ids=[1])
        if i == 3:
            assert cached is not None, "mamba state at 128 should be retained"
        else:
            assert cached is None, f"mamba hash {i} should not be cached"
    manager.free(req0)

    req1 = make_request("1", token_ids, block_size, True)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert num_computed_tokens == 4 * block_size
    assert [len(blocks) for blocks in computed_blocks.blocks] == [4, 4]


def _speculative_config(
    method: str,
    disable_lookahead_block_hashes: bool = False,
    num_mtp_layers: int = 1,
) -> SpeculativeConfig:
    speculative_config = object.__new__(SpeculativeConfig)
    object.__setattr__(speculative_config, "method", method)
    object.__setattr__(
        speculative_config,
        "disable_lookahead_block_hashes",
        disable_lookahead_block_hashes,
    )
    object.__setattr__(speculative_config, "num_speculative_tokens", 3)
    object.__setattr__(
        speculative_config,
        "draft_model_config",
        SimpleNamespace(
            hf_config=SimpleNamespace(num_nextn_predict_layers=num_mtp_layers)
        ),
    )
    return speculative_config


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        *(
            pytest.param({"method": method}, True, id=method)
            for method in ("eagle", "eagle3", "mtp", "dspark")
        ),
        # DFlash on the V2 runner keeps no trailing-block drop to begin with.
        pytest.param({"method": "dflash"}, False, id="dflash"),
        pytest.param({"method": "ngram"}, False, id="not-eagle"),
        pytest.param({"enable_prefix_caching": False}, False, id="no-prefix-caching"),
        pytest.param({"num_mtp_layers": 2}, False, id="multi-module-mtp"),
        pytest.param(
            {"disable_lookahead_block_hashes": True}, False, id="disabled-by-config"
        ),
        pytest.param({"kv_connector": True}, False, id="kv-connector"),
    ],
)
def test_lookahead_block_hashing_enabled(overrides: dict[str, Any], expected: bool):
    vllm_config = cast(
        VllmConfig,
        SimpleNamespace(
            cache_config=SimpleNamespace(
                enable_prefix_caching=overrides.get("enable_prefix_caching", True)
            ),
            speculative_config=_speculative_config(
                overrides.get("method", "mtp"),
                overrides.get("disable_lookahead_block_hashes", False),
                overrides.get("num_mtp_layers", 1),
            ),
            kv_transfer_config=object() if overrides.get("kv_connector") else None,
        ),
    )

    assert is_lookahead_block_hashing_enabled(vllm_config) == expected


def _enable_lookahead_block_hashes(scheduler) -> None:
    scheduler.use_lookahead_block_hashes = True
    scheduler.kv_cache_manager.block_pool.use_lookahead_block_hashes = True


def _scheduler_request(
    request_id: str, prompt_token_ids: list[int], block_size: int, max_tokens: int
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256, True),
    )


def test_scheduler_publishes_lookahead_blocks_at_allocation():
    """A lookahead block is published once its lookahead token exists: prompt
    blocks at their own allocation, the block ending at the first sampled
    token at the allocation after it is sampled."""
    block_size = 2
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=block_size,
        max_num_batched_tokens=16,
    )
    _enable_lookahead_block_hashes(scheduler)

    def num_cached_tokens(prompt_token_ids: list[int]) -> int:
        probe = _scheduler_request("probe", prompt_token_ids, block_size, 3)
        return scheduler.kv_cache_manager.get_computed_blocks(probe)[1]

    request = _scheduler_request("first", [0, 1, 2, 3], block_size, 3)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    # Block [2, 4) waits for its lookahead token, the first sampled one.
    assert num_cached_tokens([0, 1, 2, 3, 5, 7]) == block_size

    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[5]],
        ),
    )
    scheduler.schedule()
    assert num_cached_tokens([0, 1, 2, 3, 5, 7]) == 2 * block_size
    assert num_cached_tokens([0, 1, 2, 3, 6, 7]) == block_size


def test_finished_request_publishes_block_ending_at_last_sampled_token():
    """The last sampled token is never computed, but it is the lookahead token
    of the block ending right before it, whose draft KV the finishing step
    wrote. That block must be published before the request is freed."""
    block_size = 2
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=block_size,
        max_num_batched_tokens=16,
    )
    _enable_lookahead_block_hashes(scheduler)
    request = _scheduler_request("request", [0, 1, 2], block_size, 2)
    scheduler.add_request(request)

    for token_id in (3, 4):
        scheduler_output = scheduler.schedule()
        scheduler.update_from_output(
            scheduler_output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[token_id]],
            ),
        )
    assert request.is_finished()
    # Tokens 0-3 were computed; token 4 is the lookahead of block [2, 4).
    assert request.num_computed_tokens == 4

    probe = _scheduler_request("probe", [0, 1, 2, 3, 4, 5], block_size, 1)
    _, num_tokens = scheduler.kv_cache_manager.get_computed_blocks(probe)
    assert num_tokens == 4
