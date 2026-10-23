# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache proofs must match known inputs and survive block lifecycle correctly."""

import pytest
import torch

from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.prefill_boundary import prefill_boundary_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
init_none_hash(sha256)


def request(length=71, name="a", *, salt="same", tokens=None):
    return Request(
        name,
        list(range(length)) if tokens is None else tokens,
        SamplingParams(max_tokens=32),
        None,
        cache_salt=salt,
        block_hasher=get_request_block_hasher(8, sha256),
    )


def manager(enabled=True):
    specs = [
        FullAttentionSpec(
            block_size=32, num_kv_heads=1, head_size=1, dtype=torch.float16
        ),
        MambaSpec(
            block_size=32,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=4,
        ),
    ]
    config = KVCacheConfig(
        96, [], [KVCacheGroupSpec([str(i)], spec) for i, spec in enumerate(specs)]
    )
    return KVCacheManager(
        config,
        512,
        8,
        use_eagle=True,
        prefix_cache_retention_interval=0,
        verified_mtp_prefill_cache=enabled,
    )


def prefill(cache, req, chunk=7):
    start = 0
    while start < req.num_tokens:
        end = min(start + chunk, (start // 32 + 1) * 32, req.num_tokens)
        for group in cache.coordinator.single_type_managers:
            group.new_step_starts()
            group.remove_skipped_blocks(req.request_id, start, req.num_prompt_tokens)
            group.allocate_new_blocks(req.request_id, end, end)
        cache.coordinator.cache_blocks(req, end)
        start = end
    cache.coordinator.free(req.request_id)
    for group in cache.coordinator.single_type_managers:
        group.new_step_starts()


@pytest.mark.parametrize("length", [31, 32, 33, 63, 64, 65, 71, 95, 96, 97])
@pytest.mark.parametrize("chunk", [7, 32])
def test_known_prefix_reuses_nearest_checkpoint(length, chunk):
    cache = manager()
    req = request(length)
    prefill(cache, req, chunk)
    blocks, hit = cache.get_computed_blocks(request(length, "b"))
    assert hit == (length - 1) // 32 * 32
    if hit:
        assert not blocks.blocks[1][-1].is_null


def test_disabled_keeps_original_lookbehind():
    cache = manager(False)
    req = request()
    prefill(cache, req)
    assert cache.get_computed_blocks(request(name="b"))[1] == 32


@pytest.mark.parametrize("changed_index", [32, 64])
def test_different_next_token_never_certifies_sibling(changed_index):
    cache = manager()
    prefill(cache, request())
    tokens = list(range(71))
    tokens[changed_index] = 999
    sibling = request(name="sibling", tokens=tokens)
    _, baseline = cache.coordinator.find_longest_cache_hit(sibling.block_hashes, 70)
    assert cache.get_computed_blocks(sibling)[1] == baseline
    assert baseline < 64


def test_missing_proof_keeps_conservative_replay():
    cache = manager()
    prefill(cache, request())
    for block in cache.block_pool.blocks:
        block.prefill_boundary_hash = None
    assert cache.get_computed_blocks(request(name="b"))[1] == 32


def test_eviction_clears_proof_and_blocks_cache_hit():
    cache = manager()
    prefill(cache, request())
    blocks, _ = cache.get_computed_blocks(request(name="b"))
    block = blocks.blocks[1][-1]
    assert block.prefill_boundary_hash is not None
    cache.evict_blocks({block.block_id})
    assert block.prefill_boundary_hash is None
    assert cache.get_computed_blocks(request(name="c"))[1] < 64


def test_salt_isolation_and_prefix_extension():
    cache = manager()
    prefill(cache, request())
    assert cache.get_computed_blocks(request(name="b", salt="different"))[1] == 0
    # Extending within the next block can reuse the same certified boundary.
    assert cache.get_computed_blocks(request(80, "extension"))[1] == 64


def test_generated_tokens_and_multimodal_are_not_certified():
    req = request()
    prefix = req.block_hashes[-1]
    assert prefill_boundary_hash(req, 64, prefix) is not None
    req.append_output_token_ids(list(range(40)))
    assert prefill_boundary_hash(req, 71, prefix) is None
    assert prefill_boundary_hash(req, 96, prefix) is None
    req.mm_features = [object()]
    assert prefill_boundary_hash(req, 64, prefix) is None


def test_metadata_does_not_make_same_step_checkpoint_allocatable():
    cache = manager()
    req = request()
    # Publish the final checkpoint in this step, before worker execution.
    for group in cache.coordinator.single_type_managers:
        group.allocate_new_blocks(req.request_id, 64, 64)
    cache.coordinator.cache_blocks(req, 64)
    sibling = request(name="b")
    blocks, hit = cache.get_computed_blocks(sibling)
    assert hit == 64
    assert cache.allocate_slots(sibling, 7, hit, blocks) is None
