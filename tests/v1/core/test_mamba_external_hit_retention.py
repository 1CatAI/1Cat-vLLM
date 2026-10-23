# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A prefix restored from an external tier must stay reusable on the GPU.

With EAGLE/MTP, a GPU prefix hit on an align-mode Mamba group stops one page
before the last matched page (MambaManager.find_longest_cache_hit), and sparse
retention keeps the state there (the replay boundary). A request whose prefix
comes from the external tier receives the Mamba state at its load boundary.
These tests drive KVCacheManager the way the scheduler does for an async
external load and check what a sibling request (same document, different
question) then finds on the GPU.
"""

import pytest
import torch

from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BLOCK = 32
HASH_BLOCK = 8


def _cache_manager(eagle: bool) -> KVCacheManager:
    full = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=BLOCK,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        1000,
        [],
        [KVCacheGroupSpec(["full"], full), KVCacheGroupSpec(["mamba"], mamba)],
    )
    return KVCacheManager(
        config, 100000, HASH_BLOCK, use_eagle=eagle, prefix_cache_retention_interval=0
    )


# Once: every call draws a new random root hash.
init_none_hash(sha256)


def _request(name: str, token_ids: list[int]) -> Request:
    params = SamplingParams(max_tokens=16)
    req = Request(
        request_id=name,
        prompt_token_ids=token_ids,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(HASH_BLOCK, sha256),
    )
    return req


def _run_prefill(cm: KVCacheManager, req: Request, external: int) -> int:
    """Schedule `req` like Scheduler.schedule: local hit, then an async external
    load of `external` tokens (cached when the load finishes), then the rest in
    BLOCK-aligned chunks. Returns the local (GPU) hit length."""
    cm.new_step_starts()
    blocks, local = cm.get_computed_blocks(req)
    if external:
        assert external > local
        ext = external - local
        assert cm.allocate_slots(
            req,
            0,
            num_new_computed_tokens=local,
            new_computed_blocks=blocks,
            num_external_computed_tokens=ext,
            delay_cache_blocks=True,
        )
        req.num_computed_tokens = local + ext
        # Scheduler._update_waiting_for_remote_kv after the load completes.
        cm.new_step_starts()
        cm.cache_blocks(req, req.num_computed_tokens)
    else:
        assert local == 0
        req.num_computed_tokens = 0
    while req.num_computed_tokens < req.num_tokens:
        start = req.num_computed_tokens
        end = min((start // BLOCK + 1) * BLOCK, req.num_tokens)
        cm.new_step_starts()
        assert cm.allocate_slots(req, end - start)
        req.num_computed_tokens = end
    cm.free(req)
    return local


# A 325-token document followed by two different questions: the requests share
# 10 full blocks (320 tokens); the last full block of each prompt is block 10.
DOC = list(range(1000, 1325))
Q1 = DOC + [7001, 7002, 7003]
Q2 = DOC + [8001, 8002]


@pytest.mark.parametrize("eagle", [False, True])
def test_sibling_hits_gpu_after_computed_prefix(eagle):
    """Reference: the first request computes the document itself."""
    cm = _cache_manager(eagle)
    assert _run_prefill(cm, _request("a", Q1), external=0) == 0
    _, hit = cm.get_computed_blocks(_request("b", Q2))
    assert hit == (288 if eagle else 320)


@pytest.mark.parametrize("eagle", [False, True])
def test_restored_state_on_replay_boundary_serves_siblings(eagle):
    """The first request restores the prefix up to the page the GPU retains
    for it (where the offloading connector ends the external hit): 288 with
    EAGLE/MTP, 320 without. A sibling then hits on the GPU as if the first
    request had computed the document itself."""
    cm = _cache_manager(eagle)
    external = 288 if eagle else 320
    assert _run_prefill(cm, _request("a", Q1), external=external) == 0
    _, hit = cm.get_computed_blocks(_request("b", Q2))
    assert hit == (288 if eagle else 320)


def test_restored_state_on_last_page_is_not_reusable_with_eagle():
    """With EAGLE/MTP, a state restored on the last matched page (320) is not
    a GPU hit for a sibling: the GPU search stops one page earlier and nothing
    is retained there. The sibling misses entirely and would reload the whole
    prefix from the external tier. This is why the connector ends EAGLE
    external hits one page early."""
    cm = _cache_manager(eagle=True)
    assert _run_prefill(cm, _request("a", Q1), external=320) == 0
    _, hit = cm.get_computed_blocks(_request("b", Q2))
    assert hit == 0
