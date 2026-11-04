# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External prefix hits of a hybrid model with a drafter stay on the Mamba
``align`` page grid.

Full attention and Mamba states use 32-token blocks; the drafter's sliding
window (45 tokens) uses 16-token blocks, so its lookup can end a hit in the
middle of a Mamba page.
"""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    MockOffloadingSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test

FULL, MAMBA, DRAFT = 0, 1, 2
PAGE = 32
DRAFT_BLOCK = 16
DRAFT_WINDOW_BLOCKS = 3


def _scheduler_and_request(num_tokens: int):
    kv_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=PAGE, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=PAGE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
            KVCacheGroupSpec(
                ["draft"],
                SlidingWindowSpec(
                    block_size=DRAFT_BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=45,
                ),
            ),
        ],
    )
    config = MagicMock()
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.world_size = 1
    config.cache_config.enable_prefix_caching = True
    config.cache_config.block_size = PAGE
    config.cache_config.hash_block_size = None
    config.cache_config.prefix_cache_retention_interval = 0
    config.kv_transfer_config.kv_connector_extra_config = {}
    config.speculative_config = SimpleNamespace(use_eagle_kv_cache=lambda: False)
    scheduler = OffloadingConnectorScheduler(MockOffloadingSpec(config, kv_config))
    req = MagicMock()
    req.request_id = "req"
    req.skip_reading_prefix_cache = False
    req.kv_transfer_params = None
    req.block_hashes = [
        BlockHash(f"h{i}".encode()) for i in range(num_tokens // DRAFT_BLOCK)
    ]
    req.num_computed_tokens = 0
    req.num_prompt_tokens = req.num_tokens = num_tokens
    req.shared_prefix_boundary = 0
    req.is_finished.return_value = False
    scheduler.on_new_request(req)
    state = scheduler._req_status["req"]
    state.update_offload_keys()
    return scheduler, req, state.group_states


def _lookup(present_blocks: dict[int, range | list[int]], num_tokens: int = 320):
    scheduler, req, groups = _scheduler_and_request(num_tokens)
    present = {
        groups[g].offload_keys[b]
        for g, blocks in present_blocks.items()
        for b in blocks
    }
    scheduler.manager.lookup.side_effect = lambda key, ctx: key in present
    num_hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    return num_hit


def test_hit_not_cut_inside_a_mamba_page():
    # The draft run ends at block 13 (208 tokens, half-way into page 6). The
    # state at 224 must not be restored as the state at 208: the hit falls
    # back to 192, where the state and the whole draft window exist.
    assert _lookup({FULL: range(10), MAMBA: [5, 6], DRAFT: range(9, 13)}) == 192


def test_cut_back_hit_rechecks_the_draft_window():
    # Same, but draft block 9 is missing: the window before 192 is incomplete
    # and no other complete window exists, so there is no hit at all.
    assert _lookup({FULL: range(10), MAMBA: [5, 6], DRAFT: range(10, 13)}) == 0


@pytest.mark.parametrize("seed", range(60))
def test_hit_has_state_and_complete_draft_window(seed):
    """Random host-tier contents: any reported hit is on the page grid and has
    every full-attention block, the recurrent state at the hit and the whole
    draft window before it."""
    rng = random.Random(seed)
    scheduler, req, groups = _scheduler_and_request(320)
    present = set()
    for group_idx in (FULL, MAMBA, DRAFT):
        for key in groups[group_idx].offload_keys:
            if rng.random() < (0.9 if group_idx == FULL else 0.6):
                present.add(key)
    scheduler.manager.lookup.side_effect = lambda key, ctx: key in present
    num_hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    if not num_hit:
        return
    assert num_hit % PAGE == 0
    assert all(k in present for k in groups[FULL].offload_keys[: num_hit // PAGE])
    assert groups[MAMBA].offload_keys[num_hit // PAGE - 1] in present
    end = num_hit // DRAFT_BLOCK
    window = groups[DRAFT].offload_keys[max(0, end - DRAFT_WINDOW_BLOCKS) : end]
    assert all(k in present for k in window)
