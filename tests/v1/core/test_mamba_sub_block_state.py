# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real block allocation and worker state movement with small chunks."""

from types import SimpleNamespace

import pytest

from vllm.config import VllmConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker import mamba_utils

from .test_mamba_sparse_retention import _mamba, _request

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("budget,threshold", [(512, 0), (2048, 512)])
def test_align_config_accepts_short_chunks(budget, threshold):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=1024, mamba_cache_mode="align"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=budget,
            long_prefill_token_threshold=threshold,
            disable_chunked_mm_input=False,
        ),
    )
    VllmConfig.validate_block_size(config)


@pytest.mark.parametrize("block_size,chunk", [(1024, 560), (8192, 512)])
def test_partial_chunks_keep_running_state_and_immutable_prefix_snapshots(
    monkeypatch, block_size, chunk
):
    pool = BlockPool(32, True, 8)
    manager = _mamba(pool, block_size, 0)
    req = _request("small-chunks", 3 * block_size + 32)
    req.num_computed_tokens = 0
    state_blocks: dict[int, int] = {}
    state_idx: dict[str, int] = {}
    snapshots = {}
    copy_bufs = SimpleNamespace(
        mamba_group_ids=[0], mamba_spec=manager.kv_cache_spec, offset=0
    )
    batch = SimpleNamespace(
        req_ids=[req.request_id],
        num_accepted_tokens_cpu=[1],
        spec_num_accepted_tokens_cpu=[1],
    )

    def collect(buf, config, funcs, groups, prev, curr, bias, request, ctx, slot):
        blocks = manager.req_to_blocks[request.request_id]
        state_blocks[blocks[curr].block_id] = state_blocks[blocks[prev].block_id]

    monkeypatch.setattr(mamba_utils, "collect_mamba_copy_meta", collect)
    monkeypatch.setattr(mamba_utils, "do_mamba_copy_block", lambda buf: None)
    monkeypatch.setattr(mamba_utils, "_debug_mamba_align", lambda *a, **kw: None)

    hashes = BlockHashListWithBlockSize(req.block_hashes, 8, block_size)
    while req.num_computed_tokens < req.num_tokens:
        start = req.num_computed_tokens
        # Stop exactly at the next checkpoint, as the scheduler does.
        count = min(chunk, block_size - start % block_size, req.num_tokens - start)
        end = start + count
        manager.new_step_starts()
        manager.remove_skipped_blocks(req.request_id, start)
        manager.allocate_new_blocks(req.request_id, end, end)
        output = SchedulerOutput.make_empty()
        output.num_scheduled_tokens = {req.request_id: count}
        mamba_utils.preprocess_mamba(
            output,
            None,
            SimpleNamespace(enable_prefix_caching=True),
            state_idx,
            batch,
            {req.request_id: req},
            {},
            None,
            copy_bufs,
        )
        block = manager.req_to_blocks[req.request_id][state_idx[req.request_id]]
        # Scalar recurrence is an oracle for state continuation and copying.
        # The arithmetic is intentionally independent of chunk boundaries.
        value = state_blocks.get(block.block_id, 0)
        for token in range(start, end):
            value = (value * 31 + token + 1) % 1000000007
        state_blocks[block.block_id] = value
        manager.cache_blocks(req, end, alignment_tokens=block_size)
        if end % block_size == 0:
            cached = pool.get_cached_block(hashes[end // block_size - 1], [0])
            assert cached is not None
            snapshots[end] = (cached[0].block_id, value)
        for block_id, snapshot in snapshots.values():
            assert state_blocks[block_id] == snapshot
        req.num_computed_tokens = end

    expected = 0
    for token in range(req.num_tokens):
        expected = (expected * 31 + token + 1) % 1000000007
    assert value == expected
    manager.free(req.request_id)
    hit = manager.find_longest_cache_hit(
        hashes, req.num_tokens - 1, [0], pool, manager.kv_cache_spec, False, block_size
    )[0]
    assert len(hit) == 3
    assert state_blocks[hit[-1].block_id] == snapshots[3 * block_size][1]


@pytest.mark.parametrize("length", [16383, 16384, 16385, 32767, 32768, 32769])
@pytest.mark.parametrize("eagle", [False, True])
def test_small_chunks_preserve_hybrid_resend_window(length, eagle):
    import torch

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    specs = [
        FullAttentionSpec(
            block_size=2048, num_kv_heads=1, head_size=1, dtype=torch.float16
        ),
        MambaSpec(
            block_size=8192,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
        SlidingWindowSpec(
            block_size=1024,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float16,
            sliding_window=2048,
        ),
    ]
    config = KVCacheConfig(
        200,
        [],
        [
            KVCacheGroupSpec([str(i)], spec, is_eagle_group=eagle and i == 2)
            for i, spec in enumerate(specs)
        ],
    )
    cache = KVCacheManager(
        config, 262144, 8, use_eagle=eagle, prefix_cache_retention_interval=0
    )
    request = _request("hybrid-resend", length)
    for start in range(0, length, 512):
        end = min(start + 512, length)
        for manager in cache.coordinator.single_type_managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(request.request_id, start, length)
            manager.allocate_new_blocks(request.request_id, end, end)
        cache.coordinator.cache_blocks(request, end)
        swa = cache.coordinator.single_type_managers[2]
        real_blocks = sum(
            not block.is_null for block in swa.req_to_blocks[request.request_id]
        )
        assert real_blocks <= specs[2].max_admission_blocks_per_request(512, 262144)
    cache.coordinator.free(request.request_id)
    _, hit = cache.get_computed_blocks(request)
    expected = (length // 8192 - 1) * 8192 if eagle else (length - 1) // 8192 * 8192
    assert hit == max(expected, 0)
