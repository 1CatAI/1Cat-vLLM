# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for superseded align-mode Mamba state blocks.

Under async scheduling the step scheduled just before the current one is still
in flight, so ``remove_skipped_blocks`` sees ``processed_computed_tokens`` one
step further behind. The block recorded by the previous allocation is then the
copy source of that in-flight step and must not be freed yet, but it must not
be forgotten either: before the fix the next allocation overwrote the single
``last_state_block_idx`` slot and one state block per chunk was held until the
request finished or was preempted.
"""

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BLOCK_SIZE = 16
RID = "req"


def _manager(num_speculative_blocks: int, num_gpu_blocks: int = 256):
    pool = BlockPool(num_gpu_blocks, False, BLOCK_SIZE)
    manager = MambaManager(
        MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((1,),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=num_speculative_blocks,
        ),
        pool,
        enable_caching=False,
        kv_cache_group_id=0,
    )
    return pool, manager


def _held(manager: MambaManager) -> int:
    return sum(not b.is_null for b in manager.req_to_blocks[RID])


class _Trace:
    def __init__(self):
        # (step, held after remove, held after alloc, block ids freed by
        #  remove_skipped_blocks, block ids that must not be freed then)
        self.rows: list[tuple[int, int, int, set[int], set[int]]] = []


def _drive(manager, pool, num_chunks, lag, chunk_tokens=BLOCK_SIZE, legacy=False):
    """Chunked prefill of ``chunk_tokens`` per step.

    Step ``i`` ends at ``n_i = (i + 1) * chunk_tokens`` tokens. ``lag`` is how
    many earlier steps are still in flight when step ``i`` is scheduled: 0 is
    sync scheduling (``processed = n_{i-1}``), 1 is async scheduling
    (``processed = n_{i-2}``).

    ``legacy`` restores the pre-fix single-slot semantics by keeping only the
    newest pending index after each allocation, i.e. the old overwrite.
    """
    spec_blocks = manager.num_speculative_blocks
    trace = _Trace()
    # running-state block id at the end of each step
    state_ids: list[int] = []
    for i in range(num_chunks):
        n_i = (i + 1) * chunk_tokens
        processed = max(0, i - lag) * chunk_tokens
        # The running state of step i-1 is live. With lag 1, step i-1 is in
        # flight and copies the running state of step i-2 into its new block,
        # so that block must survive this call too.
        protected = {state_ids[j] for j in range(max(0, i - 1 - lag), i)}
        freed: set[int] = set()
        real_free = pool.free_blocks

        def spy(blks, *a, _real=real_free, _freed=freed, **kw):
            blks = list(blks)
            _freed.update(b.block_id for b in blks)
            return _real(blks, *a, **kw)

        pool.free_blocks = spy
        try:
            manager.new_step_starts()
            manager.remove_skipped_blocks(RID, processed)
        finally:
            pool.free_blocks = real_free
        after_remove = _held(manager)
        manager.allocate_new_blocks(RID, n_i, n_i)
        if legacy and RID in manager.last_state_block_idx:
            manager.last_state_block_idx[RID] = manager.last_state_block_idx[RID][-1:]
        blocks = manager.req_to_blocks[RID]
        state_ids.append(blocks[len(blocks) - 1 - spec_blocks].block_id)
        trace.rows.append((i, after_remove, _held(manager), freed, protected))
    return trace


# A chunk shorter than or equal to a block never leaves a null gap behind the
# superseded state block, so the generic skipped-block sweep reclaims it; chunks
# spanning several blocks (e.g. ~8K-token chunks over 1616-token state
# blocks) are what exposes the single-slot overwrite.
ALL_CHUNKS = [16, 24, 32, 100]
MULTI_BLOCK_CHUNKS = [32, 100]


@pytest.mark.parametrize("chunk_tokens", ALL_CHUNKS)
@pytest.mark.parametrize("spec_blocks", [0, 1, 4])
def test_async_lag_holds_bounded_state_blocks(spec_blocks, chunk_tokens):
    pool, manager = _manager(spec_blocks)
    trace = _drive(manager, pool, 40, lag=1, chunk_tokens=chunk_tokens)
    for i, after_remove, after_alloc, freed, protected in trace.rows:
        # running state + speculative blocks + at most one superseded block
        # awaiting the in-flight step
        assert after_remove <= 1 + spec_blocks + 1, (i, after_remove)
        assert after_alloc <= 1 + spec_blocks + 2, (i, after_alloc)
        # neither the running state nor the in-flight copy source is released
        assert not (freed & protected), i
    # steady state is reached, not just bounded
    assert trace.rows[-1][1] == 2 + spec_blocks
    assert trace.rows[-1][2] == 3 + spec_blocks


@pytest.mark.parametrize("chunk_tokens", ALL_CHUNKS)
@pytest.mark.parametrize("spec_blocks", [0, 1, 4])
def test_sync_scheduling_steady_state_unchanged(spec_blocks, chunk_tokens):
    pool, manager = _manager(spec_blocks)
    fixed = _drive(manager, pool, 40, lag=0, chunk_tokens=chunk_tokens)
    pool2, legacy_manager = _manager(spec_blocks)
    legacy = _drive(
        legacy_manager, pool2, 40, lag=0, chunk_tokens=chunk_tokens, legacy=True
    )
    assert [r[1:4] for r in fixed.rows] == [r[1:4] for r in legacy.rows]
    assert all(not (r[3] & r[4]) for r in fixed.rows)
    assert fixed.rows[-1][1] == 1 + spec_blocks
    assert fixed.rows[-1][2] == 2 + spec_blocks


@pytest.mark.parametrize("chunk_tokens", MULTI_BLOCK_CHUNKS)
@pytest.mark.parametrize("spec_blocks", [0, 4])
def test_legacy_single_slot_leaks_one_block_per_chunk_under_async(
    spec_blocks, chunk_tokens
):
    """Documents the bug: the old overwrite loses one block per chunk."""
    pool, manager = _manager(spec_blocks, num_gpu_blocks=512)
    trace = _drive(manager, pool, 40, lag=1, chunk_tokens=chunk_tokens, legacy=True)
    held = [r[2] for r in trace.rows]
    # growth is exactly one block per chunk once the pipeline is full
    assert [b - a for a, b in zip(held[4:], held[5:])] == [1] * (len(held) - 5)
    assert held[-1] > 35
    # the same drive with the fix stays flat
    pool, manager = _manager(spec_blocks, num_gpu_blocks=512)
    trace = _drive(manager, pool, 40, lag=1, chunk_tokens=chunk_tokens)
    assert trace.rows[-1][2] == 3 + spec_blocks


def test_preemption_mid_prefill_returns_every_block():
    pool, manager = _manager(4)
    free_before = pool.get_num_free_blocks()
    _drive(manager, pool, 12, lag=1, chunk_tokens=40)
    assert pool.get_num_free_blocks() < free_before
    assert manager.last_state_block_idx.get(RID)
    manager.free(RID)
    assert RID not in manager.last_state_block_idx
    assert RID not in manager.req_to_blocks
    assert RID not in manager._allocated_block_reqs
    assert pool.get_num_free_blocks() == free_before
    # a resumed (recomputed) request starts from a clean slate
    trace = _drive(manager, pool, 12, lag=1, chunk_tokens=40)
    assert trace.rows[-1][1] == 2 + 4
    manager.free(RID)
    assert pool.get_num_free_blocks() == free_before


def test_finished_request_returns_every_block_with_pending_entries():
    pool, manager = _manager(1)
    free_before = pool.get_num_free_blocks()
    _drive(manager, pool, 25, lag=1, chunk_tokens=40)
    manager.free(RID)
    assert pool.get_num_free_blocks() == free_before
