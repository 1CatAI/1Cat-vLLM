# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

MAMBA_BLOCK_SIZE = 816


def _split(request, num_new_tokens: int, *, interval=None, eagle=False) -> int:
    coordinator = SimpleNamespace(eagle_group_ids={0} if eagle else set())
    coordinator.get_replay_boundaries = lambda request, block: (
        KVCacheCoordinator.get_replay_boundaries(coordinator, request, block)
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        mamba_state_block_size=MAMBA_BLOCK_SIZE,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=eagle,
        mamba_state_retention_interval=interval,
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
    )
    return Scheduler._mamba_block_aligned_split(scheduler, request, num_new_tokens)


def test_scheduler_records_mamba_group_block_size() -> None:
    mamba_spec = MambaSpec(
        block_size=MAMBA_BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )

    scheduler = create_scheduler(
        block_size=16,
        num_blocks=16,
        kv_cache_spec=mamba_spec,
    )

    assert scheduler.mamba_state_block_size == MAMBA_BLOCK_SIZE


@pytest.mark.parametrize("state_block_size", [816, 8192, 16384])
def test_bulk_prefill_only_when_multiple_states_fit_in_a_chunk(state_block_size):
    spec = MambaSpec(
        block_size=state_block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    original = create_scheduler(
        max_num_batched_tokens=8192,
        block_size=state_block_size,
        num_blocks=16,
        kv_cache_spec=spec,
    )
    config = original.vllm_config
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.enable_prefix_caching = True
    scheduler = Scheduler(
        vllm_config=config,
        kv_cache_config=KVCacheConfig(16, [], [KVCacheGroupSpec(["mamba"], spec)]),
        structured_output_manager=StructuredOutputManager(config),
        block_size=state_block_size,
        hash_block_size=state_block_size,
    )
    assert scheduler.mamba_state_retention_interval == (
        0 if state_block_size < 8192 else None
    )


@pytest.mark.parametrize("mixed_alignment", [False, True])
def test_scheduler_only_batches_sparse_compatible_state_groups(mixed_alignment):
    mamba_spec = MambaSpec(
        block_size=MAMBA_BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    original = create_scheduler(block_size=16, num_blocks=16, kv_cache_spec=mamba_spec)
    config = original.vllm_config
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.enable_prefix_caching = True
    groups = [KVCacheGroupSpec(["mamba"], mamba_spec)]
    if mixed_alignment:
        groups.append(
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=512,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        )
    scheduler = Scheduler(
        vllm_config=config,
        kv_cache_config=KVCacheConfig(16, [], groups),
        structured_output_manager=StructuredOutputManager(config),
        block_size=16,
        hash_block_size=16 if mixed_alignment else MAMBA_BLOCK_SIZE,
    )
    assert scheduler.mamba_state_retention_interval == (None if mixed_alignment else 0)


def test_chunks_stop_at_every_mamba_state_boundary() -> None:
    prompt_len = 3 * MAMBA_BLOCK_SIZE + 30
    (request,) = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        block_size=16,
    )
    position = 0
    chunk_ends = []

    while position < prompt_len:
        request.num_computed_tokens = position
        num_new_tokens = _split(request, prompt_len - position)
        assert num_new_tokens > 0
        position += num_new_tokens
        chunk_ends.append(position)

    assert chunk_ends == [816, 1632, 2448, prompt_len]


@pytest.mark.parametrize("start", [0, 100, 816, 1000])
def test_sub_block_encoder_budget_makes_progress(start: int) -> None:
    request = SimpleNamespace(
        num_computed_tokens=start,
        num_prompt_tokens=3 * MAMBA_BLOCK_SIZE,
        num_tokens=3 * MAMBA_BLOCK_SIZE,
    )
    # An encoder-cache boundary can limit this request even when the global
    # token budget can accommodate a whole Mamba block.
    assert _split(request, 79) == 79


def test_repeated_sub_block_chunks_preserve_state_boundaries() -> None:
    prompt_len = 2 * MAMBA_BLOCK_SIZE + 30
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
    )
    boundaries = []
    while request.num_computed_tokens < prompt_len:
        remaining = prompt_len - request.num_computed_tokens
        chunk = _split(request, min(100, remaining))
        assert 0 < chunk <= min(100, remaining)
        previous = request.num_computed_tokens
        request.num_computed_tokens += chunk
        next_boundary = (previous // MAMBA_BLOCK_SIZE + 1) * MAMBA_BLOCK_SIZE
        assert request.num_computed_tokens <= next_boundary
        if request.num_computed_tokens == next_boundary:
            boundaries.append(next_boundary)

    assert boundaries == [816, 1632]


def _chunk_ends(length, *, interval=0, eagle=False, shared=0):
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=length,
        num_tokens=length,
        shared_prefix_boundary=shared,
    )
    ends = []
    while request.num_computed_tokens < length:
        remaining = length - request.num_computed_tokens
        chunk = _split(request, min(8192, remaining), interval=interval, eagle=eagle)
        assert 0 < chunk <= min(8192, remaining)
        request.num_computed_tokens += chunk
        ends.append(request.num_computed_tokens)
    return ends


def test_sparse_mtp_preserves_fine_resend_boundary_with_bulk_prefill():
    assert _chunk_ends(8192, eagle=True) == [7344, 8192]


@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize("length", [8160, 8161, 32768, 131040])
def test_sparse_keeps_both_replay_boundaries(eagle, length):
    ends = _chunk_ends(length, eagle=eagle)
    coordinator = SimpleNamespace(eagle_group_ids={0} if eagle else set())
    boundaries = KVCacheCoordinator.get_replay_boundaries(
        coordinator, SimpleNamespace(num_tokens=length), MAMBA_BLOCK_SIZE
    )
    for boundary in boundaries:
        aligned = boundary // MAMBA_BLOCK_SIZE * MAMBA_BLOCK_SIZE
        if aligned:
            assert aligned in ends
    assert len(ends) <= length // 8000 + 3


def test_sparse_keeps_detected_shared_prefix_boundary():
    assert 4896 in _chunk_ends(32768, eagle=True, shared=4896)


def test_periodic_retention_stops_at_all_retained_checkpoints():
    interval = 5 * MAMBA_BLOCK_SIZE
    ends = _chunk_ends(32768, eagle=True, interval=interval)
    assert set(range(interval, 32768, interval)) <= set(ends)
    assert 31824 in ends


def test_dense_retention_still_materializes_every_checkpoint():
    length = 3 * MAMBA_BLOCK_SIZE + 30
    assert _chunk_ends(length, interval=None) == [816, 1632, 2448, length]
