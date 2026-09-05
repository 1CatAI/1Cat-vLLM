# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

MAMBA_BLOCK_SIZE = 816


def _split(request, num_new_tokens: int) -> int:
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        mamba_state_block_size=MAMBA_BLOCK_SIZE,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=False,
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
