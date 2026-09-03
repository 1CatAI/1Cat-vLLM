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
