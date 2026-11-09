# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill chunk size by prompt length (SchedulerConfig.short_prompt_*)."""

import pytest

from vllm.config import SchedulerConfig

from .utils import create_requests, create_scheduler


def _chunks(scheduler, request, steps: int) -> list[int]:
    # Each schedule() advances num_computed_tokens, so consecutive calls hand
    # out the prompt's chunks as a pipeline would take them.
    chunks = []
    for _ in range(steps):
        output = scheduler.schedule()
        chunks.append(output.num_scheduled_tokens[request.request_id])
    return chunks


@pytest.mark.parametrize(
    ("prompt_tokens", "expected"),
    [
        # Shorter than the threshold: small chunks, also once it is running.
        (1500, [512, 512, 476]),
        # Longer: the whole budget. (Later chunks of a request scheduled again
        # without model output follow the scheduler's own rules, not this one.)
        (3000, [1024]),
    ],
)
def test_chunk_size_follows_the_prompt_length(prompt_tokens, expected):
    scheduler = create_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        max_model_len=8192,
        short_prompt_chunk_tokens=512,
        short_prompt_threshold=2048,
    )
    (request,) = create_requests(num_requests=1, num_tokens=prompt_tokens)
    scheduler.add_request(request)
    assert _chunks(scheduler, request, len(expected)) == expected
    assert request.prefill_chunk_cap == (512 if prompt_tokens < 2048 else 0)


def test_off_by_default():
    scheduler = create_scheduler(max_num_seqs=1, max_num_batched_tokens=1024)
    (request,) = create_requests(num_requests=1, num_tokens=1500)
    scheduler.add_request(request)
    assert _chunks(scheduler, request, 2) == [1024, 476]


@pytest.mark.parametrize(
    ("chunk", "threshold", "match"),
    [
        (512, 0, "set together"),
        (0, 2048, "set together"),
        (1024, 2048, "smaller than max_num_batched_tokens"),
    ],
)
def test_config_rejects_inconsistent_settings(chunk, threshold, match):
    with pytest.raises(ValueError, match=match):
        SchedulerConfig(
            max_num_batched_tokens=1024,
            max_model_len=8192,
            is_encoder_decoder=False,
            short_prompt_chunk_tokens=chunk,
            short_prompt_threshold=threshold,
        )
