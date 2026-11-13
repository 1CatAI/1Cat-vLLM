# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder outputs stay cached until their tokens are confirmed (#1485)."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler

pytestmark = pytest.mark.cpu_test


class _EncoderCache:
    def __init__(self):
        self.cached = {0}
        self.freed: list[int] = []

    def get_cached_input_ids(self, request):
        return self.cached

    def free_encoder_input(self, request, input_id):
        self.cached.discard(input_id)
        self.freed.append(input_id)


def _free(num_computed_tokens: int, num_in_flight_tokens: int) -> list[int]:
    cache = _EncoderCache()
    scheduler = SimpleNamespace(encoder_cache_manager=cache, is_encoder_decoder=False)
    image = SimpleNamespace(mm_position=SimpleNamespace(offset=10, length=4096))
    request = SimpleNamespace(
        mm_features=[image],
        num_computed_tokens=num_computed_tokens,
        num_in_flight_tokens=num_in_flight_tokens,
    )
    Scheduler._free_encoder_inputs(scheduler, request)
    return cache.freed


def test_in_flight_speculative_step_keeps_the_encoder_output():
    # The image ends at 4106. A step scheduled ahead with 1 + 4 draft tokens
    # moved the optimistic count past it, but those drafts may be rejected and
    # the next draft pass re-gathers the image tail.
    assert _free(num_computed_tokens=4108, num_in_flight_tokens=5) == []


def test_confirmed_tokens_release_the_encoder_output():
    assert _free(num_computed_tokens=4108, num_in_flight_tokens=0) == [0]
    assert _free(num_computed_tokens=4111, num_in_flight_tokens=5) == [0]
