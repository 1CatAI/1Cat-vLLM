# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for VLLM_1CAT_PREFILL_PACE_STEPS (1CatAI/1Cat-vLLM#490)."""

import pytest

from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

NUM_STEPS = 9


def _step(scheduler) -> dict[str, int]:
    """Run one engine step and return the tokens each request was given."""
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    sampled = []
    for req_id in req_ids:
        request = scheduler.requests[req_id]
        prefill_done = (
            request.num_computed_tokens + output.num_scheduled_tokens[req_id]
            >= request.num_prompt_tokens
        )
        sampled.append([0] if prefill_done else [])
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return dict(output.num_scheduled_tokens)


@pytest.mark.parametrize("pace_steps", [0, 2, 4])
def test_prefill_pacing_leaves_decode_only_steps(monkeypatch, pace_steps: int):
    """A paced prefill yields the engine step to the resident decoder."""
    monkeypatch.setenv("VLLM_1CAT_PREFILL_PACE_STEPS", str(pace_steps))
    scheduler = create_scheduler(max_num_batched_tokens=256, max_num_seqs=4)
    decoder = create_requests(1, num_tokens=32, max_tokens=64, req_ids=["decoder"])[0]
    prefiller = create_requests(
        1, num_tokens=2048, max_tokens=64, req_ids=["prefiller"]
    )[0]

    # Bring the short request into the decode phase first.
    scheduler.add_request(decoder)
    _step(scheduler)

    scheduler.add_request(prefiller)
    steps = [_step(scheduler) for _ in range(NUM_STEPS)]

    # The resident decoder keeps its one token in every step, paced or not.
    assert all(step.get("decoder") == 1 for step in steps)

    chunk_steps = [i for i, step in enumerate(steps) if "prefiller" in step]
    if pace_steps == 0:
        assert chunk_steps == list(range(NUM_STEPS))
        return
    # The admission step keeps its chunk; pacing starts once the request is in
    # the running queue, and then a chunk lands only every pace_steps steps.
    gaps = [b - a for a, b in zip(chunk_steps[1:], chunk_steps[2:])]
    assert gaps, "the paced prefill never resumed"
    assert all(gap == pace_steps for gap in gaps)
    assert len(chunk_steps) < NUM_STEPS


def test_prefill_pacing_leaves_a_lone_prefill_alone(monkeypatch):
    """With nothing decoding, a paced prefill still gets every step."""
    monkeypatch.setenv("VLLM_1CAT_PREFILL_PACE_STEPS", "4")
    scheduler = create_scheduler(max_num_batched_tokens=256, max_num_seqs=4)
    prefiller = create_requests(
        1, num_tokens=2048, max_tokens=64, req_ids=["prefiller"]
    )[0]
    scheduler.add_request(prefiller)

    steps = [_step(scheduler) for _ in range(NUM_STEPS)]

    assert all("prefiller" in step for step in steps)
