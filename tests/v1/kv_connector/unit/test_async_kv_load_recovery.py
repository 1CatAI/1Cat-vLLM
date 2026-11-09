# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("drafts", [0, 2])
@pytest.mark.parametrize("pending_frames", [0, 1, 2])
@pytest.mark.parametrize("prefix_reset_discard", [0, 1])
def test_failed_prefix_discards_pending_frames_without_scheduling_placeholders(
    drafts, pending_frames, prefix_reset_discard
):
    config = create_vllm_config(kv_load_failure_policy="recompute", max_model_len=128)
    config.scheduler_config.async_scheduling = True
    scheduler = create_scheduler(config)
    assert isinstance(scheduler, AsyncScheduler)
    scheduler.num_spec_tokens = drafts
    scheduler._spec_token_placeholders = (-1,) * drafts
    request = create_request(num_tokens=33)
    scheduler.add_request(request)
    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (32, False)
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    failed = scheduler.schedule()
    rid = request.request_id
    assert failed.num_scheduled_tokens[rid] == 1
    pending = [scheduler.schedule() for _ in range(pending_frames)]
    for output in pending:
        assert len(output.scheduled_spec_decode_tokens.get(rid, [])) == drafts

    blocks = failed.scheduled_new_reqs[0].block_ids[0]
    request.async_tokens_to_discard = prefix_reset_discard
    result = scheduler.update_from_output(
        failed,
        create_model_runner_output(
            [request], invalid_block_ids={blocks[0]}, token_id=17
        ),
    )
    assert all(not item.outputs for item in result.values())
    assert request.num_computed_tokens == 0
    assert request.num_output_placeholders == 0
    assert request.num_kv_load_outputs_to_discard == pending_frames
    assert request.async_tokens_to_discard == 0
    assert request.num_in_flight_steps == pending_frames
    assert request.spec_token_ids == []

    replay = scheduler.schedule()
    # The replay contains exactly the prompt, never phantom output IDs (-1).
    assert replay.num_scheduled_tokens[rid] == 33
    assert request.num_output_placeholders == 1
    assert request.num_computed_tokens == 33
    for index, output in enumerate(pending):
        result = scheduler.update_from_output(
            output, create_model_runner_output([request], token_id=17)
        )
        assert all(not item.outputs for item in result.values())
        assert request.num_computed_tokens == 33
        assert request.num_output_placeholders == 1
        assert request.num_kv_load_outputs_to_discard == pending_frames - index - 1
        assert request.async_tokens_to_discard == 0
        assert request.num_output_tokens == 0

    scheduler.update_from_output(
        replay, create_model_runner_output([request], token_id=18)
    )
    assert list(request.output_token_ids) == [18]
    assert request.num_output_placeholders == 0
    assert request.num_in_flight_steps == 0


def test_prefix_reset_discard_stays_owned_by_async_scheduler():
    config = create_vllm_config(max_model_len=128)
    config.scheduler_config.async_scheduling = True
    scheduler = create_scheduler(config)
    request = create_request(num_tokens=33)
    scheduler.add_request(request)
    output = scheduler.schedule()
    request.async_tokens_to_discard = 1
    update = Mock(wraps=scheduler._update_request_with_output)
    scheduler._update_request_with_output = update

    result = scheduler.update_from_output(
        output, create_model_runner_output([request], token_id=17)
    )

    update.assert_called_once()
    assert all(not item.outputs for item in result.values())
    assert request.async_tokens_to_discard == 0
    assert request.num_output_tokens == 0
