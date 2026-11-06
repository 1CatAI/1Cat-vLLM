# SPDX-License-Identifier: Apache-2.0
"""Real scheduler acceptance for multiple synchronous PP speculative requests."""

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import DeviceConfig
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput


@pytest.mark.parametrize("num_requests", [1, 2, 3])
def test_inflight_verify_does_not_schedule_negative_token_count(
    monkeypatch, num_requests
):
    monkeypatch.setattr("vllm.config.parallel.current_platform.device_count", lambda: 2)
    scheduler = create_scheduler(
        pipeline_parallel_size=2,
        device_config=DeviceConfig(device="cpu"),
        num_speculative_tokens=3,
    )
    requests = create_requests(num_requests=num_requests, num_tokens=37, max_tokens=200)
    for request in requests:
        scheduler.add_request(request)
    for round_id in range(5):
        scheduled = scheduler.schedule()
        ids = list(scheduled.num_scheduled_tokens)
        assert ids == [r.request_id for r in requests]
        in_flight = scheduler.schedule()
        assert in_flight.total_num_scheduled_tokens == 0
        sampled = [
            list(scheduled.scheduled_spec_decode_tokens.get(req, []))[: round_id % 4]
            + [1000 + round_id]
            for req in ids
        ]
        scheduler.update_from_output(
            scheduled,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={r: i for i, r in enumerate(ids)},
                sampled_token_ids=sampled,
            ),
        )
        scheduler.update_draft_token_ids(
            DraftTokenIds(ids, [[10, 20, 30] for _ in ids])
        )
