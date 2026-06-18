# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds


class _FakeRequest:
    is_prefill_chunk = False
    structured_output_request = None

    def __init__(self):
        self.spec_token_ids: list[int] = []

    def is_finished(self) -> bool:
        return False


class _FakeStructuredOutputManager:
    @staticmethod
    def should_advance(_request) -> bool:
        return False


def _fake_scheduler(request: _FakeRequest):
    return SimpleNamespace(
        requests={"r0": request},
        structured_output_manager=_FakeStructuredOutputManager(),
    )


def test_update_draft_token_ids_snapshots_worker_rows():
    request = _FakeRequest()
    scheduler = _fake_scheduler(request)
    worker_row = [11, 12, 13, 14]

    Scheduler.update_draft_token_ids(
        scheduler,
        DraftTokenIds(req_ids=["r0"], draft_token_ids=[worker_row]),
    )
    worker_row[0] = 99

    assert request.spec_token_ids == [11, 12, 13, 14]


def test_update_draft_token_ids_in_output_does_not_mutate_worker_rows():
    request = _FakeRequest()
    scheduler = _fake_scheduler(request)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"r0": [-1, -1]},
        num_invalid_spec_tokens={},
    )
    worker_row = [21, 22, 23, 24]

    Scheduler.update_draft_token_ids_in_output(
        scheduler,
        DraftTokenIds(req_ids=["r0"], draft_token_ids=[worker_row]),
        scheduler_output,
    )

    assert worker_row == [21, 22, 23, 24]
    assert scheduler_output.scheduled_spec_decode_tokens["r0"] == [21, 22]
