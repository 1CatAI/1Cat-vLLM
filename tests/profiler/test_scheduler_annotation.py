# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.profiler.scheduler import scheduler_step_annotation


def test_scheduler_annotation_joins_new_and_cached_requests_without_token_values():
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="new", num_computed_tokens=0, prompt_token_ids=[42])
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["cached"], num_computed_tokens=[32000], new_token_ids=[[999]]
        ),
        num_scheduled_tokens={"new": 768, "cached": 5},
        total_num_scheduled_tokens=773,
    )
    before = repr(output)
    data = json.loads(scheduler_step_annotation(output))
    assert data == {
        "kind": "model",
        "requests": [
            {"id": "cached", "computed": 32000, "scheduled": 5},
            {"id": "new", "computed": 0, "scheduled": 768},
        ],
    }
    assert repr(output) == before


def test_zero_token_control_is_distinct_and_missing_positions_remain_unknown():
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
    )
    assert json.loads(scheduler_step_annotation(output)) == {
        "kind": "zero_token_control",
        "requests": [],
    }
    output.num_scheduled_tokens = {"missing-position": 5}
    output.total_num_scheduled_tokens = 5
    assert json.loads(scheduler_step_annotation(output))["requests"] == [
        {"id": "missing-position", "computed": None, "scheduled": 5}
    ]
