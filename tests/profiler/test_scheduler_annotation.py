# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from urllib.parse import unquote

import torch

from vllm.profiler.scheduler import scheduler_step_annotation


def empty_scheduler_output():
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
    )


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
    assert scheduler_step_annotation(output) == (
        "kind=model;requests=cached@32000+5,new@0+768"
    )
    assert repr(output) == before


def test_zero_token_control_is_distinct_and_missing_positions_remain_unknown():
    output = empty_scheduler_output()
    assert scheduler_step_annotation(output) == "kind=zero_token_control;requests="
    output.num_scheduled_tokens = {"missing-position": 5}
    output.total_num_scheduled_tokens = 5
    assert scheduler_step_annotation(output) == (
        "kind=model;requests=missing-position@?+5"
    )


def test_scheduler_annotation_exports_valid_trace_with_arbitrary_request_id(tmp_path):
    request_id = 'quoted"\\\n\t,;@+%=\u6a21\u578b'
    output = empty_scheduler_output()
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=[request_id], num_computed_tokens=[32000]
    )
    output.num_scheduled_tokens = {request_id: 5}
    output.total_num_scheduled_tokens = 5
    annotation = scheduler_step_annotation(output)
    encoded_id = annotation.split("requests=", 1)[1].split("@", 1)[0]
    assert unquote(encoded_id) == request_id
    assert '"' not in annotation and "\\" not in annotation
    assert all(32 <= ord(char) < 127 for char in annotation)
    with (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p,
        torch.profiler.record_function(annotation),
    ):
        torch.ones(4).add_(1)
    path = tmp_path / "trace.json"
    p.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    assert any(
        event.get("cat") == "user_annotation" and event["name"] == annotation
        for event in events
    )
