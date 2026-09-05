# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import pytest

from benchmarks.benchmark_sm70_batch_tool_quality import run_cases, score_case
from benchmarks.benchmark_sm70_tool_protocol import _bfcl_argument_matches

pytestmark = pytest.mark.skip_global_cleanup


def schema_case(index):
    return {
        "id": str(index),
        "suite": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        "request": {"messages": [{"role": "user", "content": str(index)}]},
    }


def good_response():
    return {"ok": True, "finish_reason": "stop", "content": '{"value": 3}'}


@pytest.mark.parametrize("concurrency", (1, 4, 8, 16))
def test_requests_really_overlap_and_keep_dataset_order_and_seeds(concurrency):
    barrier = threading.Barrier(concurrency, timeout=10)

    def request(url, payload):
        barrier.wait()
        return good_response()

    cases = [schema_case(i) for i in range(concurrency)]
    result = run_cases(cases, "unused", {"seed": 17}, concurrency, request)
    assert result["peak_inflight_client_requests"] == concurrency
    assert [r["payload"]["seed"] for r in result["cases"]] == list(
        range(17, 17 + concurrency)
    )
    assert [r["id"] for r in result["cases"]] == [str(i) for i in range(concurrency)]
    assert result["suites"]["json_schema"] == {
        "correct": concurrency,
        "total": concurrency,
    }
    assert all("seed" not in c["request"] for c in cases)


def test_transport_failure_is_retained_and_not_retried():
    calls = []

    def request(url, payload):
        calls.append(1)
        raise TimeoutError("test")

    result = run_cases([schema_case(0)], "unused", {"seed": 0}, 1, request)
    assert len(calls) == 1
    assert result["suites"]["json_schema"] == {"correct": 0, "total": 1}
    assert "TimeoutError" in result["cases"][0]["response"]["error"]


@pytest.mark.parametrize("finish_reason", (None, "length", "abort"))
def test_truncated_or_unfinished_valid_json_does_not_pass(finish_reason):
    response = {**good_response(), "finish_reason": finish_reason}
    assert score_case(schema_case(0), response)


@pytest.mark.parametrize("content", ('{"value": "bad"}', "{}", "not json"))
def test_structurally_invalid_json_does_not_pass(content):
    assert score_case(schema_case(0), {**good_response(), "content": content})


def test_bfcl_uses_tool_name_and_arguments_not_just_valid_json():
    case = {
        "suite": "bfcl/simple_python",
        "entry": {
            "function": [
                {
                    "name": "sum",
                    "parameters": {
                        "type": "object",
                        "required": ["a"],
                        "properties": {"a": {"type": "integer"}},
                    },
                }
            ],
        },
        "ground_truth": [{"sum": {"a": [3]}}],
        "irrelevance": False,
    }
    function = {"name": "sum", "arguments": '{"a": 3}'}
    response = {
        "ok": True,
        "finish_reason": "tool_calls",
        "tool_calls": [{"function": function}],
    }
    assert not score_case(case, response)
    response["finish_reason"] = "stop"
    assert score_case(case, response)
    response["finish_reason"] = "tool_calls"
    function["name"] = '{"name": "sum"}'
    assert score_case(case, response)


@pytest.mark.parametrize(
    "value,expected,valid",
    [
        ({"min": 3, "max": 4}, [{"min": [3], "max": [4]}], True),
        ({"min": 3}, [{"min": [2, 3], "max": [4, ""]}], True),
        ({"name": "New-York"}, [{"name": ["new york"]}], True),
        ({"values": [1, 2]}, [{"values": [[1, 2]]}], True),
        ({"values": [2, 1]}, [{"values": [[1, 2]]}], False),
        ({"min": [3]}, [{"min": [3]}], False),
        ({"min": 5}, [{"min": [3, 4]}], False),
        ({"min": 3}, [{"min": [3], "max": [4]}], False),
        ({"min": 3, "extra": 4}, [{"min": [3]}], False),
        ({"min": 3}, ["", {"min": [3]}], True),
        ({"min": 3}, [""], False),
    ],
)
def test_bfcl_dictionary_alternatives_not_literal_values(value, expected, valid):
    assert _bfcl_argument_matches(value, expected, {"type": "dict"}) == valid


def test_bfcl_list_of_dicts_preserves_order_count_and_literal_array_semantics():
    schema = {"type": "array", "items": {"type": "dict"}}
    allowed = [[{"a": [1]}, {"a": [2]}]]
    assert _bfcl_argument_matches([{"a": 1}, {"a": 2}], allowed, schema)
    assert not _bfcl_argument_matches([{"a": 2}, {"a": 1}], allowed, schema)
    assert not _bfcl_argument_matches([{"a": 1}], allowed, schema)
    assert not _bfcl_argument_matches([{"a": 1}, 2], allowed, schema)
    ordinary = {"type": "array", "items": {"type": "integer"}}
    assert _bfcl_argument_matches([1, 2], [[1, 2]], ordinary)
    assert not _bfcl_argument_matches([1, 2], [1, 2], ordinary)
