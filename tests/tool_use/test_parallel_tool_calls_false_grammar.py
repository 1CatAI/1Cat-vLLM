# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""parallel_tool_calls=false limits the required-tool grammar to one call.

Before, the flag only dropped extra calls after generation, so the model could
still generate (and the client pay for) several calls. An explicit null keeps
the default (parallel calls allowed), as in the OpenAI API.
"""

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.utils import maybe_filter_parallel_tool_calls
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import get_json_schema_from_tools

pytestmark = pytest.mark.cpu_test

TOOL = {
    "type": "function",
    "function": {
        "name": "f",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
    },
}
CALL = {"name": "f", "parameters": {"x": "v"}}


def _request(**kwargs):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "call f"}],
        tools=[TOOL],
        **kwargs,
    )


def _adjusted(**kwargs):
    request = _request(**kwargs)
    return ToolParser.adjust_request(ToolParser.__new__(ToolParser), request)


@pytest.mark.parametrize(
    "parallel, max_items", [(False, 1), (True, None), (None, None)]
)
def test_required_schema_limits_calls_only_when_false(parallel, max_items):
    request = _adjusted(tool_choice="required", parallel_tool_calls=parallel)
    schema = request.structured_outputs.json
    assert schema.get("maxItems") == max_items
    assert schema["minItems"] == 1


def test_required_schema_default_is_unchanged():
    schema = _adjusted(tool_choice="required").structured_outputs.json
    assert schema == get_json_schema_from_tools("required", _request().tools)
    assert "maxItems" not in schema


@pytest.mark.parametrize(
    "choice", ["auto", {"type": "function", "function": {"name": "f"}}]
)
def test_other_tool_choices_are_unchanged(choice):
    for parallel in (True, False):
        request = _adjusted(tool_choice=choice, parallel_tool_calls=parallel)
        if choice == "auto":
            assert request.structured_outputs is None
        else:
            assert "maxItems" not in request.structured_outputs.json


def test_xgrammar_accepts_one_call_and_rejects_two():
    xgr = pytest.importorskip("xgrammar")
    from xgrammar.testing import _is_grammar_accept_string

    schema = _adjusted(
        tool_choice="required", parallel_tool_calls=False
    ).structured_outputs.json
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema))
    assert _is_grammar_accept_string(grammar, json.dumps([CALL]))
    assert not _is_grammar_accept_string(grammar, json.dumps([CALL, CALL]))

    unlimited = _adjusted(tool_choice="required").structured_outputs.json
    grammar = xgr.Grammar.from_json_schema(json.dumps(unlimited))
    assert _is_grammar_accept_string(grammar, json.dumps([CALL, CALL]))


def _choice(n):
    calls = [
        ToolCall(id=f"call_{i}", function=FunctionCall(name="f", arguments="{}"))
        for i in range(n)
    ]
    return ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="", tool_calls=calls),
        finish_reason="tool_calls",
    )


@pytest.mark.parametrize("parallel, kept", [(False, 1), (True, 2), (None, 2)])
def test_post_filter_treats_null_as_default(parallel, kept):
    request = _request(tool_choice="auto", parallel_tool_calls=parallel)
    choice = maybe_filter_parallel_tool_calls(_choice(2), request)
    assert len(choice.message.tool_calls) == kept


def test_responses_echoes_default_for_null_parallel_tool_calls():
    from vllm.entrypoints.openai.responses.protocol import (
        ResponsesRequest,
        ResponsesResponse,
    )
    from vllm.sampling_params import SamplingParams

    request = ResponsesRequest(model="test", input="hi", parallel_tool_calls=None)
    response = ResponsesResponse.from_request(
        request,
        SamplingParams(max_tokens=8),
        model_name="test",
        created_time=0,
        output=[],
        status="completed",
        usage=None,
    )
    assert response.parallel_tool_calls is True
