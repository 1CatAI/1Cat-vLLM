# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A forced named tool without parameters must still get an object grammar.

OpenAI tools may omit `parameters`. The named tool_choice path returned the
tool's parameters as the JSON schema, so a missing schema installed no grammar
at all and the raw model text (for Qwen3 Coder its XML call) became the
arguments. Like tool_choice="required", constrain the arguments to an object.
"""

import pytest
from openai.types.responses import FunctionTool, ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.utils import get_json_schema_from_tools

pytestmark = pytest.mark.cpu_test

EMPTY_OBJECT = {"type": "object", "properties": {}}
PARAMS = {"type": "object", "properties": {"x": {"type": "string"}}}


def _chat_tool(params):
    function = {"name": "ping"}
    if params is not ...:
        function["parameters"] = params
    return ChatCompletionToolsParam.model_validate(
        {"type": "function", "function": function}
    )


@pytest.mark.parametrize("params", [..., None, {}])
def test_chat_named_empty_params_constrain_an_object(params):
    choice = ChatCompletionNamedToolChoiceParam.model_validate(
        {"type": "function", "function": {"name": "ping"}}
    )
    assert get_json_schema_from_tools(choice, [_chat_tool(params)]) == EMPTY_OBJECT


@pytest.mark.parametrize("params", [None, {}])
def test_responses_named_empty_params_constrain_an_object(params):
    tool = FunctionTool(type="function", name="ping", parameters=params)
    choice = ToolChoiceFunction(type="function", name="ping")
    assert get_json_schema_from_tools(choice, [tool]) == EMPTY_OBJECT


def test_named_params_are_unchanged():
    choice = ChatCompletionNamedToolChoiceParam.model_validate(
        {"type": "function", "function": {"name": "ping"}}
    )
    assert get_json_schema_from_tools(choice, [_chat_tool(PARAMS)]) == PARAMS
    tool = FunctionTool(type="function", name="ping", parameters=PARAMS)
    choice = ToolChoiceFunction(type="function", name="ping")
    assert get_json_schema_from_tools(choice, [tool]) == PARAMS


def test_required_schema_is_unchanged():
    schema = get_json_schema_from_tools("required", [_chat_tool(None)])
    assert schema["items"]["anyOf"][0]["properties"]["parameters"] == EMPTY_OBJECT


def test_named_request_without_params_gets_a_grammar():
    from vllm.tool_parsers.abstract_tool_parser import ToolParser

    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "ping"}],
        tools=[{"type": "function", "function": {"name": "ping"}}],
        tool_choice={"type": "function", "function": {"name": "ping"}},
    )
    request = ToolParser.adjust_request(ToolParser.__new__(ToolParser), request)
    assert request.structured_outputs is not None
    assert request.structured_outputs.json == EMPTY_OBJECT
