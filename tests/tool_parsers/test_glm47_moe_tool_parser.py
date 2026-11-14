# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Tests for the GLM-4.7 tool call parser."""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

MODEL = "zai-org/GLM-4.5"


@pytest.fixture(scope="module")
def glm47_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="get_current_date", parameters={}),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "date": {"type": "string"},
                    },
                },
            ),
        ),
    ]


@pytest.fixture
def glm47_tool_parser(glm47_tokenizer, sample_tools):
    return Glm47MoeModelToolParser(glm47_tokenizer, tools=sample_tools)


@pytest.fixture
def mock_request(sample_tools) -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = sample_tools
    request.tool_choice = "auto"
    return request


class TestGlm47ExtractToolCalls:
    def test_no_tool_call(self, glm47_tool_parser, mock_request):
        out = "This is a plain response."
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert not r.tools_called
        assert r.content == out

    def test_zero_arg_inline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"
        assert json.loads(r.tool_calls[0].function.arguments) == {}
        assert r.content is None

    def test_zero_arg_newline(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.tool_calls[0].function.name == "get_current_date"

    def test_args_same_line(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_args_with_newlines(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "Beijing"}

    def test_whitespace_preserved_in_arg_values(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>  Beijing  </arg_value></tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert json.loads(r.tool_calls[0].function.arguments) == {"city": "  Beijing  "}

    def test_content_before(self, glm47_tool_parser, mock_request):
        out = "Checking.<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.tools_called
        assert r.content == "Checking."

    def test_multiple(self, glm47_tool_parser, mock_request):
        out = (
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>"
        )
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert len(r.tool_calls) == 2

    def test_empty_content_none(self, glm47_tool_parser, mock_request):
        out = "<tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None

    def test_whitespace_content_none(self, glm47_tool_parser, mock_request):
        out = "  \n  <tool_call>get_current_date</tool_call>"
        r = glm47_tool_parser.extract_tool_calls(out, request=mock_request)
        assert r.content is None


def _reset(parser):
    parser.current_tool_name_sent = False
    parser.prev_tool_call_arr = []
    parser.current_tool_id = -1
    parser.streamed_args_for_tool = []
    parser._tool_call_ids = []
    parser._sent_content_idx = 0


class TestGlm47Streaming:
    def test_no_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = ["<tool_call>", "get_current_date", "</tool_call>"]
        current_text = ""
        for chunk in chunks:
            current_text += chunk
            glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
        assert len(glm47_tool_parser.prev_tool_call_arr) >= 1

    def test_with_args(self, glm47_tool_parser, mock_request):
        _reset(glm47_tool_parser)
        chunks = [
            "<tool_call>",
            "get_weather\n",
            "<arg_key>city</arg_key>",
            "<arg_value>",
            "Beijing",
            "</arg_value>",
            "</tool_call>",
        ]
        current_text = ""
        for chunk in chunks:
            current_text += chunk
            glm47_tool_parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=mock_request,
            )
        args = json.loads(glm47_tool_parser.prev_tool_call_arr[0]["arguments"])
        assert args["city"] == "Beijing"


TYPED_TOOL = ChatCompletionToolsParam(
    function=FunctionDefinition(
        name="delegate",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "count": {"type": "integer"},
                "flag": {"type": "boolean"},
                "items": {"type": "array"},
                "note": {"type": ["string", "array"]},
                "optional": {"type": ["string", "null"]},
            },
        },
    ),
)


@pytest.mark.parametrize(
    "body",
    [
        "<arg_key>task</arg_key><arg_value>Fasse die Daten zusammen</arg_value>",
        "<arg_key>count</arg_key><arg_value>12</arg_value>",
        "<arg_key>flag</arg_key><arg_value>True</arg_value>",
        "<arg_key>flag</arg_key><arg_value>false</arg_value>",
        "<arg_key>items</arg_key><arg_value>['a', 'b']</arg_value>",
        "<arg_key>items</arg_key><arg_value>[\n  1,\n  2\n]</arg_value>",
        "<arg_key>note</arg_key><arg_value>siehe oben</arg_value>",
        "<arg_key>extra</arg_key><arg_value>frei formuliert</arg_value>",
        "<arg_key>optional</arg_key><arg_value> mit Leerzeichen </arg_value>",
        "<arg_key>count</arg_key><arg_value>3</arg_value>"
        '<arg_key>note</arg_key><arg_value>Text mit "Zitat" und \\</arg_value>'
        "<arg_key>task</arg_key><arg_value>Grüße</arg_value>",
    ],
)
def test_streamed_arguments_stay_a_prefix_of_the_finish(glm47_tokenizer, body):
    """Every delta must extend toward what extract_tool_calls returns at the
    finish; flush_tool_calls rejects a stream that does not."""
    from vllm.entrypoints.openai.chat_completion.tool_stream import (
        StreamedToolCall,
        flush_tool_calls,
        record_tool_calls,
    )

    parser = Glm47MoeModelToolParser(glm47_tokenizer, tools=[TYPED_TOOL])
    request = Mock(spec=ChatCompletionRequest)
    request.tools = [TYPED_TOOL]
    request.tool_choice = "auto"
    text = "<tool_call>delegate" + body + "</tool_call>"
    final = parser.extract_tool_calls(text, request).tool_calls[0].function.arguments
    seen: dict[int, StreamedToolCall] = {}
    previous = ""
    for end in range(1, len(text) + 1):
        current = text[:end]
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=current[len(previous) :],
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if delta is not None and delta.tool_calls:
            record_tool_calls(delta.tool_calls, seen)
        streamed = "".join(seen[0].arguments) if 0 in seen else ""
        assert final.startswith(streamed), (current, streamed, final)
        previous = current
    assert "".join(seen[0].arguments) == final
    json.loads(final)
    flushed, called = flush_tool_calls(parser, text, request, None, seen)
    assert called
    assert not any(
        call.function is not None and call.function.arguments
        for call in (flushed.tool_calls if flushed else [])
    )
