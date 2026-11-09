# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request shaping for the XML tool/JSON answer union, without model downloads."""

import copy
import json
from typing import Any
from unittest.mock import Mock

import pytest
from xgrammar import Grammar

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.tool_parsers.qwen3xml_tool_parser import Qwen3XMLToolParser

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "answer", "strict": True, "schema": ANSWER_SCHEMA},
}
JSON_OBJECT = {"type": "json_object"}
NAMED = {"type": "function", "function": {"name": "get_weather"}}

# Seven formerly broken requests and eight controls.
CASES = [
    ("auto-schema", {"tool_choice": "auto", "response_format": JSON_SCHEMA}, True),
    ("default-schema", {"response_format": JSON_SCHEMA}, True),
    ("auto-object", {"tool_choice": "auto", "response_format": JSON_OBJECT}, True),
    ("default-object", {"response_format": JSON_OBJECT}, True),
    (
        "parallel-schema",
        {"response_format": JSON_SCHEMA, "parallel_tool_calls": True},
        True,
    ),
    (
        "parallel-object",
        {"response_format": JSON_OBJECT, "parallel_tool_calls": True},
        True,
    ),
    (
        "single-call",
        {"response_format": JSON_SCHEMA, "parallel_tool_calls": False},
        True,
    ),
    ("schema-only", {"tools": None, "response_format": JSON_SCHEMA}, False),
    ("object-only", {"tools": None, "response_format": JSON_OBJECT}, False),
    ("tools-only", {"tool_choice": "auto"}, False),
    ("required", {"tool_choice": "required", "response_format": JSON_SCHEMA}, False),
    ("named", {"tool_choice": NAMED, "response_format": JSON_SCHEMA}, False),
    ("none", {"tool_choice": "none", "response_format": JSON_SCHEMA}, False),
    ("text", {"response_format": {"type": "text"}}, False),
    ("plain", {"tools": None}, False),
]


def make_request(overrides):
    return ChatCompletionRequest(
        **copy.deepcopy({"model": "test", "messages": [], "tools": TOOLS, **overrides})
    )


def make_parser(parser_cls):
    tokenizer = Mock()
    tokenizer.get_vocab.return_value = {"<tool_call>": 0, "</tool_call>": 1}
    return parser_cls(tokenizer)


@pytest.mark.parametrize("parser_cls", [Qwen3CoderToolParser, Qwen3XMLToolParser])
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("name,overrides,joint", CASES, ids=[c[0] for c in CASES])
def test_request_matrix(
    monkeypatch, parser_cls, reasoning, strict, name, overrides, joint
):
    monkeypatch.setattr(
        "vllm.tool_parsers.abstract_tool_parser.VLLM_ENFORCE_STRICT_TOOL_CALLING",
        strict,
    )
    monkeypatch.setattr(
        "vllm.tool_parsers.structural_tag_registry._enable_structured_outputs_in_reasoning",
        reasoning,
    )
    parser = make_parser(parser_cls)
    request = make_request(overrides)
    if joint:
        # Byte-for-byte oracle: the already supported qwen3_xml request path.
        expected = make_parser(Qwen3XMLToolParser).adjust_request(
            request.model_copy(deep=True)
        )
    else:
        expected = ToolParser.adjust_request(parser, request.model_copy(deep=True))
    result = parser.adjust_request(request)
    assert result is request
    assert result.model_dump_json() == expected.model_dump_json()
    if joint:
        assert result.response_format is None
        assert result.structured_outputs is not None
        tag_json = result.structured_outputs.structural_tag
        Grammar.from_structural_tag(tag_json)
        tag = json.loads(tag_json)["format"]
        if reasoning:
            tag = tag["elements"][1]
        whitespace, union = tag["elements"]
        assert whitespace["type"] == "optional"
        assert union["type"] == "or"
        tools_format, answer_format = union["elements"]
        assert tools_format["type"] == "tags_with_separator"
        assert tools_format["separator"] == "\n"
        assert tools_format["at_least_one"] is True
        assert answer_format["type"] == "json_schema"
        # Checks the response_format was cleared before sampling validation.
        params = result.to_sampling_params(256, {})
        assert params.structured_outputs.structural_tag == tag_json


@pytest.mark.parametrize("parser_cls", [Qwen3CoderToolParser, Qwen3XMLToolParser])
def test_incompatible_tool_schema_relaxes_only_arguments(parser_cls):
    request = make_request({"response_format": JSON_SCHEMA})
    request.tools[0].function.parameters["properties"]["city"]["pattern"] = "(?=Tokyo)"
    original_tools = request.tools[0].model_dump_json()
    result = make_parser(parser_cls).adjust_request(request)
    assert result.response_format is None
    tag_json = result.structured_outputs.structural_tag
    Grammar.from_structural_tag(tag_json)
    tool_tag = json.loads(tag_json)["format"]["elements"][1]["elements"][0]["tags"][0]
    assert tool_tag["content"]["json_schema"] is True
    assert request.tools[0].model_dump_json() == original_tools


@pytest.mark.parametrize("parser_cls", [Qwen3CoderToolParser, Qwen3XMLToolParser])
def test_incompatible_answer_falls_back_without_mutating_request(parser_cls):
    request = make_request({"response_format": JSON_SCHEMA})
    request.response_format.json_schema.json_schema["properties"]["answer"][
        "pattern"
    ] = "["
    before = request.model_dump_json()
    assert make_parser(parser_cls).adjust_request(request) is request
    assert request.model_dump_json() == before


@pytest.mark.parametrize("parser_cls", [Qwen3CoderToolParser, Qwen3XMLToolParser])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"tool_choice": None},
        {"tool_choice": "none"},
        {"tool_choice": "required", "response_format": JSON_OBJECT},
        {"tool_choice": NAMED, "response_format": JSON_OBJECT},
        {"structured_outputs": StructuredOutputsParams(regex="[0-9]+")},
        {
            "response_format": {
                "type": "structural_tag",
                "format": {"type": "const_string", "value": "hello"},
            }
        },
    ],
)
def test_other_shapes_delegate_to_base(
    monkeypatch, parser_cls, strict, overrides: dict[str, Any]
):
    monkeypatch.setattr(
        "vllm.tool_parsers.abstract_tool_parser.VLLM_ENFORCE_STRICT_TOOL_CALLING",
        strict,
    )
    request = make_request({"response_format": JSON_SCHEMA, **overrides})
    calls = []

    def base_adjust(self, incoming):
        calls.append(incoming)
        return incoming

    monkeypatch.setattr(ToolParser, "adjust_request", base_adjust)
    assert make_parser(parser_cls).adjust_request(request) is request
    assert calls == [request]


@pytest.mark.parametrize("parser_cls", [Qwen3CoderToolParser, Qwen3XMLToolParser])
def test_empty_structured_outputs_preserves_options(parser_cls):
    # Empty constraints cannot be constructed through the public API. Exercise
    # the existing guard for an internal caller that clears a constraint.
    options = StructuredOutputsParams(regex=".", disable_any_whitespace=True)
    options.regex = None
    request = make_request({"response_format": JSON_OBJECT})
    request.structured_outputs = options
    original = request.structured_outputs
    result = make_parser(parser_cls).adjust_request(request)
    assert result.structured_outputs is original
    assert result.structured_outputs.disable_any_whitespace
    assert result.structured_outputs.structural_tag is not None
    assert result.response_format is None


@pytest.mark.parametrize("parser_cls", [Qwen3CoderToolParser, Qwen3XMLToolParser])
@pytest.mark.parametrize("choice", ["auto", "required", "none"])
def test_responses_request_keeps_base_behavior(parser_cls, choice):
    request = ResponsesRequest(
        model="test",
        input="Hello",
        tools=[{"type": "function", **TOOLS[0]["function"]}],
        tool_choice=choice,
    )
    parser = make_parser(parser_cls)
    expected = ToolParser.adjust_request(parser, request.model_copy(deep=True))
    assert (
        parser.adjust_request(request).model_dump_json() == expected.model_dump_json()
    )
