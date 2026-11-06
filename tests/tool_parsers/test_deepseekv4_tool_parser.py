# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DeepSeekV4ToolParser."""

import json
from unittest.mock import MagicMock

import pytest
from xgrammar import StructuralTag

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {}

TC_START = "<｜DSML｜tool_calls>"
TC_END = "</｜DSML｜tool_calls>"
INV_START = '<｜DSML｜invoke name="'
INV_END = "</｜DSML｜invoke>"
PARAM_START = '<｜DSML｜parameter name="'
PARAM_END = "</｜DSML｜parameter>"


@pytest.fixture
def sample_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "state": {"type": "string", "description": "The state code"},
                        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
                    },
                    "required": ["city", "state"],
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "calculate_area",
                "description": "Calculate area of a shape",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shape": {"type": "string"},
                        "dimensions": {"type": "object"},
                        "precision": {"type": "integer"},
                    },
                },
            },
        ),
    ]


def make_parser(tools=None) -> DeepSeekV4ToolParser:
    return DeepSeekV4ToolParser(MOCK_TOKENIZER, tools=tools)


def make_request(tools=None) -> MagicMock:
    req = MagicMock()
    req.tools = tools
    return req


def build_tool_call(func_name: str, params: dict[str, str]) -> str:
    param_strs = "".join(
        f'{PARAM_START}{k}" string="true">{v}{PARAM_END}\n' for k, v in params.items()
    )
    return f'{TC_START}\n{INV_START}{func_name}">\n{param_strs}{INV_END}\n{TC_END}'


def stream(parser: DeepSeekV4ToolParser, full_text: str, chunk_size: int = 7):
    return stream_chunks(
        parser,
        [
            full_text[start : start + chunk_size]
            for start in range(0, len(full_text), chunk_size)
        ],
    )


def stream_chunks(parser: DeepSeekV4ToolParser, chunks: list[str]):
    deltas = []
    previous_text = ""
    for delta_text in chunks:
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=make_request(),
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
    return deltas


def reconstruct_args(deltas, tool_index: int = 0) -> str:
    fragments = []
    for delta in deltas:
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                if (
                    tool_call.index == tool_index
                    and tool_call.function
                    and tool_call.function.arguments
                ):
                    fragments.append(tool_call.function.arguments)
    return "".join(fragments)


def test_registered():
    assert ToolParserManager.get_tool_parser("deepseek_v4") is DeepSeekV4ToolParser


def test_extract_tool_calls():
    parser = make_parser()
    model_output = "Let me check. " + build_tool_call(
        "get_weather", {"location": "Beijing", "unit": "celsius"}
    )

    result = parser.extract_tool_calls(model_output, make_request())

    assert result.tools_called
    assert result.content == "Let me check. "
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {
        "location": "Beijing",
        "unit": "celsius",
    }


def test_function_calls_wrapper_is_not_recognized():
    # The V3.2 wrapper is not a V4 terminal, so it passes through as content,
    # but the invoke inside it is still parsed (tool calls anchor on the invoke).
    parser = make_parser()
    model_output = build_tool_call("search", {"query": "vllm"}).replace(
        "tool_calls", "function_calls"
    )

    result = parser.extract_tool_calls(model_output, make_request())

    assert result.tools_called
    assert result.tool_calls[0].function.name == "search"
    assert result.content == "<｜DSML｜function_calls>\n"


def test_missing_tool_calls_wrapper_is_recovered():
    # Regression for #48931: at long context the model omits the
    # <｜DSML｜tool_calls> START token but still emits a complete invoke.
    parser = make_parser()
    model_output = build_tool_call("search", {"query": "vllm"}).replace(
        TC_START + "\n", ""
    )
    assert TC_START not in model_output

    result = parser.extract_tool_calls(model_output, make_request())

    assert result.tools_called
    assert result.tool_calls[0].function.name == "search"
    assert json.loads(result.tool_calls[0].function.arguments) == {"query": "vllm"}
    assert result.content is None

    deltas = stream(make_parser(), model_output, chunk_size=3)
    names = [
        tc.function.name
        for d in deltas
        if d.tool_calls
        for tc in d.tool_calls
        if tc.function.name
    ]
    assert names == ["search"]
    assert json.loads(reconstruct_args(deltas)) == {"query": "vllm"}
    assert not any(d.content and "DSML" in d.content for d in deltas)


def test_streaming_extracts_complete_invokes():
    parser = make_parser()
    full_text = build_tool_call("search", {"query": "deepseek v4"})

    deltas = stream(parser, full_text, chunk_size=5)

    names = [
        tool_call.function.name
        for delta in deltas
        if delta.tool_calls
        for tool_call in delta.tool_calls
        if tool_call.function.name
    ]
    assert names == ["search"]
    assert json.loads(reconstruct_args(deltas)) == {"query": "deepseek v4"}


def test_streaming_emits_incremental_argument_chunks():
    tool = ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="plan_trip",
            parameters={
                "type": "object",
                "properties": {
                    "days": {"type": "integer"},
                    "flexible": {"type": "boolean"},
                    "cities": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
        ),
    )
    parser = make_parser(tools=[tool])
    full_text = (
        f"{TC_START}\n"
        f'{INV_START}plan_trip">\n'
        f'{PARAM_START}days" string="false">3{PARAM_END}\n'
        f'{PARAM_START}flexible" string="false">false{PARAM_END}\n'
        f'{PARAM_START}cities" string="false">'
        f'["Beijing","Shanghai","Tokyo","New York"]{PARAM_END}\n'
        f'{PARAM_START}notes" string="true">靠窗座位{PARAM_END}\n'
        f"{INV_END}\n"
        f"{TC_END}"
    )

    deltas = stream(parser, full_text, chunk_size=4)
    arg_chunks = [
        tool_call.function.arguments
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function and tool_call.function.arguments is not None
    ]

    assert len([chunk for chunk in arg_chunks if chunk]) > 2
    assert json.loads("".join(arg_chunks)) == {
        "days": 3,
        "flexible": False,
        "cities": ["Beijing", "Shanghai", "Tokyo", "New York"],
        "notes": "靠窗座位",
    }


def test_get_vllm_registry_structural_tag_returns_structural_tag(
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    parser = make_parser()
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="auto",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="required",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    if sample_tools:
        tool = sample_tools[0]
        req = ChatCompletionRequest(
            messages=[],
            model="m",
            tools=sample_tools,
        )
        req.tool_choice = ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name=tool.function.name)
        )
        tag = parser.get_structural_tag(req)
        assert isinstance(tag, StructuralTag)


def test_extract_tool_calls_arguments_wrapper():
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {}

    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    )

    parser = DeepSeekV4ToolParser(mock_tokenizer, tools=[tool])
    request = MagicMock()
    request.tools = [tool]

    model_output = (
        f"{TC_START}"
        f'{INV_START}get_weather">'
        f'{PARAM_START}arguments" string="false">{{"location":"Beijing"}}{PARAM_END}'
        f"{INV_END}"
        f"{TC_END}"
    )

    result = parser.extract_tool_calls(model_output, request)
    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"location": "Beijing"}


@pytest.mark.skip_global_cleanup
def test_composed_schema_converts_object_and_array_params():
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "set_timer",
            "parameters": {
                "type": "object",
                "properties": {
                    "wait": {
                        "anyOf": [
                            {"type": "object"},
                            {"type": "null"},
                        ],
                    },
                    "patches": {
                        "allOf": [
                            {"type": "array", "items": {"type": "object"}},
                        ],
                    },
                },
            },
        },
    )
    parser = make_parser(tools=[tool])
    request = make_request(tools=[tool])
    model_output = (
        f"{TC_START}\n"
        f'{INV_START}set_timer">\n'
        f'{PARAM_START}wait" string="false">'
        f'{{"type":"for","minutes":2880}}'
        f"{PARAM_END}\n"
        f'{PARAM_START}patches" string="false">'
        f'[{{"op":"replace","path":"/schedule","value":"quiet"}}]'
        f"{PARAM_END}\n"
        f"{INV_END}\n"
        f"{TC_END}"
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {
        "wait": {"type": "for", "minutes": 2880},
        "patches": [{"op": "replace", "path": "/schedule", "value": "quiet"}],
    }


def test_missing_parameter_close_does_not_swallow_next_parameter():
    """Without </｜DSML｜parameter>, the next parameter's opener ends the value
    (batch and streaming, the opener split across deltas)."""
    chunks = [
        TC_START,
        f'{INV_START}get_weather">\n',
        f'{PARAM_START}location" string="true">Paris a<b><｜DSML｜parameter',
        ' name="date" string="true">tomorrow',
        PARAM_END,
        f"\n{INV_END}",
        f"\n{TC_END}",
    ]
    expected = {"location": "Paris a<b>", "date": "tomorrow"}

    result = make_parser().extract_tool_calls("".join(chunks), make_request())
    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == expected

    deltas = stream_chunks(make_parser(), chunks)
    arguments = reconstruct_args(deltas)
    assert "DSML" not in arguments
    assert json.loads(arguments) == expected


_ORPHAN_INVOKE = (
    f'{INV_START}terminal">\n'
    f'{PARAM_START}command" string="true">echo hi{PARAM_END}\n'
    f"{INV_END}\n"
)


@pytest.mark.parametrize("trailing_end", [True, False])
def test_missing_wrapper_with_preamble_and_trailing_text(trailing_end):
    """Upstream issue #48931: an invoke without <｜DSML｜tool_calls> is a call,
    the text before it is content and text after the block is dropped,
    whether or not the closing wrapper is there (batch and streaming)."""
    model_output = "Let me run it.\n" + _ORPHAN_INVOKE
    if trailing_end:
        model_output += TC_END
    model_output += "Done."

    result = make_parser().extract_tool_calls(model_output, make_request())
    assert [call.function.name for call in result.tool_calls] == ["terminal"]
    assert json.loads(result.tool_calls[0].function.arguments) == {"command": "echo hi"}
    assert result.content == "Let me run it.\n"

    deltas = stream_chunks(make_parser(), list(model_output))
    names = [
        tool_call.function.name
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function and tool_call.function.name
    ]
    assert names == ["terminal"]
    assert json.loads(reconstruct_args(deltas)) == {"command": "echo hi"}
    assert "".join(delta.content or "" for delta in deltas) == "Let me run it.\n"


def test_wrapped_block_after_orphan_invoke():
    model_output = _ORPHAN_INVOKE + TC_START + "\n" + _ORPHAN_INVOKE + TC_END

    result = make_parser().extract_tool_calls(model_output, make_request())
    assert [call.function.name for call in result.tool_calls] == [
        "terminal",
        "terminal",
    ]
    assert result.content is None

    deltas = stream_chunks(make_parser(), list(model_output))
    assert [
        tool_call.index
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function and tool_call.function.name
    ] == [0, 1]
    assert not any(delta.content for delta in deltas)


_WEATHER_BLOCK_BODY = (
    f'{INV_START}get_weather">\n'
    f'{PARAM_START}city" string="true">Seoul{PARAM_END}\n'
    f"{INV_END}\n"
    f"{TC_END}"
)


@pytest.mark.parametrize("wrapper", DeepSeekV4ToolParser.tool_call_start_token_variants)
@pytest.mark.parametrize("preamble", ["", "Let me check.\n"])
def test_misspelled_wrapper_behaves_like_real_one(wrapper, preamble):
    """Upstream issue #51914: known misspellings of the opener open a block
    like the real wrapper, so the markup never reaches the content."""
    model_output = f"{preamble}{wrapper}\n{_WEATHER_BLOCK_BODY}"
    reference = make_parser().extract_tool_calls(
        f"{preamble}{TC_START}\n{_WEATHER_BLOCK_BODY}", make_request()
    )

    result = make_parser().extract_tool_calls(model_output, make_request())
    assert [call.function.name for call in result.tool_calls] == ["get_weather"]
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Seoul"}
    assert result.content == reference.content

    deltas = stream_chunks(make_parser(), list(model_output))
    assert json.loads(reconstruct_args(deltas)) == {"city": "Seoul"}
    assert "".join(delta.content or "" for delta in deltas) == (reference.content or "")


@pytest.mark.parametrize(
    "model_output",
    [
        "see <｜DSML｜foo> and <｜DSML｜tool_call> here",
        "see <｜DSML｜ here\nnext line",
        "see <|DSML|tool> and <|DSML|toolcalls> there",
    ],
)
def test_unregistered_dsml_markup_is_content(model_output):
    result = make_parser().extract_tool_calls(model_output, make_request())
    assert not result.tools_called
    assert result.content == model_output

    deltas = stream_chunks(make_parser(), list(model_output))
    assert "".join(delta.content or "" for delta in deltas) == model_output
