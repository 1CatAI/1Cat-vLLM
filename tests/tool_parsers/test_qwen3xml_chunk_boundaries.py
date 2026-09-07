# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A closing tag in a chunk must not close the next tool call."""

import json
import random
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3xml_tool_parser import (
    Qwen3XMLToolParser,
    StreamingXMLToolCallParser,
)

PARALLEL_CALLS = (
    "<tool_call>\n<function=read>\n<parameter=path>/tmp/a</parameter>\n"
    "</function>\n</tool_call>\n"
    "<tool_call>\n<function=bash>\n<parameter=command>pwd</parameter>\n"
    "</function>\n</tool_call>"
)


def assert_parallel_calls(chunks):
    parser = StreamingXMLToolCallParser()
    calls = {}
    for chunk in chunks:
        delta = parser.parse_single_streaming_chunks(chunk)
        for call in delta.tool_calls or []:
            entry = calls.setdefault(
                call.index, {"id": call.id, "name": "", "arguments": ""}
            )
            assert call.id == entry["id"]
            if call.function:
                entry["name"] += call.function.name or ""
                entry["arguments"] += call.function.arguments or ""
    assert list(calls) == [0, 1]
    assert calls[0]["id"] != calls[1]["id"]
    assert [call["name"] for call in calls.values()] == ["read", "bash"]
    assert [json.loads(call["arguments"]) for call in calls.values()] == [
        {"path": "/tmp/a"},
        {"command": "pwd"},
    ]


@pytest.mark.parametrize("width", range(1, len(PARALLEL_CALLS) + 1))
def test_parallel_calls_fixed_chunks(width):
    assert_parallel_calls(
        PARALLEL_CALLS[pos : pos + width]
        for pos in range(0, len(PARALLEL_CALLS), width)
    )


@pytest.mark.parametrize("split", range(1, len(PARALLEL_CALLS)))
def test_parallel_calls_single_split(split):
    assert_parallel_calls([PARALLEL_CALLS[:split], PARALLEL_CALLS[split:]])


@pytest.mark.parametrize("seed", range(100))
def test_parallel_calls_irregular_chunks(seed):
    rng = random.Random(seed)
    chunks = []
    pos = 0
    while pos < len(PARALLEL_CALLS):
        width = rng.randint(1, 50)
        chunks.append(PARALLEL_CALLS[pos : pos + width])
        pos += width
    assert_parallel_calls(chunks)


@pytest.mark.parametrize("width", [1, 25, 33, 58, 135, len(PARALLEL_CALLS)])
def test_public_streaming_matches_nonstreaming(width):
    parser = Qwen3XMLToolParser(Mock())
    request = ChatCompletionRequest(model="test", messages=[])
    expected = parser.extract_tool_calls(PARALLEL_CALLS, request)
    assert len(expected.tool_calls) == 2
    calls = {}
    previous = ""
    for pos in range(0, len(PARALLEL_CALLS), width):
        chunk = PARALLEL_CALLS[pos : pos + width]
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous, current, chunk, [], [], [], request
        )
        previous = current
        if delta is None:
            continue
        for call in delta.tool_calls or []:
            entry = calls.setdefault(call.index, {"name": "", "arguments": ""})
            if call.function:
                entry["name"] += call.function.name or ""
                entry["arguments"] += call.function.arguments or ""
    assert list(calls) == [0, 1]
    for index, call in enumerate(expected.tool_calls):
        assert calls[index]["name"] == call.function.name
        assert json.loads(calls[index]["arguments"]) == json.loads(
            call.function.arguments
        )
    assert [c["name"] for c in parser.prev_tool_call_arr] == ["read", "bash"]


@pytest.mark.parametrize("width", range(1, 81))
def test_empty_and_multiline_calls(width):
    text = (
        "<tool_call><function=refresh></function></tool_call>\n"
        "<tool_call><function=write><parameter=path>/tmp/a</parameter>"
        '<parameter=content>line 1\n  "quoted" & x &lt; y\nline 3</parameter>'
        "</function></tool_call>"
    )
    parser = StreamingXMLToolCallParser()
    calls = {}
    for pos in range(0, len(text), width):
        delta = parser.parse_single_streaming_chunks(text[pos : pos + width])
        for call in delta.tool_calls or []:
            entry = calls.setdefault(call.index, {"name": "", "arguments": ""})
            if call.function:
                entry["name"] += call.function.name or ""
                entry["arguments"] += call.function.arguments or ""
    assert list(calls) == [0, 1]
    assert [call["name"] for call in calls.values()] == ["refresh", "write"]
    assert json.loads(calls[0]["arguments"]) == {}
    assert json.loads(calls[1]["arguments"]) == {
        "path": "/tmp/a",
        "content": 'line 1\n  "quoted" & x &lt; y\nline 3',
    }
