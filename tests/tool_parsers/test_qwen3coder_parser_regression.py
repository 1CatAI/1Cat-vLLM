# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


class _DummyTokenizer:

    def get_vocab(self) -> dict[str, int]:
        return {
            "<tool_call>": 248058,
            "</tool_call>": 248059,
        }


def _tool(name: str, properties: dict):
    return SimpleNamespace(
        type="function",
        function=SimpleNamespace(
            name=name,
            parameters={
                "type": "object",
                "properties": properties,
            },
        ),
    )


def test_qwen3coder_parser_keeps_duplicate_calls_and_typed_arguments() -> None:
    parser = Qwen3CoderToolParser(_DummyTokenizer())
    request = SimpleNamespace(
        tools=[
            _tool(
                "read",
                {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                    "payload": {"anyOf": [{"type": "object"}, {"type": "string"}]},
                },
            )
        ],
    )
    output = (
        "<tool_call>\n"
        "<function=read>\n"
        "<parameter=path>\n/a\n</parameter>\n"
        "<parameter=limit>\n2\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
        "<tool_call>\n"
        "<function=read>\n"
        "<parameter=payload>\n{\"x\": 1}\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    parsed = parser.extract_tool_calls(output, request)

    assert parsed.tools_called
    assert len(parsed.tool_calls) == 2
    assert [tool.function.name for tool in parsed.tool_calls] == ["read", "read"]
    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "path": "/a",
        "limit": 2,
    }
    assert json.loads(parsed.tool_calls[1].function.arguments) == {
        "payload": {"x": 1},
    }


def test_qwen3coder_parser_ignores_malformed_function_header() -> None:
    parser = Qwen3CoderToolParser(_DummyTokenizer())
    request = SimpleNamespace(tools=[_tool("read", {"path": {"type": "string"}})])

    parsed = parser.extract_tool_calls("<tool_call><function=read", request)

    assert not parsed.tools_called
    assert parsed.tool_calls == []
