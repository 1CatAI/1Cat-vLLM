# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import _WrappedParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from vllm.tool_parsers.streaming import extract_required_tool_call_streaming


def _collect(messages):
    calls = {}
    for message in messages:
        if message is None:
            continue
        for call in message.tool_calls:
            if call.id is not None:
                assert call.index not in calls, "duplicate call header"
                assert call.function.name is not None
                calls[call.index] = {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": "",
                }
            else:
                assert call.index in calls, "arguments arrived before header"
                assert call.function.name is None
            calls[call.index]["arguments"] += call.function.arguments or ""
    return calls


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 19, 10000])
@pytest.mark.parametrize("id_type", ["random", "kimi_k2"])
def test_required_multiple_calls_keep_exact_arguments(chunk_size, id_type):
    arguments = [
        '{"text": "brace }, bracket ], quote \\" and slash \\\\"}',
        '{ "nested": [{"parameters": "inside"}], "empty": {} }',
        "{}",
    ]
    # Keep whitespace and escaping: reserializing the arguments is insufficient.
    text = (
        "["
        + ",".join(
            f'{{"name":"f{i}","parameters":{args}}}' for i, args in enumerate(arguments)
        )
        + "]\nDONE"
    )
    previous = ""
    returned = False
    count = 7
    messages = []
    for offset in range(0, len(text), chunk_size):
        delta = text[offset : offset + chunk_size]
        current = previous + delta
        message, returned = extract_required_tool_call_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            function_name_returned=returned,
            tool_call_idx=count,
            tool_call_id_type=id_type,
        )
        if message:
            count += sum(call.id is not None for call in message.tool_calls)
        messages.append(message)
        previous = current
    calls = _collect(messages)
    assert list(calls) == [0, 1, 2]
    assert count == 10
    assert len({call["id"] for call in calls.values()}) == 3
    for i, args in enumerate(arguments):
        assert calls[i]["name"] == f"f{i}"
        assert calls[i]["arguments"] == args
        json.loads(calls[i]["arguments"])
        if id_type == "kimi_k2":
            assert calls[i]["id"] == f"functions.f{i}:{i + 7}"


@pytest.mark.parametrize("split", [False, True])
def test_qwen3_coder_required_counts_every_header(monkeypatch, split):
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", None)
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {"<tool_call>": 1, "</tool_call>": 2}
    parser = _WrappedParser(tokenizer)
    parser._stream_state.tool_call_id_type = "kimi_k2"
    parser._stream_state.history_tool_call_cnt = 4
    other = _WrappedParser(tokenizer)
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "call tools"}],
        tool_choice="required",
        tools=[
            {"type": "function", "function": {"name": name}} for name in ("a", "b", "c")
        ],
    )
    first = '[{"name":"a","parameters":{}},{"name":"b","parameters":{}}'
    second = ',{"name":"c","parameters":{"x":1}}]'
    chunks = [first, second] if split else [first + second]
    messages = [
        parser.parse_delta(chunk, [3], request, prompt_token_ids=[]) for chunk in chunks
    ]
    calls = _collect(messages)
    assert [call["id"] for call in calls.values()] == [
        "functions.a:4",
        "functions.b:5",
        "functions.c:6",
    ]
    assert [json.loads(call["arguments"]) for call in calls.values()] == [
        {},
        {},
        {"x": 1},
    ]
    assert parser._stream_state.history_tool_call_cnt == 7
    assert other._stream_state.history_tool_call_cnt == 0
