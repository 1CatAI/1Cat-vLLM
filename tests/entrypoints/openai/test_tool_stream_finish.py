# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.chat_completion.tool_stream import (
    StreamedToolCall,
    flush_tool_calls,
    record_tool_calls,
    remaining_tool_args,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    RequestResponseMetadata,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


@pytest.mark.parametrize(
    "expected,actual,suffix",
    [
        ('{"x":1}', '{"x":', "1}"),
        ('{"x": 1}', '{ "x" : ', " 1}"),
        ('{"x":"a b"}', '{"x":"a ', 'b"}'),
        ('{"x":"a\\" b"}', '{"x":"a\\" ', 'b"}'),
        ('{"x":1,"y":2}', '{"y":2,"x":1}', ""),
        ('{"x":1.0,"y":2}', '{"y":2,"x":1}', ""),
        ('{"x":1}', '{"x":1}', ""),
    ],
)
def test_finish_suffix_preserves_json_strings(expected, actual, suffix):
    assert remaining_tool_args(expected, actual) == suffix
    assert json.loads(actual + suffix) == json.loads(expected)


@pytest.mark.parametrize("actual", ['{"x":2}', "not a prefix", '{"x":"ab'])
def test_incompatible_arguments_are_never_duplicated(actual):
    with pytest.raises(ValueError, match="streamed prefix"):
        remaining_tool_args('{"x":"a b"}', actual)


@pytest.mark.parametrize(
    "expected,actual",
    [
        ('{"x": false}', '{"x":0}'),
        ('{"x": true}', '{"x":1}'),
        ('{"x":123456789012345678901e0}', '{"x":123456789012345678902}'),
    ],
)
def test_finish_comparison_preserves_json_types_and_exact_numbers(expected, actual):
    with pytest.raises(ValueError, match="streamed prefix"):
        remaining_tool_args(expected, actual)


def _request(n=1):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "call tools"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {"type": "object"},
                        },
                    },
                },
            }
            for name in ["first", "second"]
        ],
        tool_choice="auto",
        stream=True,
        n=n,
        stream_options={"include_usage": True},
    )


def test_finish_supplies_missing_id_type_and_remaining_name():
    request = _request()
    parser = Qwen3CoderToolParser(_tokenizer(), tools=request.tools)
    payload = {"nested": {"x": 1}}
    arguments = json.dumps({"payload": payload})
    seen = {0: StreamedToolCall("", "fir", [arguments])}
    result, complete = flush_tool_calls(parser, _xml(payload), request, None, seen)
    assert complete and result is not None
    record_tool_calls(result.tool_calls, seen)
    assert seen[0].id
    assert seen[0].type == "function"
    assert seen[0].name == "first"
    assert "".join(seen[0].arguments) == arguments


def _tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<tool_call>": 1,
        "</tool_call>": 2,
        "<think>": 100,
        "</think>": 101,
    }
    return tokenizer


def _xml(payload):
    return "".join(
        f"<tool_call><function={name}><parameter=payload>"
        + json.dumps(payload)
        + "</parameter></function></tool_call>"
        for name in ["first", "second"]
    )


def test_finish_uses_actual_wire_state_after_parser_state_reset():
    request = _request()
    parser = Qwen3CoderToolParser(_tokenizer(), tools=request.tools)
    payload = {"nested": {"text": "a b", "values": [1, 2, 3]}}
    args = json.dumps({"payload": payload})
    seen = {0: StreamedToolCall("wire-id", "first", [args[:15]])}
    current = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments=args[15:24]),
            )
        ]
    )
    result, complete = flush_tool_calls(parser, _xml(payload), request, current, seen)
    assert complete and result is not None
    assert seen[0].arguments == [args[:15]]
    record_tool_calls(result.tool_calls, seen)
    assert seen[0].id == "wire-id"
    assert json.loads("".join(seen[0].arguments)) == {"payload": payload}
    assert seen[1].name == "second"
    assert json.loads("".join(seen[1].arguments)) == {"payload": payload}


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("reasoning", [False, True])
async def test_serving_flushes_every_choice_once_with_real_complete_parser(
    n, partial, reasoning
):
    tokenizer = _tokenizer()
    payload = {"nested": {"text": "a b", "values": [1, 2, 3]}}
    expected_args = json.dumps({"payload": payload})

    class DeferredParser:
        def __init__(self, tokenizer, tools, **kwargs):
            self.tool_parser = Qwen3CoderToolParser(tokenizer, tools=tools)
            self._stream_state = SimpleNamespace()
            self.first = True

        def parse_delta(self, **kwargs):
            if partial and self.first:
                self.first = False
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=0,
                            id="wire-id",
                            type="function",
                            function=DeltaFunctionCall(
                                name="first", arguments=expected_args[:24]
                            ),
                        )
                    ]
                )
            return None

    owner = object.__new__(OpenAIServingChat)
    for name, value in dict(
        use_harmony=False,
        parser_cls=DeferredParser,
        tool_parser=Qwen3CoderToolParser,
        enable_auto_tools=True,
        tool_call_id_type="random",
        response_role="assistant",
        enable_force_include_usage=False,
        enable_prompt_tokens_details=True,
        system_fingerprint=None,
        enable_log_outputs=False,
        request_logger=None,
    ).items():
        setattr(owner, name, value)

    def output(text, finished):
        return RequestOutput(
            request_id="request",
            prompt=None,
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            finished=finished,
            outputs=[
                CompletionOutput(
                    index=i,
                    text=text,
                    token_ids=([3, 4] if finished else [9, 101])
                    if reasoning
                    else [9, 3],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop" if finished else None,
                )
                for i in range(n)
            ],
        )

    async def source():
        text = _xml(payload)
        yield output(("thinking</think>" if reasoning else "") + text[:20], False)
        yield output(text[20:], True)
        yield output("", True)  # Defensive: repeated finish must not re-emit.

    metadata = RequestResponseMetadata(request_id="request")
    frames = [
        frame
        async for frame in owner.chat_completion_stream_generator(
            _request(n),
            source(),
            "request",
            "test",
            [],
            tokenizer,
            metadata,
            reasoning_parser=Qwen3ReasoningParser(tokenizer) if reasoning else None,
        )
    ]
    assert frames[-1] == "data: [DONE]\n\n"
    seen: list[dict[int, StreamedToolCall]] = [{} for _ in range(n)]
    finishes = [0] * n
    for frame in frames[:-1]:
        row = json.loads(frame.removeprefix("data: "))
        assert "error" not in row, row
        for choice in row["choices"]:
            i = choice["index"]
            if choice.get("finish_reason"):
                assert choice["finish_reason"] == "tool_calls"
                finishes[i] += 1
            calls = [
                DeltaToolCall.model_validate(call)
                for call in choice["delta"].get("tool_calls", [])
            ]
            record_tool_calls(calls, seen[i])
    assert finishes == [1] * n
    for emitted in seen:
        assert set(emitted) == {0, 1}
        for index, name in enumerate(["first", "second"]):
            assert emitted[index].name == name
            assert emitted[index].id
            assert json.loads("".join(emitted[index].arguments)) == {"payload": payload}
    assert metadata.final_usage_info.completion_tokens == 4 * n
    final = json.loads(frames[-2].removeprefix("data: "))["usage"]
    if reasoning:
        assert final["completion_tokens_details"]["reasoning_tokens"] == n
        assert metadata.final_usage_info.completion_tokens_details.reasoning_tokens == n
    else:
        assert "completion_tokens_details" not in final
