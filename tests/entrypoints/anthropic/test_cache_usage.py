# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import pytest

from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import UsageInfo


@pytest.fixture
def serving():
    serving = object.__new__(AnthropicServingMessages)
    serving.stop_reason_map = {"stop": "end_turn"}
    return serving


def _usage(details):
    return UsageInfo(
        prompt_tokens=24,
        completion_tokens=3,
        total_tokens=27,
        prompt_tokens_details=details,
    )


@pytest.mark.parametrize(
    "details", [None, {}, {"cached_tokens": 0}, {"cached_tokens": 16}]
)
def test_full_response_cache_usage(serving, details):
    response = ChatCompletionResponse(
        id="chatcmpl-cache-test",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0, message=ChatMessage(role="assistant", content="hello")
            )
        ],
        usage=_usage(details),
    )
    result = serving.messages_full_converter(response)

    expected = details.get("cached_tokens") if details is not None else None
    assert result.usage.cache_read_input_tokens == expected
    assert result.usage.input_tokens == 24
    assert result.usage.output_tokens == 3
    assert result.usage.cache_creation_input_tokens is None
    assert result.id == response.id
    assert result.model == response.model
    assert result.content[0].text == "hello"
    assert result.stop_reason == "end_turn"


@pytest.mark.parametrize(
    "initial_details,final_details",
    [
        (None, None),
        ({}, {}),
        ({"cached_tokens": 0}, {"cached_tokens": 0}),
        ({"cached_tokens": 8}, {"cached_tokens": 16}),
        (None, {"cached_tokens": 16}),
    ],
)
def test_stream_response_cache_usage(serving, initial_details, final_details):
    async def source():
        for details in [initial_details, final_details]:
            chunk = ChatCompletionStreamResponse(
                id="chatcmpl-cache-test",
                model="test-model",
                choices=[],
                usage=_usage(details),
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    async def convert():
        return [item async for item in serving.message_stream_converter(source())]

    events = [json.loads(item.split("data: ", 1)[1]) for item in asyncio.run(convert())]
    assert [event["type"] for event in events] == [
        "message_start",
        "message_delta",
        "message_stop",
    ]
    message = events[0]["message"]
    assert message["id"] == "chatcmpl-cache-test"
    assert message["model"] == "test-model"
    for usage, details, output_tokens in [
        (message["usage"], initial_details, 0),
        (events[1]["usage"], final_details, 3),
    ]:
        assert usage["input_tokens"] == 24
        assert usage["output_tokens"] == output_tokens
        expected = details.get("cached_tokens") if details is not None else None
        if expected is None:
            assert "cache_read_input_tokens" not in usage
        else:
            assert usage["cache_read_input_tokens"] == expected
        assert "cache_creation_input_tokens" not in usage


def test_stream_without_usage_preserves_zero_counts(serving):
    async def source():
        for _ in range(2):
            chunk = ChatCompletionStreamResponse(model="test-model", choices=[])
            yield f"data: {chunk.model_dump_json()}\n\n"

    async def convert():
        return [item async for item in serving.message_stream_converter(source())]

    events = [json.loads(item.split("data: ", 1)[1]) for item in asyncio.run(convert())]
    assert events[0]["message"]["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert events[1]["usage"] == {"input_tokens": 0, "output_tokens": 0}
