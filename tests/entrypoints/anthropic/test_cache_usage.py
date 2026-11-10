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


def _usage(details, timings=None):
    return UsageInfo(
        prompt_tokens=24,
        completion_tokens=3,
        total_tokens=27,
        prompt_tokens_details=details,
        timings=timings,
    )


@pytest.mark.parametrize(
    "details", [None, {}, {"cached_tokens": 0}, {"cached_tokens": 16}]
)
@pytest.mark.parametrize("timings", [None, {"queue_s": 0.0, "ttft_s": 0.125}])
def test_full_response_cache_usage(serving, details, timings):
    response = ChatCompletionResponse(
        id="chatcmpl-cache-test",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0, message=ChatMessage(role="assistant", content="hello")
            )
        ],
        usage=_usage(details, timings),
    )
    result = serving.messages_full_converter(response)

    expected = details.get("cached_tokens") if details is not None else None
    assert result.usage.cache_read_input_tokens == expected
    assert result.usage.input_tokens == 24 - (expected or 0)
    assert result.usage.output_tokens == 3
    assert result.usage.cache_creation_input_tokens is None
    assert result.id == "msg_cache-test"
    assert result.model == response.model
    serialized = result.model_dump(exclude_unset=True)
    assert serialized["type"] == "message"
    assert serialized["role"] == "assistant"
    assert result.content[0].text == "hello"
    assert result.stop_reason == "end_turn"
    assert result.usage.timings == timings
    assert ("timings" in serialized["usage"]) == (timings is not None)


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
@pytest.mark.parametrize("timings", [None, {"queue_s": 0.0, "ttft_s": 0.125}])
def test_stream_response_cache_usage(serving, initial_details, final_details, timings):
    async def source():
        for details in [initial_details, final_details]:
            chunk = ChatCompletionStreamResponse(
                id="chatcmpl-cache-test",
                model="test-model",
                choices=[],
                usage=_usage(details, timings),
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
    assert message["id"] == "msg_cache-test"
    assert message["model"] == "test-model"
    assert message["type"] == "message"
    assert message["role"] == "assistant"
    for usage, details, output_tokens in [
        (message["usage"], initial_details, 0),
        (events[1]["usage"], final_details, 3),
    ]:
        expected_cached = details.get("cached_tokens") if details is not None else None
        assert usage["input_tokens"] == 24 - (expected_cached or 0)
        assert usage["output_tokens"] == output_tokens
        expected = details.get("cached_tokens") if details is not None else None
        if expected is None:
            assert "cache_read_input_tokens" not in usage
        else:
            assert usage["cache_read_input_tokens"] == expected
        assert "cache_creation_input_tokens" not in usage
        # message_start reports zero output tokens; complete phase timings
        # belong to the final message_delta, even if its source was prefilled.
        if timings is None or output_tokens == 0:
            assert "timings" not in usage
        else:
            assert usage["timings"] == timings


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
