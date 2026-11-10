# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage, UsageInfo


def _request(**kwargs):
    return AnthropicMessagesRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        **kwargs,
    )


@pytest.fixture
def serving():
    owner = object.__new__(AnthropicServingMessages)
    owner.stop_reason_map = {"stop": "end_turn", "length": "max_tokens"}
    return owner


def _response():
    return ChatCompletionResponse(
        id="chatcmpl-123",
        model="test",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello"),
                finish_reason="stop",
                token_ids=[11, 12],
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        prompt_token_ids=[1, 2, 3],
    )


@pytest.mark.parametrize(
    "output_ids,prompt_ids",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_full_response_extensions_are_independently_opted_in(
    serving, output_ids, prompt_ids
):
    result = serving.messages_full_converter(
        _response(),
        _request(return_token_ids=output_ids, return_prompt_token_ids=prompt_ids),
    ).model_dump()
    assert result["id"] == "msg_123"
    assert ("token_ids" in result) is output_ids
    assert ("prompt_token_ids" in result) is prompt_ids
    if output_ids:
        assert result["token_ids"] == [11, 12]
    if prompt_ids:
        assert result["prompt_token_ids"] == [1, 2, 3]
    assert result["usage"]["input_tokens"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header,body,prompt,want_output",
    [
        ("true", False, False, True),
        ("0", True, False, True),
        ("false", False, True, False),
        (None, False, False, False),
    ],
)
async def test_header_opt_in_survives_request_conversion(
    serving, header, body, prompt, want_output
):
    serving.create_chat_completion = AsyncMock(return_value=_response())
    headers = [] if header is None else [(b"x-return-token-ids", header.encode())]
    raw = Request({"type": "http", "headers": headers})
    request = _request(return_token_ids=body, return_prompt_token_ids=prompt)
    result = await serving.create_messages(request, raw)
    generated_request = serving.create_chat_completion.call_args.args[0]
    assert generated_request.return_token_ids is (want_output or prompt)
    assert ("token_ids" in result.model_dump()) is want_output
    assert ("prompt_token_ids" in result.model_dump()) is prompt
    assert request.return_token_ids is body


def _chunk(*, delta=None, ids=None, usage=None):
    return ChatCompletionStreamResponse(
        id="chatcmpl-123",
        model="test",
        prompt_token_ids=[1, 2, 3],
        choices=[]
        if delta is None
        else [
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(**delta),
                token_ids=ids,
                finish_reason="stop" if delta == {} else None,
            )
        ],
        usage=usage,
    )


def _payloads(frames):
    return [
        json.loads(line[5:].strip())
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("data:")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize(
    "output_ids,prompt_ids",
    [(False, False), (True, False), (False, True), (True, True)],
)
async def test_fragmented_sse_and_control_tokens_preserve_usage(
    serving, fragmented, output_ids, prompt_ids
):
    chunks = [
        _chunk(delta={"content": "你好"}, ids=[11]),
        _chunk(delta={}, ids=[12]),
        _chunk(
            usage=UsageInfo(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
                prompt_tokens_details={"cached_tokens": 2},
                timings={"queue_s": 0.0, "ttft_s": 0.1},
            )
        ),
    ]
    text = (
        ": keepalive\n\nevent: message\nid: upstream\n"
        + "".join("data: " + chunk.model_dump_json() + "\n\n" for chunk in chunks)
        + "data: [DONE]"
    )  # The internal producer may omit the last blank line.
    text = text.replace("\n", "\r\n")

    async def source():
        if fragmented:
            for i in range(0, len(text), 7):
                yield text[i : i + 7]
        else:
            yield text

    frames = [
        frame
        async for frame in serving.message_stream_converter(
            source(),
            _request(return_token_ids=output_ids, return_prompt_token_ids=prompt_ids),
        )
    ]
    assert frames[0] == ": keepalive\n\n"
    events = _payloads(frames)
    assert not any("error" in event for event in events)
    assert events[0]["type"] == "message_start"
    assert events[0]["message"]["id"] == "msg_123"
    assert events[-1]["type"] == "message_stop"
    assert any(event.get("delta", {}).get("text") == "你好" for event in events)
    message = events[0]["message"]
    assert ("prompt_token_ids" in message) is prompt_ids
    assert [token for event in events for token in event.get("token_ids", [])] == (
        [11, 12] if output_ids else []
    )
    usage = next(event["usage"] for event in events if event["type"] == "message_delta")
    assert usage["input_tokens"] == 1
    assert usage["cache_read_input_tokens"] == 2
    assert usage["output_tokens"] == 2
    assert usage["timings"] == {"queue_s": 0.0, "ttft_s": 0.1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [{"message": "backend failed", "type": "server_error"}, "backend failed"]
)
async def test_upstream_error_keeps_its_message(serving, error):
    async def source():
        yield "data: " + json.dumps({"error": error}) + "\n\n"
        pytest.fail("converter must stop consuming after an upstream error")

    events = _payloads(
        [frame async for frame in serving.message_stream_converter(source())]
    )
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["error"]["message"] == "backend failed"
