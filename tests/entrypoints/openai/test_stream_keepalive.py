# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.outputs import STREAM_KEEPALIVE, CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import async_llm
from vllm.v1.engine.output_processor import RequestOutputCollector


def _finished_output():
    return RequestOutput(
        request_id="request",
        prompt=None,
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="2",
                token_ids=[4],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
        num_cached_tokens=2,
    )


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(async_llm, "_STREAM_KEEPALIVE_INTERVAL", 0.001)
    queue = RequestOutputCollector(RequestOutputKind.DELTA, "internal-request")
    owner = SimpleNamespace(
        add_request=AsyncMock(return_value=queue),
        abort=AsyncMock(),
        log_requests=False,
    )
    return owner, queue


def _generate(owner, **kwargs):
    return async_llm.AsyncLLM.generate(
        owner, {"prompt_token_ids": [1]}, SamplingParams(), "request", **kwargs
    )


@pytest.mark.asyncio
async def test_idle_heartbeat_then_real_output_has_no_completion_or_queue_loss(engine):
    owner, queue = engine
    source = _generate(owner, keepalive=True)
    assert await anext(source) is STREAM_KEEPALIVE
    assert await anext(source) is STREAM_KEEPALIVE
    output = _finished_output()
    queue.put(output)
    assert await anext(source) is output
    with pytest.raises(StopAsyncIteration):
        await anext(source)
    owner.abort.assert_not_awaited()
    assert not queue.ready._waiters


@pytest.mark.asyncio
async def test_default_callers_only_receive_real_outputs(engine):
    owner, queue = engine
    source = _generate(owner)
    pending = asyncio.create_task(anext(source))
    await asyncio.sleep(0.01)
    assert not pending.done()
    output = _finished_output()
    queue.put(output)
    assert await pending is output
    await source.aclose()
    assert not queue.ready._waiters


@pytest.mark.asyncio
async def test_close_after_heartbeat_aborts_internal_request_once(engine):
    owner, queue = engine
    source = _generate(owner, keepalive=True)
    assert await anext(source) is STREAM_KEEPALIVE
    await source.aclose()
    owner.abort.assert_awaited_once_with("internal-request", internal=True)
    assert not queue.ready._waiters


@pytest.mark.asyncio
async def test_cancel_during_heartbeat_wait_aborts_without_orphan_waiter(engine):
    owner, queue = engine
    source = _generate(owner, keepalive=True)
    pending = asyncio.create_task(anext(source))
    # add_request completes before the timed queue wait is cancelled.
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    owner.abort.assert_awaited_once_with("internal-request", internal=True)
    assert not queue.ready._waiters


@pytest.mark.asyncio
async def test_engine_validation_error_is_not_a_heartbeat(engine):
    owner, queue = engine
    source = _generate(owner, keepalive=True)
    queue.put(ValueError("bad input"))
    with pytest.raises(ValueError, match="bad input"):
        await anext(source)
    owner.abort.assert_not_awaited()


def _serving():
    serving = object.__new__(OpenAIServingChat)
    for name, value in dict(
        use_harmony=False,
        parser_cls=None,
        tool_parser=None,
        enable_auto_tools=False,
        tool_call_id_type="random",
        response_role="assistant",
        enable_force_include_usage=False,
        enable_prompt_tokens_details=True,
        system_fingerprint=None,
        enable_log_outputs=False,
        request_logger=None,
    ).items():
        setattr(serving, name, value)
    return serving


@pytest.mark.asyncio
@pytest.mark.parametrize("anthropic", [False, True])
async def test_sse_heartbeat_preserves_content_usage_and_end_marker(anthropic):
    async def engine_outputs():
        yield STREAM_KEEPALIVE
        yield _finished_output()

    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "1+1"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    wire = _serving().chat_completion_stream_generator(
        request,
        engine_outputs(),
        "request",
        "test",
        [],
        SimpleNamespace(),
        RequestResponseMetadata(request_id="request"),
    )
    if anthropic:
        converter = object.__new__(AnthropicServingMessages)
        converter.stop_reason_map = {"stop": "end_turn", "length": "max_tokens"}
        wire = converter.message_stream_converter(wire)
    frames = [frame async for frame in wire]
    assert frames[0] == ": keepalive\n\n"
    data = [
        json.loads(line[5:].strip())
        for frame in frames[1:]
        for line in frame.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]
    assert not any("error" in row for row in data)
    if anthropic:
        assert data[-1]["type"] == "message_stop"
        assert any(row.get("delta", {}).get("text") == "2" for row in data)
        usage = next(row["usage"] for row in data if row["type"] == "message_delta")
        assert usage["output_tokens"] == 1
    else:
        assert frames[-1] == "data: [DONE]\n\n"
        assert data[-1]["usage"]["prompt_tokens"] == 3
        assert data[-1]["usage"]["completion_tokens"] == 1
        assert data[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
        assert any(
            choice["delta"].get("content") == "2"
            for row in data
            for choice in row["choices"]
        )
