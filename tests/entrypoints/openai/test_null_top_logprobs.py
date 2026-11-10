# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest
from pydantic import ValidationError

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _generate,
    _output,
    _request,
    _serving,
    _stream,
    _tokenizer,
)
from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import BatchChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("top", [None, 0, 1, -1])
def test_null_top_logprobs_requests_sampled_token(enabled, top):
    if not enabled and top not in (None, 0):
        with pytest.raises(ValidationError, match="logprobs.*must be set to true"):
            _request(logprobs=enabled, top_logprobs=top)
        return
    request = _request(logprobs=enabled, top_logprobs=top)
    params = request.to_sampling_params(10, {})
    assert params.logprobs == ((top or 0) if enabled else None)


@pytest.mark.parametrize("mode", ["stream", "full", "batch"])
@pytest.mark.parametrize("top", [None, 0, 1, -1])
def test_logprobs_rendered_when_top_is_null(mode, top):
    serving = _serving()
    request = _request(
        logprobs=True, top_logprobs=top, tool_choice="none", stream=mode == "stream"
    )
    output = _output("answer", [20], "stop", logprobs=True)
    metadata = RequestResponseMetadata(request_id="test")
    if mode == "stream":
        chunks = _stream(serving, request, [output])
        probs = [
            choice["logprobs"]
            for chunk in chunks
            for choice in chunk["choices"]
            if choice.get("logprobs")
        ]
        assert len(probs) == 1
        content = probs[0]["content"]
    else:
        if mode == "full":
            response = asyncio.run(
                serving.chat_completion_full_generator(
                    request,
                    _generate([output]),
                    "test",
                    "test",
                    [],
                    _tokenizer(),
                    metadata,
                )
            )
        else:
            batch = OpenAIServingChatBatch.__new__(OpenAIServingChatBatch)
            batch.__dict__.update(serving.__dict__)
            batch_request = BatchChatCompletionRequest(
                model="test",
                messages=[[{"role": "user", "content": "x"}]],
                logprobs=True,
                top_logprobs=top,
            )
            single = batch_request.to_chat_completion_request(batch_request.messages[0])
            assert single.to_sampling_params(10, {}).logprobs == (top or 0)
            response = asyncio.run(
                batch.chat_completion_full_generator_batch(
                    batch_request,
                    [_generate([output])],
                    "test",
                    "test",
                    [[]],
                    _tokenizer(),
                    metadata,
                )
            )
        assert response.choices[0].logprobs is not None
        content = [item.model_dump() for item in response.choices[0].logprobs.content]
    assert len(content) == 1
    assert content[0]["token"] == "answer"
    assert content[0]["logprob"] == -0.5
    assert len(content[0]["top_logprobs"]) == (1 if top in (-1, 1) else 0)
