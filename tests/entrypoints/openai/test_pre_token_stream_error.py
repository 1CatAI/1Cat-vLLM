# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import pytest

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _generate,
    _output,
    _request,
    _serving,
    _tokenizer,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata


@pytest.mark.parametrize("api", ["chat", "completion"])
@pytest.mark.parametrize("has_tokens", [False, True])
@pytest.mark.parametrize("error", [False, True])
def test_stream_error_checked_before_empty_prefill(api, has_tokens, error):
    outputs = [_output("", [], None)]
    if has_tokens:
        outputs.append(_output("answer", [20]))
    outputs.append(_output("", [], "error" if error else "stop"))
    metadata = RequestResponseMetadata(request_id="test")

    async def collect():
        if api == "chat":
            serving = _serving()
            generator = serving.chat_completion_stream_generator(
                _request(stream=True, tool_choice="none"),
                _generate(outputs),
                "test",
                "test",
                [],
                _tokenizer(),
                metadata,
            )
        else:
            serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
            serving.enable_force_include_usage = False
            serving.enable_prompt_tokens_details = False
            serving.system_fingerprint = None

            async def indexed():
                for output in outputs:
                    yield 0, output

            generator = serving.completion_stream_generator(
                CompletionRequest(model="test", prompt="x", stream=True),
                [],
                indexed(),
                "test",
                0,
                "test",
                1,
                _tokenizer(),
                metadata,
            )
        return [line async for line in generator]

    lines = asyncio.run(collect())
    assert lines[-1] == "data: [DONE]\n\n"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    errors = [chunk for chunk in chunks if "error" in chunk]
    assert bool(errors) is error
    if error:
        assert errors[0]["error"]["type"] == "InternalServerError"
