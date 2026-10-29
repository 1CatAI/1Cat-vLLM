# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    RequestResponseMetadata,
)
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


def _serving(parser_factory=None):
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.use_harmony = False
    serving.enable_auto_tools = True
    serving.tool_parser = Qwen3CoderToolParser
    serving.response_role = "assistant"
    serving.tool_call_id_type = "random"
    serving.parser_cls = parser_factory
    serving.enable_force_include_usage = False
    serving.enable_log_outputs = False
    serving.enable_prompt_tokens_details = True
    serving.system_fingerprint = None
    serving.return_tokens_as_token_ids = False
    return serving


def _output(text, ids, finish=None, logprobs=False, index=0):
    return RequestOutput(
        request_id="test",
        prompt="x",
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text=text,
                token_ids=ids,
                cumulative_logprob=-0.5 * len(ids),
                logprobs=[{tid: Logprob(-0.5, 1, text)} for tid in ids]
                if logprobs
                else None,
                finish_reason=finish,
            )
        ],
        finished=finish is not None,
    )


async def _generate(outputs):
    for output in outputs:
        yield output


def _request(**kwargs):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "call f"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
        ],
        **kwargs,
    )


def _tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<think>": 10,
        "</think>": 11,
        "<tool_call>": 12,
        "</tool_call>": 13,
    }
    return tokenizer


def _stream(serving, request, outputs, tokenizer=None):
    async def collect():
        lines = [
            line
            async for line in serving.chat_completion_stream_generator(
                request,
                _generate(outputs),
                "test",
                "test",
                [],
                tokenizer or _tokenizer(),
                RequestResponseMetadata(request_id="test"),
            )
        ]
        assert lines[-1] == "data: [DONE]\n\n"
        chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
        assert not any("error" in chunk for chunk in chunks), chunks
        return chunks

    return asyncio.run(collect())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required", "named"])
@pytest.mark.parametrize("finish", ["stop", "length", "repetition"])
def test_tool_finish_reason_preserves_truncation(stream, choice, finish, monkeypatch):
    # Exercise the real qwen3_coder parser and the required/named bypasses.
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", None)
    serving = _serving(_WrappedParser)
    tool_choice = (
        {"type": "function", "function": {"name": "f"}} if choice == "named" else choice
    )
    request = _request(stream=stream, tool_choice=tool_choice)
    if choice == "auto":
        text = "<tool_call>\n<function=f>\n<parameter=x>value"
        if finish == "stop":
            text += "</parameter>\n</function>\n</tool_call>"
    elif choice == "required":
        text = '[{"name":"f","parameters":{"x":"value'
        if finish == "stop":
            text += '"}}]'
    else:
        text = '{"x":"value' + ('"}' if finish == "stop" else "")
    output = _output(text, [30, 31], finish)
    if stream:
        if choice == "auto":
            pieces = ["<tool_call>\n", "<function=f>\n", "<parameter=x>value"]
            if finish == "stop":
                pieces.append("</parameter>\n</function>\n</tool_call>")
            outputs = [
                _output(
                    piece,
                    [12] if i == 0 else [30 + i],
                    finish if i == len(pieces) - 1 else None,
                )
                for i, piece in enumerate(pieces)
            ]
        else:
            outputs = [output]
        chunks = _stream(serving, request, outputs)
        reasons = [
            c["finish_reason"]
            for chunk in chunks
            for c in chunk["choices"]
            if c.get("finish_reason")
        ]
        assert any(
            c["delta"].get("tool_calls") for chunk in chunks for c in chunk["choices"]
        )
        actual = reasons[0]
    else:
        response = asyncio.run(
            serving.chat_completion_full_generator(
                request,
                _generate([output]),
                "test",
                "test",
                [],
                _tokenizer(),
                RequestResponseMetadata(request_id="test"),
            )
        )
        if choice != "required" or finish == "stop":
            assert response.choices[0].message.tool_calls
        actual = response.choices[0].finish_reason
    expected = "tool_calls" if finish == "stop" and choice != "named" else finish
    assert actual == expected


@pytest.mark.parametrize("return_token_ids", [False, True])
@pytest.mark.parametrize("include_reasoning", [False, True])
def test_parser_suppressed_logprobs_are_preserved(return_token_ids, include_reasoning):
    parser = MagicMock()
    parser.tool_parser = None
    parser.parse_delta.side_effect = [None, None, DeltaMessage(content="answer")]
    serving = _serving(MagicMock(return_value=parser))
    chunks = _stream(
        serving,
        _request(
            stream=True,
            logprobs=True,
            return_token_ids=return_token_ids,
            include_reasoning=include_reasoning,
        ),
        [
            _output("<think>", [10], logprobs=True),
            _output("</think>", [11], logprobs=True),
            _output("answer", [20], "stop", logprobs=True),
        ],
    )
    tokens = [
        lp["token"]
        for chunk in chunks
        for c in chunk["choices"]
        if c.get("logprobs")
        for lp in c["logprobs"]["content"]
    ]
    assert tokens == ["<think>", "</think>", "answer"]


@pytest.mark.parametrize("parser_kind", ["reasoning", "tool", "both"])
def test_real_qwen_parser_control_token_logprobs(parser_kind, monkeypatch):
    monkeypatch.setattr(
        _WrappedParser,
        "tool_parser_cls",
        Qwen3CoderToolParser if parser_kind in ("tool", "both") else None,
    )
    monkeypatch.setattr(
        _WrappedParser,
        "reasoning_parser_cls",
        Qwen3ReasoningParser if parser_kind in ("reasoning", "both") else None,
    )
    marker, token_id = ("<tool_call>", 12) if parser_kind == "tool" else ("<think>", 10)
    parser = _WrappedParser(_tokenizer())
    request = _request(stream=True, logprobs=True, tool_choice="auto")
    assert parser.parse_delta(marker, [token_id], request, prompt_token_ids=[1]) is None
    serving = _serving(_WrappedParser)
    chunks = _stream(
        serving,
        request,
        [
            _output(marker, [token_id], logprobs=True),
            _output("", [], "length", logprobs=True),
        ],
    )
    tokens = [
        lp["token"]
        for chunk in chunks
        for c in chunk["choices"]
        if c.get("logprobs")
        for lp in c["logprobs"]["content"]
    ]
    assert tokens == [marker]


def test_suppressed_logprobs_are_isolated_per_choice():
    parsers = [MagicMock(), MagicMock()]
    for index, parser in enumerate(parsers):
        parser.tool_parser = None
        parser.parse_delta.side_effect = [None, DeltaMessage(content=f"answer{index}")]
    serving = _serving(MagicMock(side_effect=parsers))
    first = _output("marker0", [10], logprobs=True)
    first.outputs += _output("marker1", [11], logprobs=True, index=1).outputs
    last = _output("answer0", [20], "stop", logprobs=True)
    last.outputs += _output("answer1", [21], "stop", logprobs=True, index=1).outputs
    chunks = _stream(serving, _request(stream=True, n=2, logprobs=True), [first, last])
    tokens: dict[int, list[str]] = {0: [], 1: []}
    for chunk in chunks:
        for choice in chunk["choices"]:
            if choice.get("logprobs"):
                tokens[choice["index"]].extend(
                    lp["token"] for lp in choice["logprobs"]["content"]
                )
    assert tokens == {0: ["marker0", "answer0"], 1: ["marker1", "answer1"]}
