# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tool_choice="none" must not produce tool calls in streaming either.

The non-streaming path never runs the tool parser for tool_choice="none" and
returns any tool-call markup the model writes as content. Streaming used to
run the tool parser on every delta regardless of tool_choice, so the same
output became a tool call with finish_reason "tool_calls".
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


# Local CPU fixtures keep this regression independent of later API backports.
def _byte_tokenizer():
    # Byte-level BPE without merges: every UTF-8 byte is its own token, so a
    # CJK character or an emoji spans three or four tokens, as rare characters
    # do in the Qwen vocabularies.
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.decoder = decoders.ByteLevel()
    backend.add_tokens(
        [
            AddedToken(token, special=False, normalized=False)
            for token in ("<think>", "</think>", "<tool_call>", "</tool_call>")
        ]
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def _choices(chunks):
    choices: dict[int, list[dict]] = {}
    for chunk in chunks:
        for choice in chunk["choices"]:
            if "role" not in choice["delta"]:
                choices.setdefault(choice["index"], []).append(choice)
    return choices


async def _generate(outputs):
    for output in outputs:
        yield output


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


def _tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<think>": 10,
        "</think>": 11,
        "<tool_call>": 12,
        "</tool_call>": 13,
    }
    spellings = {10: "<think>", 11: "</think>", 12: "<tool_call>", 13: "</tool_call>"}
    tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
        spellings.get(i, "") for i in ids
    )
    return tokenizer


def _stream(serving, request, outputs, tokenizer=None, chat_template_kwargs=None):
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
                chat_template_kwargs=chat_template_kwargs,
            )
        ]
        assert lines[-1] == "data: [DONE]\n\n"
        chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
        assert not any("error" in chunk for chunk in chunks), chunks
        return chunks

    return asyncio.run(collect())


def _text(choices, field):
    return "".join(choice["delta"].get(field) or "" for choice in choices)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    }
]
XML_CALL = (
    "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>\n"
    "</tool_call>"
)


@pytest.fixture
def tok():
    return _byte_tokenizer()


@pytest.fixture(autouse=True)
def wrapped_parser(monkeypatch):
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", Qwen3ReasoningParser)


def _request(stream, with_tools):
    kwargs = {"tools": TOOLS, "tool_choice": "none"} if with_tools else {}
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "call f"}],
        stream=stream,
        **kwargs,
    )


def _with_prompt(output: RequestOutput, tok, thinking: bool) -> RequestOutput:
    # The chat template ends the prompt with an open <think> when thinking is
    # on, and with a closed empty block when it is off.
    prompt = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    output.prompt_token_ids = tok.encode(prompt, add_special_tokens=False)
    return output


def _steps(tok, text, thinking=False, finish="stop", eos_alone=False):
    """One engine step per token; optionally a final text-less EOS step."""
    ids = tok.encode(text, add_special_tokens=False)
    outputs = [
        _output(
            tok.decode([tid]), [tid], None if eos_alone or i < len(ids) - 1 else finish
        )
        for i, tid in enumerate(ids)
    ]
    if eos_alone:
        outputs.append(_output("", [0], finish))
    return [_with_prompt(output, tok, thinking) for output in outputs]


def _full(tok, request, text, thinking):
    ids = tok.encode(text, add_special_tokens=False)
    kwargs = {"enable_thinking": thinking}
    response = asyncio.run(
        _serving(_WrappedParser).chat_completion_full_generator(
            request,
            _generate([_with_prompt(_output(text, ids, "stop"), tok, thinking)]),
            "test",
            "test",
            [],
            tok,
            RequestResponseMetadata(request_id="test"),
            reasoning_parser=Qwen3ReasoningParser(tok, chat_template_kwargs=kwargs),
        )
    )
    return response.choices[0]


@pytest.mark.parametrize("with_tools", [True, False])
def test_streamed_tool_markup_stays_content(tok, with_tools):
    request = _request(stream=True, with_tools=with_tools)
    chunks = _stream(
        _serving(_WrappedParser),
        request,
        _steps(tok, XML_CALL),
        tokenizer=tok,
        chat_template_kwargs={"enable_thinking": False},
    )
    choices = _choices(chunks)[0]
    assert not any(choice["delta"].get("tool_calls") for choice in choices)
    assert _text(choices, "content") == XML_CALL
    assert choices[-1]["finish_reason"] == "stop"

    full = _full(tok, _request(stream=False, with_tools=with_tools), XML_CALL, False)
    assert full.message.content == XML_CALL
    assert not full.message.tool_calls
    assert full.finish_reason == "stop"


def test_streamed_tool_markup_after_reasoning_stays_content(tok):
    text = "Think first.</think>\n\n" + XML_CALL
    chunks = _stream(
        _serving(_WrappedParser),
        _request(stream=True, with_tools=True),
        _steps(tok, text, thinking=True),
        tokenizer=tok,
        chat_template_kwargs={"enable_thinking": True},
    )
    choices = _choices(chunks)[0]
    assert not any(choice["delta"].get("tool_calls") for choice in choices)
    assert _text(choices, "reasoning") == "Think first."
    assert _text(choices, "content") == "\n\n" + XML_CALL
    assert choices[-1]["finish_reason"] == "stop"

    full = _full(tok, _request(stream=False, with_tools=True), text, True)
    assert full.message.content == "\n\n" + XML_CALL
    assert full.message.reasoning == "Think first."


@pytest.mark.parametrize("with_tools", [True, False])
def test_plain_text_keeps_final_empty_content_chunk(tok, with_tools):
    # A text-less EOS step still ends with an empty content delta, as before.
    text = "Plain answer: a < b and c > d."
    chunks = _stream(
        _serving(_WrappedParser),
        _request(stream=True, with_tools=with_tools),
        _steps(tok, text, eos_alone=True),
        tokenizer=tok,
        chat_template_kwargs={"enable_thinking": False},
    )
    choices = _choices(chunks)[0]
    assert _text(choices, "content") == text
    assert choices[-1]["delta"] == {"content": ""}
    assert choices[-1]["finish_reason"] == "stop"
