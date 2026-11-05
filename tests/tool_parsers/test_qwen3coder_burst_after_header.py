# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A qwen3_coder call whose body arrives in one burst after its header.

The streaming parser sends the header, then "{", then the parameters, one
stage per delta. With speculative decoding a delta carries several tokens, so
a short call (for example one without parameters) can finish its whole body,
`</function>\n</tool_call>`, in the delta right after the header. The parser
then sent only "{" and returned; when the stream ended with an EOS-only delta
nothing completed the arguments, and the client was left with "{".
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


TOOLS = [
    {"type": "function", "function": {"name": "list_files"}},
    {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    },
]
NOARG = "<tool_call>\n<function=list_files>\n</function>\n</tool_call>"
SHORT = (
    "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>\n"
    "</tool_call>"
)

CASES = {
    # header | whole body | EOS alone
    "noarg": ["<tool_call>", "\n<function=list_files>\n", "</function>\n</tool_call>"],
    "short": ["<tool_call>\n<function=f>", SHORT[len("<tool_call>\n<function=f>") :]],
    "text+noarg": [
        "Listing.\n\n<tool_call>\n<function=list_files>",
        "\n</function>\n</tool_call>",
    ],
    "call+noarg": [
        SHORT[:30],
        SHORT[30:],
        "\n<tool_call>\n<function=list_files>",
        "\n</function>\n</tool_call>",
    ],
}


@pytest.fixture
def tok():
    return _byte_tokenizer()


@pytest.fixture(autouse=True)
def wrapped_parser(monkeypatch):
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", Qwen3ReasoningParser)


def _request(stream):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "go"}],
        tools=TOOLS,
        tool_choice="auto",
        stream=stream,
    )


def _steps(tok, pieces, eos_alone):
    outputs = []
    for i, piece in enumerate(pieces):
        last = i == len(pieces) - 1 and not eos_alone
        ids = tok.encode(piece, add_special_tokens=False)
        outputs.append(_output(piece, ids, "stop" if last else None))
    if eos_alone:
        outputs.append(_output("", [0], "stop"))
    # Thinking is off: the prompt already closed the reasoning block.
    prompt = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    for output in outputs:
        output.prompt_token_ids = prompt
    return outputs


def _non_streaming_calls(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    response = asyncio.run(
        _serving(_WrappedParser).chat_completion_full_generator(
            _request(stream=False),
            _generate([_output(text, ids, "stop")]),
            "test",
            "test",
            [],
            tok,
            RequestResponseMetadata(request_id="test"),
            reasoning_parser=Qwen3ReasoningParser(
                tok, chat_template_kwargs={"enable_thinking": False}
            ),
        )
    )
    return [
        (t.function.name, t.function.arguments)
        for t in response.choices[0].message.tool_calls
    ]


def _streamed_calls(choices):
    calls: dict[int, list[str]] = {}
    for choice in choices:
        for tc in choice["delta"].get("tool_calls") or []:
            call = calls.setdefault(tc["index"], ["", ""])
            call[0] += tc["function"].get("name") or ""
            call[1] += tc["function"].get("arguments") or ""
    return [tuple(call) for _, call in sorted(calls.items())]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("eos_alone", [True, False])
def test_chat_stream_completes_burst_arguments(tok, case, eos_alone):
    pieces = CASES[case]
    expected = _non_streaming_calls(tok, "".join(pieces))
    chunks = _stream(
        _serving(_WrappedParser),
        _request(stream=True),
        _steps(tok, pieces, eos_alone),
        tokenizer=tok,
        chat_template_kwargs={"enable_thinking": False},
    )
    choices = _choices(chunks)[0]
    assert _streamed_calls(choices) == expected
    assert choices[-1]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("case", CASES)
def test_parse_delta_completes_burst_arguments(tok, case):
    # The Responses API streams parse_delta() output without the chat
    # serving layer's end-of-stream argument flush.
    pieces = CASES[case]
    expected = _non_streaming_calls(tok, "".join(pieces))
    request = _request(stream=True)
    parser = _WrappedParser(
        tok, request.tools, chat_template_kwargs={"enable_thinking": False}
    )
    calls: dict[int, list[str]] = {}
    for output in _steps(tok, pieces, eos_alone=True):
        completion = output.outputs[0]
        delta = parser.parse_delta(
            completion.text,
            list(completion.token_ids),
            request,
            prompt_token_ids=output.prompt_token_ids,
        )
        for tc in (delta.tool_calls if delta else None) or []:
            call = calls.setdefault(tc.index, ["", ""])
            call[0] += tc.function.name or ""
            call[1] += tc.function.arguments or ""
    assert [tuple(call) for _, call in sorted(calls.items())] == expected


def test_token_by_token_stream_is_unchanged(tok):
    # One token per delta: the body is never complete at the "{" stage.
    text = "".join(CASES["call+noarg"])
    ids = tok.encode(text, add_special_tokens=False)
    prompt = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    outputs = [
        _output(tok.decode([tid]), [tid], "stop" if i == len(ids) - 1 else None)
        for i, tid in enumerate(ids)
    ]
    for output in outputs:
        output.prompt_token_ids = prompt
    chunks = _stream(
        _serving(_WrappedParser),
        _request(stream=True),
        outputs,
        tokenizer=tok,
        chat_template_kwargs={"enable_thinking": False},
    )
    choices = _choices(chunks)[0]
    arguments = [
        tc["function"].get("arguments")
        for choice in choices
        for tc in choice["delta"].get("tool_calls") or []
    ]
    assert _streamed_calls(choices) == _non_streaming_calls(tok, text)
    # "{" is still sent on its own before the parameters arrive.
    assert arguments.count("{") == 2
