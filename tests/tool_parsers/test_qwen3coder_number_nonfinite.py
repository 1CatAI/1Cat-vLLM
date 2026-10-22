# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""qwen3_coder: a non-finite number argument must not drop the tool call.

``int(float("inf"))`` raised OverflowError while converting a "number"
parameter. Non-streaming parsing then returned every call of the response as
raw content; streaming aborted. Non-finite values are kept as strings
(vllm-project/vllm#43984).
"""

import json

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

pytestmark = pytest.mark.cpu_test


@pytest.fixture(scope="module")
def byte_tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_tokens(
        [
            AddedToken(token, special=False, normalized=False)
            for token in ("<tool_call>", "</tool_call>")
        ]
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scale",
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {"type": "number"},
                    "name": {"type": "string"},
                },
            },
        },
    }
]


def _call(factor):
    return (
        "<tool_call>\n<function=scale>\n"
        f"<parameter=factor>\n{factor}\n</parameter>\n"
        "<parameter=name>\nx\n</parameter>\n"
        "</function>\n</tool_call>"
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("schema_type", ["number", ["number", "string"]])
@pytest.mark.parametrize(
    "factor,expected",
    [
        ("inf", "inf"),
        ("-inf", "-inf"),
        ("nan", "nan"),
        ("Infinity", "Infinity"),
        ("1e999", "1e999"),
        ("2.5", 2.5),
        ("5.0", 5),
        ("1e3", 1000),
    ],
)
def test_number_arguments(byte_tokenizer, streaming, schema_type, factor, expected):
    request = ChatCompletionRequest(model="test", messages=[], tools=TOOLS)
    request.tools[0].function.parameters["properties"]["factor"]["type"] = schema_type
    parser = Qwen3CoderToolParser(byte_tokenizer, request.tools)
    output = _call(factor)
    if not streaming:
        info = parser.extract_tool_calls(output, request)
        assert info.tools_called
        (call,) = info.tool_calls
        arguments = call.function.arguments
    else:
        ids = byte_tokenizer.encode(output, add_special_tokens=False)
        arguments = ""
        previous = ""
        for i in range(len(ids)):
            current = byte_tokenizer.decode(ids[: i + 1])
            delta = parser.extract_tool_calls_streaming(
                previous,
                current,
                current[len(previous) :],
                ids[:i],
                ids[: i + 1],
                ids[i : i + 1],
                request,
            )
            if delta:
                for call in delta.tool_calls or []:
                    if call.function and call.function.arguments is not None:
                        arguments += call.function.arguments
            previous = current
    parsed = json.loads(arguments)
    assert parsed == {"factor": expected, "name": "x"}
    # Reject NaN/Infinity even if the JSON implementation accepts extensions.
    json.dumps(parsed, allow_nan=False)
