# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)


@pytest.mark.parametrize("exclude_unset", [False, True])
@pytest.mark.parametrize("exclude_none", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_explicit_empty_tool_calls_are_omitted(stream, exclude_unset, exclude_none):
    cls = DeltaMessage if stream else ChatMessage
    message = cls(role="assistant", content="answer", tool_calls=[])
    options = dict(exclude_unset=exclude_unset, exclude_none=exclude_none)
    assert "tool_calls" not in message.model_dump(**options)
    assert "tool_calls" not in json.loads(message.model_dump_json(**options))


@pytest.mark.parametrize("stream", [False, True])
def test_nonempty_tool_calls_are_preserved(stream):
    call = (
        DeltaToolCall(
            index=0,
            id="call_1",
            type="function",
            function=DeltaFunctionCall(name="f", arguments="{}"),
        )
        if stream
        else ToolCall(id="call_1", function=FunctionCall(name="f", arguments="{}"))
    )
    cls = DeltaMessage if stream else ChatMessage
    message = cls(role="assistant", tool_calls=[call])
    assert message.model_dump()["tool_calls"][0]["id"] == "call_1"
