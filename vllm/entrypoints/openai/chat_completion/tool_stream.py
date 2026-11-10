# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconcile finish-time tool parsing with the deltas already sent."""

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.abstract_tool_parser import ToolParser


@dataclass
class StreamedToolCall:
    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)
    type: Literal["function"] | None = None


def record_tool_calls(calls: list[DeltaToolCall], seen: dict[int, StreamedToolCall]):
    for call in calls:
        state = seen.setdefault(call.index, StreamedToolCall())
        state.id += call.id or ""
        if "type" in call.model_fields_set:
            state.type = call.type
        if call.function is not None:
            state.name += call.function.name or ""
            state.arguments.append(call.function.arguments or "")


def remaining_tool_args(expected: str, actual: str) -> str:
    """Append only a prefix-aligned suffix; JSON string whitespace is data."""

    def normalized(text):
        chars, ends = [], []
        quoted = escaped = False
        for index, char in enumerate(text):
            if quoted or char not in " \t\r\n":
                chars.append(char)
                ends.append(index + 1)
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
        return "".join(chars), ends

    expected_key, ends = normalized(expected)
    actual_key, _ = normalized(actual)
    if expected_key.startswith(actual_key):
        return expected[ends[len(actual_key) - 1] :] if actual_key else expected

    def same_value(left, right):
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(
                same_value(value, right[key]) for key, value in left.items()
            )
        if isinstance(left, list):
            return len(left) == len(right) and all(
                same_value(a, b) for a, b in zip(left, right)
            )
        return left == right

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def parsed(text):
        # Preserve exact JSON numbers and distinguish false/true from 0/1.
        return json.loads(
            text,
            parse_int=Decimal,
            parse_float=Decimal,
            parse_constant=reject_constant,
        )

    try:
        if same_value(parsed(expected), parsed(actual)):
            return ""
    except (ValueError, TypeError):
        pass
    raise ValueError("Tool arguments at finish do not extend the streamed prefix")


def flush_tool_calls(
    parser: ToolParser,
    text: str,
    request: ChatCompletionRequest,
    delta: DeltaMessage | None,
    seen: dict[int, StreamedToolCall],
) -> tuple[DeltaMessage | None, bool]:
    # The non-streaming parser may reset its partial state. The client-visible
    # record belongs to this response, independently of parser internals.
    snapshot = {
        index: StreamedToolCall(state.id, state.name, list(state.arguments), state.type)
        for index, state in seen.items()
    }
    result = delta.model_copy(deep=True) if delta is not None else DeltaMessage()
    record_tool_calls(result.tool_calls or [], snapshot)
    complete = parser.extract_tool_calls(text, request)
    if not complete.tools_called:
        return delta, False
    result.tool_calls = list(result.tool_calls or [])
    for index, call in enumerate(complete.tool_calls):
        sent = snapshot.get(index)
        if sent is None:
            result.tool_calls.append(
                DeltaToolCall(
                    index=index,
                    id=call.id,
                    type="function",
                    function=DeltaFunctionCall(
                        name=call.function.name,
                        arguments=call.function.arguments,
                    ),
                )
            )
            continue
        if not call.function.name.startswith(sent.name):
            raise ValueError("Tool name at finish differs from the streamed call")
        name_suffix = call.function.name[len(sent.name) :]
        suffix = remaining_tool_args(call.function.arguments, "".join(sent.arguments))
        if suffix or name_suffix or not sent.id or sent.type is None:
            current = next(
                (item for item in result.tool_calls if item.index == index), None
            )
            if current is None:
                current = DeltaToolCall(index=index)
                result.tool_calls.append(current)
            if not sent.id:
                current.id = call.id
            if sent.type is None:
                current.type = "function"
            if current.function is None:
                current.function = DeltaFunctionCall()
            if name_suffix:
                current.function.name = (current.function.name or "") + name_suffix
            if suffix:
                current.function.arguments = (current.function.arguments or "") + suffix
    return result, True
