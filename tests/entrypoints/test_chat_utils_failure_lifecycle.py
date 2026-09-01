# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import gc
import warnings
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.chat_utils import (
    AsyncMultiModalContentParser,
    AsyncMultiModalItemTracker,
    _postprocess_messages,
)


def _assistant_tool_call(arguments, name: str = "write"):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }
    ]


def test_tool_call_arguments_malformed_json_is_recoverable(caplog):
    messages = _assistant_tool_call('{"cmd": "unterminated', name="exec")

    _postprocess_messages(messages)

    arguments = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert arguments == {}
    assert len(caplog.records) == 1
    assert "exec" in caplog.records[0].message
    assert "coercing to an empty object" in caplog.records[0].message


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [({"path": "a"}, {"path": "a"}), ('{"path": "a"}', {"path": "a"}), (None, {})],
)
def test_tool_call_argument_objects_are_preserved(arguments, expected):
    messages = _assistant_tool_call(arguments)

    _postprocess_messages(messages)

    normalized = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert normalized == expected


@pytest.mark.parametrize("arguments", ["[]", "42", "true", '"text"', "null", []])
def test_tool_call_arguments_must_be_an_object(arguments):
    messages = _assistant_tool_call(arguments)

    _postprocess_messages(messages)

    normalized = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert normalized == {}


def test_async_parser_does_not_create_coroutine_before_limit_validation():
    class RejectingTracker:
        def add(self, *_args, **_kwargs):
            raise ValueError("too many image items")

    parser = object.__new__(AsyncMultiModalContentParser)
    parser._tracker = RejectingTracker()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(ValueError, match="too many image items"):
            parser.parse_image("https://example.com/image.png")
        gc.collect()

    unawaited = [
        warning for warning in caught if "was never awaited" in str(warning.message)
    ]
    assert not unawaited


def test_resolve_items_drains_siblings_after_partial_failure():
    completed = 0

    async def fetch(should_fail: bool, delay: float):
        nonlocal completed
        await asyncio.sleep(delay)
        if should_fail:
            raise ValueError("simulated fetch failure")
        completed += 1
        return "decoded", None

    async def run_test():
        tracker = AsyncMultiModalItemTracker(MagicMock())
        tracker._items_by_modality["image"] = [
            lambda: fetch(True, 0.01),
            lambda: fetch(False, 0.05),
            lambda: fetch(False, 0.05),
        ]

        tasks_before = asyncio.all_tasks()
        with pytest.raises(ValueError, match="simulated fetch failure"):
            await tracker.resolve_items()

        leaked_tasks = asyncio.all_tasks() - tasks_before
        assert not leaked_tasks
        assert completed == 2

    asyncio.run(run_test())
