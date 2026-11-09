# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.entrypoints.chat_utils import parse_chat_messages
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from vllm.tokenizers.registry import TokenizerRegistry

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "deepseek_v4"


class FakeHfTokenizer:
    vocab_size = 100

    def get_added_vocab(self) -> dict[str, int]:
        return {"</think>": 100}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        self.last_encode = (text, add_special_tokens, kwargs)
        return [len(text)]


def _tokenizer():
    return get_deepseek_v4_tokenizer(FakeHfTokenizer())


def _model_config():
    return SimpleNamespace(
        multimodal_config=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
        enable_prompt_embeds=False,
    )


def _load_reference_case(case_id: int):
    data = json.loads((FIXTURES_DIR / f"test_input_{case_id}.json").read_text())
    if isinstance(data, dict):
        return data["messages"], data.get("tools")
    return data, None


def _render_reference_case(case_id: int, **kwargs):
    messages, tools = _load_reference_case(case_id)
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )
    return _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tools=tools,
        tokenize=False,
        **kwargs,
    )


def test_deepseek_v4_tokenizer_registered():
    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v4").__name__ == (
        "DeepseekV4Tokenizer"
    )
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v4").__name__ == (
        "DeepseekV4Renderer"
    )


def test_deepseek_v4_defaults_to_chat_mode():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")


@pytest.mark.parametrize("kwargs", [{"thinking": True}, {"enable_thinking": True}])
def test_deepseek_v4_enables_thinking_with_compatible_kwargs(kwargs):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜><think>")


@pytest.mark.parametrize(
    ("enable_thinking", "thinking_token"),
    [(False, "</think>"), (True, "<think>")],
)
def test_deepseek_v4_appends_assistant_transition_after_trailing_system(
    enable_thinking, thinking_token
):
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "system", "content": "Be concise."},
        ],
        tokenize=False,
        enable_thinking=enable_thinking,
        reasoning_effort="low",
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜>You are helpful."
        f"<｜User｜>What is 2+2?Be concise.<｜Assistant｜>{thinking_token}"
    )


def test_deepseek_v4_does_not_transition_after_mid_conversation_system():
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "6"},
            {"role": "system", "content": "Answer in French."},
            {"role": "user", "content": "What is 5+5?"},
        ],
        tokenize=False,
        enable_thinking=False,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>What is 3+3?"
        "<｜Assistant｜></think>6<｜end▁of▁sentence｜>"
        "Answer in French.<｜User｜>What is 5+5?"
        "<｜Assistant｜></think>"
    )


@pytest.mark.parametrize(
    ("enable_thinking", "expected"),
    [
        (
            False,
            (
                "<｜begin▁of▁sentence｜>rules<｜Assistant｜></think>answer"
                "<｜end▁of▁sentence｜>"
            ),
        ),
        (
            True,
            (
                "<｜begin▁of▁sentence｜>rules<｜Assistant｜><think>reason</think>answer"
                "<｜end▁of▁sentence｜>"
            ),
        ),
    ],
)
def test_deepseek_v4_transitions_from_system_to_assistant(enable_thinking, expected):
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "system", "content": "rules"},
            {
                "role": "assistant",
                "reasoning": "reason",
                "content": "answer",
            },
        ],
        tokenize=False,
        enable_thinking=enable_thinking,
        reasoning_effort="low",
    )

    assert prompt == expected


def test_deepseek_v4_opens_assistant_turn_after_compressed_history_summary():
    """AIfred's history compression yields system prompt, summary as system,
    then the remaining turns, which may start with assistant."""
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "system", "content": "You are AIfred."},
            {"role": "system", "content": "[Summary]\nEarlier: weather."},
            {"role": "assistant", "content": "It will be sunny."},
            {"role": "user", "content": "And tomorrow?"},
        ],
        tokenize=False,
        enable_thinking=True,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜>You are AIfred.[Summary]\nEarlier: weather."
        "<｜Assistant｜></think>It will be sunny.<｜end▁of▁sentence｜>"
        "<｜User｜>And tomorrow?<｜Assistant｜><think>"
    )


def test_deepseek_v4_uses_v4_tool_prompt_from_request_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Weather?"}],
        tools=tools,
        tokenize=False,
    )

    assert "## Tools" in prompt
    assert "<｜DSML｜tool_calls>" in prompt
    assert "</｜DSML｜tool_calls>" in prompt
    assert "function_calls" not in prompt
    assert '"name": "get_weather"' in prompt
    assert prompt.endswith("<｜User｜>Weather?<｜Assistant｜></think>")


def test_deepseek_v4_renders_parsed_history_tool_arguments():
    messages = [
        {"role": "user", "content": "List the repo"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": '{"command": "view", "path": "/testbed"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file list",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "Edit files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["command", "path"],
                },
            },
        }
    ]
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )

    prompt = _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tools=tools,
        tokenize=False,
    )

    assert '<｜DSML｜parameter name="command" string="true">view' in prompt
    assert '<｜DSML｜parameter name="path" string="true">/testbed' in prompt
    assert 'parameter name="arguments"' not in prompt


@pytest.mark.parametrize("reasoning_effort", ["minimal", "low", "medium", "high"])
def test_deepseek_v4_accepts_openai_reasoning_effort_values(reasoning_effort):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert prompt.endswith("<｜Assistant｜><think>")
    assert "Reasoning Effort: Absolute maximum" not in prompt


def test_deepseek_v4_none_reasoning_effort_disables_thinking():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="none",
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_mode", "expected_effort"),
    [
        ("none", "chat", None),
        ("minimal", "thinking", "high"),
        ("low", "thinking", "high"),
        ("medium", "thinking", "high"),
        ("high", "thinking", "high"),
        ("xhigh", "thinking", "max"),
        ("max", "thinking", "max"),
        ("unexpected", "thinking", "high"),
    ],
)
def test_deepseek_v4_maps_compatible_thinking_reasoning_effort_values(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort,
    expected_mode,
    expected_effort,
):
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_kwargs.append(kwargs)
        return "prompt"

    monkeypatch.setattr(
        "vllm.tokenizers.deepseek_v4.encode_messages",
        fake_encode_messages,
    )

    _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert captured_kwargs[-1]["thinking_mode"] == expected_mode
    assert captured_kwargs[-1]["reasoning_effort"] == expected_effort


def test_deepseek_v4_preserves_reference_max_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="max",
    )

    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )


def test_deepseek_v4_maps_xhigh_to_reference_max_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="xhigh",
    )

    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )


@pytest.mark.parametrize(
    ("case_id", "kwargs"),
    [
        (1, {"thinking": True}),
        (2, {"thinking": True}),
        (3, {"thinking": True}),
        (4, {}),
    ],
)
def test_deepseek_v4_matches_reference_golden_fixtures(case_id, kwargs):
    prompt = _render_reference_case(case_id, **kwargs)

    expected = (FIXTURES_DIR / f"test_output_{case_id}.txt").read_text()
    assert prompt == expected


def _request_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def _encode_reference(messages):
    from vllm.tokenizers.deepseek_v4_encoding import encode_messages

    # The wrapper maps OpenAI's "low" to "high"; this fork's encoder knows only
    # None, "high" and "max".
    return encode_messages(messages, thinking_mode="thinking", reasoning_effort="high")


def test_deepseek_v4_attaches_request_tools_to_existing_system_message():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in Paris?"},
    ]

    prompt = _tokenizer().apply_chat_template(
        messages,
        tools=_request_tools(),
        tokenize=False,
        thinking=True,
        reasoning_effort="low",
    )

    expected = _encode_reference(
        [{**messages[0], "tools": _request_tools()}, messages[1]]
    )
    assert prompt == expected
    assert prompt.startswith("<｜begin▁of▁sentence｜>You are helpful.\n\n## Tools")
    assert prompt.count("## Tools") == 1


def test_deepseek_v4_synthetic_system_only_when_no_system_message():
    """Without a system message, request tools still get a synthetic leading
    system entry."""
    messages = [{"role": "user", "content": "Weather in Paris?"}]

    prompt = _tokenizer().apply_chat_template(
        messages,
        tools=_request_tools(),
        tokenize=False,
        thinking=True,
        reasoning_effort="low",
    )

    expected = _encode_reference(
        [{"role": "system", "tools": _request_tools()}, messages[0]]
    )
    assert prompt == expected
    assert prompt.startswith("<｜begin▁of▁sentence｜>\n\n## Tools")


def test_deepseek_v4_request_tools_override_system_message_tools():
    message_tools = [
        {
            "type": "function",
            "function": {
                "name": "message_level_tool",
                "description": "Should not be rendered",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    system_message = {
        "role": "system",
        "content": "You are helpful.",
        "tools": message_tools,
    }
    user_message = {"role": "user", "content": "Weather in Paris?"}

    prompt = _tokenizer().apply_chat_template(
        [system_message, user_message],
        tools=_request_tools(),
        tokenize=False,
        thinking=True,
        reasoning_effort="low",
    )

    expected = _encode_reference(
        [{**system_message, "tools": _request_tools()}, user_message]
    )
    assert prompt == expected
    assert '"name": "get_weather"' in prompt
    assert "message_level_tool" not in prompt


def test_deepseek_v4_attaches_request_tools_to_first_system_message_only():
    """Only the first system message, at any index, gets the request tools."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "First system."},
        {"role": "system", "content": "Second system."},
        {"role": "user", "content": "Weather in Paris?"},
    ]

    prompt = _tokenizer().apply_chat_template(
        messages,
        tools=_request_tools(),
        tokenize=False,
        thinking=True,
        reasoning_effort="low",
    )

    expected = _encode_reference(
        [
            messages[0],
            {**messages[1], "tools": _request_tools()},
            messages[2],
            messages[3],
        ]
    )
    assert prompt == expected
    assert "First system.\n\n## Tools" in prompt
    assert prompt.count("## Tools") == 1


def test_deepseek_v4_request_tools_do_not_mutate_caller_messages():
    import copy

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in Paris?"},
    ]
    snapshot = copy.deepcopy(messages)

    _tokenizer().apply_chat_template(messages, tools=_request_tools(), tokenize=False)

    assert messages == snapshot
