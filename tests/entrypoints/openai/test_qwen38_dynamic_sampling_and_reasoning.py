# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Qwen3.8 dynamic thinking/non-thinking sampling and reasoning fixes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.config.model import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser

pytestmark = pytest.mark.cpu_test


class _DummyQwenTokenizer:
    def __init__(self) -> None:
        self._vocab = {
            "<think>": 248068,
            "</think>": 248069,
            "<tool_call>": 248056,
            "</tool_call>": 248057,
        }

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def _dummy_qwen_model_config(
    *,
    model: str = "Qwen/Qwen3.8-27B-NVFP4",
    model_type: str = "qwen3_5",
    generation_config: str = "auto",
    override_generation_config: dict | None = None,
    disk_gen_cfg: dict | None = None,
):
    dummy = SimpleNamespace(
        model=model,
        served_model_name=None,
        generation_config=generation_config,
        override_generation_config=override_generation_config or {},
        hf_config=SimpleNamespace(model_type=model_type),
        hf_text_config=SimpleNamespace(model_type=model_type),
    )
    dummy.supports_dynamic_thinking_sampling = (
        ModelConfig.supports_dynamic_thinking_sampling.__get__(dummy)
    )
    dummy.try_get_generation_config = lambda: dict(
        disk_gen_cfg
        if disk_gen_cfg is not None
        else {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    )
    dummy.get_diff_sampling_param = ModelConfig.get_diff_sampling_param.__get__(dummy)
    dummy.get_effective_sampling_param = (
        ModelConfig.get_effective_sampling_param.__get__(dummy)
    )
    return dummy


@pytest.mark.parametrize(
    ("model", "model_type"),
    [
        ("Qwen/Qwen3.8-27B-NVFP4", "qwen3_5"),
        ("RadixArk/Qwen3.8-Flash-Next-NVFP4", "qwen4_exp"),
    ],
)
def test_qwen38_dynamic_sampling_thinking_vs_non_thinking(
    model: str, model_type: str
) -> None:
    mc = _dummy_qwen_model_config(model=model, model_type=model_type)
    base = mc.get_diff_sampling_param()

    # Thinking mode (default) -> 1.0 / 0.95 / 20 / presence_penalty=0.0
    think_defaults = mc.get_effective_sampling_param(
        thinking_enabled=True, base_defaults=base
    )
    req_think = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
    sp_think = req_think.to_sampling_params(128, think_defaults)
    assert sp_think.temperature == pytest.approx(1.0)
    assert sp_think.top_p == pytest.approx(0.95)
    assert sp_think.top_k == 20
    assert sp_think.presence_penalty == pytest.approx(0.0)
    assert sp_think.repetition_penalty == pytest.approx(1.0)

    # Non-thinking mode -> 0.7 / 0.80 / 20 / presence_penalty=1.5
    nothink_defaults = mc.get_effective_sampling_param(
        thinking_enabled=False, base_defaults=base
    )
    req_nothink = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
    sp_nothink = req_nothink.to_sampling_params(128, nothink_defaults)
    assert sp_nothink.temperature == pytest.approx(0.7)
    assert sp_nothink.top_p == pytest.approx(0.80)
    assert sp_nothink.top_k == 20
    assert sp_nothink.presence_penalty == pytest.approx(1.5)
    assert sp_nothink.repetition_penalty == pytest.approx(1.0)

    # Explicit client fields override non-thinking defaults cleanly (including 0.0)
    req_explicit = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        presence_penalty=0.0,
    )
    sp_explicit = req_explicit.to_sampling_params(128, nothink_defaults)
    assert sp_explicit.temperature == pytest.approx(0.2)
    assert sp_explicit.presence_penalty == pytest.approx(0.0)
    assert sp_explicit.top_p == pytest.approx(0.80)


def test_override_generation_config_and_vllm_mode_precedence() -> None:
    mc_override = _dummy_qwen_model_config(
        override_generation_config={"presence_penalty": 0.4, "frequency_penalty": 0.1}
    )
    base = mc_override.get_diff_sampling_param()
    assert base["presence_penalty"] == pytest.approx(0.4)
    assert base["frequency_penalty"] == pytest.approx(0.1)

    eff = mc_override.get_effective_sampling_param(
        thinking_enabled=False, base_defaults=base
    )
    assert eff["presence_penalty"] == pytest.approx(0.4)
    assert eff["temperature"] == pytest.approx(0.7)

    comp_req = CompletionRequest(prompt="hello")
    comp_sp = comp_req.to_sampling_params(64, base)
    assert comp_sp.presence_penalty == pytest.approx(0.4)
    assert comp_sp.frequency_penalty == pytest.approx(0.1)

    mc_vllm = _dummy_qwen_model_config(generation_config="vllm")
    assert mc_vllm.get_diff_sampling_param() == {}
    assert mc_vllm.get_effective_sampling_param(thinking_enabled=False) == {}


def test_qwen3_reasoning_parser_implicit_tool_call_and_reasoning_effort() -> None:
    tok = _DummyQwenTokenizer()

    parser_none = Qwen3ReasoningParser(
        tok, chat_template_kwargs={"reasoning_effort": "none"}
    )
    assert parser_none.thinking_enabled is False
    req_plain = ChatCompletionRequest(messages=[{"role": "user", "content": "x"}])
    assert parser_none.extract_reasoning("plain answer", req_plain) == (
        None,
        "plain answer",
    )

    parser = Qwen3ReasoningParser(tok, chat_template_kwargs={"enable_thinking": True})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    # Auto tool_choice preserves <tool_call> tag for XML/Qwen3Coder parsers
    req_auto = ChatCompletionRequest(
        messages=[{"role": "user", "content": "call foo"}],
        tools=tools,
        tool_choice="auto",
    )
    r_auto, c_auto = parser.extract_reasoning(
        "Let me call foo.\n<tool_call>\n<function=foo></function>\n</tool_call>",
        req_auto,
    )
    assert r_auto == "Let me call foo.\n"
    assert c_auto is not None and c_auto.startswith("<tool_call>")

    # Named tool_choice strips implicit <tool_call> delimiter so JSON stays valid
    req_named = ChatCompletionRequest(
        messages=[{"role": "user", "content": "weather"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert isinstance(req_named.tool_choice, ChatCompletionNamedToolChoiceParam)
    r_named, c_named = parser.extract_reasoning(
        'Check weather.\n<tool_call>{"city":"Beijing"}', req_named
    )
    assert r_named == "Check weather.\n"
    assert c_named == '{"city":"Beijing"}'
    fcs_named, rem_named = OpenAIServing._parse_tool_calls_from_content(
        request=req_named,
        tokenizer=None,
        enable_auto_tools=True,
        tool_parser_cls=None,
        content=c_named,
    )
    assert rem_named is None
    assert fcs_named is not None and len(fcs_named) == 1
    assert fcs_named[0].name == "get_weather"
    assert fcs_named[0].arguments == '{"city":"Beijing"}'

    # Required tool_choice strips implicit <tool_call> and parses list JSON
    req_required = ChatCompletionRequest(
        messages=[{"role": "user", "content": "weather"}],
        tools=tools,
        tool_choice="required",
    )
    raw_req_out = (
        "Need weather.\n<tool_call>"
        '[{"name":"get_weather","parameters":{"city":"Shanghai"}}]'
    )
    r_req, c_req = parser.extract_reasoning(raw_req_out, req_required)
    assert r_req == "Need weather.\n"
    fcs_req, rem_req = OpenAIServing._parse_tool_calls_from_content(
        request=req_required,
        tokenizer=None,
        enable_auto_tools=True,
        tool_parser_cls=None,
        content=c_req,
    )
    assert rem_req is None
    assert fcs_req is not None and len(fcs_req) == 1
    assert fcs_req[0].name == "get_weather"
    assert fcs_req[0].arguments == '{"city": "Shanghai"}'


def test_greedy_temperature_zero_skips_default_penalties() -> None:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.models.sampling_defaults import QWEN38_NON_THINKING_SAMPLING_DEFAULTS

    chat_greedy = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
    ).to_sampling_params(64, dict(QWEN38_NON_THINKING_SAMPLING_DEFAULTS))
    assert chat_greedy.temperature == 0.0
    assert chat_greedy.presence_penalty == 0.0
    assert chat_greedy.frequency_penalty == 0.0
    assert chat_greedy.repetition_penalty == 1.0

    # Explicit penalty alongside temperature=0 is still honored.
    chat_explicit = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        presence_penalty=1.25,
    ).to_sampling_params(64, dict(QWEN38_NON_THINKING_SAMPLING_DEFAULTS))
    assert chat_explicit.presence_penalty == 1.25

    comp_greedy = CompletionRequest(
        prompt="hi",
        temperature=0.0,
    ).to_sampling_params(64, dict(QWEN38_NON_THINKING_SAMPLING_DEFAULTS))
    assert comp_greedy.presence_penalty == 0.0

    resp_greedy = ResponsesRequest(
        input="hi",
        temperature=0.0,
    ).to_sampling_params(64, dict(QWEN38_NON_THINKING_SAMPLING_DEFAULTS))
    assert resp_greedy.presence_penalty == 0.0


def test_sm70_qwen4exp_bare_serve_defaults_max_num_seqs(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from vllm.engine import arg_utils
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    monkeypatch.setattr(arg_utils.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        arg_utils.current_platform,
        "get_device_capability",
        lambda *a, **k: SimpleNamespace(major=7, minor=0),
    )
    monkeypatch.setattr(
        arg_utils.current_platform,
        "get_device_total_memory",
        lambda *a, **k: 32 * (1 << 30),
    )
    model_cfg = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            model_type="qwen4_exp_text",
            mtp_num_hidden_layers=1,
            layer_types=["linear_attention", "full_attention"],
        ),
        is_moe=True,
    )
    args = EngineArgs(model=str(tmp_path))
    assert args.max_num_seqs is None
    args._maybe_apply_sm70_mtp_defaults(UsageContext.OPENAI_API_SERVER, model_cfg)
    assert args.max_num_seqs == 8

    explicit = EngineArgs(model=str(tmp_path), max_num_seqs=4)
    explicit._maybe_apply_sm70_mtp_defaults(UsageContext.OPENAI_API_SERVER, model_cfg)
    assert explicit.max_num_seqs == 4


def test_truncated_required_tool_message_and_auto_tools_response_format() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage

    # Empty tool_calls list must construct ChatMessage without ValidationError
    # and omit empty tool_calls on serialization.
    msg = ChatMessage(role="assistant", content='[{"name":"w"', tool_calls=[])
    dumped = msg.model_dump()
    assert "tool_calls" not in dumped
    assert dumped["content"] == '[{"name":"w"'

    # Auto tool_choice + response_format must preserve <tool_call> tags for XML parser.
    tok = _DummyQwenTokenizer()
    parser = Qwen3ReasoningParser(tok, chat_template_kwargs={"enable_thinking": True})
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "weather"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
        response_format={"type": "json_object"},
    )
    r_txt, c_txt = parser.extract_reasoning(
        "Thinking.\n<tool_call>\n<function=get_weather></function>\n</tool_call>",
        req,
    )
    assert r_txt == "Thinking.\n"
    assert c_txt is not None and c_txt.startswith("<tool_call>")
