# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest

from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer
from vllm.tokenizers.registry import TokenizerRegistry

BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
USER, ASSISTANT, SYSTEM = "<｜User｜>", "<｜Assistant｜>", "<｜System｜>"


class _Tokenizer:
    vocab_size = 17

    def encode(self, text, **kwargs):
        assert kwargs.get("add_special_tokens", False) is False
        return list(text.encode())

    def get_added_vocab(self):
        return {BOS: 0, EOS: 1}

    def __len__(self):
        return 17  # Special tokens are already part of this vocabulary.


def test_v41_tokenizer_and_chat_transport_are_independently_registered():
    assert (
        TokenizerRegistry.load_tokenizer_cls("deepseek_v41").__name__
        == "DeepseekV41Tokenizer"
    )
    assert (
        RENDERER_REGISTRY.load_renderer_cls("deepseek_v41").__name__
        == "DeepseekV41Renderer"
    )


def test_chat_mid_system_and_wrapper_copy_match_v41_boundaries():
    tokenizer = get_deepseek_v41_tokenizer(_Tokenizer())
    messages = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
        {"role": "system", "content": "新指令"},
    ]
    expected = (
        BOS
        + USER
        + "问题"
        + ASSISTANT
        + "</think>回答"
        + EOS
        + SYSTEM
        + "新指令"
        + ASSISTANT
        + "</think>"
    )
    assert tokenizer.apply_chat_template(messages, tokenize=False) == expected
    assert tokenizer.apply_chat_template(messages) == list(expected.encode())
    cloned = copy.deepcopy(tokenizer)
    assert cloned.apply_chat_template(messages, tokenize=False) == expected
    assert len(cloned) == len(tokenizer) == 17
    assert messages[-1] == {"role": "system", "content": "新指令"}


@pytest.mark.parametrize(
    "effort,budget", [(1, 1), (100, 100), ("low", 50), ("high", 75), ("max", 100)]
)
def test_numeric_reasoning_effort_has_the_official_v41_prefix(effort, budget):
    tokenizer = get_deepseek_v41_tokenizer(_Tokenizer())
    actual = tokenizer.apply_chat_template(
        [{"role": "user", "content": "问题"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=effort,
    )
    prefix = (
        f"Reasoning Effort: {budget} "
        "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    )
    assert actual == BOS + SYSTEM + prefix + USER + "问题" + ASSISTANT + "<think>"


@pytest.mark.parametrize("budget", [0, 101, True, "unknown"])
def test_invalid_numeric_effort_is_rejected(budget):
    tokenizer = get_deepseek_v41_tokenizer(_Tokenizer())
    with pytest.raises(AssertionError, match="Invalid reasoning effort"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "问题"}], reasoning_effort=budget
        )


def test_tool_history_keeps_spaced_tags_typed_parameters_and_result_order():
    tokenizer = get_deepseek_v41_tokenizer(_Tokenizer())
    messages = [
        {"role": "user", "content": "天气"},
        {
            "role": "assistant",
            "tool_calls": [
                dict(
                    id="call-1",
                    type="function",
                    function=dict(
                        name="weather::get_weather",
                        arguments='{"city":"北京","days":2}',
                    ),
                )
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "晴"},
        {"role": "user", "content": "继续"},
    ]
    actual = tokenizer.apply_chat_template(messages, tokenize=False)
    expected = (
        BOS + USER + "天气" + ASSISTANT + "</think>\n\n"
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="weather::get_weather">\n'
        '<｜DSML｜ parameter name="city" string="true">北京</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
        + EOS
        + USER
        + "<tool_result>晴</tool_result>\n\n继续"
        + ASSISTANT
        + "</think>"
    )
    assert actual == expected


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image_url", "image_url": {"url": "example"}}],
        "<image>example</image>",
        "<｜deepseek_image｜>",
    ],
)
def test_first_text_profile_rejects_multimodal_inputs(content):
    tokenizer = get_deepseek_v41_tokenizer(_Tokenizer())
    with pytest.raises(ValueError, match="V4.1 first profile"):
        tokenizer.apply_chat_template([{"role": "user", "content": content}])
