# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import pytest
import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.abstract_parser import _WrappedParser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

_SPECIAL_TOKENS = (
    "<|message_system|>",
    "<|message_user|>",
    "<|message_tool|>",
    "<|message_model|>",
    "<|content_thinking|>",
    "<|content_text|>",
    "<|content_invoke_tool_text|>",
    "<|content_tool_error|>",
    "<|content_invoke_tool_json|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
)
_SPECIAL_RE = re.compile(
    "(" + "|".join(re.escape(tok) for tok in _SPECIAL_TOKENS) + ")"
)


class _FakeInkling2Tokenizer:
    """Network-free character/special-token tokenizer for Inkling-2 CI tests."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {chr(i): i for i in range(128)}
        base = 200000
        for offset, tok in enumerate(_SPECIAL_TOKENS):
            self._vocab[tok] = base + offset
        self._id_to_tok: dict[int, str] = {v: k for k, v in self._vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for part in _SPECIAL_RE.split(text):
            if not part:
                continue
            if part in self._vocab:
                ids.append(self._vocab[part])
            else:
                for ch in part:
                    ids.append(self._vocab[ch])
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        special_ids = {self._vocab[t] for t in _SPECIAL_TOKENS}
        return "".join(
            self._id_to_tok[tid]
            for tid in token_ids
            if not (skip_special_tokens and tid in special_ids)
        )


@pytest.fixture(scope="module")
def inkling2_tokenizer():
    return _FakeInkling2Tokenizer()


def test_inkling2_reasoning_parser_registered_and_adjusts_request(
    inkling2_tokenizer,
):
    parser_cls = ReasoningParserManager.get_reasoning_parser("inkling2")
    parser = parser_cls(inkling2_tokenizer)
    req = ChatCompletionRequest(
        model="Inkling-2-Mini-NVFP4",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.skip_special_tokens is True
    adjusted = parser.adjust_request(req)
    assert adjusted.skip_special_tokens is False


def test_inkling2_non_streaming_extraction(inkling2_tokenizer):
    parser_cls = ReasoningParserManager.get_reasoning_parser("inkling2")
    parser = parser_cls(inkling2_tokenizer)
    req = ChatCompletionRequest(
        model="Inkling-2-Mini-NVFP4",
        messages=[{"role": "user", "content": "hi"}],
    )

    full = (
        "<|message_model|><|content_thinking|>Compute 2+2=4."
        "<|end_message|><|message_model|><|content_text|>The answer is 4."
        "<|end_message|><|content_model_end_sampling|>"
    )
    assert parser.extract_reasoning(full, req) == (
        "Compute 2+2=4.",
        "The answer is 4.",
    )

    no_think = (
        "<|message_model|><|content_text|>391"
        "<|end_message|><|content_model_end_sampling|>"
    )
    assert parser.extract_reasoning(no_think, req) == (None, "391")

    empty_think = (
        "<|message_model|><|content_thinking|><|end_message|>"
        "<|message_model|><|content_text|>OK<|end_message|>"
        "<|content_model_end_sampling|>"
    )
    assert parser.extract_reasoning(empty_think, req) == (None, "OK")

    truncated = "<|message_model|><|content_thinking|>Still thinking"
    assert parser.extract_reasoning(truncated, req) == ("Still thinking", None)


def test_inkling2_streaming_and_prompt_is_reasoning_end(inkling2_tokenizer):
    parser_cls = ReasoningParserManager.get_reasoning_parser("inkling2")
    parser = parser_cls(inkling2_tokenizer)

    # Prompt ends with <|end_message|>, so backward scan must stop at <|message_user|>.
    prompt_str = (
        "<|message_system|><|content_text|>Thinking effort level: 0.9"
        "<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"
    )
    prompt_ids = inkling2_tokenizer.encode(prompt_str, add_special_tokens=False)
    assert parser.is_reasoning_end(prompt_ids) is False

    output_str = (
        "<|message_model|><|content_thinking|>Step 1."
        "<|end_message|><|message_model|><|content_text|>42"
        "<|end_message|><|content_model_end_sampling|>"
    )
    output_ids = inkling2_tokenizer.encode(output_str, add_special_tokens=False)
    assert parser.count_reasoning_tokens(output_ids) > 0
    assert parser.is_reasoning_end(output_ids) is True

    prev_ids: list[int] = []
    prev_text = ""
    reasoning_buf = []
    content_buf = []
    for tid in output_ids:
        cur_ids = [*prev_ids, tid]
        cur_text = inkling2_tokenizer.decode(cur_ids, skip_special_tokens=False)
        delta_text = cur_text[len(prev_text) :]
        delta = parser.extract_reasoning_streaming(
            prev_text, cur_text, delta_text, prev_ids, cur_ids, [tid]
        )
        if delta is not None:
            if delta.reasoning:
                reasoning_buf.append(delta.reasoning)
            if delta.content:
                content_buf.append(delta.content)
        prev_ids = cur_ids
        prev_text = cur_text

    assert "".join(reasoning_buf) == "Step 1."
    assert "".join(content_buf) == "42"


def test_inkling2_streaming_multitoken_and_tool_split_delimiters(inkling2_tokenizer):
    """Multi-token speculative deltas, empty reasoning, and split-tag tool EOF."""
    parser = ReasoningParserManager.get_reasoning_parser("inkling2")(inkling2_tokenizer)
    req = ChatCompletionRequest(
        model="Inkling-2-Mini-NVFP4",
        messages=[{"role": "user", "content": "hi"}],
    )

    # 1) Multi-token (speculative-style) deltas crossing channel boundaries.
    full = (
        "<|message_model|><|content_thinking|>AB<|end_message|>"
        "<|message_model|><|content_text|>CD<|end_message|>"
        "<|content_model_end_sampling|>"
    )
    all_ids = inkling2_tokenizer.encode(full, add_special_tokens=False)
    prev_ids: list[int] = []
    prev_text = ""
    r_parts: list[str] = []
    c_parts: list[str] = []
    for i in range(0, len(all_ids), 3):
        delta_ids = all_ids[i : i + 3]
        cur_ids = [*prev_ids, *delta_ids]
        cur_text = inkling2_tokenizer.decode(cur_ids, skip_special_tokens=False)
        delta_text = cur_text[len(prev_text) :]
        delta = parser.extract_reasoning_streaming(
            prev_text, cur_text, delta_text, prev_ids, cur_ids, delta_ids
        )
        if delta is not None:
            if delta.reasoning:
                r_parts.append(delta.reasoning)
            if delta.content:
                c_parts.append(delta.content)
        prev_ids = cur_ids
        prev_text = cur_text

    assert ("".join(r_parts) or None, "".join(c_parts) or None) == (
        parser.extract_reasoning(full, req)
    )

    # 2) Tool parser streaming with delimiters split mid-tag + mid-delimiter EOF.
    tool_parser = ToolParserManager.get_tool_parser("inkling2")(inkling2_tokenizer)
    chunks = [
        "<|message_",
        "model|><|content_",
        "text|>Hi<|end_",
        "message|><|message_model|>get_weather<|content_invoke_tool_",
        'json|>{"name":"get_weather","args":{"city":"Paris"}}<|end_',
        "message|><|content_model_end_",
    ]
    t_prev = ""
    t_content: list[str] = []
    t_calls: list[tuple[str, dict]] = []
    for ch in chunks:
        t_cur = t_prev + ch
        dm = tool_parser.extract_tool_calls_streaming(
            t_prev, t_cur, ch, [], [], [], req
        )
        if dm is not None:
            if dm.content:
                t_content.append(dm.content)
            for tc in dm.tool_calls or []:
                t_calls.append((tc.function.name, json.loads(tc.function.arguments)))
        t_prev = t_cur
    assert "".join(t_content) == "Hi"
    assert t_calls == [("get_weather", {"city": "Paris"})]


def test_inkling2_wrapped_parser_strips_all_delimiters(inkling2_tokenizer):
    class _Inkling2Wrapped(_WrappedParser):
        reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser("inkling2")
        tool_parser_cls = None

    prompt_str = (
        "<|message_system|><|content_text|>Thinking effort level: 0.9"
        "<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"
    )
    prompt_ids = inkling2_tokenizer.encode(prompt_str, add_special_tokens=False)

    for full_str, exp_r, exp_c in (
        (
            (
                "<|message_model|><|content_thinking|>Think hard.<|end_message|>"
                "<|message_model|><|content_text|>Final answer<|end_message|>"
                "<|content_model_end_sampling|>"
            ),
            "Think hard.",
            "Final answer",
        ),
        (
            (
                '<|message_model|><|content_text|>{"ok":1}<|end_message|>'
                "<|content_model_end_sampling|>"
            ),
            "",
            '{"ok":1}',
        ),
        (
            (
                "<|message_model|><|content_thinking|><|end_message|>"
                "<|message_model|><|content_text|>Empty think<|end_message|>"
                "<|content_model_end_sampling|>"
            ),
            "",
            "Empty think",
        ),
    ):
        parser = _Inkling2Wrapped(inkling2_tokenizer)
        req = ChatCompletionRequest(
            model="Inkling-2-Mini-NVFP4",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        parser._reasoning_parser.adjust_request(req)
        out_ids = inkling2_tokenizer.encode(full_str, add_special_tokens=False)
        r_parts: list[str] = []
        c_parts: list[str] = []
        for tid in out_ids:
            d_txt = inkling2_tokenizer.decode([tid], skip_special_tokens=False)
            dm = parser.parse_delta(d_txt, [tid], req, prompt_ids)
            if dm is not None:
                if dm.reasoning:
                    r_parts.append(dm.reasoning)
                if dm.content:
                    c_parts.append(dm.content)
        assert "".join(r_parts) == exp_r
        assert "".join(c_parts) == exp_c


def test_inkling2_rejects_incompatible_tool_parser(inkling2_tokenizer):
    """Fail closed when Inkling2 reasoning is paired with another tool parser."""
    with pytest.raises(ValueError, match="cannot be combined with tool parser"):
        ParserManager.get_parser(
            tool_parser_name="hermes",
            reasoning_parser_name="inkling2",
            enable_auto_tools=True,
        )

    # Also verify defense-in-depth in _WrappedParser.__init__ directly.
    class _IncompatibleWrapped(_WrappedParser):
        reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser("inkling2")
        tool_parser_cls = ToolParserManager.get_tool_parser("hermes")

    with pytest.raises(ValueError, match="cannot be combined with tool parser"):
        _IncompatibleWrapped(inkling2_tokenizer)


def test_inkling2_tool_parser_and_combined_streaming(inkling2_tokenizer):
    tool_cls = ToolParserManager.get_tool_parser("inkling2")
    tool_spec = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    req = ChatCompletionRequest(
        model="Inkling-2-Mini-NVFP4",
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=tool_spec,
        tool_choice="auto",
        stream=True,
    )

    # Non-streaming: reasoning_parser -> tool_parser chained as serving.py does.
    r_parser = ReasoningParserManager.get_reasoning_parser("inkling2")(
        inkling2_tokenizer
    )
    t_parser = tool_cls(inkling2_tokenizer)
    raw = (
        "<|message_model|><|content_thinking|>Need weather.<|end_message|>"
        "<|message_model|><|content_text|>Checking now.<|end_message|>"
        "<|message_model|>get_weather<|content_invoke_tool_json|>"
        '{"name":"get_weather","args":{"city":"Paris"}}<|end_message|>'
        "<|content_model_end_sampling|>"
    )
    reasoning, content = r_parser.extract_reasoning(raw, req)
    assert reasoning == "Need weather."
    extracted = t_parser.extract_tool_calls(content or "", req)
    assert extracted.tools_called is True
    assert extracted.content == "Checking now."
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].function.name == "get_weather"
    assert json.loads(extracted.tool_calls[0].function.arguments) == {"city": "Paris"}

    # Streaming via ParserManager.get_parser("inkling2", "inkling2"):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="inkling2",
        reasoning_parser_name="inkling2",
        enable_auto_tools=True,
    )
    assert parser_cls is _WrappedParser

    prompt_str = (
        "<|message_system|><|content_text|>Thinking effort level: 0.9"
        "<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"
    )
    prompt_ids = inkling2_tokenizer.encode(prompt_str, add_special_tokens=False)

    cases = [
        (
            "auto",
            (
                "<|message_model|><|content_thinking|>14*12=168.<|end_message|>"
                "<|message_model|><|content_text|>168<|end_message|>"
                "<|content_model_end_sampling|>"
            ),
            "14*12=168.",
            "168",
            [],
        ),
        (
            "auto",
            raw,
            "Need weather.",
            "Checking now.",
            [("get_weather", {"city": "Paris"})],
        ),
        (
            "auto",
            (
                "<|message_model|>get_weather<|content_invoke_tool_json|>"
                '{"name":"get_weather","args":{"city":"Tokyo"}}<|end_message|>'
                "<|content_model_end_sampling|>"
            ),
            "",
            "",
            [("get_weather", {"city": "Tokyo"})],
        ),
        (
            "none",
            (
                "<|message_model|><|content_thinking|>Think.<|end_message|>"
                "<|message_model|><|content_text|>Plain text<|end_message|>"
                "<|content_model_end_sampling|>"
            ),
            "Think.",
            "Plain text",
            [],
        ),
        (
            "none",
            (
                "<|message_model|>get_weather<|content_invoke_tool_json|>"
                '{"name":"get_weather","args":{"city":"Paris"}}<|end_message|>'
                "<|content_model_end_sampling|>"
            ),
            "",
            '{"name":"get_weather","args":{"city":"Paris"}}',
            [],
        ),
    ]

    for tc_mode, full_str, exp_r, exp_c, exp_tcs in cases:
        parser = parser_cls(inkling2_tokenizer, req.tools)
        case_req = ChatCompletionRequest(
            model="Inkling-2-Mini-NVFP4",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tool_spec if tc_mode != "none" else None,
            tool_choice=tc_mode,
            stream=True,
        )
        parser.adjust_request(case_req)
        assert case_req.skip_special_tokens is False
        out_ids = inkling2_tokenizer.encode(full_str, add_special_tokens=False)
        r_parts: list[str] = []
        c_parts: list[str] = []
        tcs: list[tuple[str, dict]] = []
        for tid in out_ids:
            d_txt = inkling2_tokenizer.decode([tid], skip_special_tokens=False)
            dm = parser.parse_delta(d_txt, [tid], case_req, prompt_ids)
            if dm is not None:
                if dm.reasoning:
                    r_parts.append(dm.reasoning)
                if dm.content:
                    c_parts.append(dm.content)
                for tc in dm.tool_calls or []:
                    tcs.append((tc.function.name, json.loads(tc.function.arguments)))
        assert "".join(r_parts) == exp_r
        assert "".join(c_parts) == exp_c
        assert tcs == exp_tcs


@pytest.mark.skipif(
    not os.environ.get("INKLING2_MODEL_DIR"),
    reason="Set INKLING2_MODEL_DIR to run real-tokenizer integration parity.",
)
def test_inkling2_real_tokenizer_integration():
    from vllm.tokenizers.registry import get_tokenizer

    tok = get_tokenizer(os.environ["INKLING2_MODEL_DIR"])
    parser = ReasoningParserManager.get_reasoning_parser("inkling2")(tok)
    req = ChatCompletionRequest(
        model="Inkling-2-Mini-NVFP4",
        messages=[{"role": "user", "content": "hi"}],
    )
    raw = (
        "<|message_model|><|content_thinking|>Check.<|end_message|>"
        "<|message_model|><|content_text|>OK<|end_message|>"
        "<|content_model_end_sampling|>"
    )
    assert parser.extract_reasoning(raw, req) == ("Check.", "OK")
