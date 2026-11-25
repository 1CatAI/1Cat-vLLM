# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling reasoning + tool parsers through the parser stack (CPU only).

Needs the Inkling-2-Mini tokenizer: a local path or hub id in INKLING2_MINI_PATH.
"""

import json
import os

import pytest
from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager

MODEL = os.environ.get("INKLING2_MINI_PATH", "thinkingmachines/Inkling-2-Mini")
try:
    tok = AutoTokenizer.from_pretrained(MODEL)
except Exception:  # offline or gated: nothing to test against
    pytest.skip("Inkling-2-Mini tokenizer not available", allow_module_level=True)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "run",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "w",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    },
]
MM, TH, TX, IV, END, EOS = (
    "<|message_model|>",
    "<|content_thinking|>",
    "<|content_text|>",
    "<|content_invoke_tool_json|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
)


def call(name, args):
    return (
        MM
        + name
        + IV
        + json.dumps(
            {"name": name, "args": args}, separators=(",", ":"), ensure_ascii=False
        )
        + END
    )


CASES = {
    "think_text_two_calls": (
        MM
        + TH
        + "I should list files, then write.\nOk."
        + END
        + MM
        + TX
        + "Let me look."
        + END
        + call("bash", {"command": "ls -la"})
        + call("write_file", {"path": "a.txt", "content": "x\ny"})
        + EOS,
        "I should list files, then write.\nOk.",
        "Let me look.",
        [
            ("bash", {"command": "ls -la"}),
            ("write_file", {"path": "a.txt", "content": "x\ny"}),
        ],
    ),
    "text_only": (MM + TX + "Hello there!" + END + EOS, None, "Hello there!", []),
    "think_nested_unicode": (
        MM
        + TH
        + "Plan: édit"
        + END
        + call(
            "write_file",
            {"path": "ü.json", "content": json.dumps({"k": [1, {"z": None}]})},
        )
        + EOS,
        "Plan: édit",
        None,
        [
            (
                "write_file",
                {"path": "ü.json", "content": json.dumps({"k": [1, {"z": None}]})},
            )
        ],
    ),
}


parser_cls = ParserManager.get_parser(
    tool_parser_name="inkling",
    reasoning_parser_name="inkling",
    enable_auto_tools=True,
    model_name="inkling",
)

# A multi-turn prompt (earlier assistant turn with a tool call, then a tool result).
PROMPT_IDS = tok.encode(
    tok.apply_chat_template(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Sure.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"command": "ls"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
        ],
        tools=TOOLS,
        tokenize=False,
    ),
    add_special_tokens=False,
)


def markup(name, args):
    body = json.dumps(
        {"name": name, "args": args}, separators=(",", ":"), ensure_ascii=False
    )
    return IV.join([MM + name, body]) + END


def decode(ids):
    return tok.decode(
        ids, skip_special_tokens=False, spaces_between_special_tokens=False
    )


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("tools_on", [True, False])
@pytest.mark.parametrize("mode", ["stream", "full"])
def test_parse(case, tools_on, mode):
    text, want_reasoning, want_content, want_calls = CASES[case]
    ids = tok.encode(text, add_special_tokens=False)
    parser = parser_cls(tok, TOOLS if tools_on else None)
    kw = {"tools": TOOLS, "tool_choice": "auto"} if tools_on else {}
    req = ChatCompletionRequest(
        model="inkling", messages=[{"role": "user", "content": "x"}], **kw
    )
    req = parser.adjust_request(req)
    assert req.skip_special_tokens is False
    assert req.spaces_between_special_tokens is False
    if mode == "stream":
        reasoning, content, calls = "", "", []
        prev = ""
        for k in range(len(ids)):
            cur = decode(ids[: k + 1])
            d = parser.parse_delta(
                delta_text=cur[len(prev) :],
                delta_token_ids=[ids[k]],
                request=req,
                prompt_token_ids=PROMPT_IDS,
            )
            prev = cur
            if d is None:
                continue
            reasoning += d.reasoning or ""
            content += d.content or ""
            for tc in d.tool_calls:
                calls.append((tc.function.name, json.loads(tc.function.arguments)))
    else:
        # As the chat server does: tool calls replace the content only when some
        # were called.
        reasoning, content = parser.extract_reasoning(decode(ids), req)
        calls = []
        if tools_on:
            info = parser.extract_tool_calls(content or "", req)
            if info.tools_called:
                content = info.content
                calls = [
                    (c.function.name, json.loads(c.function.arguments))
                    for c in info.tool_calls
                ]
    got = (reasoning or None, content or None, calls)
    if tools_on:
        want = (want_reasoning, want_content, want_calls)
    else:  # no tools allowed: calls stay in the content as model markup
        plain = (want_content or "") + "".join(markup(n, a) for n, a in want_calls)
        want = (want_reasoning, plain or None, [])
    assert got == want


def test_prompt_is_not_reasoning_end():
    """A prompt that ends after a tool result must not count as ended reasoning."""
    assert not parser_cls(tok).is_reasoning_end(PROMPT_IDS)
