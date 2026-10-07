# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 tokenizer with the official prompt encoder (``--tokenizer-mode deepseek_v41``).

Port of upstream vLLM b6d8e8af ``vllm/tokenizers/deepseek_v41.py`` onto 1Cat's V4 wrapper pattern. Prompts are
rendered by the official ``encoding/encoding.py`` (vendored verbatim as ``deepseek_v41_encoding``); the serving
controls follow the checkpoint's ``chat_template.jinja``:

* ``thinking_mode`` "thinking" | "chat" (default "thinking"); ``thinking`` / ``enable_thinking`` booleans are
  accepted as aliases (an explicit False selects chat);
* ``reasoning_effort`` int in [1, 100] or "low" 50 / "medium" 62 / "high" 75 / "max" 100 (default "high");
  "minimal" -> 50 and "xhigh" -> 100 (OpenAI request values; 1Cat V4 compatibility); "none" selects chat mode;
* ``drop_thinking`` (default True); top-level ``tools`` attach to a leading system message, or to a new empty one.
"""

from __future__ import annotations

import copy
from typing import Any

from transformers import PreTrainedTokenizerFast

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .deepseek_v41_encoding import encode_messages
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike

# chat_template.jinja: {"low": 50, "medium": 62, "high": 75, "max": 100}. The OpenAI request field also allows
# "minimal" (-> the smallest named budget, 50) and "xhigh" (-> "max", as 1Cat's V4 wrapper does).
REASONING_EFFORT_BUDGETS: dict[str, int] = {"minimal": 50, "low": 50, "medium": 62, "high": 75, "max": 100,
                                            "xhigh": 100}
DEFAULT_REASONING_EFFORT = "high"


def resolve_reasoning(kwargs: dict[str, Any]) -> tuple[str, int]:
    """(thinking_mode, budget) from chat-template kwargs, with the jinja template's defaults."""
    thinking_mode = kwargs.get("thinking_mode")
    if thinking_mode is None:
        flags = [kwargs[k] for k in ("thinking", "enable_thinking") if kwargs.get(k) is not None]
        thinking_mode = "chat" if flags and not any(bool(f) for f in flags) else "thinking"
    if thinking_mode not in ("thinking", "chat"):
        raise ValueError(f"DeepSeek V4.1 thinking_mode must be 'thinking' or 'chat', got {thinking_mode!r}")
    effort = kwargs.get("reasoning_effort")
    if effort == "none":
        return "chat", REASONING_EFFORT_BUDGETS[DEFAULT_REASONING_EFFORT]
    if effort is None:
        effort = DEFAULT_REASONING_EFFORT
    if isinstance(effort, str) and effort in REASONING_EFFORT_BUDGETS:
        return thinking_mode, REASONING_EFFORT_BUDGETS[effort]
    if type(effort) is int and 1 <= effort <= 100:
        return thinking_mode, effort
    raise ValueError("DeepSeek V4.1 reasoning_effort must be an integer in [1, 100] or one of "
                     f"{sorted(REASONING_EFFORT_BUDGETS)} / 'none'; got {effort!r}")


def normalize_messages(messages: list[ChatCompletionMessageParam],
                       tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI messages -> the encoder's message dicts (text-only; list content joined by blank lines)."""
    result = [dict(message) for message in copy.deepcopy(messages)]
    for message in result:
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool", "latest_reminder"):
            raise ValueError(f"Invalid role: {role}")
        if role == "developer":
            message["role"] = "system"
        if "reasoning" in message and message.get("reasoning_content") is None:
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type not in ("text", "input_text", "output_text"):
                    raise ValueError(f"the SM70 DeepSeek V4.1 port is text-only; got a {part_type!r} content part")
                parts.append(block.get("text", ""))
            message["content"] = "\n\n".join(parts)
    if tools:
        if result and result[0]["role"] == "system":
            result[0].setdefault("tools", tools)
        else:
            result.insert(0, {"role": "system", "content": "", "tools": tools})
    return result


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """Wrap an HF tokenizer with the V4.1 prompt encoder."""
    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            # The generic renderer's ``conversation`` flattened text parts with '\n'; V4.1 joins them with '\n\n',
            # so the original messages are encoded (as upstream does).
            thinking_mode, budget = resolve_reasoning(kwargs)
            prompt = encode_messages(
                normalize_messages(messages, tools),
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=budget,
            )
            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {k: kwargs[k] for k in ("truncation", "max_length") if k in kwargs}
                return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)
            return prompt

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer,)

    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"
    wrapped.__class__ = _DeepseekV41Tokenizer
    return wrapped


class DeepseekV41Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
