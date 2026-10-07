# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 reasoning parser (``--reasoning-parser deepseek_v41``).

Output format (checkpoint encoding/encoding.py): in thinking mode the prompt ends with ``<think>`` and the model
writes its reasoning, ``</think>``, then the answer; in chat mode the prompt ends with ``</think>`` and the output is
the answer. Unlike V3/V4, V4.1 defaults to THINKING (chat_template.jinja), so the mode is resolved with the exact
rules the tokenizer applies (``vllm.tokenizers.deepseek_v41.resolve_reasoning``): ``thinking_mode``,
``thinking`` / ``enable_thinking`` (explicit False -> chat) and ``reasoning_effort="none"`` -> chat.
Upstream vLLM implements this as an adapter over ``vllm.parser.engine`` (absent in 1Cat); this delegates to 1Cat's
DeepSeekR1 (``<think>``/``</think>``) or identity parser like DeepSeekV3ReasoningParser does.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from vllm.reasoning.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from vllm.tokenizers.deepseek_v41 import resolve_reasoning


class DeepSeekV41EngineReasoningParser(DeepSeekV3ReasoningParser):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        chat_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        thinking_mode, _ = resolve_reasoning(chat_kwargs)
        self.thinking_mode = thinking_mode
        chat_kwargs["thinking"] = chat_kwargs["enable_thinking"] = thinking_mode == "thinking"
        kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, *args, **kwargs)
