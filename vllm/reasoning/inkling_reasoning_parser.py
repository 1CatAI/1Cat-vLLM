# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for Inkling 2 (`--reasoning-parser inkling`).

Reasoning is the body of the turn's `<|content_thinking|>` segments. Content
is the visible text without segment markers; tool calls stay as model markup
so the `inkling` tool parser can read them (vLLM keeps this content as-is when
no tool is called). Streaming content after the reasoning goes through the
unified `inkling` parser (`vllm/parser/inkling_parser.py`). See
`vllm/models/inkling2/parsing.py` for the format.
"""

from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.models.inkling2.parsing import (
    THINKING_ID,
    content_start_index,
    parse_segments,
    reasoning_ended,
    split_reasoning,
    visible_content,
)
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser


def keep_special_tokens(request):
    """Inkling's segment markers are special tokens: keep them in the text."""
    if hasattr(request, "skip_special_tokens"):
        request.skip_special_tokens = False
    if hasattr(request, "spaces_between_special_tokens"):
        request.spaces_between_special_tokens = False
    return request


class InklingReasoningParser(ReasoningParser):
    def adjust_request(self, request):
        return keep_special_tokens(request)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return reasoning_ended(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        return reasoning_ended(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        k = content_start_index(input_ids)
        if k is not None:
            return list(input_ids[k:])
        # The boundary `<|message_model|>` came in an earlier delta.
        if input_ids and input_ids[0] != THINKING_ID:
            return list(input_ids)
        return []

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        k = content_start_index(token_ids)
        return len(token_ids) if k is None else k

    def extract_reasoning(
        self, model_output: str, request
    ) -> tuple[str | None, str | None]:
        segs = parse_segments(model_output)
        reasoning = "".join(s.body for s in segs if s.kind == "thinking")
        content = visible_content(segs)
        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        prev_reasoning, prev_start = split_reasoning(previous_text)
        cur_reasoning, cur_start = split_reasoning(current_text)
        reasoning = (
            cur_reasoning[len(prev_reasoning) :]
            if cur_reasoning.startswith(prev_reasoning)
            else ""
        )
        content = None
        if cur_start is not None:
            content = current_text[cur_start:] if prev_start is None else delta_text
        if not reasoning and not content:
            return None
        return DeltaMessage(reasoning=reasoning or None, content=content or None)
