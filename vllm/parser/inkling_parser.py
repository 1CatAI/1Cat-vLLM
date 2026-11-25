# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Inkling 2 parser, picked when both `--reasoning-parser` and
`--tool-call-parser` are `inkling`.

It is the usual reasoning + tool parser pair, with one change for streams
whose request allows no tools (`tool_choice` "none", or no tools at all).
The generic path forwards that content raw, which would show Inkling's segment
markers to the client. This parser sends the visible text instead; tool calls
the model writes anyway stay as model markup, the same as without streaming.
"""

from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.models.inkling2.parsing import parse_segments, visible_content
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.inkling_reasoning_parser import InklingReasoningParser
from vllm.tool_parsers.inkling_tool_parser import InklingToolParser


class InklingParser(_WrappedParser):
    reasoning_parser_cls = InklingReasoningParser
    tool_parser_cls = InklingToolParser

    def __init__(self, tokenizer, tools=None, **kwargs):
        super().__init__(tokenizer, tools, **kwargs)
        self._plain_sent = 0

    def _extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
        tool_call_idx: int | None = None,
        tool_call_id_type: str = "random",
        function_name_returned: bool = False,
    ) -> tuple[DeltaMessage | None, bool]:
        if request.tool_choice == "none" or not getattr(request, "tools", None):
            text = visible_content(
                parse_segments(current_text), closed_invokes_only=True
            )
            out = text[self._plain_sent :]
            self._plain_sent = max(self._plain_sent, len(text))
            if out:
                return DeltaMessage(content=out), False
            # Like the generic path: an empty content delta keeps the final chunk.
            if delta_token_ids and current_text:
                return DeltaMessage(content=""), False
            return None, False
        return super()._extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
            tool_call_idx=tool_call_idx,
            tool_call_id_type=tool_call_id_type,
            function_name_returned=function_name_returned,
        )
