# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool-call parser for Inkling 2 (`--tool-call-parser inkling`).

Tool calls are `<|message_model|>{name}<|content_invoke_tool_json|>
{"name": ..., "args": {...}}<|end_message|>` segments; visible text is in
`<|content_text|>` segments (see `vllm/models/inkling2/parsing.py`).
Streaming sends text as it arrives and each tool call whole (name and
complete arguments) once its segment closes, so clients never see partial
JSON.

Named / required tool choice is not enforced with a grammar: the request
is left unconstrained (a JSON-schema constraint would break the format)
and the model's calls are parsed as for `auto`.
"""

from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.models.inkling2.parsing import parse_segments, parse_tool_call, render_raw
from vllm.reasoning.inkling_reasoning_parser import keep_special_tokens
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)


def _visible_text(segs) -> str:
    """Text the user sees: text segments and stray raw text (no thinking,
    tool calls or incomplete headers)."""
    return "".join(s.body for s in segs if s.kind in ("text", "raw"))


class InklingToolParser(ToolParser):
    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        try:
            super().__init__(tokenizer, tools)
        except TypeError:
            super().__init__(tokenizer)
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self.current_tool_id = -1
        self._content_sent = 0
        self._calls_sent = 0

    def adjust_request(self, request):
        return keep_special_tokens(request)

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        segs = parse_segments(model_output)
        calls: list[ToolCall] = []
        content_parts: list[str] = []
        for seg in segs:
            if seg.kind == "invoke":
                parsed = parse_tool_call(seg)
                if parsed is None:
                    logger.warning("Inkling: unparsable tool call: %.200s", seg.body)
                    content_parts.append(render_raw(seg))
                    continue
                name, arguments = parsed
                calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(name=name, arguments=arguments),
                    )
                )
            elif seg.kind in ("text", "raw"):
                content_parts.append(seg.body)
        content = "".join(content_parts)
        return ExtractedToolCallInformation(
            tools_called=bool(calls), tool_calls=calls, content=content or None
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        segs = parse_segments(current_text)
        visible = _visible_text(segs)
        content = visible[self._content_sent :]
        self._content_sent = max(self._content_sent, len(visible))
        deltas: list[DeltaToolCall] = []
        closed_calls = [s for s in segs if s.kind == "invoke" and s.closed]
        for seg in closed_calls[self._calls_sent :]:
            self._calls_sent += 1
            parsed = parse_tool_call(seg)
            if parsed is None:
                logger.warning("Inkling: unparsable tool call: %.200s", seg.body)
                content += render_raw(seg)
                continue
            name, arguments = parsed
            self.current_tool_id += 1
            self.prev_tool_call_arr.append({"name": name, "arguments": arguments})
            self.streamed_args_for_tool.append(arguments)
            deltas.append(
                DeltaToolCall(
                    index=self.current_tool_id,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(name=name, arguments=arguments),
                )
            )
        if not content and not deltas:
            return None
        return DeltaMessage(content=content or None, tool_calls=deltas)
