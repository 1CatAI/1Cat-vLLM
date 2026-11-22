# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object

logger = init_logger(__name__)

_TOOL_JSON_TAG = "<|content_invoke_tool_json|>"
_TEXT_TAGS = frozenset(
    ("<|content_text|>", "<|content_invoke_tool_text|>", "<|content_tool_error|>")
)
_THINKING_TAG = "<|content_thinking|>"
_END_MESSAGE_TAG = "<|end_message|>"
_CLOSE_TAGS = frozenset(
    ("<|message_model|>", _END_MESSAGE_TAG, "<|content_model_end_sampling|>")
)

# Matches Inkling-2 role/content channel headers and end delimiters:
#   <|message_model|>[header_name]<|content_...|>
#   or standalone <|message_model|> / <|end_message|> / <|content_model_end_sampling|>
_INKLING2_TOKEN_RE = re.compile(
    r"(?:<\|message_model\|>([^<]*))?"
    r"(<\|content_(?:thinking|text|invoke_tool_text|tool_error|"
    r"invoke_tool_json)\|>)"
    r"|(<\|message_model\|>|<\|end_message\|>|<\|content_model_end_sampling\|>)"
)


def _find_partial_tag_prefix(text: str) -> int:
    """Return index of an incomplete trailing `<|...` tag, or len(text) if none."""
    idx = text.rfind("<")
    if idx == -1:
        return len(text)
    suffix = text[idx:]
    if suffix == "<" or suffix.startswith("<|") and "|>" not in suffix:
        return idx
    return len(text)


def _build_tool_call_parts(raw_json: str, fallback_name: str) -> tuple[str, str] | None:
    """Parse an Inkling-2 tool JSON block `{"name": ..., "args": {...}}`."""
    raw = raw_json.strip()
    if not raw:
        return None
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Inkling-2 tool call JSON: %r", raw[:200])
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        name = fallback_name.strip()
    if not name:
        return None
    args = payload.get("args", payload.get("arguments", {}))
    if isinstance(args, str):
        arguments = args
    else:
        arguments = json.dumps(args if args is not None else {}, ensure_ascii=False)
    return name, arguments


class Inkling2ToolParser(ToolParser):
    """Tool call parser for Inkling-2 models.

    Inkling-2 emits tool calls as:
        <|message_model|><fn_name><|content_invoke_tool_json|>{"name":...,"args":{...}}<|end_message|>
    and plain assistant text as:
        <|message_model|><|content_text|>...<|end_message|><|content_model_end_sampling|>
    """

    # Required/named tool_choice must use the Inkling-2 delimited wrapper rather
    # than bare OpenAI JSON schema mode.
    supports_required_and_named = False
    supports_content_delimiters = True

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools)
        # Streaming FSM states:
        # "pre" | "between" | "header" | "text" | "thinking" | "tool_json"
        self._stream_state: str = "pre"
        self._buf: str = ""
        self._header_name: str = ""
        self._tool_json_buf: str = ""

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        # Preserve Inkling-2 channel delimiters (<|content_invoke_tool_json|>, etc.).
        request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        del request
        if not model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        state = "pre"
        header_name = ""
        tool_buf: list[str] = []
        pos = 0

        for match in _INKLING2_TOKEN_RE.finditer(model_output):
            chunk = model_output[pos : match.start()]
            if chunk:
                if state in ("pre", "text"):
                    content_parts.append(chunk)
                elif state == "tool_json":
                    tool_buf.append(chunk)

            msg_header = match.group(1) or ""
            open_tag = match.group(2)
            close_tag = match.group(3)

            if open_tag == _TOOL_JSON_TAG:
                state = "tool_json"
                header_name = msg_header
                tool_buf = []
            elif open_tag in _TEXT_TAGS:
                state = "text"
            elif open_tag == _THINKING_TAG:
                state = "thinking"
            elif close_tag in _CLOSE_TAGS:
                if state == "tool_json" and close_tag == _END_MESSAGE_TAG:
                    parsed = _build_tool_call_parts("".join(tool_buf), header_name)
                    if parsed is not None:
                        name, arguments = parsed
                        tool_calls.append(
                            ToolCall(
                                id=make_tool_call_id(),
                                type="function",
                                function=FunctionCall(name=name, arguments=arguments),
                            )
                        )
                    tool_buf = []
                    header_name = ""
                if close_tag == "<|message_model|>":
                    state = "header"
                    header_name = ""
                else:
                    state = "between"
            pos = match.end()

        tail = model_output[pos:]
        if tail:
            if state in ("pre", "text"):
                content_parts.append(tail)
            elif state == "tool_json":
                tool_buf.append(tail)

        cleaned_content = "".join(content_parts)
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_content or None,
            )
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=cleaned_content,
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
        del (
            previous_text,
            current_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
        if not delta_text and not self._buf:
            return None

        work = self._buf + delta_text
        self._buf = ""

        safe_end = _find_partial_tag_prefix(work)
        if safe_end < len(work):
            self._buf = work[safe_end:]
            work = work[:safe_end]

        if not work:
            return None

        out_content: list[str] = []
        out_tool_calls: list[DeltaToolCall] = []
        pos = 0

        for match in _INKLING2_TOKEN_RE.finditer(work):
            chunk = work[pos : match.start()]
            if chunk:
                if self._stream_state in ("pre", "text"):
                    out_content.append(chunk)
                elif self._stream_state == "header":
                    self._header_name += chunk
                elif self._stream_state == "tool_json":
                    self._tool_json_buf += chunk

            msg_header = match.group(1) or ""
            open_tag = match.group(2)
            close_tag = match.group(3)

            if open_tag == _TOOL_JSON_TAG:
                if msg_header:
                    self._header_name = msg_header
                self._stream_state = "tool_json"
                self._tool_json_buf = ""
            elif open_tag in _TEXT_TAGS:
                self._header_name = ""
                self._stream_state = "text"
            elif open_tag == _THINKING_TAG:
                self._header_name = ""
                self._stream_state = "thinking"
            elif close_tag in _CLOSE_TAGS:
                if self._stream_state == "tool_json" and close_tag == _END_MESSAGE_TAG:
                    parsed = _build_tool_call_parts(
                        self._tool_json_buf, self._header_name
                    )
                    self._tool_json_buf = ""
                    self._header_name = ""
                    if parsed is not None:
                        name, arguments = parsed
                        self.current_tool_id += 1
                        tool_idx = self.current_tool_id
                        call_id = make_tool_call_id()
                        self.prev_tool_call_arr.append(
                            {"name": name, "arguments": arguments}
                        )
                        self.streamed_args_for_tool.append(arguments)
                        out_tool_calls.append(
                            DeltaToolCall(
                                index=tool_idx,
                                id=call_id,
                                type="function",
                                function=DeltaFunctionCall(
                                    name=name, arguments=arguments
                                ),
                            )
                        )
                if close_tag == "<|message_model|>":
                    self._stream_state = "header"
                    self._header_name = ""
                else:
                    self._stream_state = "between"
            pos = match.end()

        tail = work[pos:]
        if tail:
            if self._stream_state in ("pre", "text"):
                out_content.append(tail)
            elif self._stream_state == "header":
                self._header_name += tail
            elif self._stream_state == "tool_json":
                self._tool_json_buf += tail

        content_str = "".join(out_content) if out_content else None
        if content_str is None and not out_tool_calls:
            return None
        if out_tool_calls and content_str is not None:
            return DeltaMessage(content=content_str, tool_calls=out_tool_calls)
        if out_tool_calls:
            return DeltaMessage(tool_calls=out_tool_calls)
        return DeltaMessage(content=content_str)
