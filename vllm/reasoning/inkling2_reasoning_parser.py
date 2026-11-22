# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

_MARKER_RE = re.compile(
    r"(?:<\|message_model\|>[^<]*)?"
    r"(<\|content_(?:thinking|text|invoke_tool_text|tool_error|"
    r"invoke_tool_json)\|>)"
    r"|(<\|message_model\|>|<\|end_message\|>|<\|content_model_end_sampling\|>)"
)

_THINKING_TAG = "<|content_thinking|>"
_CONTENT_TAGS = frozenset(
    ("<|content_text|>", "<|content_invoke_tool_text|>", "<|content_tool_error|>")
)
_TOOL_JSON_TAG = "<|content_invoke_tool_json|>"
_CLOSE_TAGS = frozenset(
    ("<|message_model|>", "<|end_message|>", "<|content_model_end_sampling|>")
)


class Inkling2ReasoningParser(ReasoningParser):
    """Reasoning parser for Inkling-2 models.

    Inkling-2 emits special delimiter tokens:
        <|message_model|><|content_thinking|>...<|end_message|>
        <|message_model|><|content_text|>...<|end_message|><|content_model_end_sampling|>
    When `reasoning_effort=0.0`, the thinking block is omitted and the model
    begins directly with `<|message_model|><|content_text|>`.
    """

    parses_content_delimiters = True

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to Inkling2ReasoningParser."
            )
        vocab = self.vocab
        self._thinking_id = vocab.get(_THINKING_TAG)
        self._end_msg_id = vocab.get("<|end_message|>")
        self._end_sampling_id = vocab.get("<|content_model_end_sampling|>")
        self._msg_model_id = vocab.get("<|message_model|>")
        self._content_ids = {
            tid for tag in _CONTENT_TAGS if (tid := vocab.get(tag)) is not None
        }
        self._tool_json_id = vocab.get(_TOOL_JSON_TAG)
        self._non_assistant_role_ids = {
            tid
            for tag in ("<|message_user|>", "<|message_system|>", "<|message_tool|>")
            if (tid := vocab.get(tag)) is not None
        }
        if self._thinking_id is None or self._end_msg_id is None:
            raise RuntimeError(
                "Inkling2ReasoningParser could not locate "
                "<|content_thinking|>/<|end_message|> in tokenizer vocab."
            )
        self._emit_tool_json_marker: bool = False
        self._all_special_ids = {
            tid
            for tid in (
                self._thinking_id,
                self._end_msg_id,
                self._end_sampling_id,
                self._msg_model_id,
                self._tool_json_id,
                *self._content_ids,
            )
            if tid is not None
        }

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        """Preserve special delimiter tokens needed to separate channels."""
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Locate the last message-role boundary. Prompts end with a user/system/
        # tool turn whose trailing <|end_message|> must never end reasoning.
        role_start = 0
        for idx in range(len(input_ids) - 1, -1, -1):
            tid = input_ids[idx]
            if tid in self._non_assistant_role_ids:
                return False
            if tid == self._msg_model_id:
                role_start = idx + 1
                break
        saw_thinking = False
        for tid in input_ids[role_start:]:
            if tid in self._content_ids or (
                self._tool_json_id is not None and tid == self._tool_json_id
            ):
                return True
            if tid == self._thinking_id:
                saw_thinking = True
            elif tid == self._end_msg_id and saw_thinking:
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        trigger = {self._end_msg_id, *self._content_ids}
        if self._tool_json_id is not None:
            trigger.add(self._tool_json_id)
        if not any(tid in trigger for tid in delta_ids):
            return False
        return self.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        state = "pre"
        out: list[int] = []
        for tid in input_ids:
            if tid == self._thinking_id:
                state = "thinking"
            elif tid in self._content_ids:
                state = "text"
            elif self._tool_json_id is not None and tid == self._tool_json_id:
                state = "text"
                if self._emit_tool_json_marker:
                    out.append(tid)
            elif tid in self._all_special_ids:
                state = "between"
            elif state == "text":
                out.append(tid)
        return out

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        in_thinking = False
        count = 0
        for tid in token_ids:
            if tid == self._thinking_id:
                in_thinking = True
            elif tid in self._all_special_ids:
                in_thinking = False
            elif in_thinking:
                count += 1
        return count

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if not model_output:
            return None, None
        preserve_tool_tags = bool(
            getattr(request, "tools", None)
            and getattr(request, "tool_choice", None) != "none"
        )
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        state = "pre"
        pos = 0
        saw_marker = False
        for match in _MARKER_RE.finditer(model_output):
            saw_marker = True
            chunk = model_output[pos : match.start()]
            if chunk:
                if state == "thinking":
                    reasoning_parts.append(chunk)
                elif state in ("text", "pre", "tool_raw"):
                    content_parts.append(chunk)
            open_tag = match.group(1)
            close_tag = match.group(2)
            if state == "tool_raw" and close_tag == "<|end_message|>":
                content_parts.append(close_tag)
                state = "between"
            elif open_tag == _THINKING_TAG:
                state = "thinking"
            elif open_tag in _CONTENT_TAGS:
                state = "text"
            elif open_tag == _TOOL_JSON_TAG:
                if preserve_tool_tags:
                    content_parts.append(match.group(0))
                    state = "tool_raw"
                else:
                    state = "text"
            elif close_tag in _CLOSE_TAGS:
                state = "between"
            pos = match.end()
        tail = model_output[pos:]
        if tail:
            if state == "thinking":
                reasoning_parts.append(tail)
            elif state in ("text", "pre", "tool_raw"):
                content_parts.append(tail)
        if not saw_marker:
            return None, model_output or None
        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        return reasoning, content

    def _state_from_prefix(self, previous_token_ids: Sequence[int]) -> str:
        for tid in reversed(previous_token_ids):
            if tid == self._thinking_id:
                return "thinking"
            if tid in self._content_ids or (
                self._tool_json_id is not None and tid == self._tool_json_id
            ):
                return "text"
            if tid in self._all_special_ids:
                return "between"
        return "pre"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del previous_text, current_text, current_token_ids
        state = self._state_from_prefix(previous_token_ids)
        has_special_id = any(tid in self._all_special_ids for tid in delta_token_ids)
        has_marker_text = "<|" in delta_text

        if not has_special_id and not has_marker_text:
            if not delta_text:
                return None
            if state == "thinking":
                return DeltaMessage(reasoning=delta_text)
            if state in ("text", "pre"):
                return DeltaMessage(content=delta_text)
            return None

        if has_marker_text:
            r_parts: list[str] = []
            c_parts: list[str] = []
            pos = 0
            for match in _MARKER_RE.finditer(delta_text):
                chunk = delta_text[pos : match.start()]
                if chunk:
                    if state == "thinking":
                        r_parts.append(chunk)
                    elif state in ("text", "pre"):
                        c_parts.append(chunk)
                open_tag = match.group(1)
                close_tag = match.group(2)
                if open_tag == _THINKING_TAG:
                    state = "thinking"
                elif open_tag in _CONTENT_TAGS:
                    state = "text"
                elif open_tag == _TOOL_JSON_TAG:
                    state = "text"
                    if self._emit_tool_json_marker:
                        c_parts.append(open_tag)
                elif close_tag in _CLOSE_TAGS:
                    state = "between"
                pos = match.end()
            tail = delta_text[pos:]
            if tail:
                if state == "thinking":
                    r_parts.append(tail)
                elif state in ("text", "pre"):
                    c_parts.append(tail)
            reasoning = "".join(r_parts) or None
            content = "".join(c_parts) or None
            if reasoning is None and content is None:
                return None
            return DeltaMessage(reasoning=reasoning, content=content)

        # Fallback if caller stripped special tokens from delta_text but kept IDs.
        r_ids: list[int] = []
        c_ids: list[int] = []
        for tid in delta_token_ids:
            if tid == self._thinking_id:
                state = "thinking"
            elif tid in self._content_ids:
                state = "text"
            elif self._tool_json_id is not None and tid == self._tool_json_id:
                state = "text"
                if self._emit_tool_json_marker:
                    c_ids.append(tid)
            elif tid in self._all_special_ids:
                state = "between"
            elif state == "thinking":
                r_ids.append(tid)
            elif state in ("text", "pre"):
                c_ids.append(tid)
        reasoning = (
            self.model_tokenizer.decode(r_ids, skip_special_tokens=True)
            if r_ids
            else None
        )
        content = (
            self.model_tokenizer.decode(c_ids, skip_special_tokens=True)
            if c_ids
            else None
        )
        if not reasoning and not content:
            return None
        return DeltaMessage(
            reasoning=reasoning or None,
            content=content or None,
        )
