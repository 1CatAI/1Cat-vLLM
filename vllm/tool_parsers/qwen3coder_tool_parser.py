# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

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
from vllm.envs import VLLM_ENFORCE_STRICT_TOOL_CALLING
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    find_tool_properties,
)

logger = init_logger(__name__)


class Qwen3CoderToolParser(ToolParser):
    supports_required_and_named: bool = not VLLM_ENFORCE_STRICT_TOOL_CALLING

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Qwen3 XML Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

        logger.debug(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        # Store accumulated parameters for type conversion
        self.accumulated_params = {}
        self.streaming_request = None
        self.json_tool_calls_streamed = 0

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        if not isinstance(param_value, str):
            return param_value
        param_schema = param_config.get(param_name, {})
        param_types = extract_types_from_schema(param_schema)
        return coerce_to_schema_type(param_value, param_types)

    def _parameter_start_positions(
        self,
        text: str,
        param_config: dict,
    ) -> list[int]:
        """Find parameter tags without treating inline code as markup.

        Qwen's wire format puts parameter tags on structural boundaries. Tool
        values, especially code and JSON passed to Write/Edit, may themselves
        contain strings such as ``<parameter=x>``. A raw substring scan treats
        those literals as new arguments and corrupts the tool call.

        Unknown names at a bare line boundary are ignored when a schema is
        available, but remain accepted after a real closing tag. This preserves
        malformed-output recovery without interpreting an inline literal as a
        schema field.
        """
        positions: list[int] = []
        search_idx = 0
        while True:
            position = text.find(self.parameter_prefix, search_idx)
            if position == -1:
                break
            search_idx = position + len(self.parameter_prefix)

            name_end = text.find(">", search_idx)
            if name_end == -1:
                break
            name = text[search_idx:name_end]

            line_start = text.rfind("\n", 0, position) + 1
            at_line_boundary = not text[line_start:position].strip()
            after_close = text[:position].rstrip().endswith(self.parameter_end_token)
            after_function_header = False
            if not positions:
                function_start = text.rfind(self.tool_call_prefix, 0, position)
                if function_start != -1:
                    function_header_end = text.find(">", function_start)
                    after_function_header = (
                        function_header_end != -1
                        and not text[function_header_end + 1 : position].strip()
                    )
                elif not text[:position].strip():
                    after_function_header = True

            is_structural = at_line_boundary or after_close or after_function_header
            if not is_structural:
                continue
            if (
                param_config
                and name not in param_config
                and not after_close
                and not after_function_header
            ):
                continue
            positions.append(position)
        return positions

    def _find_structural_parameter_end(
        self,
        text: str,
        *,
        value_start: int,
        next_parameter_start: int | None,
        container_end: int | None,
        allow_text_end: bool,
    ) -> int | None:
        """Return a closing tag only when its suffix is structural.

        Literal ``</parameter>`` text is common in generated XML and source
        files. It closes the current value only when followed by the next real
        parameter, the function/tool boundary, or (for a complete non-streamed
        body) the end of the text. Streaming deliberately waits for that one-
        token lookahead instead of publishing a value that may later truncate.
        """
        search_idx = value_start
        limit = next_parameter_start
        if limit is None:
            limit = container_end
        if limit is None:
            limit = len(text)

        while True:
            position = text.find(self.parameter_end_token, search_idx)
            if position == -1 or position >= limit:
                return None
            suffix = position + len(self.parameter_end_token)
            while suffix < len(text) and text[suffix].isspace():
                suffix += 1
            if next_parameter_start is not None and suffix == next_parameter_start:
                return position
            if container_end is not None and suffix == container_end:
                return position
            if allow_text_end and suffix == len(text):
                return position
            search_idx = position + len(self.parameter_end_token)

    def _parse_xml_function_call(self, function_call_str: str) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.find(">")
        # If there's no ">" character, this is not a valid xml function call
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        param_config = find_tool_properties(self.tools, function_name)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        param_starts = self._parameter_start_positions(parameters, param_config)
        for param_index, position in enumerate(param_starts):
            name_start = position + len(self.parameter_prefix)
            name_end = parameters.find(">", name_start)
            if name_end == -1:
                continue
            param_name = parameters[name_start:name_end]
            value_start = name_end + 1
            next_start = (
                param_starts[param_index + 1]
                if param_index + 1 < len(param_starts)
                else None
            )
            value_end = self._find_structural_parameter_end(
                parameters,
                value_start=value_start,
                next_parameter_start=next_start,
                container_end=len(parameters),
                allow_text_end=True,
            )
            if value_end is None:
                value_end = next_start if next_start is not None else len(parameters)
            param_value = parameters[value_start:value_end]
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)
            ),
        )

    @staticmethod
    def _parse_json_function_call(payload: str) -> ToolCall | None:
        """Parse the alternate Qwen/OpenCode JSON body inside ``<tool_call>``.

        Some Qwen coding prompts teach ``{"name": ..., "arguments": ...}``
        while the native model template teaches XML ``<function=...>``.  The
        wrapper token is unambiguous, so accepting the JSON body keeps both
        clients interoperable without treating arbitrary assistant JSON as a
        tool call.
        """
        try:
            raw_call = json.loads(payload.strip())
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(raw_call, dict):
            return None

        name = raw_call.get("name")
        arguments = raw_call.get("arguments", {})
        if not isinstance(name, str) or not name:
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None

        return ToolCall(
            type="function",
            function=FunctionCall(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def _get_json_function_calls(self, model_output: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for match in self.tool_call_complete_regex.finditer(model_output):
            parsed = self._parse_json_function_call(match.group(1))
            if parsed is not None:
                calls.append(parsed)
        return calls

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [
            match[0] if match[0] else match[1] for match in raw_function_calls
        ]
        return function_calls

    def _tool_start_positions(self, text: str) -> list[int]:
        positions: list[int] = []
        idx = 0
        while True:
            idx = text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            positions.append(idx)
            idx += len(self.tool_call_start_token)

        # Qwen3-Coder can emit a bare <function=...> block without the
        # surrounding <tool_call> tag. Non-streaming already falls back to this
        # format; keep streaming behavior consistent.
        if not positions:
            func_idx = text.find(self.tool_call_prefix)
            if func_idx != -1:
                positions.append(func_idx)
        return positions

    def _trailing_tool_marker_prefix(self, text: str) -> str:
        """Return the unfinished suffix of a possible tool-call marker."""
        markers = (self.tool_call_start_token, self.tool_call_prefix)
        for marker in markers:
            max_len = min(len(marker) - 1, len(text))
            for prefix_len in range(max_len, 0, -1):
                if text.endswith(marker[:prefix_len]):
                    return marker[:prefix_len]
        return ""

    def _pending_tool_marker_delta_prefix(
        self, previous_text: str, current_text: str, delta_text: str
    ) -> str | None:
        """Hold only an unfinished marker prefix without dropping plain text.

        A streamed ``<`` may be the first token of ``<tool_call>`` or a
        normal HTML tag. The old code held the prefix but emitted only the
        next delta when it stopped matching a marker, permanently dropping
        the ``<``. Reconstruct the previously held suffix before emitting the
        newly confirmed non-tool text.
        """
        previous_prefix = self._trailing_tool_marker_prefix(previous_text)
        current_prefix = self._trailing_tool_marker_prefix(current_text)
        if not previous_prefix and not current_prefix:
            return None

        pending_and_delta = previous_prefix + delta_text
        if current_prefix:
            return pending_and_delta[: -len(current_prefix)]
        return pending_and_delta

    def _finish_streaming_function(self, tool_text: str) -> None:
        """Finalize one streamed XML function and its JSON argument state."""
        self.json_closed = True

        func_start = tool_text.find(self.tool_call_prefix) + len(self.tool_call_prefix)
        func_content_end = tool_text.find(self.function_end_token, func_start)
        if func_content_end != -1:
            func_content = tool_text[func_start:func_content_end]
            try:
                parsed_tool = self._parse_xml_function_call(func_content)
                if parsed_tool and self.current_tool_index < len(
                    self.prev_tool_call_arr
                ):
                    self.prev_tool_call_arr[self.current_tool_index]["arguments"] = (
                        parsed_tool.function.arguments
                    )
            except Exception:
                logger.debug(
                    "Failed to parse tool call during streaming: %s",
                    tool_text,
                    exc_info=True,
                )

        if self.current_tool_index < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[self.current_tool_index] += "}"
        else:
            logger.warning(
                "streamed_args_for_tool out of sync: index=%d len=%d",
                self.current_tool_index,
                len(self.streamed_args_for_tool),
            )

        self.in_function = False
        self.accumulated_params = {}

    def _extract_json_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
    ) -> DeltaMessage | None:
        """Emit complete alternate-format JSON calls as atomic deltas.

        Buffering until ``</tool_call>`` deliberately favors correctness over
        partial display: downstream agents must never observe malformed JSON
        arguments and then persist that broken call into the next turn.
        """
        matches = list(self.tool_call_complete_regex.finditer(current_text))
        if self.json_tool_calls_streamed >= len(matches):
            return None

        deltas: list[DeltaToolCall] = []
        first_new_match = None
        for match_index in range(self.json_tool_calls_streamed, len(matches)):
            match = matches[match_index]
            parsed = self._parse_json_function_call(match.group(1))
            if parsed is None:
                continue
            if first_new_match is None:
                first_new_match = match

            call_index = len(self.prev_tool_call_arr)
            call_id = self._generate_tool_call_id()
            arguments = parsed.function.arguments
            self.prev_tool_call_arr.append(
                {
                    "name": parsed.function.name,
                    "arguments": arguments,
                }
            )
            self.streamed_args_for_tool.append(arguments)
            deltas.append(
                DeltaToolCall(
                    index=call_index,
                    id=call_id,
                    function=DeltaFunctionCall(
                        name=parsed.function.name,
                        arguments=arguments,
                    ),
                    type="function",
                )
            )

        # Complete wrappers, including malformed ones, are consumed once.
        self.json_tool_calls_streamed = len(matches)
        if not deltas:
            return None

        self.is_tool_call_started = False
        self.header_sent = False
        self.in_function = False
        self.json_started = True
        self.json_closed = True

        content = None
        if first_new_match is not None and first_new_match.start() > len(previous_text):
            content = current_text[len(previous_text) : first_new_match.start()]
            if not content:
                content = None
        return DeltaMessage(content=content, tool_calls=deltas)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Quick check to avoid unnecessary processing
        if self.tool_call_prefix not in model_output:
            json_tool_calls = self._get_json_function_calls(model_output)
            if json_tool_calls:
                self.prev_tool_call_arr = [
                    {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                    for call in json_tool_calls
                ]
                content_index = model_output.find(self.tool_call_start_token)
                content = model_output[:content_index]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=json_tool_calls,
                    content=content if content else None,
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = [
                self._parse_xml_function_call(function_call_str)
                for function_call_str in function_calls
            ]
            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    )

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            content = model_output[:content_index]  # .rstrip()
            valid_tool_calls = [tc for tc in tool_calls if tc is not None]
            return ExtractedToolCallInformation(
                tools_called=(len(valid_tool_calls) > 0),
                tool_calls=valid_tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
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
        # Store request for type conversion
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, return None unless it's an EOS token after tools
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            # Check for tool calls in text even if is_tool_call_started
            # is False (might have been reset after processing all tools)
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                # If we have completed tool calls and populated
                # prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(
                        self.tool_call_start_token
                    ) - current_text.count(self.tool_call_end_token)
                    if open_calls == 0:
                        # Return empty delta for finish_reason processing
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        # Update accumulated text
        self.accumulated_text = current_text

        # Qwen/OpenCode may emit a JSON body inside the same unambiguous
        # <tool_call> wrapper used by the XML format. Handle a complete JSON
        # call before the XML state machine consumes the wrapper and waits for
        # a <function= header that will never arrive.
        json_delta = self._extract_json_tool_calls_streaming(
            previous_text,
            current_text,
        )
        if json_delta is not None:
            return json_delta

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # Check if this tool call has ended
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                # This tool has ended, advance to next
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}

                # Check if there are more tool calls
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    # No more tool calls
                    self.is_tool_call_started = False
                # Continue processing next tool
                return None

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            # Check if tool call is starting
            tool_start_candidates = [
                pos
                for pos in (
                    current_text.find(self.tool_call_start_token),
                    current_text.find(self.tool_call_prefix),
                )
                if pos != -1
            ]
            tool_start = min(tool_start_candidates) if tool_start_candidates else -1
            if self.tool_call_start_token_id in delta_token_ids or tool_start != -1:
                self.is_tool_call_started = True
                # Return any content before the tool call
                if tool_start > len(previous_text):
                    content_before = (
                        self._trailing_tool_marker_prefix(previous_text)
                        + delta_text[: tool_start - len(previous_text)]
                    )
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                pending_prefix = self._pending_tool_marker_delta_prefix(
                    previous_text, current_text, delta_text
                )
                if pending_prefix is not None:
                    if pending_prefix:
                        return DeltaMessage(content=pending_prefix)
                    return None
                # Check if we're between tool calls - skip whitespace
                if (
                    current_text.rstrip().endswith(self.tool_call_end_token)
                    and delta_text.strip() == ""
                ):
                    # We just ended a tool call, skip whitespace
                    return None
                # Normal content, no tool call
                return DeltaMessage(content=delta_text)

        # Check if we're between tool calls (waiting for next one)
        # Count tool calls we've seen vs processed
        tool_start_positions = self._tool_start_positions(current_text)
        tool_starts_count = len(tool_start_positions)
        if self.current_tool_index >= tool_starts_count:
            # We're past all tool calls, shouldn't be here
            return None

        # We're in a tool call, find the current tool call portion
        # Need to find the correct tool call based on current_tool_index
        tool_start_idx = tool_start_positions[self.current_tool_index]
        # Find where this tool call ends (or current position if not ended yet)
        tool_end_idx = current_text.find(self.tool_call_end_token, tool_start_idx)
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx : tool_end_idx + len(self.tool_call_end_token)
            ]

        # Looking for function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    # Found complete function name
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    # A speculative burst can finish the entire function
                    # before the parser has emitted its header. Do not rely on
                    # a later delta to revisit the already-buffered body: the
                    # next delta may only contain </tool_call> plus EOS. Parse
                    # the complete body now and emit its arguments together
                    # with the header.
                    complete_arguments: str | None = None
                    func_content_end = tool_text.find(
                        self.function_end_token, func_start
                    )
                    if func_content_end != -1:
                        parsed_tool = self._parse_xml_function_call(
                            tool_text[func_start:func_content_end]
                        )
                        if parsed_tool is not None:
                            complete_arguments = parsed_tool.function.arguments

                    # Always append — each tool call is a separate
                    # invocation even if the function name is the same
                    # (e.g. two consecutive "read" calls).
                    self.prev_tool_call_arr.append(
                        {
                            "name": self.current_function_name,
                            "arguments": complete_arguments or "{}",
                        }
                    )

                    # Initialize streamed args tracking for this tool.
                    # The serving layer reads streamed_args_for_tool to
                    # compute remaining arguments at stream end. Without
                    # this, IndexError occurs when the serving layer
                    # accesses streamed_args_for_tool[index].
                    self.streamed_args_for_tool.append(complete_arguments or "")

                    if complete_arguments is not None:
                        self.json_started = True
                        self.json_closed = True
                        self.in_function = False

                    # Send header with function info
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                id=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    name=self.current_function_name,
                                    arguments=complete_arguments or "",
                                ),
                                type="function",
                            )
                        ]
                    )
            return None

        # We've sent header, now handle function body
        if self.in_function:
            # Always send opening brace first, regardless of whether
            # parameter_prefix is in the current delta. With speculative
            # decoding, a single delta may contain both the opening brace
            # and parameter data; skipping "{" here would desync
            # json_started from what was actually streamed.
            if not self.json_started:
                self.json_started = True
                self.streamed_args_for_tool[self.current_tool_index] += "{"
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="{"),
                        )
                    ]
                )

            param_config = find_tool_properties(
                self.tools, self.current_function_name or ""
            )
            param_starts = self._parameter_start_positions(tool_text, param_config)

            # Process ALL complete params in a loop (spec decode fix).
            # With speculative decoding a single delta can deliver
            # multiple complete parameters at once. The old single-pass
            # code would process one and ``return None`` if the next was
            # incomplete — skipping any already-complete params that
            # preceded it. Using a loop with ``break`` instead ensures
            # we emit every complete parameter before yielding control.
            json_fragments = []
            while not self.in_param and self.param_count < len(param_starts):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" not in remaining:
                    break

                name_end = remaining.find(">")
                current_param_name = remaining[:name_end]

                value_start = param_start + name_end + 1
                next_param_idx = (
                    param_starts[self.param_count + 1]
                    if self.param_count + 1 < len(param_starts)
                    else None
                )
                function_end_idx = tool_text.find(self.function_end_token, value_start)
                tool_end_idx = tool_text.find(self.tool_call_end_token, value_start)
                boundaries = [
                    boundary
                    for boundary in (function_end_idx, tool_end_idx)
                    if boundary != -1
                ]
                container_end = min(boundaries) if boundaries else None
                value_end = self._find_structural_parameter_end(
                    tool_text,
                    value_start=value_start,
                    next_parameter_start=next_param_idx,
                    container_end=container_end,
                    allow_text_end=False,
                )
                if value_end is None:
                    if next_param_idx is not None:
                        # Recover a missing </parameter> before a real next tag.
                        value_end = next_param_idx
                    elif container_end is not None:
                        # Recover a missing final close before function/tool end.
                        value_end = container_end
                    else:
                        # Wait for structural lookahead. The current suffix may
                        # be literal </parameter> text inside a long value.
                        break

                param_value = tool_text[value_start:value_end]
                if param_value.startswith("\n"):
                    param_value = param_value[1:]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                self.current_param_name = current_param_name
                self.accumulated_params[current_param_name] = param_value

                converted_value = self._convert_param_value(
                    param_value,
                    current_param_name,
                    param_config,
                    self.current_function_name or "",
                )

                serialized_value = json.dumps(converted_value, ensure_ascii=False)

                if self.param_count == 0:
                    json_fragment = f'"{current_param_name}": {serialized_value}'
                else:
                    json_fragment = f', "{current_param_name}": {serialized_value}'

                self.param_count += 1
                json_fragments.append(json_fragment)

            if json_fragments:
                combined = "".join(json_fragments)

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += combined
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )

                # A speculative burst (or stream_interval > 1) can complete
                # the final parameter and close the function in one parser
                # call. Returning only the parameter fragment loses the final
                # JSON brace because the next delta may be EOS/empty.
                if not self.json_closed and self.function_end_token in tool_text:
                    self._finish_streaming_function(tool_text)
                    combined += "}"

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments=combined),
                        )
                    ]
                )

            # Check for function end AFTER processing parameters.
            # This ordering is critical: with speculative decoding a
            # burst can deliver the final parameter value together with
            # </function>. If the close check ran first it would emit
            # "}" and set in_function=False before the parameter loop
            # ever ran, causing the parameter to be silently dropped.
            if not self.json_closed and self.function_end_token in tool_text:
                self._finish_streaming_function(tool_text)

                result = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="}"),
                        )
                    ]
                )

                return result

        return None

    def get_structural_tag(self, request: ChatCompletionRequest):
        return get_model_structural_tag(
            model="qwen_3_5",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )
