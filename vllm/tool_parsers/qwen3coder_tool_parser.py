# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
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
from vllm.envs import (
    VLLM_ENFORCE_STRICT_TOOL_CALLING,
    VLLM_QWEN3X_TOOL_FIX,
)
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
    find_common_prefix,
    find_tool_properties,
)

logger = init_logger(__name__)


# opt23 debug log (disabled by default; enable via VLLM_OPT23_DEBUG_LOG=1)
_OPT23_LOG = None

def _log(msg: str, *args: Any) -> None:
    global _OPT23_LOG
    import os as _os
    if not _os.environ.get("VLLM_OPT23_DEBUG_LOG"):
        return
    if _OPT23_LOG is None:
        _OPT23_LOG = open("/tmp/log/vllm-tool-log.txt", "a", buffering=1)
    _OPT23_LOG.write(msg % args if args else msg)
    _OPT23_LOG.write("\n")


class Qwen3CoderToolParser(ToolParser):
    supports_required_and_named: bool = not VLLM_ENFORCE_STRICT_TOOL_CALLING

    # opt23 Path B fake streaming (=5): partial streaming only for
    # Write/Edit (tools with long string content params that benefit
    # from incremental display).  Path B Native (=1,2) handles all
    # XML tools via diff-based XML streaming — no whitelist needed.
    _STREAMING_TOOLS: frozenset = frozenset({"Write", "Edit"})

    @property
    def _fix_force_json(self) -> bool:
        """opt23 §11.22: fix=2 强制 JSON 真流式增量路由（XML 输出也最终按 JSON 输出）。"""
        return VLLM_QWEN3X_TOOL_FIX == 2

    @property
    def _fix_force_xml(self) -> bool:
        """opt23 §11.22: fix=3/4 强制 XML 路由（JSON 输出也走 XML 解析）。"""
        return VLLM_QWEN3X_TOOL_FIX in (3, 4)

    @property
    def _is_streaming_tool(self) -> bool:
        """Streaming optimizations apply to =1,2,5 for Write/Edit tools.

        Guards _pending_brace, partial streaming, and remaining delta
        paths for tools with long string content params that benefit
        from incremental display.
        """
        return (VLLM_QWEN3X_TOOL_FIX in (1, 2, 4, 5)
                and not self._json_tool_active
                and self.current_function_name in self._STREAMING_TOOLS)

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
        # opt23: partial streaming for long string params (write/edit content)
        self._partial_emit_offset: int = 0
        self._partial_param_start: int = -1
        self._partial_emitted: str = ""
        self._partial_prefix: str = ""
        self._last_partial_time: float = 0.0
        # opt23: defer standalone "{" to combine with first real delta
        self._pending_brace: bool = False
        # opt23: flag for model duplication of <function=...> in streaming
        self._func_dup_detected: bool = False
        # opt23 Path C: JSON-format tool call (not XML) detected in streaming
        self._json_tool_active: bool = False
        # opt23 =3 JSON path: end position of processed tool call for skip
        self._json_processed_end: int = 0
        # opt23 §11.42 (v122 框架迁入): resolved tool-call format
        # ('xml'|'json', or None when unconfigured → auto-detect);
        # config-first via request.chat_template_kwargs.tool_call_format.
        self._tool_format: str | None = None
        # opt23 =2 XML path: incomplete param saved from parsing loop
        self._stream_partial_name: str | None = None
        self._stream_partial_value: str | None = None
        # opt23 =2 优化3: 状态指纹，无变化时跳过 diff
        self._stream_last_fp: tuple | None = None

    def _compute_json_diff(self, current_json: str, prev_streamed: str) -> str:
        """Safe-prefix diff on JSON strings for incremental tool args.

        Mirrors upstream ``_compute_arg_delta`` from vLLM PR #45413.
        Returns only the suffix of *current_json* that is not already
        present in *prev_streamed*, i.e. a valid JSON continuation.

        When *prev_streamed* is empty the entire *current_json* is
        returned (first emission).
        """
        if not prev_streamed:
            return current_json
        if not current_json:
            return ""
        common = find_common_prefix(prev_streamed, current_json)
        return current_json[len(common):]

    @staticmethod
    def _safe_content_length(text: str) -> int:
        """Return length of *text* prefix safe to emit as partial content.

        Detects trailing XML close-tag fragments (``</parameter>``,
        ``</function>``, ``</tool_call>``) that the model may generate
        during streaming and truncates before them.  Characters that
        look like a legitimate XML close tag are **not** emitted as
        string content — when the tag completes the XML parser will
        detect it and the streaming-param handler will emit the full
        corrected value.
        """
        _pos = text.rfind('</')
        if _pos < 0:
            return len(text)

        _after = text[_pos + 2:]       # text after '</'
        _after_clean = _after.rstrip('>')

        # Valid XML close-tag names in qwen3coder format.
        for _tag in ('parameter', 'function', 'tool_call'):
            if _tag.startswith(_after_clean) or _after_clean.startswith(_tag):
                _result = _pos
                if _result > 0 and text[_result - 1] == '\n':
                    _result -= 1
                return _result

        # '</' not followed by a known close-tag name (e.g. </div>)
        # — treat as legitimate string content.
        return len(text)

    def _is_string_param(self, param_name: str) -> bool:
        """True when *param_name* is typed as ``string`` in the tool schema.

        Used by the boundary detection in the param-completion loop to
        decide whether ``<parameter=next>`` can serve as a fallback
        delimiter — string params must wait for an explicit
        ``</parameter>`` to avoid premature completion with a partial
        value.
        """
        func = self.current_function_name
        if not func:
            return False
        _pc = find_tool_properties(self.tools, func)
        _ps = _pc.get(param_name, {})
        _pt = extract_types_from_schema(_ps)
        return bool(_pt and "string" in _pt)

    @staticmethod
    def _is_inside_json_string(json_text: str) -> bool:
        """Return True if the final character position in *json_text*
        is inside a JSON string value (between an unescaped ``"`` and
        its matching closing ``"``).

        Walks *json_text* character by character, tracking ``in_string``
        and ``escape_next`` state.  Returns the string state at the
        end of the text.
        """
        in_string = False
        escape_next = False
        for c in json_text:
            if escape_next:
                escape_next = False
                continue
            if c == '\\':
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
        return in_string

    @staticmethod
    def _unescape_display_chars(s: str) -> str:
        """Replace JSON escape sequences ``\\n``, ``\\t``, ``\\r`` with
        literal newline / tab / carriage-return characters for frontend
        display.

        Preserves structural escapes ``\\\"`` and ``\\\\`` so the
        accumulated JSON remains structurally parseable.
        """
        result: list[str] = []
        i = 0
        while i < len(s):
            c = s[i]
            if c == '\\' and i + 1 < len(s):
                nxt = s[i + 1]
                if nxt == '\\' or nxt == '"':
                    # Structural escapes — keep as-is
                    result.append(c)
                    result.append(nxt)
                    i += 2
                    continue
                elif nxt == 'n':
                    result.append('\n')
                    i += 2
                    continue
                elif nxt == 't':
                    result.append('\t')
                    i += 2
                    continue
                elif nxt == 'r':
                    result.append('\r')
                    i += 2
                    continue
            result.append(c)
            i += 1
        return ''.join(result)

    @staticmethod
    def _escape_raw_control_chars(s: str) -> str:
        """模型可能把真实换行/控制字符（XML 换行习惯）直接输出进 JSON
        字符串值，导致 JSON 非法（Write content 参数丢失根因）。仅在
        字符串值内部把控制字符转义为 JSON 合法序列（\\n → \\\\n）。"""
        out = []
        i = 0
        n = len(s)
        in_str = False
        _esc_map = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
        while i < n:
            c = s[i]
            if c == "\\":
                out.append(c)
                if i + 1 < n:
                    out.append(s[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            if c == '"':
                in_str = not in_str
                out.append(c)
                i += 1
                continue
            if in_str and ord(c) < 0x20:
                out.append(_esc_map.get(c, "\\u%04x" % ord(c)))
                i += 1
                continue
            out.append(c)
            i += 1
        return "".join(out)

    def _handle_json_tool_streaming(
        self, current_text: str, delta_text: str
    ) -> DeltaMessage | None:
        """Handle JSON-format tool calls with safe-prefix JSON diffing.

        Supported format (single tool)::

            {"name": "Write", "arguments": {"file_path": "/x", "content": "..."}}

        Each emitted delta is a valid JSON continuation of the arguments
        object, suitable for Anthropic ``input_json_delta`` accumulation.
        """
        # --- name extraction (first time only) ---
        if not self.header_sent:
            _name_kw = current_text.find('"name"')
            if _name_kw == -1:
                return None
            _colon = current_text.find(":", _name_kw)
            if _colon == -1:
                return None
            _q1 = current_text.find('"', _colon + 1)
            if _q1 == -1:
                return None
            _q2 = current_text.find('"', _q1 + 1)
            if _q2 == -1:
                return None
            name = current_text[_q1 + 1:_q2]
            if not name:
                return None

            self.current_function_name = name
            self.current_tool_id = self._generate_tool_call_id()
            self.header_sent = True
            self.in_function = True
            self.json_started = True
            self.accumulated_params = {}

            self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})
            self.streamed_args_for_tool.append("")

            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        id=self.current_tool_id,
                        function=DeltaFunctionCall(name=name, arguments=""),
                        type="function",
                    )
                ]
            )

        # --- arguments JSON extraction ---
        _args_kw = current_text.find('"arguments"')
        if _args_kw == -1:
            return None

        _args_colon = current_text.find(":", _args_kw)
        if _args_colon == -1:
            return None

        # skip whitespace after colon
        _val_start = _args_colon + 1
        while _val_start < len(current_text) and current_text[_val_start] in " \t\n\r":
            _val_start += 1
        if _val_start >= len(current_text):
            return None
        if current_text[_val_start] != "{":
            return None

        # Walk brace depth to find the end of the arguments JSON object.
        # For incomplete streaming text the closing "}" may be absent;
        # in that case we take everything from _val_start to end-of-text.
        _depth = 0
        _in_str = False
        _esc = False
        _json_end = -1
        for i in range(_val_start, len(current_text)):
            c = current_text[i]
            if _esc:
                _esc = False
                continue
            if c == "\\":
                _esc = True
                continue
            if c == '"' and not _esc:
                _in_str = not _in_str
                continue
            if _in_str:
                continue
            if c == "{":
                _depth += 1
            elif c == "}":
                _depth -= 1
                if _depth == 0:
                    _json_end = i + 1
                    break

        if _json_end > 0:
            current_args = current_text[_val_start:_json_end]
        else:
            current_args = current_text[_val_start:]

        # opt23: 模型把真实换行/控制字符（XML 换行习惯）直接输出进 JSON
        # 字符串值时 → 字符串值内转义为 JSON 合法序列，否则参数提取
        # 非法/截断（Write content 丢失、工具调用参数不完整根因）。
        current_args = self._escape_raw_control_chars(current_args)

        # --- diff against previously streamed ---
        idx = self.current_tool_index
        prev = self.streamed_args_for_tool[idx] if idx < len(self.streamed_args_for_tool) else ""
        # opt23 §11.29: 对齐主线 hermes `_compute_args_diff`（前缀截取增量）。
        # 主线用 `args[len(prev):]`（信任流式累积，current 是 prev 的超集），
        # 替代本地 `_compute_json_diff`（find_common_prefix 手工 diff）。
        if len(current_args) <= len(prev):
            return None
        delta = current_args[len(prev):]
        if not delta:
            return None

        # Store raw (escaped) delta for future diffs and serving-layer
        # remaining-delta computation.  We emit an unescaped version
        # for display when inside a JSON string value.
        if idx < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[idx] += delta

        # Update prev_tool_call_arr when JSON is complete so the serving
        # layer can compute the correct remaining delta at stream end.
        # 优化 A: 标记参数完整。确认外层工具调用也闭合了（}} 连续）
        # 或文本正好在参数闭括号处结束（EOS 场景）。
        if (_json_end > 0 and idx < len(self.prev_tool_call_arr)
                and (_json_end == len(current_text)
                     or current_text[_json_end] == '}')):
            self.prev_tool_call_arr[idx]["arguments"] = current_args
            self.json_closed = True
            self.in_function = False
            self._json_processed_end = _json_end
            _log(
                "opt23 JSON handler 优化A: idx=%s current_args=%s "
                "prev_tool_call_arr=%s",
                idx, current_args[:200],
                self.prev_tool_call_arr)

        # opt23: 之前为前端显示打字机效果把转义序列（\n \t \r）反转义为
        # 真实控制字符——但前端累积拼接后 JSON 非法（真实换行在字符串值
        # 内），工具执行 InputValidationError（Write content 参数失败
        # 根因）。直接发射转义版增量（字面 \n），前端拼接即可 json.parse。
        display_delta = delta

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=display_delta),
                )
            ]
        )

    def _handle_xml_tool_streaming(
        self, current_text: str, delta_text: str
    ) -> DeltaMessage | None:
        """Path B Native: XML-native true streaming with safe-prefix diffing.

        Isomorphic to ``_handle_json_tool_streaming``.  Builds an XML
        intermediate state from the current ``tool_text`` and emits the
        diff against previously streamed XML.  Pure XML output — no JSON
        involvement.

        Algorithm:
        1. HEADER EXTRACTION — find ``<function=NAME>``, send name delta
        2. BUILD XML INTERMEDIATE STATE — parse completed + in-progress params
        3. DIFF — ``find_common_prefix(prev_xml, current_xml)``
        4. EMIT — return XML continuation delta

        Activated when ``VLLM_QWEN3X_TOOL_FIX`` is 1 or 2 and
        the model is generating XML-format tool calls.
        """

        # Find tool positions
        tool_start_positions = self._tool_start_positions(current_text)
        if self.current_tool_index >= len(tool_start_positions):
            return None

        tool_start_idx = tool_start_positions[self.current_tool_index]
        tool_end_idx = current_text.find(
            self.tool_call_end_token, tool_start_idx
        )
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx:tool_end_idx + len(self.tool_call_end_token)
            ]

        # ---- Step 1: Header Extraction ----
        if not self.header_sent:
            if self.tool_call_prefix not in tool_text:
                return None
            func_start = tool_text.find(self.tool_call_prefix) + len(
                self.tool_call_prefix
            )
            func_end = tool_text.find(">", func_start)
            if func_end == -1:
                return None

            name = tool_text[func_start:func_end]
            if not name:
                return None

            self.current_function_name = name
            self.current_tool_id = self._generate_tool_call_id()
            self.header_sent = True
            self.in_function = True
            self.json_started = True
            self.accumulated_params = {}
            self._pending_brace = False

            self.prev_tool_call_arr.append(
                {"name": name, "arguments": "{}"}
            )
            self.streamed_args_for_tool.append("")

            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        id=self.current_tool_id,
                        function=DeltaFunctionCall(
                            name=name, arguments=""
                        ),
                        type="function",
                    )
                ]
            )

        # ---- Step 2: Build XML Intermediate State ----
        # Find all <parameter=...> positions in tool_text
        param_starts: list[int] = []
        search_idx = 0
        while True:
            search_idx = tool_text.find(
                self.parameter_prefix, search_idx
            )
            if search_idx == -1:
                break
            param_starts.append(search_idx)
            search_idx += len(self.parameter_prefix)

        # Filter out stale params before the LAST <function=...> tag
        # (model duplication artifact in streaming).
        _func_positions: list[int] = []
        _fsi = 0
        while True:
            _fsi = tool_text.find(self.tool_call_prefix, _fsi)
            if _fsi == -1:
                break
            _close = tool_text.find(
                ">", _fsi + len(self.tool_call_prefix)
            )
            if _close != -1:
                _func_positions.append(_fsi)
            _fsi += len(self.tool_call_prefix)

        if param_starts and _func_positions:
            _func_positions = [
                p for p in _func_positions if p < param_starts[0]
            ]
        if len(_func_positions) > 1:
            _last_func = _func_positions[-1]
            param_starts = [
                p for p in param_starts if p > _last_func
            ]

        # Build XML from parsed parameters
        xml_parts: list[str] = []

        for param_start_pos in param_starts:
            param_start = param_start_pos + len(self.parameter_prefix)
            remaining = tool_text[param_start:]

            if ">" not in remaining:
                break

            name_end = remaining.find(">")
            param_name = remaining[:name_end]

            value_start = param_start + name_end + 1
            value_text = tool_text[value_start:]
            if value_text.startswith("\n"):
                value_text = value_text[1:]

            # Find end of this param
            # 优化1: detect false-positive </parameter> inside content
            # (e.g. Write content that contains "</parameter>" literally).
            # Scan forward until a true close-tag is confirmed by the
            # text that follows it (another <parameter=, </function>,
            # </tool_call>, or end-of-text).
            param_end_idx = -1
            _pe_search = 0
            while _pe_search < len(value_text):
                _pe = value_text.find(self.parameter_end_token, _pe_search)
                if _pe == -1:
                    break
                _after = value_text[_pe + len(self.parameter_end_token):].lstrip()
                if (_after.startswith(self.parameter_prefix)
                        or _after.startswith(self.function_end_token)
                        or _after.startswith(self.tool_call_end_token)
                        or not _after):
                    param_end_idx = _pe
                    break
                # Content contained a </parameter> fragment — keep scanning
                _pe_search = _pe + 1

            if param_end_idx != -1:
                # Complete param — include closing </parameter> tag
                param_value = value_text[:param_end_idx]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]
                xml_parts.append(
                    f"<parameter={param_name}>\n{param_value}"
                    f"\n</parameter>"
                )
                if param_name not in self.accumulated_params:
                    # opt23 =2: 对完整参数做类型转换 (bool/int/string)，
                    # 确保 json.dumps 输出正确的 JSON 类型
                    param_config = find_tool_properties(
                        self.tools, self.current_function_name or "")
                    converted = self._convert_param_value(
                        param_value, param_name,
                        param_config, self.current_function_name or "")
                    self.accumulated_params[param_name] = converted
            else:
                # Incomplete param — trim trailing XML tags from value
                next_param = value_text.find(self.parameter_prefix)
                func_end_v = value_text.find(self.function_end_token)
                tool_end_v = value_text.find(self.tool_call_end_token)

                end_candidates = []
                if next_param != -1:
                    end_candidates.append(next_param)
                if func_end_v != -1:
                    end_candidates.append(func_end_v)
                if tool_end_v != -1:
                    end_candidates.append(tool_end_v)

                if end_candidates:
                    param_value = value_text[:min(end_candidates)]
                else:
                    param_value = value_text

                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                # opt23: guard against partial XML close-tag fragments
                # (e.g. "</param", "</funct") leaking into the XML
                # intermediate state.
                _safe_len = self._safe_content_length(param_value)
                if _safe_len < len(param_value):
                    param_value = param_value[:_safe_len]
                    if param_value.endswith("\n"):
                        param_value = param_value[:-1]

                xml_parts.append(
                    f"<parameter={param_name}>\n{param_value}"
                )
                # Save incomplete param for Step 3 JSON building
                self._stream_partial_name = param_name
                self._stream_partial_value = param_value
                break  # stop at first incomplete param

        else:
            # 优化2: for-else — all params complete this cycle, clear
            # stale partial state so Step 3 doesn't attach a redundant
            # in-progress fragment that was already fully parsed.
            self._stream_partial_name = None
            self._stream_partial_value = None

        current_xml = "\n".join(xml_parts)

        # ---- Step 3: Build JSON from accumulated params ----
        # =2 and =1 XML path: XML input → JSON output (standard tool args).
        # Build partial JSON from accumulated_params + current in-progress
        # param, diff against previously streamed.
        idx = self.current_tool_index
        prev_raw = (
            self.streamed_args_for_tool[idx]
            if idx < len(self.streamed_args_for_tool)
            else ""
        )

        # 优化3: state fingerprint — skip build+diff when nothing changed
        _state_fp = (len(self.accumulated_params),
                     len(self._stream_partial_value or ""))
        if _state_fp == self._stream_last_fp:
            delta = ""
        else:
            self._stream_last_fp = _state_fp

            # Build partial JSON (no closing } — added at function-end)
            parts: list[str] = []
            first = True
            for name, value in self.accumulated_params.items():
                serialized = json.dumps(value, ensure_ascii=False)
                if first:
                    parts.append(f'"{name}": {serialized}')
                    first = False
                else:
                    parts.append(f', "{name}": {serialized}')

            # Attach the current in-progress param (trimmed by parsing loop)
            partial_name = self._stream_partial_name
            partial_value = self._stream_partial_value

            # Guard: skip stale partial when the param was fully completed
            # in this cycle (now in accumulated_params) to avoid duplicate keys.
            if partial_name is not None and partial_value is not None:
                if partial_name not in self.accumulated_params:
                    # Only stream partial values for string-type params.
                    # Bool/int params change JSON representation after type
                    # conversion (e.g. "false" → false), which breaks the
                    # safe-prefix diff and produces garbage deltas.
                    _is_string = True
                    if partial_name:
                        _pc = find_tool_properties(
                            self.tools, self.current_function_name or "")
                        _ps = _pc.get(partial_name, {})
                        _pt = extract_types_from_schema(_ps)
                        if _pt and "string" not in _pt:
                            _is_string = False
                    if _is_string:
                        escaped = json.dumps(partial_value, ensure_ascii=False)[1:-1]
                        if first:
                            parts.append(f'"{partial_name}": "{escaped}')
                        else:
                            parts.append(f', "{partial_name}": "{escaped}')

            current_partial = "{" + "".join(parts)
            delta = self._compute_json_diff(current_partial, prev_raw)

        # ---- Step 4: Detect function end ----
        func_ended = (
            self.function_end_token in tool_text
            and not self.json_closed
        )

        if func_ended:
            # Build complete JSON from accumulated_params
            full_json = json.dumps(
                self.accumulated_params, ensure_ascii=False)

            if idx < len(self.prev_tool_call_arr):
                self.prev_tool_call_arr[idx]["arguments"] = full_json

            _log(
                "opt23 Step4 func_ended: tool=%s idx=%s "
                "accumulated=%s full_json=%s prev_raw=%s delta_before=%s",
                self.current_function_name, idx,
                self.accumulated_params, full_json,
                prev_raw, delta)

            # Compute closing delta (the } suffix)
            streamed_total = prev_raw + delta if delta else prev_raw
            if full_json.startswith(streamed_total):
                delta += full_json[len(streamed_total):]
            elif full_json.startswith(prev_raw):
                delta = full_json[len(prev_raw):]

            self.json_closed = True
            self.in_function = False
            self.accumulated_params = {}

        if not delta:
            return None


        # Update streamed args
        if idx < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[idx] += delta

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=delta),
                )
            ]
        )

    def _emit_xml_json_diff(
        self, tool_text: str, param_starts: list[int],
        tool_start_idx: int, current_text: str,
    ) -> DeltaMessage | None:
        """Phase 3: build partial args JSON from XML params, emit via
        safe-prefix JSON-diff.

        Constructs a PARTIAL JSON string (no closing ``}``, trailing
        string value has no closing ``"``) so that each diff is a valid
        JSON continuation that can be safely accumulated by the client.

        Activated when ``VLLM_QWEN3X_TOOL_FIX == 4`` (=4 互转, 待设计).
        """
        partial_name = None
        partial_value = None

        # Extract partial param value when a param is still forming
        if self.param_count < len(param_starts):
            incomplete_start = param_starts[self.param_count]
            param_text = tool_text[
                incomplete_start + len(self.parameter_prefix):
            ]
            if ">" not in param_text:
                return None
            name_end = param_text.find(">")
            partial_name = param_text[:name_end]

            # Type gate: only string params get partial-inclusion
            _pc = find_tool_properties(
                self.tools, self.current_function_name or "",
            )
            _ps = _pc.get(partial_name, {})
            _pt = extract_types_from_schema(_ps)
            if _pt and "string" not in _pt:
                return None

            value_start = (
                incomplete_start + len(self.parameter_prefix) + name_end + 1
            )
            _abs_value_start = tool_start_idx + value_start
            if (
                _abs_value_start < len(current_text)
                and current_text[_abs_value_start:_abs_value_start + 1] == "\n"
            ):
                _abs_value_start += 1
            partial_value = current_text[_abs_value_start:]

        # --- build partial JSON string ---
        # Uses the same format as json_fragments: starts with "{",
        # completed params have fully-formed key-value pairs, and the
        # trailing string param (if any) is left open (no closing ").
        parts = []
        first = True
        for name, value in self.accumulated_params.items():
            serialized = json.dumps(value, ensure_ascii=False)
            if first:
                parts.append(f'"{name}": {serialized}')
                first = False
            else:
                parts.append(f', "{name}": {serialized}')

        if partial_name and partial_value is not None:
            escaped = json.dumps(partial_value, ensure_ascii=False)[1:-1]
            if first:
                parts.append(f'"{partial_name}": "{escaped}')
            else:
                parts.append(f', "{partial_name}": "{escaped}')

        current_partial = "{" + "".join(parts)

        # --- diff ---
        idx = self.current_tool_index
        prev = (
            self.streamed_args_for_tool[idx]
            if idx < len(self.streamed_args_for_tool) else ""
        )
        delta = self._compute_json_diff(current_partial, prev)
        if not delta:
            return None

        if idx < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[idx] += delta

        # _pending_brace is never used here — the "{" is always part
        # of the partial JSON string built above.  Mark it consumed
        # so the func-end handler won't re-emit it.
        self._pending_brace = False

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=delta),
                )
            ]
        )

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        if not isinstance(param_value, str):
            return param_value
        param_schema = param_config.get(param_name, {})
        param_types = extract_types_from_schema(param_schema)
        return coerce_to_schema_type(param_value, param_types)

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
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            try:
                idx = match_text.index(">")
            except ValueError:
                # Truncated parameter tag (e.g. <parameter=name without >).
                # Skip gracefully instead of crashing the entire extraction.
                continue
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
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

    def _extract_tool_calls_json(self, model_output: str) -> list[ToolCall]:
        """opt23 fix=2: 提取 JSON 格式工具调用（<tool_call>{...}</tool_call> 或裸 JSON）。"""
        raw_calls: list[dict] = []
        for _m in re.finditer(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            model_output,
            re.DOTALL,
        ):
            try:
                raw_calls.append(json.loads(_m.group(1)))
            except Exception:
                pass
        if not raw_calls:
            try:
                _start = model_output.find("{")
                if _start != -1:
                    _d = json.loads(model_output[_start:])
                    if isinstance(_d, dict) and (
                        "name" in _d or "arguments" in _d
                    ):
                        raw_calls.append(_d)
            except Exception:
                pass
        tcs: list[ToolCall] = []
        for _d in raw_calls:
            _name = _d.get("name")
            _args = _d.get("arguments")
            if _name is None:
                continue
            if not isinstance(_args, str):
                _args = json.dumps(_args, ensure_ascii=False)
            tcs.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(name=str(_name), arguments=_args),
                )
            )
        return tcs

    @staticmethod
    def _detect_tool_format(text: str) -> str:
        """Auto-detect tool-call format from generated text."""
        if text.find("<function=") != -1:
            return "xml"
        if '"name"' in text and '"arguments"' in text:
            return "json"
        return "xml"

    @staticmethod
    def _detect_tool_format_if_present(text: str) -> str | None:
        """Detect an explicit tool-call format; None when absent."""
        if text.find("<function=") != -1:
            return "xml"
        if '"name"' in text and '"arguments"' in text:
            return "json"
        return None

    def _resolve_tool_format(
        self,
        request: ChatCompletionRequest,
        text: str = "",
    ) -> str | None:
        """Resolve the client-selected tool-call format (config-first)."""
        kwargs = getattr(request, "chat_template_kwargs", None) or {}
        cfg = kwargs.get("tool_call_format")
        if cfg in ("xml", "json"):
            return cfg
        if not text:
            return None
        return self._detect_tool_format(text)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # opt23 §11.42 (v122 框架迁入): config-first 格式解析——serving 按
        # fix 注入 request.chat_template_kwargs.tool_call_format（fix=2→json、
        # fix=3→xml）优先；无配置则按输出形态检测（_detect_tool_format）。
        _fmt = self._resolve_tool_format(request, model_output)

        # Quick check to avoid unnecessary processing
        if _fmt == "json" or (_fmt is None and self.tool_call_prefix not in model_output):
            # opt23 §11.22: 所有 fix 均 fallback JSON 提取（模型输出
            # <tool_call>{"name"...}</tool_call> 或裸 JSON 时保证解析成功）
            json_tcs = self._extract_tool_calls_json(model_output)
            if json_tcs:
                self.prev_tool_call_arr = [
                    {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in json_tcs
                ]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=json_tcs,
                    content=None,
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

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # 优化 B: JSON 格式无 </tool_call> 标记，参数完整后直接推进
            if self._json_tool_active:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                self.is_tool_call_started = False
                # Clear active flag so Section 3 can re-detect or release
                # subsequent content, and Section 4 won't re-enter handler
                # for an already-processed tool call. fix=2 时保持 JSON 激活。
                self._json_tool_active = self._fix_force_json
                # opt23: reset partial streaming state
                self._partial_emit_offset = 0
                self._partial_param_start = -1
                self._partial_emitted = ""
                self._partial_prefix = ""
                self._pending_brace = False
                self._func_dup_detected = False
                self._stream_last_fp = None
                self._stream_partial_name = None
                self._stream_partial_value = None
                return None

            # Check if this tool call has ended (XML format)
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                # This tool has ended, advance to next
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                # opt23: reset partial streaming state for next tool call
                self._partial_emit_offset = 0
                self._partial_param_start = -1
                self._partial_emitted = ""
                self._partial_prefix = ""
                self._pending_brace = False
                self._func_dup_detected = False
                self._stream_last_fp = None
                self._stream_partial_name = None
                self._stream_partial_value = None

                # Check if there are more tool calls
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    # No more tool calls
                    self.is_tool_call_started = False
                # Continue processing next tool
                return None

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            # opt23 fix=2: 强制 JSON 路由——<tool_call> 标记（模板 json 分支
            # 引导模型输出 <tool_call> 包裹 JSON）或 JSON 特征（"name"/
            # "arguments"）出现即激活。不能只等 "name" 特征：模型输出
            # <tool_call>\n{"name":...} 时 <tool_call> 先到达，XML 检测会
            # 抢先激活（expat 解析 JSON 失败吞掉工具调用）。无条件强制
            # 则会吞纯文本正文（think 后无正文根因）。<function= 表示
            # 模型实际输出 XML，不在此激活（走 XML 解析路径）。
            if self._fix_force_json:
                _json_name = current_text.find('"name"')
                _json_args = current_text.find('"arguments"')
                if (current_text.find(self.tool_call_start_token) != -1
                        or (_json_name != -1 and _json_args != -1
                            and _json_name < _json_args)):
                    self._json_tool_active = True
                    self.is_tool_call_started = True
                    return None
            # opt23 Path C: detect JSON-format tool calls before falling
            # into XML detection.  JSON format looks like:
            #   {"name": "Write", "arguments": {...}}
            # Only activate when no XML markers are present (otherwise
            # a JSON-looking substring inside XML content could trigger).
            # 优化 C: 从已处理 JSON 工具调用的结束位置之后搜索，
            # 避免重新检测到同一个已完成的工具调用。
            _json_search = max(self._json_processed_end, 0)
            _json_name = current_text.find('"name"', _json_search)
            _json_args = current_text.find('"arguments"', _json_search)
            if (_json_name != -1
                    and _json_args != -1
                    and _json_name < _json_args
                    and current_text.find(self.tool_call_prefix) == -1):
                self._json_tool_active = True
                self.is_tool_call_started = True
                return None  # routing takes effect on next call

            # Check if tool call is starting (XML)
            tool_start_candidates = [
                pos
                for pos in (
                    current_text.find(self.tool_call_start_token),
                    current_text.find(self.tool_call_prefix),
                )
                if pos != -1
            ]
            tool_start = min(tool_start_candidates) if tool_start_candidates else -1
            if (
                self.tool_call_start_token_id in delta_token_ids
                or tool_start != -1
            ):
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

        # opt23 §11.27: JSON 激活时用 safe-prefix diff 增量发射（fix=1/2 的
        # JSON 真流式——复刻 vLLM 主线 PR #45413 的 _compute_arg_delta）。
        # 退休时认为「JSON format does not occur in practice」，但前端 JSON
        # 指令会让模型输出 JSON——恢复接入，前端要流式 json 就提供增量。
        if self._json_tool_active:
            _json_delta = self._handle_json_tool_streaming(current_text, delta_text)
            if _json_delta is not None:
                return _json_delta
            # opt23 fix=2: JSON 路由激活后不得落入 original incremental
            # path——双路径竞争输出导致增量不一致（original path 未转义
            # 真实控制字符/解码 \n，Write content 参数非法根因）。
            return None

        # v1.2.2+: =1, =2, =3, =5 all use the original incremental path
        # (json_fragments loop + _is_streaming_tool enhancements for
        # Write/Edit).  The JSON-diff and XML-diff handlers are retired
        # — they emitted one combined delta per step which caused
        # CC-HAHA SSE truncation for long-content tool calls.
        # Qwen3Coder's structural tag is always XML, so JSON format
        # does not occur in practice regardless of client format.
        if VLLM_QWEN3X_TOOL_FIX != 0:
            _log(
                "opt23 original path: fix=%s json_closed=%s func=%s "
                "delta_len=%d",
                VLLM_QWEN3X_TOOL_FIX,
                self.json_closed, self.current_function_name,
                len(delta_text))

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

                    # Always append — each tool call is a separate
                    # invocation even if the function name is the same
                    # (e.g. two consecutive "read" calls).
                    self.prev_tool_call_arr.append(
                        {
                            "name": self.current_function_name,
                            "arguments": "{}",
                        }
                    )

                    # Initialize streamed args tracking for this tool.
                    # The serving layer reads streamed_args_for_tool to
                    # compute remaining arguments at stream end. Without
                    # this, IndexError occurs when the serving layer
                    # accesses streamed_args_for_tool[index].
                    self.streamed_args_for_tool.append("")

                    # Send header with function info
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                id=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    name=self.current_function_name, arguments=""
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
                if self._is_streaming_tool:
                    # opt23: if a parameter tag is already in tool_text,
                    # defer "{" and combine it with the first param delta
                    # so CC-HAHA receives a parseable partial_json.
                    # Otherwise fall back to bare "{" — the original
                    # behavior that the model can recover from when
                    # params arrive in a subsequent delta.
                    if tool_text.find(self.parameter_prefix) != -1:
                        self._pending_brace = True
                        # Fall through to parameter processing
                    else:
                        if self.current_tool_index < len(
                                self.streamed_args_for_tool):
                            self.streamed_args_for_tool[
                                self.current_tool_index] += "{"
                        return DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_index,
                                function=DeltaFunctionCall(arguments="{"))
                        ])
                else:
                    if self.current_tool_index < len(self.streamed_args_for_tool):
                        self.streamed_args_for_tool[self.current_tool_index] += "{"
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="{"))
                    ])

            # Find all parameter start positions in current tool_text
            param_starts = []
            search_idx = 0
            while True:
                search_idx = tool_text.find(self.parameter_prefix, search_idx)
                if search_idx == -1:
                    break
                param_starts.append(search_idx)
                search_idx += len(self.parameter_prefix)

            # opt23: detect model duplication of <function=...> within
            # the same tool_text.  Gated by env var — this is a streaming
            # artifact detection heuristic.
            _func_positions: list = []
            if VLLM_QWEN3X_TOOL_FIX:
                _known_names = {t.function.name for t in (self.tools or [])
                                if getattr(t, 'function', None)
                                and getattr(t.function, 'name', None)}
                _fsi = 0
                while True:
                    _fsi = tool_text.find(self.tool_call_prefix, _fsi)
                    if _fsi == -1:
                        break
                    _close = tool_text.find(">", _fsi + len(self.tool_call_prefix))
                    if _close != -1:
                        _name = tool_text[
                            _fsi + len(self.tool_call_prefix):_close
                        ]
                        if _name in _known_names:
                            _func_positions.append(_fsi)
                    _fsi += len(self.tool_call_prefix)

                # Exclude <function=...> tags that appear inside
                # parameter values.  Once the first <parameter= opens,
                # any subsequent "<function=" is content text, not a
                # real function tag.  Without this filter the detector
                # falsely triggers when Write/Edit content happens to
                # mention tool-call syntax (e.g. documenting tool usage).
                if param_starts:
                    _func_positions = [p for p in _func_positions
                                       if p < param_starts[0]]

                if len(_func_positions) > 1:
                    # Model re-emitted the function tag. Use the LAST
                    # <function=...> as the authoritative position and
                    # ignore parameters that appear before it (they
                    # belong to the stale/duplicated first function).
                    _last_func = _func_positions[-1]
                    param_starts = [p for p in param_starts if p > _last_func]
                    # Only reset param_count the first time we detect
                    # the duplication for this tool call.
                    if not self._func_dup_detected:
                        self._func_dup_detected = True
                        self.param_count = 0


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
                value_text = tool_text[value_start:]
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                param_end_idx = value_text.find(self.parameter_end_token)
                if param_end_idx == -1:
                    next_param_idx = value_text.find(self.parameter_prefix)
                    func_end_idx = value_text.find(self.function_end_token)

                    if next_param_idx != -1 and (
                        func_end_idx == -1 or next_param_idx < func_end_idx
                    ):
                        # String params (file_path, old_string, etc.)
                        # must wait for an explicit </parameter> tag.
                        # Using the next <parameter= as a fallback
                        # boundary would complete the param with a
                        # partial value (e.g. file_path="/home" before
                        # the model finishes generating the full path).
                        if self._is_string_param(current_param_name):
                            break
                        param_end_idx = next_param_idx
                    elif func_end_idx != -1:
                        param_end_idx = func_end_idx
                    else:
                        # Fallback for malformed XML where </function>
                        # is missing. Use </tool_call> as a delimiter
                        # if present in the value so we don't include
                        # the closing tag as part of the param value.
                        tool_end_in_value = value_text.find(self.tool_call_end_token)
                        if tool_end_in_value != -1:
                            param_end_idx = tool_end_in_value
                        else:
                            # Parameter incomplete — break so we still
                            # emit any fragments accumulated by earlier
                            # loop iterations.
                            break

                if param_end_idx == -1:
                    break

                param_value = value_text[:param_end_idx]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                self.current_param_name = current_param_name
                self.accumulated_params[current_param_name] = param_value

                param_config = find_tool_properties(
                    self.tools, self.current_function_name or ""
                )

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

                # opt23: If this parameter was partially streamed, emit
                # only the delta (remaining content + closing quote).
                if self._partial_emitted:
                    emitted_total = len(self._partial_emitted)
                    if len(json_fragment) > emitted_total:
                        json_fragment = json_fragment[emitted_total:]
                    else:
                        # opt23 §11.40: partial 发射可能比完整参数长——
                        # XML 参数值以换行包裹（<parameter=x>\n值\n</parameter>），
                        # partial 发射保留尾部格式换行而完整参数 trim 掉，
                        # 导致 emitted_total > len(json_fragment)。此时内容
                        # 已全部发射，缺失的只是闭合引号——补发最后一个
                        # 字符（闭合 "），否则前端拼接 JSON 非法
                        # （Edit old_string 未闭合根因）。
                        json_fragment = (
                            json_fragment[-1:]
                            if json_fragment.endswith('"') else ""
                        )
                    self._partial_emitted = ""
                    self._partial_prefix = ""
                    if not json_fragment:
                        # All content was already streamed; skip
                        # duplicate emission but still advance counters.
                        self.param_count += 1
                        continue

                self.param_count += 1
                json_fragments.append(json_fragment)

            # opt23 XML→JSON 互转 (=4): build complete JSON from accumulated
            # params (+ partial value) and emit via safe-prefix diffing.
            # This bypasses the json_fragments + partial streaming paths
            # entirely, producing only valid JSON continuation deltas.
            # Only for =4 (双格式互转, 待设计), =3 is pure JSON no conversion.
            if (VLLM_QWEN3X_TOOL_FIX == 4
                    and self.current_function_name in self._STREAMING_TOOLS):
                _ph3 = self._emit_xml_json_diff(
                    tool_text, param_starts, tool_start_idx, current_text,
                )
                if _ph3 is not None:
                    return _ph3

            if json_fragments:
                combined = "".join(json_fragments)
                # opt23: when _pending_brace deferred the "{", prepend
                # it to the first json_fragment.
                if self._pending_brace:
                    combined = "{" + combined
                    self._pending_brace = False

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += combined
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments=combined),
                        )
                    ]
                )

            # opt23 Path B enhanced: partial streaming starts AFTER
            # file_path has been fully parsed (appears in accumulated_params).
            # This protects file_path from fragmentation regardless of
            # parameter order (e.g. Edit [replace_all, file_path, ...]).
            # Non-string params (replace_all boolean) are already blocked
            # by the type gate below.  All other string params (old_string,
            # content, new_string) get incremental streaming — even large
            # old_string values don't block the frontend.
            _param_config = find_tool_properties(
                self.tools, self.current_function_name or "",
            )
            _expected_total = len(_param_config) if _param_config else 0

            if (not json_fragments
                    and self._is_streaming_tool
                    and self.in_function
                    and self.json_started
                    and self.param_count > 0
                    and self.param_count < len(param_starts)
                    and _expected_total > 1
                    and self.accumulated_params.get("file_path") is not None
                    and self.current_tool_index < len(self.streamed_args_for_tool)):
                incomplete_start = param_starts[self.param_count]
                if incomplete_start != self._partial_param_start:
                    self._partial_param_start = incomplete_start
                    self._partial_emit_offset = 0
                    self._partial_emitted = ""
                    self._partial_prefix = ""

                param_text = tool_text[incomplete_start
                                       + len(self.parameter_prefix):]
                if ">" in param_text:
                    name_end = param_text.find(">")
                    param_name = param_text[:name_end]
                    # opt23: Partial streaming only for string-type
                    # parameters.  Boolean (replace_all), integer etc.
                    # must be fully parsed and type-coerced via
                    # _convert_param_value.  Raw streaming bypasses
                    # type coercion, producing "false" vs false
                    # mismatches that break remaining-delta computation
                    # and cause CC-HAHA to receive malformed JSON.
                    _pc = _param_config  # reuse from gate above
                    _ps = _pc.get(param_name, {})
                    _pt = extract_types_from_schema(_ps)
                    if _pt and "string" not in _pt:
                        return None

                    if not self._partial_prefix:
                        if self.param_count == 0:
                            _prefix = f'"{param_name}": "'
                        else:
                            _prefix = f', "{param_name}": "'
                        self._partial_prefix = _prefix
                    value_start = (incomplete_start
                                   + len(self.parameter_prefix)
                                   + name_end + 1)
                    # value_start is relative to tool_text. Adjust to
                    # absolute position in current_text for slicing.
                    _abs_value_start = tool_start_idx + value_start
                    if (_abs_value_start < len(current_text)
                            and current_text[_abs_value_start:_abs_value_start + 1]
                            == "\n"):
                        _abs_value_start += 1

                    current_value = current_text[_abs_value_start:]
                    if len(current_value) > self._partial_emit_offset:
                        _now = time.monotonic()
                        if (_now - self._last_partial_time) < 0.25:
                            return None
                        _raw_new = current_value[
                            self._partial_emit_offset:]
                        # opt23: strip trailing XML close-tag fragments
                        # (</parameter>, </function>, …) so they don't
                        # leak into the streamed JSON string value.
                        _safe_len = self._safe_content_length(_raw_new)
                        if _safe_len == 0:
                            return None
                        new_content = _raw_new[:_safe_len]
                        self._partial_emit_offset += _safe_len
                        if new_content:
                            escaped = json.dumps(
                                new_content, ensure_ascii=False)[1:-1]
                            if not self._partial_emitted:
                                fragment = self._partial_prefix + escaped
                                self._partial_emitted += fragment
                            else:
                                fragment = escaped
                                self._partial_emitted += fragment
                            if self.current_tool_index < len(
                                    self.streamed_args_for_tool):
                                self.streamed_args_for_tool[
                                    self.current_tool_index] += fragment
                            self._last_partial_time = _now
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_index,
                                        function=DeltaFunctionCall(
                                            arguments=fragment),
                                    )
                                ]
                            )
                    return None

            if (not json_fragments
                    and self.param_count < len(param_starts)
                    and self.current_tool_index < len(self.streamed_args_for_tool)):
                return None

            # Check for function end AFTER processing parameters.
            # This ordering is critical: with speculative decoding a
            # burst can deliver the final parameter value together with
            # </function>. If the close check ran first it would emit
            # "}" and set in_function=False before the parameter loop
            # ever ran, causing the parameter to be silently dropped.
            if not self.json_closed and self.function_end_token in tool_text:
                # opt23: when _pending_brace deferred "{" but no
                # <parameter=...> tags arrived, fall back to bare
                # "{" + "}" — the original behavior that the model
                # can recover from (user empirically confirmed).
                if self._pending_brace and not param_starts:
                    if self.current_tool_index < len(
                            self.streamed_args_for_tool):
                        self.streamed_args_for_tool[
                            self.current_tool_index] += "{"
                    self._pending_brace = False
                    # Fall through to emit "}" via function-end handler

                self.json_closed = True

                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = tool_text.find(self.function_end_token, func_start)
                if func_content_end != -1:
                    func_content = tool_text[func_start:func_content_end]
                    try:
                        parsed_tool = self._parse_xml_function_call(
                            func_content,
                        )
                        if parsed_tool and self.current_tool_index < len(
                            self.prev_tool_call_arr
                        ):
                            args = parsed_tool.function.arguments
                            self.prev_tool_call_arr[self.current_tool_index][
                                "arguments"
                            ] = args
                            _log(
                                "opt23 main func_end: tool=%s idx=%s "
                                "args=%s streamed=%s",
                                self.current_function_name,
                                self.current_tool_index,
                                args,
                                self.streamed_args_for_tool[
                                    self.current_tool_index]
                                if self.current_tool_index < len(
                                    self.streamed_args_for_tool)
                                else "N/A")
                            # opt23 Fix B: detect empty tool calls
                            # (model emitted <function=NAME></function>
                            # with no <parameter=...> tags).
                            if args == "{}" and not self._is_streaming_tool:
                                logger.warning(
                                    "Qwen3Coder streaming: tool '%s' "
                                    "has no parameters (empty {}) — "
                                    "model error, call will fail",
                                    self.current_function_name or "?",
                                )
                    except Exception:
                        logger.debug(
                            "Failed to parse tool call during streaming: %s",
                            tool_text,
                            exc_info=True,
                        )

                # opt23: for streaming tools (Write/Edit), compute the real
                # remaining delta from the full parsed JSON minus what was
                # already streamed, instead of emitting "}" as a bare
                # punctuation delta that clients cannot parse.
                remaining = "}"
                if self._is_streaming_tool:
                    if self.current_tool_index < len(self.prev_tool_call_arr):
                        full_json = self.prev_tool_call_arr[
                            self.current_tool_index
                        ].get("arguments", "")
                        streamed = (
                            self.streamed_args_for_tool[self.current_tool_index]
                            if self.current_tool_index < len(
                                self.streamed_args_for_tool)
                            else ""
                        )
                        if full_json and full_json.startswith(streamed):
                            remaining = full_json[len(streamed):]
                self._pending_brace = False

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += remaining
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )

                result = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments=remaining),
                        )
                    ]
                )

                self.in_function = False
                self.json_closed = True
                self.accumulated_params = {}

                return result

        return None

    def get_structural_tag(self, request: ChatCompletionRequest):
        tag = get_model_structural_tag(
            model="qwen_3_5",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )
        # v1.2.2+: 不再尝试检测客户端格式。Qwen3Coder 的 structural tag
        # 始终是 XML 格式（str(tag) 返回 Python repr，不含 "name"），
        # 所以 _json_tool_active 始终为 False。所有工具走原始路径。
        return tag

# ------------------------------------------------------------------
# Hot-reload hook: when importlib.reload() re-executes this module,
# patch any cached class references that other modules hold so the
# running process picks up code changes without a restart.
# ------------------------------------------------------------------
def _reload_patch_cached_refs() -> None:
    """After ``importlib.reload()``, the *new* Qwen3CoderToolParser
    class has been created in this module's namespace, but other
    modules (ToolParserManager, OpenAIServingRender, OpenAIServingChat,
    AnthropicServingMessages) still hold references to the *old* class
    object.  Copy every method / property / class-attribute from the
    new class onto the old class so that even existing references
    behave like the reloaded code.
    """
    try:
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
    except Exception:
        return

    old_cls = ToolParserManager.tool_parsers.get("qwen3_coder")
    if old_cls is None:
        return
    if old_cls is Qwen3CoderToolParser:
        return  # already the same object (first import, not a reload)

    # Copy attributes from the *new* class onto the *old* class object.
    # After this, ``old_cls(...)`` instantiates with the latest code.
    copied = 0
    for name in list(dir(Qwen3CoderToolParser)):
        if name in ("__dict__", "__weakref__", "__class__", "__bases__",
                     "__mro__", "__subclasshook__", "__init_subclass__",
                     "__init__"):
            continue
            continue
        try:
            attr = getattr(Qwen3CoderToolParser, name)
        except Exception:
            continue
        # Only copy callables (methods, staticmethod, classmethod) and
        # properties / data descriptors — skip plain class data that
        # is set in __init__.
        if callable(attr) or isinstance(attr, (property, staticmethod,
                                                classmethod)):
            try:
                setattr(old_cls, name, attr)
                copied += 1
            except Exception:
                pass

    # Also copy class-level data attributes (like _STREAMING_TOOLS)
    for name in ("_STREAMING_TOOLS", "supports_required_and_named"):
        try:
            setattr(old_cls, name, getattr(Qwen3CoderToolParser, name))
        except Exception:
            pass

    # Update the registry cache so fresh lookups return the new class.
    ToolParserManager.tool_parsers["qwen3_coder"] = Qwen3CoderToolParser

    import logging
    _rl = logging.getLogger(__name__)
    _rl.info(
        "Qwen3Coder hot-reload: patched %d attributes on cached class "
        "(old id=%s → new id=%s).", copied, hex(id(old_cls)),
        hex(id(Qwen3CoderToolParser)))

_reload_patch_cached_refs()
