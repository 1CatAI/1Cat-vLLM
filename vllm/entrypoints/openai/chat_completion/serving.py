# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import io
import json
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import pybase64 as base64
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    get_history_tool_calls_cnt,
    get_tool_call_id_type,
    make_tool_call_id,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.stream_harmony import (
    TokenState,
    extract_harmony_streaming_delta,
)
from vllm.entrypoints.openai.chat_completion.tool_stream import (
    StreamedToolCall,
    flush_tool_calls,
    record_tool_calls,
)
from vllm.entrypoints.openai.engine.protocol import (
    CompletionTokenUsageInfo,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    FunctionCall,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    GenerationError,
    OpenAIServing,
    clamp_prompt_logprobs,
    drop_misaligned_logprobs,
)
from vllm.entrypoints.openai.engine.usage import get_request_usage_timings
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant,
    parse_chat_output,
)
from vllm.entrypoints.openai.utils import maybe_filter_parallel_tool_calls
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import STREAM_KEEPALIVE, CompletionOutput, RequestOutput
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.collection_utils import as_list
from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser

if TYPE_CHECKING:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

logger = init_logger(__name__)


def _token_list(rendered: Any) -> list[int]:
    if isinstance(rendered, dict) or hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    return list(rendered)


def _count_reasoning_tokens(reasoning_parser, token_ids_per_choice) -> int | None:
    if reasoning_parser is None or token_ids_per_choice is None:
        return None
    counts = [
        reasoning_parser.count_reasoning_tokens(ids)
        for ids in token_ids_per_choice
        if ids
    ]
    return sum(counts) if counts else None


class _PrefixCacheBoundaries:
    """Prompt positions worth a partial prefix-cache entry.

    A follow-up chat turn repeats the conversation up to the end of the last
    message, but not necessarily the generation prompt after it: reasoning
    templates render a finished assistant turn differently from the prompt
    that started it. The end of the shared system prompt is also where
    concurrent conversations diverge. Both positions are derived by rendering
    small probes with the same template and accepted only when they reproduce
    the actual prompt tokens.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._generation: OrderedDict[str, tuple[int, ...] | None] = OrderedDict()
        self._system: OrderedDict[str, tuple[int, ...] | None] = OrderedDict()
        self._max_entries = max_entries

    # Rendering arguments this helper sets itself.
    _OWN_ARGS = (
        "add_generation_prompt",
        "continue_final_message",
        "tokenize",
        "tools",
        "chat_template",
        "conversation",
        "messages",
    )

    @staticmethod
    def _normalize_kwargs(tokenizer, chat_template: str | None, kwargs: dict[str, Any]):
        cleaned = {
            k: v for k, v in kwargs.items() if k not in _PrefixCacheBoundaries._OWN_ARGS
        }
        tmpl = chat_template or getattr(tokenizer, "chat_template", "") or ""
        if "reasoning_effort" in cleaned and "Supported types are xhigh" in str(tmpl):
            effort = cleaned.get("reasoning_effort")
            if effort == "max":
                cleaned["reasoning_effort"] = "xhigh"
            elif effort == "minimal":
                cleaned["reasoning_effort"] = "low"
            elif effort == "none":
                cleaned.pop("reasoning_effort", None)
                cleaned.setdefault("enable_thinking", False)
        return cleaned

    def _put_generation(
        self, key: str, value: tuple[int, ...] | None
    ) -> tuple[int, ...] | None:
        self._generation[key] = value
        self._generation.move_to_end(key)
        while len(self._generation) > self._max_entries:
            self._generation.popitem(last=False)
        return value

    @staticmethod
    def _render(tokenizer, messages, chat_template, kwargs, tools, generation):
        kwargs = _PrefixCacheBoundaries._normalize_kwargs(
            tokenizer, chat_template, kwargs
        )
        try:
            return _token_list(
                tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    chat_template=chat_template,
                    add_generation_prompt=generation,
                    tokenize=True,
                    **kwargs,
                )
            )
        except Exception:
            return None

    def _generation_prompt(self, tokenizer, chat_template, kwargs):
        key = json.dumps([chat_template, kwargs], sort_keys=True, default=str)
        if key in self._generation:
            self._generation.move_to_end(key)
            return self._generation[key]
        probe = [{"role": "user", "content": "x"}]
        with_prompt = self._render(tokenizer, probe, chat_template, kwargs, None, True)
        without = self._render(tokenizer, probe, chat_template, kwargs, None, False)
        suffix = None
        if (
            with_prompt is not None
            and without is not None
            and len(with_prompt) > len(without)
            and with_prompt[: len(without)] == without
        ):
            suffix = tuple(with_prompt[len(without) :])
        return self._put_generation(key, suffix)

    @staticmethod
    def _has_non_text_parts(messages: GenericSequence[dict[str, Any]]) -> bool:
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") != "text" for p in content
            ):
                return True
        return False

    @staticmethod
    def _digest_obj(obj: Any) -> str:
        payload = json.dumps(
            obj, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def _trailing_turn_suffix(
        self,
        tokenizer,
        trailing_messages: list[dict[str, Any]],
        chat_template,
        kwargs,
    ):
        if self._has_non_text_parts(trailing_messages):
            return None
        turn_key = "__turn_sfx__:" + self._digest_obj(
            [trailing_messages, chat_template, kwargs]
        )
        if turn_key in self._generation:
            self._generation.move_to_end(turn_key)
            return self._generation[turn_key]
        base_key = "__turn_base__:" + self._digest_obj([chat_template, kwargs])
        base_msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        if base_key in self._generation:
            self._generation.move_to_end(base_key)
            base = self._generation[base_key]
        else:
            base_tokens = self._render(
                tokenizer, base_msgs, chat_template, kwargs, None, False
            )
            base = self._put_generation(
                base_key, tuple(base_tokens) if base_tokens else None
            )
        if not base:
            return self._put_generation(turn_key, None)
        probed = self._render(
            tokenizer,
            base_msgs + trailing_messages,
            chat_template,
            kwargs,
            None,
            True,
        )
        suffix = None
        if (
            probed is not None
            and len(probed) > len(base)
            and tuple(probed[: len(base)]) == base
        ):
            suffix = tuple(probed[len(base) :])
        return self._put_generation(turn_key, suffix)

    def _system_prompt(
        self,
        tokenizer,
        sys_messages,
        chat_template,
        kwargs,
        tools,
        *,
        tools_digest: str | None = None,
    ):
        td = (
            tools_digest
            if tools_digest is not None
            else (self._digest_obj(tools) if tools else "")
        )
        key = self._digest_obj([sys_messages, td, chat_template, kwargs])
        if key in self._system:
            self._system.move_to_end(key)
            return self._system[key]
        prefix_msgs = list(sys_messages) if sys_messages else []
        tokens = (
            self._render(tokenizer, prefix_msgs, chat_template, kwargs, tools, False)
            if prefix_msgs
            else None
        )
        if not tokens and (prefix_msgs or tools):
            probe_a = self._render(
                tokenizer,
                prefix_msgs + [{"role": "user", "content": "A"}],
                chat_template,
                kwargs,
                tools,
                False,
            )
            probe_b = self._render(
                tokenizer,
                prefix_msgs + [{"role": "user", "content": "xyz"}],
                chat_template,
                kwargs,
                tools,
                False,
            )
            if probe_a and probe_b:
                common = 0
                limit = min(len(probe_a), len(probe_b))
                while common < limit and probe_a[common] == probe_b[common]:
                    common += 1
                if common > 0:
                    tokens = probe_a[:common]
        value = tuple(tokens) if tokens else None
        self._system[key] = value
        if len(self._system) > self._max_entries:
            self._system.popitem(last=False)
        return value

    @staticmethod
    def _common_prefix_len(a: GenericSequence[int], b: GenericSequence[int]) -> int:
        limit = min(len(a), len(b))
        idx = 0
        while idx < limit and a[idx] == b[idx]:
            idx += 1
        return idx

    def __call__(
        self,
        tokenizer,
        request: Any,
        prompt_token_ids: list[int] | None,
        chat_template: str | None,
        kwargs: dict[str, Any],
        *,
        messages: GenericSequence[Any] | None = None,
        tools: GenericSequence[Any] | None = None,
    ) -> list[int]:
        if not prompt_token_ids or not hasattr(tokenizer, "apply_chat_template"):
            return []
        n = len(prompt_token_ids)
        boundaries = []
        suffix = self._generation_prompt(tokenizer, chat_template, kwargs)
        if (
            suffix
            and n > len(suffix)
            and tuple(prompt_token_ids[-len(suffix) :]) == suffix
        ):
            boundaries.append(n - len(suffix))
        resolved_messages = (
            messages
            if messages is not None
            else (getattr(request, "messages", None) or [])
        )
        sys_msgs: list[dict[str, Any]] = []
        for msg in resolved_messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") in ("system", "developer")
                and isinstance(msg.get("content"), str | list)
            ):
                sys_msgs.append({"role": msg["role"], "content": msg["content"]})
            else:
                break
        has_non_sys = len(resolved_messages) > len(sys_msgs)
        req_tools = tools if tools is not None else getattr(request, "tools", None)
        if (sys_msgs and has_non_sys) or (resolved_messages and req_tools):
            cached_dumped = (
                getattr(request, "_cached_tool_dicts", None) if tools is None else None
            )
            dumped_tools = (
                cached_dumped
                if cached_dumped is not None
                else (
                    [
                        tool.model_dump() if hasattr(tool, "model_dump") else tool
                        for tool in req_tools
                    ]
                    if req_tools
                    else None
                )
            )
            tools_digest = self._digest_obj(dumped_tools) if dumped_tools else ""
            system = self._system_prompt(
                tokenizer,
                sys_msgs if has_non_sys else None,
                chat_template,
                kwargs,
                dumped_tools,
                tools_digest=tools_digest,
            )
            if (
                system
                and len(system) < n
                and tuple(prompt_token_ids[: len(system)]) == system
            ):
                boundaries.append(len(system))
            if dumped_tools and sys_msgs:
                tools_only = self._system_prompt(
                    tokenizer,
                    None,
                    chat_template,
                    kwargs,
                    dumped_tools,
                    tools_digest=tools_digest,
                )
                if tools_only:
                    common = self._common_prefix_len(prompt_token_ids, tools_only)
                    if 128 <= common < n:
                        boundaries.append(common)
            if (
                len(sys_msgs) == 1
                and isinstance(sys_msgs[0].get("content"), list)
                and len(sys_msgs[0]["content"]) > 1
            ):
                parts = sys_msgs[0]["content"]
                for idx_part in range(len(parts) - 1):
                    part = parts[idx_part]
                    if isinstance(part, dict) and part.get("cache_control"):
                        sub_sys = [
                            {
                                "role": sys_msgs[0]["role"],
                                "content": parts[: idx_part + 1],
                            }
                        ]
                        sub_tokens = self._system_prompt(
                            tokenizer,
                            sub_sys,
                            chat_template,
                            kwargs,
                            dumped_tools,
                            tools_digest=tools_digest,
                        )
                        if sub_tokens:
                            common = self._common_prefix_len(
                                prompt_token_ids, sub_tokens
                            )
                            if 64 <= common < n:
                                boundaries.append(common)
            sys_cache_prefixes = getattr(request, "_system_cache_prefixes", None)
            if sys_msgs and sys_cache_prefixes:
                for pfx in sys_cache_prefixes:
                    sub_tokens = self._system_prompt(
                        tokenizer,
                        [{"role": sys_msgs[0]["role"], "content": pfx}],
                        chat_template,
                        kwargs,
                        dumped_tools,
                        tools_digest=tools_digest,
                    )
                    if sub_tokens:
                        common = self._common_prefix_len(prompt_token_ids, sub_tokens)
                        if 64 <= common < n:
                            boundaries.append(common)

        if len(resolved_messages) >= 3:
            last = resolved_messages[-1]
            if isinstance(last, dict):
                last_role = last.get("role")
                if last_role == "user" and isinstance(last.get("content"), str | list):
                    turn_suffix = self._trailing_turn_suffix(
                        tokenizer,
                        [{"role": "user", "content": last["content"]}],
                        chat_template,
                        kwargs,
                    )
                    if (
                        turn_suffix
                        and len(turn_suffix) < n
                        and tuple(prompt_token_ids[-len(turn_suffix) :]) == turn_suffix
                    ):
                        boundaries.append(n - len(turn_suffix))
                elif last_role == "tool":
                    tool_start = len(resolved_messages) - 1
                    while (
                        tool_start > 0
                        and isinstance(resolved_messages[tool_start - 1], dict)
                        and resolved_messages[tool_start - 1].get("role") == "tool"
                    ):
                        tool_start -= 1
                    if tool_start >= 2:
                        trailing_tools = [
                            {
                                "role": "tool",
                                "tool_call_id": m.get("tool_call_id", ""),
                                "content": m.get("content", ""),
                            }
                            for m in resolved_messages[tool_start:]
                            if isinstance(m, dict)
                        ]
                        tool_suffix = self._trailing_turn_suffix(
                            tokenizer, trailing_tools, chat_template, kwargs
                        )
                        if (
                            tool_suffix
                            and len(tool_suffix) < n
                            and tuple(prompt_token_ids[-len(tool_suffix) :])
                            == tool_suffix
                        ):
                            boundaries.append(n - len(tool_suffix))
        return sorted(set(boundaries))


class _AssistantReasoningCache:
    """Restore omitted assistant ``reasoning_content`` across multi-turn chats.

    Many OpenAI-compatible SDKs and agent loops append only ``content`` and
    ``tool_calls`` to conversation history. For templates that retain
    ``reasoning_content`` on prior assistant turns, omitting it changes the
    rendered tokens right after ``<think>\\n`` and invalidates the KV/recurrent
    prefix generated during decode.
    """

    def __init__(self, max_entries: int = 2048) -> None:
        self._entries: OrderedDict[tuple[Any, ...], str] = OrderedDict()
        self._token_spans: OrderedDict[
            tuple[Any, ...], tuple[str, int, bytes, tuple[int, ...]]
        ] = OrderedDict()
        self._max_entries = max_entries

    @staticmethod
    def _token_digest(tokens: GenericSequence[int]) -> bytes:
        arr = np.asarray(tokens, dtype=np.int32)
        return hashlib.blake2b(arr.tobytes(), digest_size=16).digest()

    @staticmethod
    def _canonical_json(raw: Any) -> str:
        if isinstance(raw, dict | list):
            return json.dumps(
                raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        if not isinstance(raw, str):
            return str(raw or "")
        text = raw.strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
            return json.dumps(
                parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        except Exception:
            return text

    @classmethod
    def _canonical_tool_calls(cls, tool_calls: Any) -> tuple[tuple[str, str], ...]:
        if not tool_calls:
            return ()
        items: list[tuple[str, str]] = []
        for tc in tool_calls:
            if isinstance(tc, tuple) and len(tc) == 2:
                name, args = tc
            elif isinstance(tc, dict):
                fn = tc.get("function") or {}
                name = fn.get("name", "") if isinstance(fn, dict) else ""
                args = fn.get("arguments", "") if isinstance(fn, dict) else ""
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") if fn is not None else ""
                args = getattr(fn, "arguments", "") if fn is not None else ""
            items.append((str(name or ""), cls._canonical_json(args)))
        return tuple(items)

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        return ""

    @staticmethod
    def _extract_media_sig(content: Any) -> tuple[tuple[str, str], ...]:
        if not isinstance(content, list):
            return ()
        sigs: list[tuple[str, str]] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            ptype = str(p.get("type") or "")
            if ptype == "text":
                continue
            raw_val: Any = None
            for k in (
                "image_url",
                "input_image",
                "image",
                "video_url",
                "video",
                "audio_url",
                "input_audio",
                "audio",
            ):
                if k in p and p[k] is not None:
                    v = p[k]
                    raw_val = v.get("url", v) if isinstance(v, dict) else v
                    break
            if raw_val is None:
                raw_str = json.dumps(p, sort_keys=True, default=str)
            elif isinstance(raw_val, str):
                raw_str = raw_val
            else:
                raw_str = str(raw_val)
            part_hash = hashlib.blake2b(
                raw_str.encode("utf-8", errors="replace"), digest_size=12
            ).hexdigest()
            sigs.append((ptype, part_hash))
        return tuple(sigs)

    @classmethod
    def _message_digest(cls, m: Any) -> bytes:
        if isinstance(m, dict):
            role = m.get("role", "")
            tool_call_id = m.get("tool_call_id", "")
            raw_content = m.get("content")
            tcs = cls._canonical_tool_calls(m.get("tool_calls"))
        else:
            role = getattr(m, "role", "")
            tool_call_id = getattr(m, "tool_call_id", "")
            raw_content = getattr(m, "content", None)
            tcs = cls._canonical_tool_calls(getattr(m, "tool_calls", None))
        content = cls._extract_text(raw_content).strip()
        media = cls._extract_media_sig(raw_content)
        h = hashlib.blake2b(digest_size=16)
        h.update(str(role or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(str(tool_call_id or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(content.encode("utf-8"))
        h.update(b"\x00")
        if media:
            for ptype, phash in media:
                h.update(ptype.encode("utf-8"))
                h.update(b":")
                h.update(phash.encode("ascii"))
                h.update(b";")
        h.update(b"\x00")
        if tcs:
            for name, args in tcs:
                h.update(name.encode("utf-8"))
                h.update(b"=")
                h.update(args.encode("utf-8"))
                h.update(b"\x01")
        return h.digest()

    @classmethod
    def _context_digest(cls, prior_messages: GenericSequence[Any]) -> str:
        if not prior_messages:
            return ""
        h = hashlib.blake2b(digest_size=16)
        for m in prior_messages:
            h.update(cls._message_digest(m))
        return h.hexdigest()

    @classmethod
    def prefix_digests(cls, messages: GenericSequence[Any]) -> list[str]:
        """Compute ``_context_digest(messages[:idx])`` for all ``idx`` in O(L)."""
        n = len(messages)
        if n == 0:
            return []
        out = [""]
        if n == 1:
            return out
        h = hashlib.blake2b(digest_size=16)
        for idx in range(n - 1):
            h.update(cls._message_digest(messages[idx]))
            out.append(h.hexdigest())
        return out

    def _put(self, key: tuple[Any, ...], reasoning: str) -> None:
        self._entries[key] = reasoning
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def _put_token_span(
        self,
        key: tuple[Any, ...],
        value: tuple[str, int, bytes, tuple[int, ...]],
    ) -> None:
        self._token_spans[key] = value
        self._token_spans.move_to_end(key)
        while len(self._token_spans) > self._max_entries:
            self._token_spans.popitem(last=False)

    def record(
        self,
        prior_messages: GenericSequence[Any],
        content: Any,
        tool_calls: Any,
        reasoning: str | None,
        *,
        cache_salt: str | None = None,
        reasoning_ended: bool = True,
        prompt_token_ids: GenericSequence[int] | None = None,
        output_token_ids: GenericSequence[int] | None = None,
        gen_suffix: tuple[int, ...] | None = None,
        im_end_id: int | None = None,
        nl_ids: tuple[int, ...] | None = None,
    ) -> None:
        content_norm = self._extract_text(content).strip()
        tc_norm = self._canonical_tool_calls(tool_calls)
        if not content_norm and not tc_norm:
            return
        ctx = self._context_digest(prior_messages)
        salt_norm = cache_salt or ""
        ctx_key = ("ctx", salt_norm, ctx, content_norm, tc_norm)
        if reasoning and reasoning_ended:
            self._put(ctx_key, reasoning)
        if (
            reasoning_ended
            and prompt_token_ids
            and output_token_ids
            and gen_suffix
            and im_end_id is not None
            and nl_ids
            and len(prompt_token_ids) > len(gen_suffix)
            and len(output_token_ids) <= 32768
            and output_token_ids[-1] == im_end_id
            and tuple(prompt_token_ids[-len(gen_suffix) :]) == gen_suffix
        ):
            prefix_len = len(prompt_token_ids) - len(gen_suffix)
            prefix_digest = self._token_digest(prompt_token_ids[:prefix_len])
            assistant_span = tuple(gen_suffix) + tuple(output_token_ids) + tuple(nl_ids)
            self._put_token_span(
                ctx_key,
                ((reasoning or "").strip(), prefix_len, prefix_digest, assistant_span),
            )

    def lookup(
        self,
        prior_messages: GenericSequence[Any],
        content: Any,
        tool_calls: Any,
        *,
        cache_salt: str | None = None,
        ctx_digest: str | None = None,
    ) -> str | None:
        content_norm = self._extract_text(content).strip()
        tc_norm = self._canonical_tool_calls(tool_calls)
        if not content_norm and not tc_norm:
            return None
        ctx = (
            ctx_digest
            if ctx_digest is not None
            else self._context_digest(prior_messages)
        )
        salt_norm = cache_salt or ""
        ctx_key = ("ctx", salt_norm, ctx, content_norm, tc_norm)
        if ctx_key in self._entries:
            self._entries.move_to_end(ctx_key)
            return self._entries[ctx_key]
        return None

    def lookup_token_span(
        self,
        prior_messages: GenericSequence[Any],
        content: Any,
        tool_calls: Any,
        *,
        cache_salt: str | None = None,
        ctx_digest: str | None = None,
    ) -> tuple[str, int, bytes, tuple[int, ...]] | None:
        content_norm = self._extract_text(content).strip()
        tc_norm = self._canonical_tool_calls(tool_calls)
        if not content_norm and not tc_norm:
            return None
        ctx = (
            ctx_digest
            if ctx_digest is not None
            else self._context_digest(prior_messages)
        )
        salt_norm = cache_salt or ""
        ctx_key = ("ctx", salt_norm, ctx, content_norm, tc_norm)
        if ctx_key in self._token_spans:
            self._token_spans.move_to_end(ctx_key)
            return self._token_spans[ctx_key]
        return None

    def splice_prompt_tokens(
        self,
        messages: GenericSequence[Any],
        prompt_token_ids: list[int],
        tokenizer: Any,
        *,
        cache_salt: str | None = None,
        max_model_len: int | None = None,
        mm_placeholders: Any = None,
    ) -> list[int]:
        """Replace re-rendered historical assistant spans with exact decoded IDs.

        Chat templates that run ``reasoning_content|trim`` or reformat XML/JSON
        tool arguments can shift token IDs by 1-2 tokens after ``<think>\\n``,
        missing ``PartialPrefixCache`` decode checkpoints on Turn 2+. Splicing
        the exact decoded span when the prefix digest and ``<|im_start|>`` /
        ``<|im_end|>\\n`` boundaries match preserves 100% decode cache fidelity.
        """
        if not self._token_spans or len(messages) < 2 or not prompt_token_ids:
            return prompt_token_ids
        try:
            vocab = tokenizer.get_vocab()
            im_start_id = vocab.get("<|im_start|>")
            im_end_id = vocab.get("<|im_end|>")
            think_id = vocab.get("<think>")
            nl_ids = tuple(
                _token_list(tokenizer.encode("\n", add_special_tokens=False))
            )
        except Exception:
            return prompt_token_ids
        if im_start_id is None or im_end_id is None or not nl_ids:
            return prompt_token_ids
        nl_list = list(nl_ids)
        nl_len = len(nl_list)
        tokens = list(prompt_token_ids)
        ctx_digests = self.prefix_digests(messages)
        for idx in range(len(messages) - 1):
            msg = messages[idx]
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            span_entry = self.lookup_token_span(
                (),
                msg.get("content"),
                msg.get("tool_calls"),
                cache_salt=cache_salt,
                ctx_digest=ctx_digests[idx],
            )
            if span_entry is None:
                continue
            cached_reasoning_norm, prefix_len, prefix_digest, assistant_span = (
                span_entry
            )
            explicit_r = msg.get("reasoning")
            if explicit_r is None:
                explicit_r = msg.get("reasoning_content")
            if (
                isinstance(explicit_r, str)
                and explicit_r.strip()
                and explicit_r.strip() != cached_reasoning_norm
            ):
                continue
            if prefix_len >= len(tokens) or tokens[prefix_len] != im_start_id:
                continue
            if self._token_digest(tokens[:prefix_len]) != prefix_digest:
                continue
            try:
                end_pos = tokens.index(im_end_id, prefix_len + 1)
            except ValueError:
                continue
            after_msg = end_pos + 1 + nl_len
            if after_msg > len(tokens) or tokens[end_pos + 1 : after_msg] != nl_list:
                continue
            if think_id is not None and (think_id in assistant_span) != (
                think_id in tokens[prefix_len:after_msg]
            ):
                continue
            delta = len(assistant_span) - (after_msg - prefix_len)
            new_len = len(tokens) + delta
            if max_model_len is not None and new_len >= max_model_len:
                continue
            if delta != 0 and isinstance(mm_placeholders, dict) and mm_placeholders:
                overlaps = any(
                    r.offset < after_msg and r.offset + r.length > prefix_len
                    for ranges in mm_placeholders.values()
                    for r in ranges
                )
                if overlaps:
                    continue
            span_list = list(assistant_span)
            if tokens[prefix_len:after_msg] != span_list:
                tokens = tokens[:prefix_len] + span_list + tokens[after_msg:]
                if delta != 0 and isinstance(mm_placeholders, dict) and mm_placeholders:
                    for mod_key, ranges in list(mm_placeholders.items()):
                        mm_placeholders[mod_key] = [
                            type(r)(
                                offset=r.offset + delta,
                                length=r.length,
                                is_embed=r.is_embed,
                            )
                            if r.offset >= after_msg
                            else r
                            for r in ranges
                        ]
        return tokens


class OpenAIServingChat(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        openai_serving_render: "OpenAIServingRender",
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        enable_log_deltas: bool = True,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.openai_serving_render = openai_serving_render
        self.response_role = response_role
        self.chat_template = chat_template
        self._prefix_cache_boundaries = _PrefixCacheBoundaries()
        self._assistant_reasoning_cache = _AssistantReasoningCache()
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.enable_log_outputs = enable_log_outputs
        self.enable_log_deltas = enable_log_deltas

        # set up reasoning parser
        self.reasoning_parser_cls = ParserManager.get_reasoning_parser(
            reasoning_parser_name=reasoning_parser
        )
        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        self.tool_parser = ParserManager.get_tool_parser(
            tool_parser_name=tool_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
        )
        self.parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
        )
        if (
            is_mistral_tool_parser(self.tool_parser)
            and self.reasoning_parser_cls is not None
        ):
            from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

            MistralToolParser.model_can_reason = True

        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
        self.use_harmony = self.model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        self.tool_call_id_type = get_tool_call_id_type(self.model_config)

        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

    def warmup(self) -> None:
        self.renderer.warmup(
            ChatParams(
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                chat_template_kwargs=self.default_chat_template_kwargs,
            )
        )

    def _effective_chat_template_kwargs(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    def _effective_sampling_params(
        self,
        request: ChatCompletionRequest,
        chat_template_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        effort = request.reasoning_effort or chat_template_kwargs.get(
            "reasoning_effort"
        )
        enable_thinking = chat_template_kwargs.get("enable_thinking")
        thinking_enabled = (enable_thinking is not False) and (effort != "none")
        return self.model_config.get_effective_sampling_param(
            thinking_enabled=thinking_enabled,
            base_defaults=self.default_sampling_params,
        )

    def _restore_historical_assistant_reasoning(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        messages = request.messages
        if len(messages) < 2:
            return request
        updated: list[Any] | None = None
        ctx_digests: list[str] | None = None
        for idx in range(len(messages) - 1):
            msg = messages[idx]
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            existing = msg.get("reasoning")
            if existing is None:
                existing = msg.get("reasoning_content")
            if existing is not None:
                if msg.get("reasoning_content") is None:
                    if updated is None:
                        updated = list(messages)
                    updated[idx] = {**msg, "reasoning_content": existing}
                continue
            if ctx_digests is None:
                ctx_digests = self._assistant_reasoning_cache.prefix_digests(messages)
            restored = self._assistant_reasoning_cache.lookup(
                (),
                msg.get("content"),
                msg.get("tool_calls"),
                cache_salt=request.cache_salt,
                ctx_digest=ctx_digests[idx],
            )
            if restored is not None:
                if updated is None:
                    updated = list(messages)
                updated[idx] = {
                    **msg,
                    "reasoning": restored,
                    "reasoning_content": restored,
                }
        if updated is None:
            return request
        return request.model_copy(update={"messages": updated})

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """
        Validate the model and preprocess a chat completion request.

        Delegates preprocessing logic to OpenAIServingRender, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A tuple of (conversation, engine_inputs) on success,
            or an ErrorResponse on failure.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        return await self.openai_serving_render.render_chat(request)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        return await self._with_kv_transfer_rejection_cleanup(
            self._create_chat_completion(request, raw_request), request, raw_request
        )

    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        parser_chat_kwargs = (
            {**chat_template_kwargs, "reasoning_effort": request.reasoning_effort}
            if request.reasoning_effort
            and "reasoning_effort" not in chat_template_kwargs
            else chat_template_kwargs
        )
        effective_sampling_params = self._effective_sampling_params(
            request, chat_template_kwargs
        )
        reasoning_parser: ReasoningParser | None = None
        if self.reasoning_parser_cls:
            reasoning_parser = self.reasoning_parser_cls(
                tokenizer,
                chat_template_kwargs=parser_chat_kwargs,  # type: ignore[call-arg]
            )
            request = self._restore_historical_assistant_reasoning(request)
        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_inputs = result

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        effective_chat_template = (
            request.chat_template
            if (request.chat_template and self.trust_request_chat_template)
            else self.chat_template
        )
        gen_suffix = self._prefix_cache_boundaries._generation_prompt(
            tokenizer, effective_chat_template, chat_template_kwargs
        )
        im_end_id: int | None = None
        nl_ids: tuple[int, ...] | None = None
        try:
            im_end_id = tokenizer.get_vocab().get("<|im_end|>")
            nl_ids = tuple(
                _token_list(tokenizer.encode("\n", add_special_tokens=False))
            )
        except Exception:
            pass
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, engine_input in enumerate(engine_inputs):
            prompt_token_ids = self._extract_prompt_components(engine_input).token_ids
            if (
                prompt_token_ids
                and isinstance(engine_input, dict)
                and "prompt_token_ids" in engine_input
                and not engine_input.get("prompt_embeds")
            ):
                spliced = self._assistant_reasoning_cache.splice_prompt_tokens(
                    request.messages,
                    list(prompt_token_ids),
                    tokenizer,
                    cache_salt=request.cache_salt,
                    max_model_len=max_model_len,
                    mm_placeholders=engine_input.get("mm_placeholders"),
                )
                if spliced != list(prompt_token_ids):
                    cast(dict[str, Any], engine_input)["prompt_token_ids"] = spliced
                    prompt_token_ids = spliced

            # If we are creating sub requests for multiple prompts, ensure that they
            # have unique request ids.
            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_input),
                effective_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )

            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, effective_sampling_params
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    effective_sampling_params,
                )
                boundaries = self._prefix_cache_boundaries(
                    tokenizer,
                    request,
                    list(prompt_token_ids) if prompt_token_ids else None,
                    effective_chat_template,
                    chat_template_kwargs,
                )
                if boundaries:
                    sampling_params.extra_args = {
                        **(sampling_params.extra_args or {}),
                        "prefix_cache_boundaries": boundaries,
                    }

            self._log_inputs(
                sub_request_id,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_input,
                    request_id=sub_request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                if request._grammar_from_tool_parser:
                    # The Mistral grammar already includes an optional
                    # `think?` rule that handles both reasoning and
                    # non-reasoning outputs.
                    reasoning_ended = True
                elif reasoning_parser:
                    reasoning_ended = reasoning_parser.is_reasoning_end(
                        prompt_token_ids or []
                    )
                else:
                    reasoning_ended = None

                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    keepalive=bool(request.stream),
                    priority=request.priority,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=reasoning_ended,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": parser_chat_kwargs,
                    }
                    if reasoning_parser
                    else None,
                )

            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
                chat_template_kwargs=parser_chat_kwargs,
                gen_suffix=gen_suffix,
                im_end_id=im_end_id,
                nl_ids=nl_ids,
            )

        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            reasoning_parser,
            gen_suffix=gen_suffix,
            im_end_id=im_end_id,
            nl_ids=nl_ids,
        )

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        gen_suffix: tuple[int, ...] | None = None,
        im_end_id: int | None = None,
        nl_ids: tuple[int, ...] | None = None,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        generated_ids_per_choice: list[list[int]] = [[] for _ in range(num_choices)]
        sent_tool_calls: list[dict[int, StreamedToolCall]] = [
            {} for _ in range(num_choices)
        ]
        num_prompt_tokens = 0
        num_cached_tokens = None
        if self.use_harmony:
            harmony_parsers = [
                get_streamable_parser_for_assistant() for _ in range(num_choices)
            ]
            harmony_tools_streamed = [False] * num_choices
        tools_streamed = [False] * num_choices

        is_mistral_grammar_path = request._grammar_from_tool_parser

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request)
        )

        all_previous_token_ids: list[list[int]] | None
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Always track previous_texts for comprehensive output logging
        previous_texts = [""] * num_choices

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if (
            is_mistral_grammar_path
            or tool_choice_auto
            or tool_choice_function_name
            or request.tool_choice == "required"
            or reasoning_parser
        ):
            all_previous_token_ids = [[] for _ in range(num_choices)]
            reasoning_end_arr = [False] * num_choices
            prompt_is_reasoning_end_arr: list[bool | None] = [None] * num_choices
        else:
            all_previous_token_ids = None

        try:
            if self.parser_cls is not None:
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer not available when `skip_tokenizer_init=True`"
                    )
                parsers: list[Parser | None] = [
                    self.parser_cls(
                        tokenizer,
                        request.tools,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    for _ in range(num_choices)
                ]
                for p in parsers:
                    if p is not None:
                        p._stream_state.tool_call_id_type = self.tool_call_id_type
                        p._stream_state.history_tool_call_cnt = history_tool_call_cnt
            else:
                parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in parser creation.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        # Kept so the final usage can carry the engine's per-request breakdown.
        # The stats object is attached to every output, but only the finished
        # one holds a complete set of timestamps.
        last_metrics = None
        last_prompt_token_ids: list[int] | None = None
        stream_cache_recorded = False

        def _record_stream_cache() -> None:
            nonlocal stream_cache_recorded
            if stream_cache_recorded or not previous_texts:
                return
            stream_cache_recorded = True
            for i, full_raw_text in enumerate(previous_texts):
                if not full_raw_text:
                    continue
                if reasoning_parser is not None:
                    raw_reasoning, raw_content = reasoning_parser.extract_reasoning(
                        full_raw_text, request=request
                    )
                else:
                    raw_reasoning, raw_content = None, full_raw_text
                streamed_tcs = [
                    (tc.name, "".join(tc.arguments))
                    for _, tc in sorted(sent_tool_calls[i].items())
                    if tc.name
                ]
                visible_content = raw_content or ""
                choice_parser = parsers[i] if i < len(parsers) else None
                choice_tool_parser = (
                    choice_parser.tool_parser if choice_parser is not None else None
                )
                if streamed_tcs and (
                    tool_choice_function_name or request.tool_choice == "required"
                ):
                    visible_content = ""
                elif (
                    streamed_tcs and visible_content and choice_tool_parser is not None
                ):
                    extracted = choice_tool_parser.extract_tool_calls(
                        visible_content, request=request
                    )
                    if extracted.tools_called:
                        visible_content = extracted.content or ""
                reasoning_done = (
                    True
                    if reasoning_parser is None
                    else (bool(raw_content) or bool(streamed_tcs))
                )
                rcache = getattr(self, "_assistant_reasoning_cache", None)
                if rcache is not None:
                    rcache.record(
                        request.messages,
                        visible_content,
                        streamed_tcs,
                        raw_reasoning,
                        cache_salt=request.cache_salt,
                        reasoning_ended=reasoning_done,
                        prompt_token_ids=last_prompt_token_ids,
                        output_token_ids=(
                            generated_ids_per_choice[i]
                            if i < len(generated_ids_per_choice)
                            else None
                        ),
                        gen_suffix=gen_suffix,
                        im_end_id=im_end_id,
                        nl_ids=nl_ids,
                    )

        try:
            async for res in result_generator:
                if getattr(res, "metrics", None) is not None:
                    last_metrics = res.metrics

                if res.prompt_token_ids is not None:
                    last_prompt_token_ids = as_list(res.prompt_token_ids)
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                if res is STREAM_KEEPALIVE:
                    yield ": keepalive\n\n"
                    continue

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # ``res.prompt`` is the rendered chat-templated prompt
                    prompt_text = res.prompt if request.return_prompt_text else None

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(
                                res.prompt_token_ids
                                if request.return_token_ids
                                else None
                            ),
                            prompt_text=prompt_text,
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )

                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    self._raise_if_error(output.finish_reason, request_id)
                    parser = parsers[i]
                    tool_parser = parser.tool_parser if parser is not None else None

                    if (
                        reasoning_parser
                        and res.prompt_token_ids
                        and prompt_is_reasoning_end_arr[i] is None
                    ):
                        # only check once per choice, because prompt_token_ids
                        # are the same for all deltas in that choice
                        prompt_is_reasoning_end_arr[i] = (
                            reasoning_parser.is_reasoning_end(res.prompt_token_ids)
                        )
                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs:
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs or 0,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    if self.use_harmony:
                        harmony_parser = harmony_parsers[i]
                        prev_recipient = harmony_parser.current_recipient

                        # Track accumulated content per token with their state
                        token_states: list[TokenState] = []
                        for token_id in output.token_ids:
                            harmony_parser.process(token_id)
                            token_delta = harmony_parser.last_content_delta or ""
                            token_states.append(
                                TokenState(
                                    harmony_parser.current_channel,
                                    harmony_parser.current_recipient,
                                    token_delta,
                                )
                            )
                        delta_text = "".join(delta for _, _, delta in token_states)
                        cur_channel = harmony_parser.current_channel

                        # handle the case where several tokens where generated at once
                        # including the final token, leading to a delta in the text
                        # but the current channel to be empty (start state)
                        if not cur_channel and delta_text:
                            cur_channel = "final"
                    else:
                        delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    # just update previous_texts and previous_token_ids
                    if (
                        is_mistral_grammar_path
                        or tool_choice_auto
                        or tool_choice_function_name
                        or request.tool_choice == "required"
                        or reasoning_parser
                    ):
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        # avoid the None + list error.
                        if previous_token_ids:
                            current_token_ids = previous_token_ids + as_list(
                                output.token_ids
                            )
                        else:
                            current_token_ids = as_list(output.token_ids)

                    if self.use_harmony:
                        delta_message, tools_streamed_flag = (
                            extract_harmony_streaming_delta(
                                harmony_parser=harmony_parser,
                                token_states=token_states,
                                prev_recipient=prev_recipient,
                                include_reasoning=request.include_reasoning,
                            )
                        )
                        harmony_tools_streamed[i] |= tools_streamed_flag
                    # Mistral grammar path: combined reasoning + tool streaming
                    elif is_mistral_grammar_path:
                        from vllm.tool_parsers.mistral_tool_parser import (
                            MistralToolParser,
                        )

                        assert tool_parser is not None
                        assert isinstance(tool_parser, MistralToolParser)
                        assert reasoning_end_arr is not None
                        output_token_ids = as_list(output.token_ids)
                        result = tool_parser.extract_maybe_reasoning_and_tool_streaming(
                            reasoning_parser=reasoning_parser,
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            previous_token_ids=previous_token_ids,
                            current_token_ids=current_token_ids,
                            output_token_ids=output_token_ids,
                            reasoning_ended=reasoning_end_arr[i],
                            prompt_is_reasoning_end=(prompt_is_reasoning_end_arr[i]),
                            request=request,
                        )
                        delta_message = result.delta_message
                        reasoning_end_arr[i] = result.reasoning_ended
                        current_text = result.current_text
                        current_token_ids = result.current_token_ids
                        if result.tools_called:
                            tools_streamed[i] = True

                    elif parser is not None:
                        delta_message = parser.parse_delta(
                            delta_text=delta_text,
                            delta_token_ids=as_list(output.token_ids),
                            request=request,
                            prompt_token_ids=res.prompt_token_ids,
                        )
                        if delta_message and delta_message.tool_calls:
                            tools_streamed[i] = True
                    # handle streaming just a content delta (no parsers)
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    if (
                        is_mistral_grammar_path
                        or tool_choice_auto
                        or tool_choice_function_name
                        or request.tool_choice == "required"
                        or reasoning_parser
                    ) and not self.use_harmony:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids
                    else:
                        # Update for comprehensive logging even in simple case
                        assert previous_texts is not None
                        previous_texts[i] += delta_text

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)
                    if generated_ids_per_choice is not None:
                        generated_ids_per_choice[i].extend(output.token_ids)

                    if (
                        delta_message is not None
                        and not request.include_reasoning
                        and (
                            delta_message.reasoning is not None
                            or delta_message.reasoning_content is not None
                        )
                    ):
                        delta_message.reasoning = None
                        delta_message.reasoning_content = None
                        if (
                            delta_message.content is None
                            and not delta_message.tool_calls
                            and delta_message.role is None
                        ):
                            delta_message = None

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        # NOTE: If return_token_ids is enabled, we still need to
                        # send a chunk with token_ids even if delta_message is None
                        # to ensure all tokens are included in the response
                        if (
                            output.finish_reason is None
                            and not request.return_token_ids
                            and logprobs is None
                        ):
                            continue
                        delta_message = DeltaMessage()

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content_parts = []
                        if delta_message.content:
                            delta_content_parts.append(delta_message.content)
                        if delta_message.reasoning:
                            reasoning = delta_message.reasoning
                            delta_content_parts.append(f"[reasoning: {reasoning}]")
                        if delta_message.tool_calls:
                            tool_args = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments
                            )
                            if tool_args:
                                delta_content_parts.append(f"[tool_calls: {tool_args}]")

                        if delta_content_parts and self.enable_log_deltas:
                            delta_content = " ".join(delta_content_parts)
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                    # if the model is finished generating
                    else:
                        # check for error finish reason and abort streaming
                        # finish_reason='error' indicates a retryable error
                        self._raise_if_error(output.finish_reason, request_id)

                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using structured outputs
                        index = 0
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                            index = (
                                len(tool_parser.prev_tool_call_arr) - 1
                                if auto_tools_called
                                else 0
                            )
                        flushed = False
                        if tool_parser and tool_choice_auto:
                            text = previous_texts[i]
                            if reasoning_parser is not None:
                                _, content = reasoning_parser.extract_reasoning(
                                    text, request=request
                                )
                                text = content or ""
                            delta_message, flushed = flush_tool_calls(
                                tool_parser,
                                text,
                                request,
                                delta_message,
                                sent_tool_calls[i],
                            )
                            if delta_message is None:
                                delta_message = DeltaMessage()
                            auto_tools_called = auto_tools_called or flushed
                        should_check = (
                            not flushed
                            and self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output
                            )
                        )
                        # only check if there are any tool calls
                        # detected by partial parsing
                        # Hold-back parsers that release ALL tool calls in one
                        # final delta with full args (e.g. the DeepSeek-V4.1 DSML
                        # engine parser) have already streamed every call in full.
                        # The compensation below assumes the vanilla invariant that
                        # the final delta carries only the LAST call's incremental
                        # args tail and that it is safe to REPLACE the whole delta
                        # with a single remaining-args call; given a multi-call
                        # release delta it reads tool_calls[0] (the FIRST call),
                        # trims the last call's streamed args by the wrong length,
                        # and DROPS every other call. If the parser's own
                        # bookkeeping says the last call's args were already
                        # streamed in full, there is nothing to compensate: skip.
                        if (
                            should_check
                            and tool_parser
                            and auto_tools_called
                            and 0 <= index < len(tool_parser.prev_tool_call_arr)
                            and index < len(tool_parser.streamed_args_for_tool)
                            and tool_parser.streamed_args_for_tool[index]
                            == (
                                tool_parser.prev_tool_call_arr[index].get(
                                    "arguments", {}
                                )
                                if isinstance(
                                    tool_parser.prev_tool_call_arr[index].get(
                                        "arguments", {}
                                    ),
                                    str,
                                )
                                else json.dumps(
                                    tool_parser.prev_tool_call_arr[index].get(
                                        "arguments", {}
                                    ),
                                    ensure_ascii=False,
                                )
                            )
                        ):
                            should_check = False
                        if (
                            should_check
                            and tool_parser
                            and auto_tools_called
                            and 0 <= index < len(tool_parser.prev_tool_call_arr)
                            and index < len(tool_parser.streamed_args_for_tool)
                        ):
                            latest_delta_len = 0
                            if (
                                isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall,
                                )
                            ) and isinstance(
                                delta_message.tool_calls[0].function.arguments, str
                            ):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.arguments
                                )

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON.
                            # Tool parsers (e.g. Qwen3Coder) store
                            # arguments as a JSON string in
                            # prev_tool_call_arr. Calling json.dumps()
                            # on an already-serialized string would
                            # double-serialize it (e.g. '{"k":1}' becomes
                            # '"{\\"k\\":1}"'), which then causes the
                            # replace() below to fail and append the
                            # entire double-serialized string as a
                            # spurious final delta.
                            args = tool_parser.prev_tool_call_arr[index].get(
                                "arguments", {}
                            )
                            if isinstance(args, str):
                                expected_call = args
                            else:
                                expected_call = json.dumps(args, ensure_ascii=False)

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[index]
                            if latest_delta_len > 0:
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call.replace(actual_call, "", 1)
                            # set that as a delta message
                            delta_message = self._create_remaining_args_delta(
                                delta_message, remaining_call, index
                            )

                        # Send the finish response for each request.n only once
                        # In OpenAI's API, when a tool is called, the
                        # finish_reason is:
                        # "tool_calls" for "auto" or "required" tool calls,
                        # and "stop" for named tool calls.
                        if output.finish_reason == "stop" and (
                            (auto_tools_called and not tool_choice_function_name)
                            or (tools_streamed[i] and not tool_choice_function_name)
                            or (self.use_harmony and harmony_tools_streamed[i])
                        ):
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                        finish_reason_sent[i] = True

                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    record_tool_calls(
                        choice_data.delta.tool_calls or [], sent_tool_calls[i]
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    # Stamp the fingerprint on terminal chunks only (those with
                    # finish_reason set). When ``include_usage`` is on, the
                    # trailing usage chunk below overrides this as the true
                    # final message.
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and choice_data.finish_reason is not None
                    ):
                        chunk.system_fingerprint = self.system_fingerprint

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            _record_stream_cache()

            reasoning_tokens = _count_reasoning_tokens(
                reasoning_parser,
                generated_ids_per_choice if reasoning_parser is not None else None,
            )
            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    completion_tokens_details=(
                        CompletionTokenUsageInfo(reasoning_tokens=reasoning_tokens)
                        if reasoning_tokens is not None
                        else None
                    ),
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and num_cached_tokens is not None:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens
                    )
                # Engine-authoritative per-request timings. Omitted when the
                # engine collected no stats for this request.
                final_usage.timings = get_request_usage_timings(
                    last_metrics, num_choices=num_choices
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                    system_fingerprint=self.system_fingerprint,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )
            if reasoning_tokens is not None:
                request_metadata.final_usage_info.completion_tokens_details = (
                    CompletionTokenUsageInfo(reasoning_tokens=reasoning_tokens)
                )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts)
                        else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except GenerationError as e:
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
        except Exception as e:
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        finally:
            _record_stream_cache()
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
        gen_suffix: tuple[int, ...] | None = None,
        im_end_id: int | None = None,
        nl_ids: tuple[int, ...] | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        if final_res is None:
            return self.create_error_response(
                "No output received from the engine.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        choices: list[ChatCompletionResponseChoice] = []
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            # check for error finish reason and raise GenerationError
            # finish_reason='error' indicates a retryable request-level internal error
            self._raise_if_error(output.finish_reason, request_id)
            token_ids = output.token_ids
            out_logprobs = output.logprobs
            tool_call_info = None

            if request.logprobs:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs or 0,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if self.use_harmony:
                reasoning, content, _ = parse_chat_output(token_ids)
                if not request.include_reasoning:
                    reasoning = None

                if self.tool_parser is not None:
                    if tokenizer is None:
                        raise ValueError(
                            "Tokenizer not available when `skip_tokenizer_init=True`"
                        )

                    tool_parser = self.tool_parser(tokenizer, request.tools)
                    # NOTE: We use token_ids for openai tool parser
                    tool_call_info = tool_parser.extract_tool_calls(
                        "",
                        request=request,
                        token_ids=token_ids,  # type: ignore
                    )
                    content = tool_call_info.content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_info.tool_calls,
                    )
                else:
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                    )

                # Encode routed_experts for transport. JSON can't carry raw
                # bytes, so we write the ndarray as a ``.npy`` byte stream
                # and base64-encode it. ``pybase64`` is ~3x faster than the
                # stdlib ``base64`` on large payloads thanks to SIMD.
                routed_experts_b64 = None
                if output.routed_experts is not None:
                    buf = io.BytesIO()
                    np.save(buf, output.routed_experts)
                    routed_experts_b64 = base64.b64encode(buf.getvalue()).decode(
                        "ascii"
                    )

                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    logprobs=logprobs,
                    finish_reason=(
                        "tool_calls"
                        if (
                            output.finish_reason == "stop"
                            and tool_call_info is not None
                            and tool_call_info.tools_called
                        )
                        else output.finish_reason
                        if output.finish_reason
                        else "stop"
                    ),
                    stop_reason=output.stop_reason,
                    token_ids=(
                        as_list(output.token_ids) if request.return_token_ids else None
                    ),
                    routed_experts=routed_experts_b64,
                )
                choices.append(choice_data)
                continue

            raw_reasoning: str | None = None
            if reasoning_parser:
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning, content = reasoning_parser.extract_reasoning(
                    output.text, request=request
                )
                raw_reasoning = reasoning
                if not request.include_reasoning:
                    reasoning = None
            else:
                reasoning = None
                content = output.text

            auto_tools_called = False
            # Do not auto-close truncated half-written XML tool calls when
            # generation hit max_tokens (finish_reason == "length").
            parse_content = content
            if (
                output.finish_reason == "length"
                and parse_content
                and "<tool_call>" in parse_content
            ):
                last_open = parse_content.rfind("<tool_call>")
                last_close = parse_content.rfind("</tool_call>")
                if (
                    last_open > last_close
                    and "</function>" not in parse_content[last_open:]
                ):
                    parse_content = parse_content[:last_open]
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            tool_calls, parsed_text = self._parse_tool_calls_from_content(
                request=request,
                tokenizer=tokenizer,
                content=parse_content,
                enable_auto_tools=self.enable_auto_tools,
                tool_parser_cls=self.tool_parser,
            )
            if parse_content == content or tool_calls:
                content = parsed_text
            if is_mistral_tokenizer(tokenizer):
                from vllm.tool_parsers.mistral_tool_parser import MistralToolCall

                tool_call_class: type[ToolCall] = MistralToolCall
            else:
                tool_call_class = ToolCall

            use_mistral_tool_parser = request._grammar_from_tool_parser
            if use_mistral_tool_parser:
                from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

                tool_call_items = MistralToolParser.build_non_streaming_tool_calls(
                    tool_calls
                )
                if tool_call_items:
                    auto_tools_called = (
                        request.tool_choice is None or request.tool_choice == "auto"
                    )
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_call_items,
                )

            elif (not self.enable_auto_tools or not self.tool_parser) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            elif (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                tool_call_class_items = []
                tool_calls = tool_calls or []
                for idx, tc in enumerate(tool_calls):
                    # Use native ID if available (e.g., Kimi K2),
                    # otherwise generate ID with correct id_type
                    if tc.id:
                        tool_call_class_items.append(
                            tool_call_class(id=tc.id, function=tc)
                        )
                    else:
                        # Generate ID using the correct format (kimi_k2 or random),
                        # but leave it to the class if it's Mistral to preserve
                        # 9-char IDs
                        if is_mistral_tokenizer(tokenizer):
                            tool_call_class_items.append(tool_call_class(function=tc))
                        else:
                            generated_id = make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tc.name,
                                idx=history_tool_call_cnt,
                            )
                            tool_call_class_items.append(
                                tool_call_class(id=generated_id, function=tc)
                            )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content="" if tool_call_class_items else content,
                    tool_calls=tool_call_class_items,
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class_items = []
                tool_calls = tool_calls or []
                for idx, tool_call in enumerate(tool_calls):
                    # Use native ID if available,
                    # otherwise generate ID with correct id_type
                    if tool_call.id:
                        tool_call_class_items.append(
                            tool_call_class(id=tool_call.id, function=tool_call)
                        )
                    else:
                        # Generate ID using the correct format (kimi_k2 or random),
                        # but leave it to the class if it's Mistral to preserve
                        # 9-char IDs
                        if is_mistral_tokenizer(tokenizer):
                            tool_call_class_items.append(
                                tool_call_class(function=tool_call)
                            )
                        else:
                            generated_id = make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tool_call.name,
                                idx=history_tool_call_cnt,
                            )
                            tool_call_class_items.append(
                                tool_call_class(id=generated_id, function=tool_call)
                            )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    content="" if tool_call_class_items else content,
                    tool_calls=tool_call_class_items,
                    reasoning=reasoning,
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and self.tool_parser
            ):
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_calls is not None and len(tool_calls) > 0
                if tool_calls:
                    tool_call_items = []
                    for idx, tc in enumerate(tool_calls):
                        # Use native ID if available (e.g., Kimi K2),
                        # otherwise generate ID with correct id_type
                        if tc.id:
                            tool_call_items.append(
                                tool_call_class(id=tc.id, function=tc)
                            )
                        else:
                            # Generate ID using the correct format (kimi_k2 or random),
                            # but leave it to the class if it's Mistral to preserve
                            # 9-char IDs
                            if is_mistral_tokenizer(tokenizer):
                                tool_call_items.append(tool_call_class(function=tc))
                            else:
                                generated_id = make_tool_call_id(
                                    id_type=self.tool_call_id_type,
                                    func_name=tc.name,
                                    idx=history_tool_call_cnt,
                                )
                                tool_call_items.append(
                                    tool_call_class(id=generated_id, function=tc)
                                )
                        history_tool_call_cnt += 1
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_items,
                    )

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    ret_content = content

                    # try to use content return from tool parser first,
                    # tool parser may do some modify for the content.
                    if content and len(content) > 0:
                        ret_content = content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=ret_content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = ChatMessage(role=role, reasoning=reasoning, content=content)
            reasoning_done = (
                True
                if reasoning_parser is None
                else (bool(message.content) or bool(message.tool_calls))
            )
            rcache = getattr(self, "_assistant_reasoning_cache", None)
            if rcache is not None:
                rcache.record(
                    request.messages,
                    message.content,
                    message.tool_calls,
                    raw_reasoning,
                    cache_salt=request.cache_salt,
                    reasoning_ended=reasoning_done,
                    prompt_token_ids=(
                        as_list(final_res.prompt_token_ids)
                        if final_res.prompt_token_ids is not None
                        else None
                    ),
                    output_token_ids=as_list(output.token_ids),
                    gen_suffix=gen_suffix,
                    im_end_id=im_end_id,
                    nl_ids=nl_ids,
                )
            # In OpenAI's API, when a tool is called, the finish_reason is:
            # "tool_calls" for "auto" or "required" tool calls,
            # and "stop" for named tool calls.
            is_finish_reason_tool_calls = output.finish_reason == "stop" and (
                auto_tools_called or request.tool_choice == "required"
            )

            # Encode routed_experts for transport. JSON can't carry raw
            # bytes, so we write the ndarray as a ``.npy`` byte stream
            # and base64-encode it. ``pybase64`` is ~3x faster than the
            # stdlib ``base64`` on large payloads thanks to SIMD.
            routed_experts_b64 = None
            if output.routed_experts is not None:
                buf = io.BytesIO()
                np.save(buf, output.routed_experts)
                routed_experts_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
                token_ids=(
                    as_list(output.token_ids) if request.return_token_ids else None
                ),
                routed_experts=routed_experts_b64,
            )
            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)

            choices.append(choice_data)

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if (
            self.enable_prompt_tokens_details
            and final_res.num_cached_tokens is not None
        ):
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )
        # Engine-authoritative per-request timings. Without this a
        # non-streaming caller has no prefill/decode split at all, and any
        # consumer that derives rates from a token stream sees the whole
        # request as one point in time.
        usage.timings = get_request_usage_timings(
            getattr(final_res, "metrics", None), num_choices=len(choices)
        )

        reasoning_tokens = _count_reasoning_tokens(
            reasoning_parser, [output.token_ids for output in final_res.outputs]
        )
        if reasoning_tokens is not None:
            usage.completion_tokens_details = CompletionTokenUsageInfo(
                reasoning_tokens=reasoning_tokens
            )
        request_metadata.final_usage_info = usage

        # ``final_res.prompt`` is the rendered chat-templated prompt text
        prompt_text = final_res.prompt if request.return_prompt_text else None

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            system_fingerprint=self.system_fingerprint,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            prompt_text=prompt_text,
            kv_transfer_params=final_res.kv_transfer_params,
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:  # type: ignore
                        function_call: FunctionCall = tc.function  # type: ignore
                        tool_call_descriptions.append(
                            f"{function_call.name}({function_call.arguments})"
                        )
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: TokenizerLike | None,
        should_return_as_token_id: bool,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            if (top_logprobs and i < top_logprobs or top_logprobs == -1)
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        tokenizer: TokenizerLike | None,
        num_output_top_logprobs: int | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        top_logprobs = drop_misaligned_logprobs(token_ids, top_logprobs)
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "Unable to get tokenizer because `skip_tokenizer_init=True`"
                        )

                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)

    def _should_stream_with_auto_tool_parsing(self, request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (
            request.tools
            and self.tool_parser
            and self.enable_auto_tools
            and request.tool_choice in ["auto", None]
        )

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: DeltaMessage | None,
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools
            and self.tool_parser
            and delta_message
            and delta_message.tool_calls
            and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )

    @staticmethod
    def _create_remaining_args_delta(
        delta_message: DeltaMessage,
        remaining_call: str,
        index: int,
    ) -> DeltaMessage:
        """
        Create a delta message for remaining tool arguments, preserving
        id/type/name from the original delta.
        """
        original_tc = next(
            (tc for tc in delta_message.tool_calls if tc.index == index),
            None,
        )
        original_fn = original_tc.function if original_tc else None
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    id=original_tc.id if original_tc else None,
                    type=original_tc.type if original_tc else None,
                    function=DeltaFunctionCall(
                        name=original_fn.name if original_fn else None,
                        arguments=remaining_call,
                    ),
                )
            ]
        )
