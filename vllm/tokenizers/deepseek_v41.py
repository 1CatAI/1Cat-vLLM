# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 text tokenizer using the checkpoint's independent prompt encoding."""

import copy
import threading
from collections import OrderedDict

from transformers import PreTrainedTokenizerFast

from vllm.third_party.deepseek_v41.encoding import encode_messages

from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike

_IMAGE = "<｜deepseek_image｜>"
# Neutralized form for literal occurrences in user/assistant/system text:
# encodes to ordinary text tokens (never image_token_id 129264) and cannot
# collide with placeholders inserted by vLLM for real image content blocks.
_IMAGE_ESCAPED = "<|deepseek_image|>"


def _vision() -> bool:
    import vllm.envs as envs

    return envs.VLLM_V41_VISION == "1"


def _escape_image_placeholder(text: str) -> str:
    if _IMAGE in text:
        return text.replace(_IMAGE, _IMAGE_ESCAPED)
    return text


def _text_only(value, images=False):
    """Validate content blocks and neutralize literal V4.1 image special tokens.

    Allows text and tool results (plus image blocks when ``images`` is true,
    i.e. ``VLLM_V41_VISION=1``). Ordinary ``<image>`` strings are not special
    tokens in V4.1 and pass through unchanged; literal ``<｜deepseek_image｜>``
    strings in text are escaped to ``<|deepseek_image|>`` (copy-on-write) so
    they neither trigger 400 errors on transcript/compaction prompts nor emit
    ``image_token_id`` or collide with real multimodal placeholders.
    """
    if isinstance(value, list):
        new_list = None
        for idx, block in enumerate(value):
            kind = block.get("type")
            if images and kind in ("image", "image_url"):
                if new_list is not None:
                    new_list.append(block)
                continue
            if kind not in ("text", "tool_result"):
                raise ValueError(
                    "V4.1 admits text, tool results"
                    + (" and images" if images else "")
                    + " only"
                )
            key = "content" if "content" in block else "text"
            raw = block.get(key)
            cleaned = _text_only(raw, images)
            if cleaned is not raw:
                if new_list is None:
                    new_list = list(value[:idx])
                updated = dict(block)
                updated[key] = cleaned
                new_list.append(updated)
            elif new_list is not None:
                new_list.append(block)
        return new_list if new_list is not None else value
    if isinstance(value, str):
        return _escape_image_placeholder(value)
    return value


def _sanitize_tool_calls(tool_calls):
    if not isinstance(tool_calls, list):
        return tool_calls
    new_calls = None
    for idx, call in enumerate(tool_calls):
        updated = call
        if isinstance(call, dict):
            fn = call.get("function")
            if isinstance(fn, dict):
                args = fn.get("arguments")
                if isinstance(args, str) and _IMAGE in args:
                    updated = dict(
                        call,
                        function=dict(fn, arguments=_escape_image_placeholder(args)),
                    )
        if updated is not call:
            if new_calls is None:
                new_calls = list(tool_calls[:idx])
            new_calls.append(updated)
        elif new_calls is not None:
            new_calls.append(call)
    return new_calls if new_calls is not None else tool_calls


def _sanitize_message(message, images=False):
    """Validate and copy-on-write neutralize literal image special tokens."""
    content = message.get("content")
    new_content = _text_only(content, images)
    content_blocks = message.get("content_blocks")
    new_blocks = _text_only(content_blocks, images)
    reasoning = message.get("reasoning_content")
    new_reasoning = (
        _escape_image_placeholder(reasoning)
        if isinstance(reasoning, str)
        else reasoning
    )
    tool_calls = message.get("tool_calls")
    new_tool_calls = _sanitize_tool_calls(tool_calls)
    if (
        new_content is content
        and new_blocks is content_blocks
        and new_reasoning is reasoning
        and new_tool_calls is tool_calls
    ):
        return message
    updated = dict(message)
    if new_content is not content:
        updated["content"] = new_content
    if new_blocks is not content_blocks:
        updated["content_blocks"] = new_blocks
    if new_reasoning is not reasoning:
        updated["reasoning_content"] = new_reasoning
    if new_tool_calls is not tool_calls:
        updated["tool_calls"] = new_tool_calls
    return updated


def _image_blocks(text):
    """vLLM's interleaved string (parts joined by newlines, one placeholder
    per image) -> the content blocks the V4.1 encoding expects."""
    parts = text.split(_IMAGE)
    blocks = []
    for i, part in enumerate(parts):
        if i > 0 and part.startswith("\n"):
            part = part[1:]
        if i < len(parts) - 1 and part.endswith("\n"):
            part = part[:-1]
        if part:
            blocks.append({"type": "text", "text": part})
        if i < len(parts) - 1:
            # The pixels travel as multi-modal data; the encoding only needs
            # the block (its source is not read).
            blocks.append({"type": "image", "url": "vllm-multimodal"})
    return blocks


_EOS_MARK = "<｜end▁of▁sentence｜>"
_USER_MARK = "<｜User｜>"
_TOK_CACHE_MAX_ENTRIES = 128
_TOK_CACHE_MAX_TOKENS = 16_000_000


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    wrapped = copy.copy(tokenizer)
    added_vocab, size = tokenizer.get_added_vocab(), len(tokenizer)
    has_eos = _EOS_MARK in added_vocab
    has_user = _USER_MARK in added_vocab
    tok_cache: OrderedDict[str, list[int]] = OrderedDict()
    tok_cache_tokens = [0]
    tok_lock = threading.Lock()

    def _cache_put(prefix: str, ids: list[int]) -> None:
        if len(ids) > _TOK_CACHE_MAX_TOKENS // 2:
            return
        with tok_lock:
            old = tok_cache.pop(prefix, None)
            if old is not None:
                tok_cache_tokens[0] -= len(old)
            tok_cache[prefix] = ids
            tok_cache_tokens[0] += len(ids)
            while tok_cache and (
                len(tok_cache) > _TOK_CACHE_MAX_ENTRIES
                or tok_cache_tokens[0] > _TOK_CACHE_MAX_TOKENS
            ):
                _, evicted = tok_cache.popitem(last=False)
                tok_cache_tokens[0] -= len(evicted)

    def _boundary_cuts(prompt: str) -> list[int]:
        cuts: list[int] = []
        n = len(prompt)
        if has_user:
            u0 = prompt.find(_USER_MARK)
            if 0 < u0 < n:
                cuts.append(u0)
        if has_eos:
            step = len(_EOS_MARK)
            pos = prompt.find(_EOS_MARK)
            while pos != -1:
                end = pos + step
                if (not cuts or end > cuts[-1]) and end < n:
                    cuts.append(end)
                pos = prompt.find(_EOS_MARK, end)
        return cuts

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def encode(self, text, *args, **kwargs):
            if (
                isinstance(text, str)
                and not args
                and len(kwargs) == 1
                and kwargs.get("add_special_tokens") is False
                and len(text) >= 2048
                and (has_eos or has_user)
            ):
                cuts = _boundary_cuts(text)
                if cuts:
                    candidates = list(reversed(cuts[-8:]))
                    if cuts[0] not in candidates:
                        candidates.append(cuts[0])
                    hit_cut = None
                    hit_ids = None
                    with tok_lock:
                        for c in candidates:
                            cached = tok_cache.get(text[:c])
                            if cached is not None:
                                tok_cache.move_to_end(text[:c])
                                hit_cut = c
                                hit_ids = cached
                                break
                    last_cut = cuts[-1]
                    if hit_cut is not None and hit_ids is not None:
                        if hit_cut == last_cut:
                            tail = super().encode(
                                text[last_cut:], add_special_tokens=False
                            )
                            return hit_ids + tail
                        mid = super().encode(
                            text[hit_cut:last_cut], add_special_tokens=False
                        )
                        tail = super().encode(text[last_cut:], add_special_tokens=False)
                        new_pref = hit_ids + mid
                        _cache_put(text[:last_cut], new_pref)
                        return new_pref + tail
                    first_cut = cuts[0]
                    if first_cut < last_cut:
                        p0 = super().encode(text[:first_cut], add_special_tokens=False)
                        mid = super().encode(
                            text[first_cut:last_cut], add_special_tokens=False
                        )
                        tail = super().encode(text[last_cut:], add_special_tokens=False)
                        _cache_put(text[:first_cut], p0)
                        full_pref = p0 + mid
                        _cache_put(text[:last_cut], full_pref)
                        return full_pref + tail
                    p0 = super().encode(text[:last_cut], add_special_tokens=False)
                    tail = super().encode(text[last_cut:], add_special_tokens=False)
                    _cache_put(text[:last_cut], p0)
                    return p0 + tail
            return super().encode(text, *args, **kwargs)

        def apply_chat_template(self, messages, tools=None, **kwargs):
            raw_conv = kwargs.get("conversation", messages)
            conversation = []
            if tools:
                conversation.append({"role": "system", "tools": tools})
            for orig in raw_conv:
                message = orig
                content = message.get("content")
                if (
                    message.get("role") == "assistant"
                    and message.get("tool_calls")
                    and isinstance(content, str)
                    and content.endswith("\n\n")
                ):
                    content = content[:-2]
                    message = dict(message, content=content)
                if _vision() and isinstance(content, str) and _IMAGE in content:
                    # Placeholders inserted by vLLM for the request's images
                    # (any literal placeholders in user text were neutralized
                    # beforehand by DeepseekV41Renderer).
                    reasoning = message.get("reasoning_content")
                    new_reasoning = (
                        _escape_image_placeholder(reasoning)
                        if isinstance(reasoning, str)
                        else reasoning
                    )
                    tool_calls = message.get("tool_calls")
                    new_tool_calls = _sanitize_tool_calls(tool_calls)
                    message = dict(message, content=_image_blocks(content))
                    if new_reasoning is not reasoning:
                        message["reasoning_content"] = new_reasoning
                    if new_tool_calls is not tool_calls:
                        message["tool_calls"] = new_tool_calls
                    conversation.append(message)
                    continue
                message = _sanitize_message(message, images=False)
                conversation.append(message)
            thinking = kwargs.get("thinking", False) or kwargs.get(
                "enable_thinking", False
            )
            effort = kwargs.get("reasoning_effort")
            if effort == "none":
                thinking, effort = False, None
            elif effort == "minimal":
                # OpenAI clients send "minimal"; V4.1's lightest named level.
                effort = "low"
            elif effort == "medium":
                effort = 75
            elif effort == "xhigh":
                effort = 100
            try:
                prompt = encode_messages(
                    conversation,
                    thinking_mode="thinking" if thinking else "chat",
                    drop_thinking=kwargs.get("drop_thinking", True),
                    reasoning_effort=effort,
                )
            except AssertionError as e:
                # The encoding validates with asserts: a bad request, not a 500.
                raise ValueError(str(e)) from e
            if not kwargs.get("tokenize", True):
                return prompt
            options = {
                k: kwargs[k] for k in ("truncation", "max_length") if k in kwargs
            }
            return self.encode(prompt, add_special_tokens=False, **options)

        def num_special_tokens_to_add(self):
            return len(self.encode(""))

        def __len__(self):
            return size

        def get_added_vocab(self):
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
