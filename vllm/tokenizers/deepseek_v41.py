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


def _vision() -> bool:
    import vllm.envs as envs

    return envs.VLLM_V41_VISION == "1"


def _text_only(value, images=False):
    """Text and tool results; image blocks too when `images` (VLLM_V41_VISION).
    Placeholder tokens typed into text are always refused."""
    if isinstance(value, list):
        for block in value:
            kind = block.get("type")
            if images and kind in ("image", "image_url"):
                continue
            if kind not in ("text", "tool_result"):
                raise ValueError(
                    "V4.1 admits text, tool results"
                    + (" and images" if images else "")
                    + " only"
                )
            _text_only(block.get("content", block.get("text")), images)
    elif isinstance(value, str) and (_IMAGE in value or "<image>" in value):
        raise ValueError("V4.1 does not admit image placeholders in text")


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
                    # Placeholders inserted by vLLM for the request's images.
                    message = dict(message, content=_image_blocks(content))
                    conversation.append(message)
                    continue
                _text_only(content)
                _text_only(message.get("content_blocks"))
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
