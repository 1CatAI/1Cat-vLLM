# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 text tokenizer using the checkpoint's independent prompt encoding."""

import copy

from transformers import PreTrainedTokenizerFast

from vllm.third_party.deepseek_v41.encoding import encode_messages

from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike


def _text_only(value):
    if isinstance(value, list):
        for block in value:
            if block.get("type") not in ("text", "tool_result"):
                raise ValueError("V4.1 first profile admits text and tool results only")
            _text_only(block.get("content", block.get("text")))
    elif isinstance(value, str) and (
        "<｜deepseek_image｜>" in value or "<image>" in value
    ):
        raise ValueError("V4.1 first profile does not admit image inputs")


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    wrapped = copy.copy(tokenizer)
    added_vocab, size = tokenizer.get_added_vocab(), len(tokenizer)

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(self, messages, tools=None, **kwargs):
            conversation = copy.deepcopy(kwargs.get("conversation", messages))
            for message in conversation:
                _text_only(message.get("content"))
                _text_only(message.get("content_blocks"))
            if tools:
                conversation.insert(0, {"role": "system", "tools": tools})
            thinking = kwargs.get("thinking", False) or kwargs.get(
                "enable_thinking", False
            )
            effort = kwargs.get("reasoning_effort")
            if effort == "none":
                thinking, effort = False, None
            elif effort == "medium":
                effort = 75
            elif effort == "xhigh":
                effort = 100
            prompt = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=effort,
            )
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
