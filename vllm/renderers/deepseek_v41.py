# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse chat transport while dispatching to the independent V4.1 encoding."""

from vllm.tokenizers.deepseek_v41 import _text_only

from .deepseek_v4 import DeepseekV4Renderer


class DeepseekV41Renderer(DeepseekV4Renderer):
    @staticmethod
    def _validate_messages(messages):
        for message in messages:
            _text_only(message.get("content"))
            _text_only(message.get("content_blocks"))

    def render_messages(self, messages, params):
        self._validate_messages(messages)
        return super().render_messages(messages, params)

    async def render_messages_async(self, messages, params):
        self._validate_messages(messages)
        return await super().render_messages_async(messages, params)
