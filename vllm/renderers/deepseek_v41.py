# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse chat transport while dispatching to the independent V4.1 encoding."""

from vllm.tokenizers.deepseek_v41 import _sanitize_message, _vision

from .deepseek_v4 import DeepseekV4Renderer


class DeepseekV41Renderer(DeepseekV4Renderer):
    @staticmethod
    def _validate_messages(messages):
        images = _vision()
        sanitized = None
        for idx, message in enumerate(messages):
            cleaned = _sanitize_message(message, images)
            if cleaned is not message:
                if sanitized is None:
                    sanitized = list(messages[:idx])
                sanitized.append(cleaned)
            elif sanitized is not None:
                sanitized.append(message)
        return sanitized if sanitized is not None else messages

    def render_messages(self, messages, params):
        messages = self._validate_messages(messages)
        return super().render_messages(messages, params)

    async def render_messages_async(self, messages, params):
        messages = self._validate_messages(messages)
        return await super().render_messages_async(messages, params)
