# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Qwen3 model uses <think>...</think> tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Qwen3.5 chat templates open the <think> block in the prompt when
        # thinking is enabled, so completion tokens may only contain </think>.
        self.prompt_has_open_think = bool(chat_kwargs.get("enable_thinking", False))

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        Qwen3 has stricter requirements - it needs both start and end tokens
        to be present, unlike other models that work with just the end token.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        if self.prompt_has_open_think:
            if self.start_token in model_output:
                model_output = model_output.partition(self.start_token)[2]
            if self.end_token not in model_output:
                # Generation stopped inside reasoning. Keep reasoning out of the
                # assistant content channel instead of returning it as content.
                return (model_output or None), None

            reasoning, _, content = model_output.partition(self.end_token)
            return (reasoning or None), (content or None)

        # No prompt-open think block: only parse reasoning if the completion
        # explicitly contains a complete <think>...</think> segment.
        if self.start_token not in model_output or self.end_token not in model_output:
            return None, model_output

        model_output = model_output.partition(self.start_token)[2]
        reasoning, _, content = model_output.partition(self.end_token)
        return (reasoning or None), (content or None)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if not self.prompt_has_open_think:
            return super().extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        # The completion starts inside reasoning; the opening token lives in the
        # prompt. Keep routing deltas to `reasoning` until </think> appears.
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.end_token_id:
            return None

        if self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)

        if self.end_token_id in delta_token_ids:
            end_index = delta_text.find(self.end_token)
            if end_index == -1:
                return DeltaMessage(reasoning=delta_text)
            reasoning = delta_text[:end_index] or None
            content = delta_text[end_index + len(self.end_token) :] or None
            return DeltaMessage(reasoning=reasoning, content=content)

        return DeltaMessage(reasoning=delta_text)
