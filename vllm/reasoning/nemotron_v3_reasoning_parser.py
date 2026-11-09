# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class NemotronV3ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Nemotron V3 models.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # The serving layer constructs parsers with the *effective* template
        # kwargs, i.e. the request's values merged over
        # `--default-chat-template-kwargs`. Keep them: a server started with
        # e.g. `--default-chat-template-kwargs '{"enable_thinking": false}'`
        # must have that default honoured here too, not only when a request
        # spells it out.
        self._chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}

    def _resolved_switches(self, request) -> dict:
        """Template kwargs with server defaults applied.

        Prefer the construction-time values (already merged upstream over the
        server defaults). Fall back to the request when the parser was built
        without them, keeping the old behaviour for that path.
        """
        merged = dict(self._chat_template_kwargs)
        request_kwargs = getattr(request, "chat_template_kwargs", None)
        if request_kwargs:
            merged.update(request_kwargs)
        return merged

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        reasoning, final_content = super().extract_reasoning(model_output, request)
        chat_template_kwargs = self._resolved_switches(request)

        if (
            (
                chat_template_kwargs.get("enable_thinking") is False
                or chat_template_kwargs.get("force_nonempty_content") is True
            )
            and (final_content is None or not final_content.strip())
        ):
            reasoning, final_content = final_content, reasoning

        return reasoning, final_content
