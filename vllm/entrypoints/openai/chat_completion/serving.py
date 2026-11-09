# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import io
import json
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Final

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
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    FunctionCall,
    CompletionTokenUsageInfo,
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
from vllm.outputs import (
    STREAM_KEEPALIVE,
    CompletionOutput,
    RequestOutput,
)
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


def _count_reasoning_tokens(
    reasoning_parser: "ReasoningParser | None",
    token_ids_per_choice: Iterable[GenericSequence[int]] | None,
) -> int | None:
    """Count reasoning tokens among the generated ids.

    Returns None when there is nothing to measure — no reasoning parser, or no
    ids retained for any choice. The caller then omits the field instead of
    reporting zero, because a zero and "we did not look" mean different things
    to whoever is reading the number.
    """
    if reasoning_parser is None or token_ids_per_choice is None:
        return None
    total = 0
    measured = False
    for token_ids in token_ids_per_choice:
        if not token_ids:
            continue
        total += reasoning_parser.count_reasoning_tokens(token_ids)
        measured = True
    return total if measured else None


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
        self._generation: dict[str, tuple[int, ...] | None] = {}
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
    def _render(tokenizer, messages, chat_template, kwargs, tools, generation):
        kwargs = {
            k: v for k, v in kwargs.items() if k not in _PrefixCacheBoundaries._OWN_ARGS
        }
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
        if key not in self._generation:
            probe = [{"role": "user", "content": "x"}]
            with_prompt = self._render(
                tokenizer, probe, chat_template, kwargs, None, True
            )
            without = self._render(tokenizer, probe, chat_template, kwargs, None, False)
            suffix = None
            if (
                with_prompt is not None
                and without is not None
                and len(with_prompt) > len(without)
                and with_prompt[: len(without)] == without
            ):
                suffix = tuple(with_prompt[len(without) :])
            self._generation[key] = suffix
        return self._generation[key]

    def _system_prompt(self, tokenizer, message, chat_template, kwargs, tools):
        key = json.dumps(
            [message, tools, chat_template, kwargs], sort_keys=True, default=str
        )
        if key in self._system:
            self._system.move_to_end(key)
            return self._system[key]
        tokens = self._render(tokenizer, [message], chat_template, kwargs, tools, False)
        value = tuple(tokens) if tokens else None
        self._system[key] = value
        if len(self._system) > self._max_entries:
            self._system.popitem(last=False)
        return value

    def __call__(
        self,
        tokenizer,
        request: ChatCompletionRequest,
        prompt_token_ids: list[int] | None,
        chat_template: str | None,
        kwargs: dict[str, Any],
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
        messages = request.messages
        first = messages[0] if messages else None
        if (
            len(messages) > 1
            and isinstance(first, dict)
            and first.get("role") in ("system", "developer")
            and isinstance(first.get("content"), str)
        ):
            tools = (
                [tool.model_dump(exclude_none=True) for tool in request.tools]
                if request.tools
                else None
            )
            system = self._system_prompt(
                tokenizer,
                {"role": first["role"], "content": first["content"]},
                chat_template,
                kwargs,
                tools,
            )
            if (
                system
                and len(system) < n
                and tuple(prompt_token_ids[: len(system)]) == system
            ):
                boundaries.append(len(system))
        return sorted(set(boundaries))


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
        # With tool_choice="none", do not hand the tools to the template.
        #
        # The model cannot see tool_choice at all -- upstream measurements show
        # the first 12 generated token ids are identical for auto and none. So
        # as long as the template carries tools, the model writes a call body
        # even when the user asked it not to, and the engine emits it as plain
        # text: the client receives an unparsed `<tool_call>...</tool_call>`
        # blob.
        # With no tools passed at all, the model instead produces a useful
        # ```bash ...``` command example.
        #
        # Only active behind --exclude-tools-when-tool-choice-none.
        if (
            self.exclude_tools_when_tool_choice_none
            and request.tool_choice == "none"
            and request.tools
        ):
            request = request.model_copy(update={"tools": None, "tool_choice": None})
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
        reasoning_parser: ReasoningParser | None = None
        if self.reasoning_parser_cls:
            reasoning_parser = self.reasoning_parser_cls(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
            )
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
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, engine_input in enumerate(engine_inputs):
            prompt_token_ids = self._extract_prompt_components(engine_input).token_ids

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
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )

            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )
                boundaries = self._prefix_cache_boundaries(
                    tokenizer,
                    request,
                    list(prompt_token_ids) if prompt_token_ids else None,
                    request.chat_template
                    if (request.chat_template and self.trust_request_chat_template)
                    else self.chat_template,
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
                if not request.include_reasoning:
                    reasoning_ended = True
                elif request._grammar_from_tool_parser:
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
                    priority=request.priority,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=reasoning_ended,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": chat_template_kwargs,
                    }
                    if reasoning_parser
                    else None,
                    # Only streaming requests enable the idle keep-alive
                    # sentinel -- only the chat streaming generator knows how
                    # to recognise and consume it.
                    keepalive=bool(request.stream),
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
                chat_template_kwargs=chat_template_kwargs,
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
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        # Accumulate each choice's generated token ids so reasoning_tokens
        # can be counted at finish time.
        generated_ids_per_choice: list[list[int]] = [[] for _ in range(num_choices)]
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

        # Two-sided keep-alive:
        #   the engine emits a STREAM_KEEPALIVE sentinel after 15s of idleness;
        #   the API layer, on receiving one, writes an SSE comment frame
        #   (`: keepalive`) to the client if nothing has actually been sent for
        #   >= 120s. A comment frame carries no message and does not affect
        #   client parsing, but it shows intermediaries that the connection is
        #   still alive, so long thinking does not get cut off.
        last_server_send_time = time.time()
        try:
            async for res in result_generator:
                if res is STREAM_KEEPALIVE:
                    now = time.time()
                    if now - last_server_send_time >= 120:
                        yield ": keepalive\n\n"
                        last_server_send_time = now
                    continue

                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

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
                    # Accumulate generated ids for the reasoning count. This
                    # must run on **every** frame, not inside the finish branch,
                    # or only the last frame gets accumulated.
                    # output.token_ids is incremental.
                    if 0 <= i < len(generated_ids_per_choice):
                        generated_ids_per_choice[i].extend(
                            as_list(output.token_ids) or []
                        )

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
                        # On EOS, re-parse the whole text through the
                        # **non-streaming** path and emit whatever tail the
                        # stream missed.
                        # Division of labour with `_remaining_tool_args` below:
                        #   flush     -- source is `extract_tool_calls(full text)`,
                        #                so the judgement is complete
                        #   remaining -- source is prev_tool_call_arr (partial),
                        #                the same source the stream used
                        # Measured on a 3-level nested structure: streaming and
                        # partial both saw 99B, non-streaming 157B.
                        # flush advances streamed_args_for_tool, so the
                        # subsequent `_remaining_tool_args` computes an empty
                        # string and does not re-emit.
                        _flush_done = False
                        if tool_parser and hasattr(
                            tool_parser, "flush_streaming_finish"
                        ):
                            try:
                                flush_msg = tool_parser.flush_streaming_finish()
                            except Exception:
                                flush_msg = None
                            if flush_msg and flush_msg.tool_calls:
                                # flush already advanced
                                # streamed_args_for_tool inside the parser, so
                                # only merge the delta into this frame.
                                delta_message = self._merge_flush_delta(
                                    delta_message, flush_msg
                                )
                                # If flush already covered it, skip the
                                # partial re-emission below:
                                # streamed_args_for_tool now holds the
                                # non-streaming complete value while
                                # expected_call comes from the partial path and
                                # is shorter, so the prefix test no longer
                                # holds and tolerance alignment would compute a
                                # wrong delta.
                                _flush_done = True

                        # With thinking enabled, calls to tools outside
                        # `_STREAMING_TOOLS` (Bash/Grep/Glob/...) are
                        # **deliberately deferred** by the parser
                        # (`_should_defer_parsing` returns True and sets
                        # `_deferred_used=True`) so that tag-shaped text quoted
                        # inside the thinking block is not mistaken for a real
                        # call. Those calls must be emitted at **finish time**
                        # from the complete text. Without this:
                        #   a client that sends no `chat_template_kwargs`
                        #   => `_thinking_enabled_of()` defaults to True
                        #   => every non-whitelisted tool is deferred
                        #   => nothing emits them at finish
                        #   => **the tool calls silently vanish**
                        #      (finish_reason=stop, empty content), which the
                        #      user experiences as "the model said it would run
                        #      something and nothing happened".
                        #
                        # Note that `_thinking_enabled_of` reads the **raw**
                        # request kwargs while template rendering uses the
                        # merged `with_defaults` value. Deployments that set
                        # `--default-chat-template-kwargs {"enable_thinking":
                        # false}` make the two disagree, and this emission is
                        # then the only fallback.
                        if (
                            tool_parser
                            and previous_texts is not None
                            and reasoning_parser is not None
                            and (
                                getattr(tool_parser, "_thinking_enabled", False)
                                or getattr(tool_parser, "_coalesce_mode", False)
                            )
                            and getattr(tool_parser, "_deferred_used", False)
                        ):
                            # Capture how many calls were already streamed so
                            # the deferred ones continue the numbering. This
                            # must be read here: the `extract_tool_calls_deferred`
                            # that follows resets current_tool_index inside
                            # `_reset_streaming_state()`.
                            _streamed_base = getattr(tool_parser, "current_tool_index", 0)
                            _, _deferred_content = reasoning_parser.extract_reasoning(
                                previous_texts[i], request=request
                            )
                            _deferred_info = tool_parser.extract_tool_calls_deferred(
                                _deferred_content or "", request
                            )
                            if _deferred_info.tools_called:
                                if delta_message is None:
                                    delta_message = DeltaMessage()
                                _base = max(
                                    _streamed_base,
                                    len(delta_message.tool_calls or []),
                                )
                                # Prefer the parser's recorded **original
                                # index**; `_base + _di` is only a fallback.
                                # Renumbering lets a client merge two calls'
                                # arguments under one index -> `Extra data`.
                                _deferred_indices = (
                                    getattr(tool_parser, "_deferred_indices", None) or []
                                )
                                delta_message.tool_calls = list(
                                    delta_message.tool_calls or []
                                ) + [
                                    DeltaToolCall(
                                        index=(
                                            _deferred_indices[_di]
                                            if _di < len(_deferred_indices)
                                            else _base + _di
                                        ),
                                        id=_tc.id,
                                        type="function",
                                        function=DeltaFunctionCall(
                                            name=_tc.function.name,
                                            arguments=_tc.function.arguments,
                                        ),
                                    )
                                    for _di, _tc in enumerate(_deferred_info.tool_calls)
                                ]
                                auto_tools_called = True
                                _flush_done = True
                        should_check = (
                            not _flush_done
                            and self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output
                            )
                        )
                        # only check if there are any tool calls
                        # detected by partial parsing
                        # Bounds guard.
                        # `index` was computed **above** from the length of
                        # `prev_tool_call_arr` at that moment. But
                        # `flush_streaming_finish()` and
                        # `extract_tool_calls_deferred()` may have run in
                        # between, and both call `_reset_streaming_state()`,
                        # which clears and rebuilds `prev_tool_call_arr` and
                        # `streamed_args_for_tool`. If they decided nothing
                        # needed emitting (`_flush_done` still False), control
                        # reaches here with an out-of-range index:
                        #   IndexError: list index out of range
                        # That exception is wrapped into an `{"error": {...}}`
                        # SSE frame, which the Anthropic layer feeds to
                        # `ChatCompletionStreamResponse.model_validate_json`
                        # -> ValidationError: `model`/`choices` Field required
                        # -> a 500 for the whole turn and a lost tool call.
                        # Re-check against the **current** length: if the list
                        # is too short there is no accounted call, so skip.
                        _pca = getattr(tool_parser, "prev_tool_call_arr", ()) or ()
                        _sat = getattr(tool_parser, "streamed_args_for_tool", ()) or ()
                        if (
                            should_check
                            and tool_parser
                            and auto_tools_called
                            and 0 <= index < len(_pca)
                            and 0 <= index < len(_sat)
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
                            # The subtraction must not go out of bounds.
                            # `actual_call[:-N]` **silently yields an empty
                            # string** when N > len(actual), a Python negative-
                            # slice property. Observed when the finish frame
                            # already carried a flush-emitted delta (1231B)
                            # while the streaming side had accumulated only 30B:
                            # the subtraction reduced 30B to "" -> "the client
                            # received nothing" -> remaining degraded to the
                            # full text -> duplicated head, invalid JSON.
                            # Going out of bounds means that delta was the
                            # finish-time emission, not new in this frame, so
                            # merge it into actual and use prefix alignment.
                            if latest_delta_len > 0:
                                cut = len(actual_call) - latest_delta_len
                                if cut >= 0:
                                    actual_call = actual_call[:cut]
                                else:
                                    flush_delta = (
                                        delta_message.tool_calls[0].function.arguments
                                        or ""
                                    )
                                    actual_call = actual_call + flush_delta

                            # check to see if there's anything left to stream
                            # Prefix alignment replaces
                            # `expected_call.replace(actual_call, "", 1)`, which
                            # matches nothing when actual is not a substring and
                            # then returns the whole text -> duplicated args.
                            remaining_call = self._remaining_tool_args(
                                expected_call, actual_call
                            )
                            # set that as a delta message
                            delta_message = self._create_remaining_args_delta(
                                delta_message, remaining_call, index
                            )
                            if remaining_call:
                                # Idempotence: advance "already sent" to the
                                # complete value so later finish outputs do not
                                # emit the same tail again.
                                self._advance_streamed_args(
                                    tool_parser, index, expected_call
                                )

                        # Send the finish response for each request.n only once
                        # In OpenAI's API, when a tool is called, the
                        # finish_reason is:
                        # "tool_calls" for "auto" or "required" tool calls,
                        # and "stop" for named tool calls.
                        if output.finish_reason == "stop" and (
                            auto_tools_called
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

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens
                    )
                # Split reasoning from content for the token counts.
                _rt = _count_reasoning_tokens(
                    reasoning_parser, generated_ids_per_choice
                )
                if _rt is not None:
                    final_usage.completion_tokens_details = (
                        CompletionTokenUsageInfo(reasoning_tokens=_rt)
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
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                # Skip the idle keep-alive sentinel; it carries no output.
                if res is STREAM_KEEPALIVE:
                    continue
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

            if reasoning_parser:
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning, content = reasoning_parser.extract_reasoning(
                    output.text, request=request
                )
                if not request.include_reasoning:
                    reasoning = None
            else:
                reasoning = None
                content = output.text

            auto_tools_called = False
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            tool_calls, content = self._parse_tool_calls_from_content(
                request=request,
                tokenizer=tokenizer,
                content=content,
                enable_auto_tools=self.enable_auto_tools,
                tool_parser_cls=self.tool_parser,
            )
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
                    content="",
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
                    content="",
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
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )
        # Split reasoning from content for the token counts. Non-streaming
        # output.token_ids is the **complete** id list, not a delta.
        _rt = _count_reasoning_tokens(
            reasoning_parser,
            [as_list(o.token_ids) or [] for o in final_res.outputs],
        )
        if _rt is not None:
            usage.completion_tokens_details = CompletionTokenUsageInfo(
                reasoning_tokens=_rt
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
    def _coverage_end(actual_call: str, expected_call: str) -> int | None:
        """Estimate how far ``actual`` covers ``expected``; None if uncertain.

        ``actual`` is usually a prefix of ``expected`` because the stream emits
        incrementally, but it can also be an **approximate** prefix: the
        partial path appends a trailing newline to a parameter value and
        truncates the tail, which defeats both ``startswith`` and ``replace``.

        Approach: take a slice of ``actual``'s tail as a probe, locate it in
        ``expected``, and treat the probe's end as the coverage point. Several
        probe lengths keep this robust, and a bound on (end - len(actual))
        rejects false matches. Returns None when every probe fails, so the
        caller conservatively emits nothing.
        """
        n = len(actual_call)
        if n == 0:
            return 0
        tol = max(16, n // 100)
        for probe in (128, 96, 64, 32, 16):
            if n < probe:
                continue
            pos = expected_call.rfind(actual_call[-probe:])
            if pos < 0:
                continue
            end = pos + probe
            if end <= len(expected_call) and abs(end - n) <= tol:
                return end
        return None

    @staticmethod
    def _merge_flush_delta(
        delta_message: DeltaMessage | None, flush_msg: DeltaMessage
    ) -> DeltaMessage:
        """Merge the EOS tail (``flush_streaming_finish``) into this frame's delta.

        The flush delta is the tail the stream missed, and the client has to
        receive it or the tool arguments stay incomplete. Appended by index so
        it lands on the right call.
        """
        if delta_message is None or not delta_message.tool_calls:
            return flush_msg
        by_index = {tc.index: tc for tc in delta_message.tool_calls}
        for ftc in flush_msg.tool_calls:
            cur = by_index.get(ftc.index)
            if cur is None:
                delta_message.tool_calls.append(ftc)
                continue
            if ftc.function and ftc.function.arguments:
                if cur.function is None:
                    cur.function = ftc.function
                else:
                    cur.function.arguments = (
                        cur.function.arguments or ""
                    ) + ftc.function.arguments
        return delta_message

    @staticmethod
    def _remaining_tool_args(expected_call: str, actual_call: str) -> str:
        """Return the argument tail that still needs to be sent.

        Replaces `expected_call.replace(actual_call, "", 1)`, which had two
        defects:
        1. When actual is not a substring of expected, `replace` matches
           nothing and returns the **whole** of expected -> the client receives
           duplicated arguments -> JSON parse failure.
        2. Even when it matches, an actual that appears mid-string rather than
           as a prefix leaves a malformed splice.

        Now: a true prefix emits the tail; an approximate prefix uses the
        coverage point; when neither applies nothing is emitted, so duplication
        cannot happen.
        """
        if actual_call and expected_call.startswith(actual_call):
            return expected_call[len(actual_call) :]
        n = len(actual_call)
        if n == 0:
            return expected_call
        tol = max(16, n // 100)
        # Binary-search the largest k such that act[:k] is a proper prefix of
        # exp (no guessing, O(n log n)).
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if expected_call.startswith(actual_call[:mid]):
                lo = mid
            else:
                hi = mid - 1
        k = lo
        # Only accept the "short extra tail" shape (the partial path appends
        # 1-2B in practice; 4B is the upper bound). A mid-string mismatch
        # leaves act[k:] a dozen bytes long, and mistaking that for alignment
        # would emit in the wrong place.
        if k >= n - tol and k > 0 and (n - k) <= 4:
            overflow = actual_call[k:].rstrip()
            head_raw = expected_call[k:]
            head = head_raw.lstrip()
            if (
                overflow
                and head
                and overflow[-1] == head[0]
                and overflow[-1] in '"}]:,'
            ):
                # Overlap: the client already has this closer, so skip that
                # position in expected.
                skip = (len(head_raw) - len(head)) + 1
                return expected_call[k + skip :]
            return expected_call[k:]
        end = OpenAIServingChat._coverage_end(actual_call, expected_call)
        if end is None:
            return ""
        return expected_call[end:]

    @staticmethod
    def _advance_streamed_args(tool_parser, index: int, expected_call: str) -> None:
        """Advance "already sent" to the complete value after emitting (idempotence).

        Upstream emits the delta without updating state, and the emission
        trigger is only `finish_reason is not None`. That condition holds for
        several consecutive outputs, so this block is entered repeatedly and
        the same remaining tail gets emitted again and again.
        """
        try:
            if index < len(tool_parser.streamed_args_for_tool):
                tool_parser.streamed_args_for_tool[index] = expected_call
        except Exception:
            pass

    # Without `@staticmethod`, calling
    # `self._create_remaining_args_delta(a, b, c)` counts `self` as the first
    # positional argument -> TypeError ("takes 3 positional arguments but 4
    # were given") -> the streaming generator raises and aborts, so the finish
    # frame is never sent and the argument tail is never emitted. The client
    # then holds truncated, invalid JSON (observed `{"path": "/tmp/1.txt"`
    # with no content) and the tool call is lost.
    @staticmethod
    def _create_remaining_args_delta(
        delta_message: DeltaMessage,
        remaining_call: str,
        index: int,
    ) -> DeltaMessage:
        """
        Create a delta message for remaining tool arguments, preserving
        id/type/name from the original delta.

        When `remaining_call` is empty, return `delta_message` **unchanged**.
        Unconditionally overwriting the arguments with it replaced a correct
        in-frame delta (1259B measured) with the empty string, so the client
        received a **call with empty arguments**.
        """
        if not remaining_call:
            return delta_message
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
