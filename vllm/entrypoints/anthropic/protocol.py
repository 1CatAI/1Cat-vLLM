# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pydantic models for Anthropic API protocol"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class AnthropicError(BaseModel):
    """Error structure for Anthropic API"""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API"""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage information"""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicContentBlock(BaseModel):
    """Content block in message"""

    type: Literal[
        "text",
        "image",
        "tool_use",
        "tool_result",
        "tool_reference",
        "thinking",
        "redacted_thinking",
    ]
    text: str | None = None
    # For image content
    source: dict[str, Any] | None = None
    # For tool use/result
    id: str | None = None
    tool_use_id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None
    # For tool_reference content
    tool_name: str | None = None
    # For thinking content
    thinking: str | None = None
    signature: str | None = None
    # For redacted thinking content (safety-filtered by the API)
    data: str | None = None


class AnthropicMessage(BaseModel):
    """Message structure"""

    role: Literal["user", "assistant", "system"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    """Tool definition"""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]
    defer_loading: bool | None = None

    @field_validator("input_schema")
    @classmethod
    def validate_input_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"  # Default to object type
        return v


class AnthropicToolChoice(BaseModel):
    """Tool Choice definition"""

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None

    @model_validator(mode="after")
    def validate_name_required_for_tool(self) -> "AnthropicToolChoice":
        if self.type == "tool" and not self.name:
            raise ValueError("tool_choice.name is required when type is 'tool'")
        return self


class AnthropicJsonOutputFormat(BaseModel):
    """JSON output format configuration"""

    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    type: Literal["json_schema"] = "json_schema"


class AnthropicOutputConfig(BaseModel):
    """Configuration options for the model's output, such as the output format."""

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    format: AnthropicJsonOutputFormat | None = None


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request"""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: dict[str, Any] | None = None
    output_config: AnthropicOutputConfig | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    top_k: int | None = None
    top_p: float | None = None

    # vLLM-specific fields that are not in Anthropic spec
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )
    # Per-chunk output token ids are opt-in,
    # mirroring the OpenAI protocol's `return_token_ids` (default off). A live
    # TPS meter wants them; callers that do not should not pay for them. The
    # same switch can be given as a request header (`x-return-token-ids`) for
    # proxies that forward the body verbatim — see
    # `AnthropicServingMessages._wants_token_ids`.
    return_token_ids: bool = Field(
        default=False,
        description=(
            "If true, each response carries the token ids its output decodes "
            "from (`token_ids` per `content_block_delta`, or on the message for "
            "a non-streaming reply). The Anthropic schema defines no such field; "
            "this mirrors the OpenAI protocol's `return_token_ids`."
        ),
    )
    # The prompt is typically tens of thousands of ids, so echoing it back is a
    # separate, even more explicit opt-in than the output ids above.
    return_prompt_token_ids: bool = Field(
        default=False,
        description=(
            "If true, the response also carries the prompt's token ids in "
            "`prompt_token_ids`. The Anthropic schema defines no such field; "
            "this mirrors the OpenAI protocol's `prompt_token_ids`."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class AnthropicDelta(BaseModel):
    """Delta for streaming responses"""

    type: (
        Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"]
        | None
    ) = None
    text: str | None = None
    thinking: str | None = None
    partial_json: str | None = None
    signature: str | None = None

    # Message delta
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event"""

    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: "AnthropicMessagesResponse | None" = None
    delta: AnthropicDelta | None = None
    content_block: AnthropicContentBlock | None = None
    index: int | None = None
    error: AnthropicError | None = None
    usage: AnthropicUsage | None = None
    # Not part of the Anthropic spec, but useful for
    # tracing the tokens in agent scenarios — the token ids a
    # `content_block_delta` decodes from, mirroring the OpenAI protocol's
    # per-chunk `choices[].token_ids`.
    token_ids: list[int] | None = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response"""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: AnthropicUsage | None = None

    # vLLM-specific fields that are not in Anthropic spec
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )
    # The output tokens this message produced
    # (mirroring the OpenAI protocol's `choices[].token_ids`), plus the prompt's
    # ids when the request opted in with `return_prompt_token_ids`.
    # Neither is in the Anthropic spec.
    token_ids: list[int] | None = None
    prompt_token_ids: list[int] | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _normalize_id(cls, v: Any) -> str:
        """Normalise the id to the Anthropic `msg_` prefix.

        Two things:

        1. An empty id falls back to a uuid. The previous
           `model_post_init` used `msg_{int(time.time()*1000)}`, and millisecond
           timestamps collide under load.
        2. A `chatcmpl-` prefix from upstream is rewritten. This endpoint
           forwards to the OpenAI chat path internally, so `generator.id` is
           `chatcmpl-{uuid}`; passing it through verbatim hands Anthropic
           clients an id they do not recognise as a message.

        A `field_validator(mode="before")` is used rather than
        `model_post_init`, whose `if not self.id` guard never fires because
        callers always pass an id explicitly.
        """
        from vllm.utils import random_uuid

        if not v:
            return f"msg_{random_uuid()}"
        v = str(v)
        if v.startswith("msg_"):
            return v
        for prefix in ("chatcmpl-", "cmpl-"):
            if v.startswith(prefix):
                return f"msg_{v[len(prefix):]}"
        return f"msg_{v}"


class AnthropicContextManagement(BaseModel):
    """Context management information for token counting."""

    original_input_tokens: int


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic messages.count_tokens request"""

    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None

    # vLLM-specific fields that are not in Anthropic spec
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the chat template renderer. "
            "Will be accessible by the template."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic messages.count_tokens response"""

    input_tokens: int
    context_management: AnthropicContextManagement | None = None
