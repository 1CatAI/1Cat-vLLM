# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generated logprob rows that do not line up with the output tokens.

When the number of logprob rows differs from the number of tokens, a missing
or extra row may be anywhere, so the builders must return the output without
logprobs instead of raising IndexError or attaching a probability to the
wrong token (upstream vLLM #63775). Aligned input must be unchanged.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.serving import drop_misaligned_logprobs
from vllm.entrypoints.serve.disagg.serving import ServingTokens
from vllm.logprobs import Logprob

TOKENS = [11, 12, 13]


def _row(token_id: int, logprob: float) -> dict[int, Logprob]:
    return {token_id: Logprob(logprob=logprob, rank=1, decoded_token=f"t{token_id}")}


ALIGNED = [_row(11, -0.1), _row(12, -0.2), _row(13, -0.3)]
SHORT = [_row(11, -0.1), _row(12, -0.2)]
# An extra row in the middle: zipping or truncating would bind 12 -> -2.0.
EXTRA_MIDDLE = [
    _row(11, -0.1),
    {12: Logprob(-2.0, 1, "x")},
    _row(12, -0.2),
    _row(13, -0.3),
]


def test_helper_keeps_aligned_rows():
    assert drop_misaligned_logprobs(TOKENS, ALIGNED) is ALIGNED


@pytest.mark.parametrize(
    "rows", [SHORT, EXTRA_MIDDLE, []], ids=["short", "extra", "empty"]
)
def test_helper_drops_all_rows_on_mismatch(rows):
    assert drop_misaligned_logprobs(TOKENS, rows) == [None, None, None]


def _chat() -> OpenAIServingChat:
    serving = object.__new__(OpenAIServingChat)
    serving.return_tokens_as_token_ids = True
    return serving


def _completion() -> OpenAIServingCompletion:
    serving = object.__new__(OpenAIServingCompletion)
    serving.return_tokens_as_token_ids = True
    return serving


def test_chat_builder_aligned_unchanged():
    out = _chat()._create_chat_logprobs(TOKENS, ALIGNED, None, 0, True)
    assert [c.logprob for c in out.content] == [-0.1, -0.2, -0.3]
    assert [c.token for c in out.content] == [f"token_id:{t}" for t in TOKENS]


@pytest.mark.parametrize("rows", [SHORT, EXTRA_MIDDLE], ids=["short", "extra"])
def test_chat_builder_mismatch_returns_tokens_without_logprobs(rows):
    out = _chat()._create_chat_logprobs(TOKENS, rows, None, 0, True)
    assert [c.token for c in out.content] == [f"token_id:{t}" for t in TOKENS]
    # The protocol's "no probability" default, never a borrowed value.
    default = type(out.content[0])(token="x").logprob
    assert all(c.logprob == default for c in out.content)
    assert all(not c.top_logprobs for c in out.content)


def test_completion_builder_aligned_unchanged():
    out = _completion()._create_completion_logprobs(TOKENS, ALIGNED, 0, None, 0, True)
    assert out.token_logprobs == [-0.1, -0.2, -0.3]


@pytest.mark.parametrize("rows", [SHORT, EXTRA_MIDDLE], ids=["short", "extra"])
def test_completion_builder_mismatch_returns_tokens_without_logprobs(rows):
    out = _completion()._create_completion_logprobs(TOKENS, rows, 0, None, 0, True)
    assert out.tokens == [f"token_id:{t}" for t in TOKENS]
    assert out.token_logprobs == [None, None, None]
    assert out.top_logprobs == [None, None, None]


def test_tokens_builder_mismatch_returns_tokens_without_logprobs():
    serving = object.__new__(ServingTokens)
    out = serving._create_tokens_logprobs(TOKENS, EXTRA_MIDDLE, 0)
    assert [c.token for c in out.content] == [f"token_id:{t}" for t in TOKENS]
    assert all(not c.top_logprobs for c in out.content)
    aligned = serving._create_tokens_logprobs(TOKENS, ALIGNED, 0)
    assert [c.logprob for c in aligned.content] == [-0.1, -0.2, -0.3]
