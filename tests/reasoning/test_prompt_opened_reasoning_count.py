# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generated marker boundaries without downloading a model tokenizer."""

from types import SimpleNamespace

import pytest

from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


@pytest.mark.parametrize(
    "tokens,expected",
    [
        ([], 0),
        ([11, 12], 0),
        ([11, 12, 2, 99], 2),
        ([2, 99], 0),
        ([11, 2, 99, 2], 1),
        ([1, 2, 99, 2], 0),
        ([1, 2], 0),
        ([1, 11, 12, 2, 99], 2),
        ([1, 11, 1, 12, 2, 13, 2], 3),
        ([1, 11, 2, 99, 2], 1),
        ([1, 11, 12], 2),
    ],
)
def test_generated_reasoning_marker_boundaries(tokens, expected):
    parser = SimpleNamespace(start_token_id=1, end_token_id=2)
    assert (
        BaseThinkingReasoningParser.count_reasoning_tokens(parser, tokens) == expected
    )
