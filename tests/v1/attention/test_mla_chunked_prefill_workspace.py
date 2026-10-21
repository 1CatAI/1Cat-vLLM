# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MLA chunked-prefill workspace covers the contexts one batch can hold."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadataBuilder,
)


def _config(max_model_len, max_num_seqs, block_size=64):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        cache_config=SimpleNamespace(block_size=block_size),
    )


@pytest.mark.parametrize(
    "max_model_len,max_num_seqs,expected",
    [
        # one sequence: its full context in one chunk, not eight
        (16384, 1, 16384),
        (4096, 2, 8192),
        # many sequences: eight full requests, capped at 64k tokens
        (4096, 32, 32768),
        (16384, 16, 64 * 1024),
        # at least one page per request
        (128, 2048, 2048 * 64),
    ],
)
def test_workspace_holds_the_batch_contexts(max_model_len, max_num_seqs, expected):
    config = _config(max_model_len, max_num_seqs)
    size = MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(config)
    assert size == expected
