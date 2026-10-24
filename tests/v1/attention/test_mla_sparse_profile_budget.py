# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse indexer's prefill buffer holds the sequences one batch can hold."""

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size


@pytest.mark.parametrize(
    "max_num_seqs,expected_sequences",
    [(1, 1), (4, 4), (40, 40), (256, 40)],
)
def test_indexer_gather_buffer_holds_the_sequences_a_batch_can_hold(
    max_num_seqs, expected_sequences
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=262_144),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )
    assert get_max_prefill_buffer_size(config) == 262_144 * expected_sequences
