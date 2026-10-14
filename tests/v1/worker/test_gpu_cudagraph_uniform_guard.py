# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The speculative-decode FULL cudagraph must only be replayed for real verify
batches: a prefill chunk of exactly 1 + num_draft tokens per request has the
same shape but no draft tokens, and the graph would consume stale spec-state
metadata (seen as out-of-vocabulary garbage output with DFlash2 + Mamba align
prefix caching once the freed state pages were reused by another group)."""

from vllm.v1.worker.gpu.cudagraph_utils import (
    get_uniform_token_count,
    is_speculative_uniform_batch,
)


def test_prefill_chunk_of_spec_shape_is_not_speculative() -> None:
    # DFlash2 with 7 draft tokens captures graphs for 8 tokens per request.
    assert get_uniform_token_count(1, 8, 8) == 8
    assert not is_speculative_uniform_batch(8, {"r0": 8}, {})
    assert not is_speculative_uniform_batch(8, {"r0": 8}, None)


def test_real_verify_batch_is_speculative() -> None:
    drafts = {"r0": list(range(7)), "r1": list(range(7))}
    assert is_speculative_uniform_batch(8, {"r0": 8, "r1": 8}, drafts)


def test_mixed_batch_with_one_prefill_row_is_not_speculative() -> None:
    drafts = {"r0": list(range(7))}
    assert not is_speculative_uniform_batch(8, {"r0": 8, "r1": 8}, drafts)


def test_short_or_longer_draft_lists_do_not_match() -> None:
    assert not is_speculative_uniform_batch(8, {"r0": 8}, {"r0": list(range(3))})
    assert not is_speculative_uniform_batch(8, {"r0": 8}, {"r0": list(range(9))})


def test_adaptive_draft_length_matches_its_own_graph() -> None:
    # Adaptive DFlash2 lookup captures a second, shorter query length.
    assert is_speculative_uniform_batch(4, {"r0": 4}, {"r0": list(range(3))})


def test_plain_decode_batches_stay_uniform() -> None:
    assert is_speculative_uniform_batch(1, {"r0": 1, "r1": 1}, {})
    assert is_speculative_uniform_batch(1, {"r0": 1}, None)
