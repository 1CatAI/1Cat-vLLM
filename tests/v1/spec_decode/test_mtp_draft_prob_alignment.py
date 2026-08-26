# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.draft_prob_alignment import get_aligned_draft_probs
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def _get_aligned_draft_probs(
    *,
    req_ids: list[str],
    draft_probs: torch.Tensor | None,
    draft_prob_req_ids: list[str] | None,
    draft_prob_token_ids: torch.Tensor | list[list[int]] | None,
    spec_decode_metadata: SpecDecodeMetadata,
) -> torch.Tensor | None:
    return get_aligned_draft_probs(
        req_ids=req_ids,
        draft_probs=draft_probs,
        draft_prob_req_ids=draft_prob_req_ids,
        draft_prob_token_ids=draft_prob_token_ids,
        spec_decode_metadata=spec_decode_metadata,
    )


def _metadata(draft_token_ids: list[list[int]]) -> SpecDecodeMetadata:
    return SpecDecodeMetadata.make_dummy(
        draft_token_ids=draft_token_ids,
        device=torch.device("cpu"),
    )


def _inconsistent_metadata() -> SpecDecodeMetadata:
    return SpecDecodeMetadata(
        draft_token_ids=torch.tensor([11], dtype=torch.int32),
        num_draft_tokens=[2],
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([3], dtype=torch.int32),
        target_logits_indices=torch.zeros(1, dtype=torch.int32),
        bonus_logits_indices=torch.zeros(1, dtype=torch.int32),
        logits_indices=torch.zeros(3, dtype=torch.int32),
    )


def test_get_spec_decode_draft_probs_binds_tokens_by_req_id():
    draft_probs = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    actual = _get_aligned_draft_probs(
        req_ids=["b", "a"],
        draft_probs=draft_probs,
        draft_prob_req_ids=["a", "b"],
        draft_prob_token_ids=torch.tensor(
            [[10, 11, 12], [20, 21, 22]], dtype=torch.int32
        ),
        spec_decode_metadata=_metadata([[20], [10, 11]]),
    )

    expected = torch.cat([draft_probs[1, :1], draft_probs[0, :2]], dim=0)
    assert torch.equal(actual, expected)


def test_get_spec_decode_draft_probs_rejects_missing_probability_row():
    with pytest.raises(
        RuntimeError, match="Missing cached draft probabilities for request b"
    ):
        _get_aligned_draft_probs(
            req_ids=["a", "b"],
            draft_probs=torch.zeros((1, 2, 4), dtype=torch.float32),
            draft_prob_req_ids=["a"],
            draft_prob_token_ids=torch.tensor([[1, 2]], dtype=torch.int32),
            spec_decode_metadata=_metadata([[1], [3]]),
        )


def test_get_spec_decode_draft_probs_rejects_probability_width_shortage():
    with pytest.raises(RuntimeError, match="do not have enough draft positions"):
        _get_aligned_draft_probs(
            req_ids=["a"],
            draft_probs=torch.zeros((1, 1, 4), dtype=torch.float32),
            draft_prob_req_ids=["a"],
            draft_prob_token_ids=torch.tensor([[1, 2]], dtype=torch.int32),
            spec_decode_metadata=_metadata([[1, 2]]),
        )


def test_get_spec_decode_draft_probs_rejects_probability_row_count_mismatch():
    with pytest.raises(RuntimeError, match="row count does not match request ids"):
        _get_aligned_draft_probs(
            req_ids=["a"],
            draft_probs=torch.zeros((1, 2, 4), dtype=torch.float32),
            draft_prob_req_ids=["a", "b"],
            draft_prob_token_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            spec_decode_metadata=_metadata([[1]]),
        )


def test_get_spec_decode_draft_probs_rejects_metadata_token_count_mismatch():
    with pytest.raises(RuntimeError, match="draft token count does not match"):
        _get_aligned_draft_probs(
            req_ids=["a"],
            draft_probs=torch.zeros((1, 2, 4), dtype=torch.float32),
            draft_prob_req_ids=["a"],
            draft_prob_token_ids=torch.tensor([[11, 12]], dtype=torch.int32),
            spec_decode_metadata=_inconsistent_metadata(),
        )


def test_get_spec_decode_draft_probs_rejects_cached_token_mismatch():
    with pytest.raises(RuntimeError, match="do not match verifier draft tokens"):
        _get_aligned_draft_probs(
            req_ids=["a"],
            draft_probs=torch.zeros((1, 2, 4), dtype=torch.float32),
            draft_prob_req_ids=["a"],
            draft_prob_token_ids=torch.tensor([[1, 9]], dtype=torch.int32),
            spec_decode_metadata=_metadata([[1, 2]]),
        )
