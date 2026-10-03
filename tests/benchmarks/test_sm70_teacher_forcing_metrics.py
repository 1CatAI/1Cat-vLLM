# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from benchmarks.sm70_teacher_forcing_metrics import compare_dumps, compare_logits


def test_identical_logits_and_chunk_independence():
    logits = torch.tensor([[1000.0, 999.0, -1000.0], [1.0, 0.0, -1.0]])
    result = compare_logits(logits, logits, chunk_rows=1)
    assert result == compare_logits(logits, logits, chunk_rows=2)
    assert result["passed"]
    assert result["mean_kl"] == 0
    assert result["top1_agreement"] == 1


def test_common_shift_changes_raw_error_but_not_distribution():
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    result = compare_logits(logits, logits + 1)
    assert result["mean_kl"] == pytest.approx(0, abs=1e-14)
    assert result["max_centered_logit_error"] == 0
    assert result["max_logit_error"] == 1
    assert not result["passed"]


def test_confident_token_flip_fails_with_known_kl():
    logits = torch.tensor([[2.0, 0.0]])
    result = compare_logits(logits, logits.flip(-1))
    assert result["mean_kl"] == pytest.approx(2 * torch.tanh(torch.tensor(1.0)).item())
    assert result["top1_agreement"] == 0
    assert result["high_margin_top1_agreement"] == 0
    assert not result["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_logits(torch.zeros(1, 2), torch.tensor([[0.0, value]]))


def test_empty_shape_rejected():
    with pytest.raises(ValueError):
        compare_logits(torch.empty(0, 2), torch.empty(0, 2))


@pytest.mark.parametrize(
    "field", ["prompt_sha256", "role", "position_ids", "token_ids"]
)
def test_misaligned_teacher_forcing_rejected(field):
    dump = {
        "prompt_sha256": "frozen",
        "role": "target",
        "position_ids": torch.tensor([8192]),
        "token_ids": torch.tensor([42]),
        "logits": torch.tensor([[1.0, 0.0]]),
    }
    candidate = dump.copy()
    candidate[field] = (
        dump[field] + 1 if isinstance(dump[field], torch.Tensor) else "different"
    )
    with pytest.raises(ValueError, match="mismatch"):
        compare_dumps(dump, candidate)
