# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu_model_runner import _select_dummy_sample_hidden_states


def test_pp_dummy_intermediates_are_not_sampled() -> None:
    intermediate = IntermediateTensors(
        {"hidden_states": torch.empty((8, 16), dtype=torch.float16)}
    )

    result = _select_dummy_sample_hidden_states(
        intermediate, np.array([3, 5]), torch.device("cpu")
    )

    assert result is None


def test_last_pp_rank_selects_final_scheduled_tokens() -> None:
    hidden_states = torch.arange(8 * 2).view(8, 2)

    result = _select_dummy_sample_hidden_states(
        hidden_states, np.array([3, 5]), torch.device("cpu")
    )

    assert result is not None
    torch.testing.assert_close(result, hidden_states[[2, 7]])


class _FakePPGroup:
    is_last_rank = True
    last_rank = 4
    device_group = object()


def test_pp_broadcast_pads_prefill_tokens_to_speculative_width(monkeypatch) -> None:
    broadcasts: list[torch.Tensor] = []

    monkeypatch.setattr(pp_utils, "get_pp_group", lambda: _FakePPGroup())
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, **_kwargs: broadcasts.append(tensor.clone()),
    )

    pp_utils.pp_broadcast(
        torch.tensor([[42]], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        max_sample_len=8,
    )

    assert len(broadcasts) == 2
    torch.testing.assert_close(
        broadcasts[0], torch.tensor([[42, -1, -1, -1, -1, -1, -1, -1]])
    )
    assert broadcasts[1].shape == (2, 1)


def test_pp_broadcast_rejects_tokens_wider_than_receive_buffer(monkeypatch) -> None:
    monkeypatch.setattr(pp_utils, "get_pp_group", lambda: _FakePPGroup())

    with pytest.raises(ValueError, match="exceeds PP receive width"):
        pp_utils.pp_broadcast(
            torch.zeros((1, 3), dtype=torch.int64),
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            max_sample_len=2,
        )
