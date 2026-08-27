# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.sequence import IntermediateTensors
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
