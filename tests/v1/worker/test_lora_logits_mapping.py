# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.lora.request import LoRARequest
from vllm.v1.worker.gpu.lora_utils import LoraState


@pytest.mark.parametrize("num_logits", [None, [1, 1, 1], [8, 0, 3], [1, 8, 0]])
def test_logits_and_model_tokens_keep_independent_lora_row_mappings(num_logits):
    state = LoraState(8)
    a, b = LoRARequest("a", 7, "/unused/a"), LoRARequest("b", 11, "/unused/b")
    state.add_request("a", 1, a)
    state.add_request("b", 4, b)
    state.add_request("base", 6, None)
    arguments = {} if num_logits is None else {"num_logits": np.array(num_logits)}
    prompt, token, active = state.make_lora_inputs(
        ["b", "a", "base"], np.array([4, 1, 6]), np.array([4, 5, 1]), **arguments
    )
    expected = [11, 7, 0]
    if num_logits is not None:
        expected = [
            lora for lora, count in zip(expected, num_logits) for _ in range(count)
        ]
    assert prompt == tuple(expected)
    assert token == (11,) * 4 + (7,) * 5 + (0,)
    assert active == {a, b}
    state.remove_request("b")
    state.add_request("new-base", 4, None)
    prompt, token, active = state.make_lora_inputs(
        ["new-base", "a"],
        np.array([4, 1]),
        np.array([8, 1]),
        num_logits=np.array([3, 1]),
    )
    assert prompt == (0, 0, 0, 7)
    assert token == (0,) * 8 + (7,)
    assert active == {a}
