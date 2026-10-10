# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise upstream's real live-prefill eligibility before residual verification."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize(
    ("scheduled", "computed", "prefill", "dummy", "expected"),
    [
        ({"a": 5}, [0], [8], False, None),
        ({"a": 5}, [4], [9], False, None),
        ({"a": 5, "b": 5}, [8, 0], [8, 8], False, None),
        ({"a": 5, "b": 5}, [8, 8], [8, 8], False, 5),
        ({"a": 1, "b": 1}, [8, 8], [8, 8], False, 1),
        ({"a": 1, "b": 5}, [8, 8], [8, 8], False, None),
        ({"a": 5}, [0], [8], True, 5),
    ],
)
def test_upstream_live_prefill_state_controls_uniform_graph_eligibility(
    scheduled, computed, prefill, dummy, expected
):
    runner = SimpleNamespace(
        req_states=SimpleNamespace(
            req_id_to_index={name: i for i, name in enumerate(scheduled)},
            num_computed_prefill_tokens=np.asarray(computed),
            prefill_len=SimpleNamespace(np=np.asarray(prefill)),
        )
    )
    output = SimpleNamespace(
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
    )
    assert (
        GPUModelRunner._get_uniform_decode_token_count(runner, output, dummy)
        == expected
    )
