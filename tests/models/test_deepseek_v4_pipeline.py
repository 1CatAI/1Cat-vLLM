# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.utils import get_pp_indices
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model


@pytest.mark.parametrize(
    ("pp_size", "expected_partitions"),
    [
        (2, [(0, 22), (22, 43)]),
        (4, [(0, 11), (11, 22), (22, 33), (33, 43)]),
    ],
)
def test_deepseek_v4_pipeline_partitions_keep_dspark_targets_on_last_stage(
    monkeypatch: pytest.MonkeyPatch,
    pp_size: int,
    expected_partitions: list[tuple[int, int]],
) -> None:
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)

    partitions = [get_pp_indices(43, rank, pp_size) for rank in range(pp_size)]

    assert partitions == expected_partitions
    last_start, last_end = partitions[-1]
    assert all(last_start <= layer_id < last_end for layer_id in (40, 41, 42))


def test_deepseek_v4_pipeline_intermediate_preserves_mhc_streams() -> None:
    model = SimpleNamespace(
        hc_mult=4,
        config=SimpleNamespace(hidden_size=4096),
    )

    intermediate = DeepseekV4Model.make_empty_intermediate_tensors(
        model,
        batch_size=3,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )

    assert set(intermediate.tensors) == {"hidden_states"}
    assert intermediate["hidden_states"].shape == (3, 4, 4096)
    assert intermediate["hidden_states"].dtype == torch.float16
    assert torch.count_nonzero(intermediate["hidden_states"]) == 0
