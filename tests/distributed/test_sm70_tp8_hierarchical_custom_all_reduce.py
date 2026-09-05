# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import envs
from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    _sm70_tp8_hierarchical_peer_ranks,
)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, (0, 1, 2, 3, 4)),
        (1, (0, 1, 2, 3, 5)),
        (2, (0, 1, 2, 3, 6)),
        (3, (0, 1, 2, 3, 7)),
        (4, (4, 5, 6, 7, 0)),
        (5, (4, 5, 6, 7, 1)),
        (6, (4, 5, 6, 7, 2)),
        (7, (4, 5, 6, 7, 3)),
    ],
)
def test_sm70_tp8_hierarchical_peer_ranks(rank: int, expected: tuple[int, ...]) -> None:
    assert _sm70_tp8_hierarchical_peer_ranks(rank) == expected


@pytest.mark.parametrize("rank", [-1, 8])
def test_sm70_tp8_hierarchical_peer_ranks_rejects_invalid_rank(rank: int) -> None:
    with pytest.raises(ValueError, match="must be in"):
        _sm70_tp8_hierarchical_peer_ranks(rank)


@pytest.mark.parametrize("elements", [4096, 8 * 4096])
def test_sm70_tp8_hierarchical_dispatch_accepts_exact_fp16_shapes(
    elements: int,
) -> None:
    communicator = object.__new__(CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.tp8_hierarchical = True

    assert communicator.should_custom_ar(torch.empty(elements, dtype=torch.float16))


@pytest.mark.parametrize(
    ("elements", "dtype"),
    [
        (4096, torch.bfloat16),
        (4096, torch.float32),
        (8192, torch.float16),
        (8 * 4096 - 8, torch.float16),
        (8 * 4096 + 8, torch.float16),
    ],
)
def test_sm70_tp8_hierarchical_dispatch_rejects_other_contracts(
    elements: int, dtype: torch.dtype
) -> None:
    communicator = object.__new__(CustomAllreduce)
    communicator.disabled = False
    communicator._ptr = 0
    communicator.tp8_hierarchical = True

    assert not communicator.should_custom_ar(torch.empty(elements, dtype=dtype))


def test_glm53_q8_sum2_defaults_on_and_can_be_disabled(monkeypatch):
    name = "VLLM_SM70_GLM53_MOE_SUM2_ALLREDUCE_Q8"
    monkeypatch.delenv(name, raising=False)
    assert envs.VLLM_SM70_GLM53_MOE_SUM2_ALLREDUCE_Q8

    monkeypatch.setenv(name, "0")
    assert not envs.VLLM_SM70_GLM53_MOE_SUM2_ALLREDUCE_Q8
