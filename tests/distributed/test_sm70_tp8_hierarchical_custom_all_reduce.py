# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.device_communicators.custom_all_reduce import (
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
