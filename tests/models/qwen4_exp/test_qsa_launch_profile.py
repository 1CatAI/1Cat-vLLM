# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("block_m", "base_programs", "is_sm70", "expected"),
    [
        pytest.param(8, 8, True, (16, 64, 4), id="sm70_small_block_m8"),
        pytest.param(16, 4, True, (16, 64, 4), id="sm70_small_block_m16"),
        pytest.param(16, 5, True, (16, 32, 4), id="sm70_narrow"),
        pytest.param(8, 256, True, (64, 8, 4), id="sm70_split8"),
        pytest.param(8, 512, True, (32, 4, 4), id="sm70_split4"),
        pytest.param(8, 513, False, (64, 1, 2), id="non_sm70_split1"),
        pytest.param(8, 513, True, (32, 1, 4), id="sm70_split1"),
    ],
)
def test_qsa_sparse_launch_profile(
    block_m: int,
    base_programs: int,
    is_sm70: bool,
    expected: tuple[int, int, int],
) -> None:
    assert (
        qsa_ops._qsa_sparse_launch_profile(base_programs, block_m, is_sm70) == expected
    )
