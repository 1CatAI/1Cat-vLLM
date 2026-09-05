# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU metadata checks; the benchmark import requires a Flash-V100 installation."""

import pytest
import torch

pytest.importorskip("flash_attn_v100")

from benchmarks.kernels.benchmark_qwen38_qsa_mtp5_flash_v100 import _logical_indices


@pytest.mark.parametrize("rows", (4, 5, 8, 16))
@pytest.mark.parametrize("independent", (False, True))
@pytest.mark.parametrize("overlap", (0.0, 0.82, 1.0))
def test_qsa_micro_uses_canonical_causal_selections(rows, independent, overlap):
    indices = _logical_indices(
        rows=rows,
        seq_len=8192,
        overlap=overlap,
        seed=7,
        independent=independent,
    )
    assert indices.shape == (rows, 2051)
    for row in range(rows):
        visible = 8192 if independent else 8192 - rows + row + 1
        complete = indices[row, :2048].reshape(512, 4)
        assert torch.all(complete[:, 0] % 4 == 0)
        assert torch.all(complete[:, 0] // 4 < visible // 4)
        torch.testing.assert_close(
            complete,
            complete[:, :1] + torch.arange(4, dtype=indices.dtype),
        )
        tail = indices[row, 2048:]
        count = visible % 4
        torch.testing.assert_close(
            tail[:count],
            torch.arange(visible - count, visible, dtype=indices.dtype),
        )
        assert torch.all(tail[count:] == -1)
        valid = indices[row][indices[row] >= 0]
        assert valid.unique().numel() == 2048 + count
        assert torch.all(valid < visible)
