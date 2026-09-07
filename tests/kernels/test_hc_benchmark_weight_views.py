# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU layout proof only; CUDA performance/rounding require the TP4 screen."""

import pytest
import torch

from benchmarks.kernels.benchmark_sm70_hc_batch_tp4 import shard_weights, up_projection

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("rows", (1, 4, 8, 16))
def test_hc_weight_views_alias_original_without_output_overlap(rank, rows):
    torch.manual_seed(7)
    # Integral low-magnitude inputs make the layout oracle exact on CPU;
    # this deliberately does not purport to test GPU reduction association.
    down = torch.randint(-1, 2, (336, 10240)).half()
    up = torch.randint(-1, 2, (10240, 320)).half()
    lora = torch.randint(-1, 2, (rows, 320)).half()
    wd, wu = shard_weights(down, up, rank, "views")
    pd, pu = shard_weights(down, up, rank, "packed")
    assert wd.untyped_storage().data_ptr() == down.untyped_storage().data_ptr()
    assert wu.untyped_storage().data_ptr() == up.untyped_storage().data_ptr()
    assert wd.storage_offset() == rank * 80 * 10240
    torch.testing.assert_close(wd[:80], pd[:80], rtol=0, atol=0)
    if rank == 3:
        torch.testing.assert_close(wd[80:84], pd[80:84], rtol=0, atol=0)
    output = torch.full((rows, 2560), float("nan"), dtype=torch.float16)
    result = up_projection(lora, wu, output)
    expected = torch.nn.functional.linear(lora, pu)
    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    offsets = torch.arange(rows * 2560).view(rows, 4, 640).transpose(0, 1)
    assert offsets.unique().numel() == rows * 2560
    # Refreshing weights must be visible without stale packed copies.
    up.zero_()
    up_projection(lora, wu, output)
    assert torch.count_nonzero(output).item() == 0
