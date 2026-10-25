# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Five-bit expert scales: encoding and decoding on the host."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.s5_scales import (
    S5_MAX_SPREAD,
    decode_s5,
    encode_s5,
    scale_spread,
)


def _raster(spread: int, experts=3, tiles=5, groups=12, seed=0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    low = torch.randint(90, 120, (experts, tiles, 1, 32), generator=generator)
    delta = torch.randint(
        0, spread + 1, (experts, tiles, groups, 32), generator=generator
    )
    delta[:, :, 0, :] = 0  # each position's smallest scale
    delta[:, :, 1, :] = spread  # and its largest
    return (low + delta).to(torch.uint8)


@pytest.mark.parametrize("spread", [0, 15, 16, S5_MAX_SPREAD])
def test_s5_round_trip_is_exact(spread):
    raster = _raster(spread)
    planes = encode_s5(raster.reshape(raster.shape[0], -1), raster.shape[2])
    assert torch.equal(decode_s5(planes), raster)


def test_s5_plane_shapes_and_sizes():
    raster = _raster(S5_MAX_SPREAD)
    planes = encode_s5(raster.reshape(raster.shape[0], -1), raster.shape[2])
    assert planes.base.shape == (3, 5, 32)
    assert planes.nib.shape == (3, 5, 6, 32)
    assert planes.hi.shape == (3, 5, 12)
    # five bits a scale plus the bases: 5/8 of the raster
    nbytes = sum(t.numel() * t.element_size() for t in (planes.nib, planes.hi))
    assert nbytes == raster.numel() * 5 // 8


def test_s5_high_bit_of_position_p_is_bit_p_of_the_group_word():
    raster = torch.zeros(1, 1, 2, 32, dtype=torch.uint8)
    raster[0, 0, 1, 5] = 16  # position 5, group 1: high bit
    raster[0, 0, 0, 31] = 17  # position 31, group 0: high bit and nibble 1
    planes = encode_s5(raster.reshape(1, -1), 2)
    assert int(planes.hi[0, 0, 1]) == 1 << 5
    assert int(planes.hi[0, 0, 0]) & 0xFFFFFFFF == 1 << 31 or int(
        planes.hi[0, 0, 0]
    ) == -(1 << 31)
    assert int(planes.nib[0, 0, 0, 31]) == 1  # group 0 is the low nibble


def test_s5_refuses_a_row_wider_than_five_bits():
    raster = _raster(S5_MAX_SPREAD)
    raster[1, 2, 3, 7] = raster[1, 2, 0, 7] + 32  # 32 above the position's smallest
    with pytest.raises(ValueError, match="five-bit"):
        encode_s5(raster.reshape(raster.shape[0], -1), raster.shape[2])


@pytest.mark.parametrize("spread", [0, 7, 31, 32, 56])
def test_scale_spread_reports_the_widest_row(spread):
    raster = _raster(min(spread, 31)) if spread <= 31 else _raster(31)
    if spread > 31:
        raster[2, 4, 5, 9] = raster[2, 4, 0, 9] + spread  # one row spans `spread`
    assert scale_spread(raster.reshape(raster.shape[0], -1), raster.shape[2]) == spread
