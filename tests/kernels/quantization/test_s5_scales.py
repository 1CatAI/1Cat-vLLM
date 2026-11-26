# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Five- and six-bit expert scales: encoding and decoding on the host."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.s5_scales import (
    SCALE_BITS,
    decode_scales,
    encode_scales,
    max_spread,
    scale_bits_for,
    scale_spread,
)

S5_MAX_SPREAD = max_spread(5)
S6_MAX_SPREAD = max_spread(6)


def _raster(spread: int, experts=3, tiles=5, groups=12, seed=0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    low = torch.randint(90, 120, (experts, tiles, 1, 32), generator=generator)
    delta = torch.randint(
        0, spread + 1, (experts, tiles, groups, 32), generator=generator
    )
    delta[:, :, 0, :] = 0  # each position's smallest scale
    delta[:, :, 1, :] = spread  # and its largest
    return (low + delta).to(torch.uint8)


def _flat(raster: torch.Tensor) -> torch.Tensor:
    return raster.reshape(raster.shape[0], -1)


def _reference_s5_planes(raster: torch.Tensor):
    """The five-bit encoding as it was before it became one width of many: base,
    nibble plane and one [E, tiles, groups] high-bit word plane."""
    experts, tiles, groups, _ = raster.shape
    low = raster.amin(dim=2)
    delta = raster - low[:, :, None, :]
    nibbles = delta & 15
    nib = nibbles[:, :, 0::2] | (nibbles[:, :, 1::2] << 4)
    weights = (1 << torch.arange(8)).to(torch.uint8)
    high = (delta >> 4).reshape(experts, tiles, groups, 4, 8)
    hi = (high * weights).sum(dim=-1, dtype=torch.uint8).view(torch.int32)
    return low, nib, hi.reshape(experts, tiles, groups)


@pytest.mark.parametrize("spread", [0, 15, 16, S5_MAX_SPREAD])
def test_s5_round_trip_is_exact(spread):
    raster = _raster(spread)
    planes = encode_scales(_flat(raster), raster.shape[2], 5)
    assert torch.equal(decode_scales(planes), raster)


@pytest.mark.parametrize("spread", [32, 47, 56, S6_MAX_SPREAD])
def test_s6_round_trip_is_exact(spread):
    raster = _raster(spread)
    planes = encode_scales(_flat(raster), raster.shape[2], 6)
    assert planes.bits == 6
    assert torch.equal(decode_scales(planes), raster)


@pytest.mark.parametrize("spread", [0, 16, S5_MAX_SPREAD])
def test_s5_bytes_equal_the_old_encoding(spread):
    raster = _raster(spread)
    planes = encode_scales(_flat(raster), raster.shape[2], 5)
    base, nib, hi = _reference_s5_planes(raster)
    assert planes.bits == 5
    assert torch.equal(planes.base, base)
    assert torch.equal(planes.nib, nib)
    assert planes.hi.shape == (*hi.shape, 1)
    assert torch.equal(planes.hi[..., 0], hi)
    # one word a (tile, group): the same bytes as the old [E, tiles, groups] plane
    assert planes.hi.numpy().tobytes() == hi.numpy().tobytes()


def test_s6_widens_the_s5_encoding_by_one_word_plane():
    # Same raster, spread within five bits: the six-bit planes carry the five-bit
    # planes unchanged and an all-zero second word per (tile, group).
    raster = _raster(S5_MAX_SPREAD)
    five = encode_scales(_flat(raster), raster.shape[2], 5)
    six = encode_scales(_flat(raster), raster.shape[2], 6)
    assert torch.equal(six.base, five.base)
    assert torch.equal(six.nib, five.nib)
    assert torch.equal(six.hi[..., :1], five.hi)
    assert not six.hi[..., 1].any()


def test_s5_plane_shapes_and_sizes():
    raster = _raster(S5_MAX_SPREAD)
    planes = encode_scales(_flat(raster), raster.shape[2], 5)
    assert planes.base.shape == (3, 5, 32)
    assert planes.nib.shape == (3, 5, 6, 32)
    assert planes.hi.shape == (3, 5, 12, 1)
    # five bits a scale plus the bases: 5/8 of the raster
    nbytes = sum(t.numel() * t.element_size() for t in (planes.nib, planes.hi))
    assert nbytes == raster.numel() * 5 // 8


def test_s6_plane_shapes_and_sizes():
    raster = _raster(S6_MAX_SPREAD)
    planes = encode_scales(_flat(raster), raster.shape[2], 6)
    assert planes.hi.shape == (3, 5, 12, 2)  # both words of a group adjacent
    nbytes = sum(t.numel() * t.element_size() for t in (planes.nib, planes.hi))
    assert nbytes == raster.numel() * 6 // 8


def test_s5_high_bit_of_position_p_is_bit_p_of_the_group_word():
    raster = torch.zeros(1, 1, 2, 32, dtype=torch.uint8)
    raster[0, 0, 1, 5] = 16  # position 5, group 1: high bit
    raster[0, 0, 0, 31] = 17  # position 31, group 0: high bit and nibble 1
    planes = encode_scales(raster.reshape(1, -1), 2, 5)
    assert int(planes.hi[0, 0, 1, 0]) == 1 << 5
    assert int(planes.hi[0, 0, 0, 0]) & 0xFFFFFFFF == 1 << 31
    assert int(planes.nib[0, 0, 0, 31]) == 1  # group 0 is the low nibble


def test_s6_bit_five_of_position_p_is_bit_p_of_the_second_word():
    raster = torch.zeros(1, 1, 2, 32, dtype=torch.uint8)
    raster[0, 0, 1, 5] = 32 + 16 + 3  # position 5, group 1: both high bits
    raster[0, 0, 0, 7] = 32  # position 7, group 0: second high bit only
    planes = encode_scales(raster.reshape(1, -1), 2, 6)
    assert planes.hi[0, 0, 1].tolist() == [1 << 5, 1 << 5]
    assert planes.hi[0, 0, 0].tolist() == [0, 1 << 7]
    assert int(planes.nib[0, 0, 0, 5]) == 3 << 4  # group 1 is the high nibble


def test_s5_refuses_a_row_wider_than_five_bits():
    raster = _raster(S5_MAX_SPREAD)
    raster[1, 2, 3, 7] = raster[1, 2, 0, 7] + 32  # 32 above the position's smallest
    with pytest.raises(ValueError, match="5-bit"):
        encode_scales(_flat(raster), raster.shape[2], 5)


def test_s6_refuses_a_row_wider_than_six_bits():
    raster = _raster(S6_MAX_SPREAD)
    raster[1, 2, 3, 7] = raster[1, 2, 0, 7] + 64  # 64 above the position's smallest
    with pytest.raises(ValueError, match="6-bit"):
        encode_scales(_flat(raster), raster.shape[2], 6)


def test_unsupported_widths_are_refused():
    raster = _raster(7)
    for bits in (4, 7):
        with pytest.raises(ValueError, match="scale width"):
            encode_scales(_flat(raster), raster.shape[2], bits)


@pytest.mark.parametrize(
    ("spread", "bits"),
    [(0, 5), (31, 5), (32, 6), (47, 6), (63, 6), (64, None), (255, None)],
)
def test_the_smallest_width_that_carries_the_spread_is_chosen(spread, bits):
    assert scale_bits_for(spread) == bits
    assert SCALE_BITS == (5, 6)


@pytest.mark.parametrize("spread", [0, 7, 31, 32, 56])
def test_scale_spread_reports_the_widest_row(spread):
    raster = _raster(min(spread, 31)) if spread <= 31 else _raster(31)
    if spread > 31:
        raster[2, 4, 5, 9] = raster[2, 4, 0, 9] + spread  # one row spans `spread`
    assert scale_spread(_flat(raster), raster.shape[2]) == spread
