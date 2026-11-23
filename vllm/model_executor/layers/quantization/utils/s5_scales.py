# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Five- and six-bit expert scales ("S5", "S6") that the skinny QPN MoE kernels
read in place.

A scale raster holds one fp8-e4m3 byte per (tile row, K group): ``[expert][tile
N/32][group K/16][position 32]``, the layout `qpn_prepack` gives the checkpoint
scales. The csf_32x2_v1 checkpoint planes are smaller, but a few scales per
thousand fall outside a row's four-bit window and are patched from a record list,
which a kernel cannot afford to look up in its loop (one block in six has a
record). The scale bytes of a row do span a short range, though: the group scales
of GLM-5.3 differ from their row's smallest by at most 56, and the e4m3 bytes of
positive scales order like integers. Five or six bits per scale therefore carry
every scale without exceptions:

    scale = base[expert][tile][position] + nibble + 16 * high0 [+ 32 * high1]

``nibble`` is four bits, two groups per byte (``nib``: low nibble the even group),
``high0`` (and, for six bits, ``high1``) one bit per (group, position), a 32-bit
word per (tile, group) and bit plane (``hi``, last axis = bit plane, so a kernel
reads both words of a group with one 64-bit load). Positions are the raster
positions of the card's fragment order, so a lane reads the bytes it would read
from the raster and decodes them with a few integer operations. Five bits take
5/8 of the raster, six bits 6/8, and no raster has to exist for the experts of a
layer.

Not every projection of GLM-5.3 fits five bits: the last MoE layers have rows that
span up to 56 (layers 43 and 44; a few rows in layers 36, 40 to 42). `scale_spread`
tells, `scale_bits_for` picks the smallest width that carries a projection, and a
projection that fits neither keeps its csf planes.
"""

from dataclasses import dataclass

import torch

# The widths a kernel instance exists for, smallest first.
SCALE_BITS = (5, 6)
# The nibble plane carries the low four bits; every further bit is a word plane.
_NIBBLE_BITS = 4
# Experts per encoding step: bounds the temporaries of the load-time conversion.
_ENCODE_EXPERTS = 16


def max_spread(bits: int) -> int:
    """The widest span between a row's largest and smallest scale `bits` carry."""
    return (1 << bits) - 1


def scale_bits_for(spread: int) -> int | None:
    """The smallest supported width that carries `spread`, None if none does."""
    for bits in SCALE_BITS:
        if spread <= max_spread(bits):
            return bits
    return None


@dataclass
class PackedScales:
    """One projection set's scale planes; ``tiles`` are N/32 tiles per expert."""

    groups: int
    base: torch.Tensor  # u8 [E, tiles, 32]
    nib: torch.Tensor  # u8 [E, tiles, groups / 2, 32]
    hi: torch.Tensor  # i32 [E, tiles, groups, bits - 4]

    @property
    def num_experts(self) -> int:
        return self.base.shape[0]

    @property
    def tiles(self) -> int:
        return self.base.shape[1]

    @property
    def bits(self) -> int:
        return _NIBBLE_BITS + self.hi.shape[-1]


def scale_spread(raster: torch.Tensor, groups: int) -> int:
    """The widest span, over all rows, between a row's largest and smallest scale
    in a u8 scale raster [E, tiles * groups * 32] (kernel positions)."""
    experts = raster.shape[0]
    widest = 0
    for first in range(0, experts, _ENCODE_EXPERTS):
        r = raster[first : first + _ENCODE_EXPERTS].reshape(-1, groups, 32)
        widest = max(widest, int((r.amax(dim=1) - r.amin(dim=1)).max()))
    return widest


def encode_scales(raster: torch.Tensor, groups: int, bits: int) -> PackedScales:
    """Encode a u8 scale raster [E, tiles * groups * 32] (kernel positions) in
    `bits` bits per scale.

    Raises if a row of any expert spans more than `max_spread(bits)` between its
    smallest and largest scale: the code would not carry it.
    """
    if bits not in SCALE_BITS:
        raise ValueError(f"scale width {bits} is not one of {SCALE_BITS}")
    if raster.dtype != torch.uint8 or groups % 2:
        raise ValueError("packed scales encode a u8 raster with an even group count")
    experts = raster.shape[0]
    tiles = raster.numel() // (experts * groups * 32)
    if tiles * groups * 32 * experts != raster.numel():
        raise ValueError("raster size is not experts x tiles x groups x 32")
    device = raster.device
    planes = bits - _NIBBLE_BITS
    base = torch.empty(experts, tiles, 32, dtype=torch.uint8, device=device)
    nib = torch.empty(experts, tiles, groups // 2, 32, dtype=torch.uint8, device=device)
    hi = torch.empty(experts, tiles, groups, planes, dtype=torch.int32, device=device)
    bit_weights = (1 << torch.arange(8, device=device)).to(torch.uint8)
    widest = 0
    for first in range(0, experts, _ENCODE_EXPERTS):
        last = min(first + _ENCODE_EXPERTS, experts)
        r = raster[first:last].reshape(last - first, tiles, groups, 32)
        low = r.amin(dim=2)
        delta = r - low[:, :, None, :]
        widest = max(widest, int(delta.max()))
        base[first:last] = low
        nibbles = delta & 15
        nib[first:last] = nibbles[:, :, 0::2] | (nibbles[:, :, 1::2] << 4)
        for plane in range(planes):
            # bit p of a (tile, group) word is position p's bit `4 + plane`
            bit = ((delta >> (_NIBBLE_BITS + plane)) & 1).reshape(
                last - first, tiles, groups, 4, 8
            )
            hi[first:last, :, :, plane] = (
                (bit * bit_weights).sum(dim=-1, dtype=torch.uint8).view(torch.int32)
            ).reshape(last - first, tiles, groups)
    if widest > max_spread(bits):
        raise ValueError(
            f"a scale row spans {widest} above its smallest scale, the {bits}-bit "
            f"code carries {max_spread(bits)}"
        )
    return PackedScales(groups=groups, base=base, nib=nib, hi=hi)


def decode_scales(scales: PackedScales) -> torch.Tensor:
    """The u8 raster [E, tiles, groups, 32] a `PackedScales` stands for."""
    experts, tiles, groups = scales.num_experts, scales.tiles, scales.groups
    device = scales.base.device
    nibbles = torch.stack((scales.nib & 15, scales.nib >> 4), dim=3)
    nibbles = nibbles.reshape(experts, tiles, groups, 32)
    positions = torch.arange(32, device=device, dtype=torch.int32)
    plane_bits = (scales.hi[..., None] >> positions) & 1  # [E, T, G, planes, 32]
    plane_weights = 1 << (
        _NIBBLE_BITS + torch.arange(scales.hi.shape[-1], device=device)
    )
    high = (plane_bits * plane_weights[:, None]).sum(dim=-2).to(torch.uint8)
    return scales.base[:, :, None, :] + nibbles + high
