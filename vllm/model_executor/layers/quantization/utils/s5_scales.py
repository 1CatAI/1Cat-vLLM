# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Five-bit expert scales ("S5") that the skinny QPN MoE kernels read in place.

A scale raster holds one fp8-e4m3 byte per (tile row, K group): ``[expert][tile
N/32][group K/16][position 32]``, the layout `qpn_prepack` gives the checkpoint
scales. The csf_32x2_v1 checkpoint planes are smaller, but a few scales per
thousand fall outside a row's four-bit window and are patched from a record list,
which a kernel cannot afford to look up in its loop (one block in six has a
record). The scale bytes of a row do span a short range, though: the group scales
of GLM-5.3 differ from their row's smallest by at most 23, and the e4m3 bytes of
positive scales order like integers. Five bits per scale therefore carry every
scale without exceptions:

    scale = base[expert][tile][position] + nibble + 16 * high

``nibble`` is four bits, two groups per byte (``nib``: low nibble the even group),
``high`` one bit per (group, position), a 32-bit word per (tile, group) (``hi``).
Positions are the raster positions of the card's fragment order, so a lane
reads the bytes it would read from the raster and decodes them with three integer
operations. The planes take 5/8 of the raster, and no raster has to exist for the
experts of a layer.
"""

from dataclasses import dataclass

import torch

S5_MAX_SPREAD = 31
# Experts per encoding step: bounds the temporaries of the load-time conversion.
_ENCODE_EXPERTS = 16


@dataclass
class S5Scales:
    """One projection set's scale planes; ``tiles`` are N/32 tiles per expert."""

    groups: int
    base: torch.Tensor  # u8 [E, tiles, 32]
    nib: torch.Tensor  # u8 [E, tiles, groups / 2, 32]
    hi: torch.Tensor  # i32 [E, tiles, groups]

    @property
    def num_experts(self) -> int:
        return self.base.shape[0]

    @property
    def tiles(self) -> int:
        return self.base.shape[1]


def encode_s5(raster: torch.Tensor, groups: int) -> S5Scales:
    """Encode a u8 scale raster [E, tiles * groups * 32] (kernel positions).

    Raises if a row of any expert spans more than `S5_MAX_SPREAD` between its
    smallest and largest scale: the five-bit code would not carry it.
    """
    if raster.dtype != torch.uint8 or groups % 2:
        raise ValueError("S5 encodes a u8 raster with an even group count")
    experts = raster.shape[0]
    tiles = raster.numel() // (experts * groups * 32)
    if tiles * groups * 32 * experts != raster.numel():
        raise ValueError("raster size is not experts x tiles x groups x 32")
    device = raster.device
    base = torch.empty(experts, tiles, 32, dtype=torch.uint8, device=device)
    nib = torch.empty(experts, tiles, groups // 2, 32, dtype=torch.uint8, device=device)
    hi = torch.empty(experts, tiles, groups, dtype=torch.int32, device=device)
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
        # bit p of a (tile, group) word is position p's high bit
        high = (delta >> 4).reshape(last - first, tiles, groups, 4, 8)
        hi[first:last] = (
            (high * bit_weights).sum(dim=-1, dtype=torch.uint8).view(torch.int32)
        ).reshape(last - first, tiles, groups)
    if widest > S5_MAX_SPREAD:
        raise ValueError(
            f"a scale row spans {widest} above its smallest scale, the five-bit "
            f"code carries {S5_MAX_SPREAD}"
        )
    return S5Scales(groups=groups, base=base, nib=nib, hi=hi)


def decode_s5(scales: S5Scales) -> torch.Tensor:
    """The u8 raster [E, tiles, groups, 32] an `S5Scales` stands for."""
    experts, tiles, groups = scales.num_experts, scales.tiles, scales.groups
    device = scales.base.device
    nibbles = torch.stack((scales.nib & 15, scales.nib >> 4), dim=3)
    nibbles = nibbles.reshape(experts, tiles, groups, 32)
    positions = torch.arange(32, device=device, dtype=torch.int32)
    high = ((scales.hi[..., None] >> positions) & 1).to(torch.uint8)
    return scales.base[:, :, None, :] + nibbles + 16 * high
