# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expand ``csf_32x2_v1`` expert scales into the skinny QPN scale raster.

The compressed parts of one MoE layer stay on the GPU. Before each grouped
skinny call, the experts of its routing groups are expanded into a shared
raster laid out as `qpn_prepack` leaves the checkpoint scales:
``[expert][tile N/32][group K/16][lane 32]`` bytes. Group slots whose range
is empty (``goff[g] >= num_slots``) are skipped, so the launch has a fixed
shape and is CUDA-graph safe.

Only full CSF tiles are supported (rows % 128 == 0, groups % 4 == 0), which
covers every routed-expert shape of the GLM-5.3 checkpoints.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from vllm.triton_utils import tl, triton

CSF32_SCALE_ENCODING = "csf_32x2_v1"

# A CSF block is 32 rows x 2 groups; a tile 4 block rows x 2 column pairs.
_BLOCK_SCALES = 64
_TILE_BLOCKS = 8
_TILE_SCALES = _TILE_BLOCKS * _BLOCK_SCALES
_RECORDS_PER_STEP = 256


@dataclass
class CsfExpertScales:
    """One layer's compressed scale planes, matrices in (expert, projection) order.

    ``projections`` matrices per expert share one geometry (rows x groups);
    their rows stack in the destination raster (gate then up for w13).
    """

    rows: int
    groups: int
    projections: int
    bases: torch.Tensor  # u8 [M, rows]
    codes: torch.Tensor  # u8 [M, rows * groups / 2]
    page_bases: torch.Tensor  # i32 [M, pages]
    offsets: torch.Tensor  # i32 [M, blocks + pages] (u16 widened)
    exceptions: torch.Tensor  # i32 [total records] (u16 widened)
    exception_start: torch.Tensor  # i64 [M]
    exception_count: torch.Tensor  # i32 [M]

    @property
    def num_blocks(self) -> int:
        return self.rows // 32 * (self.groups // 2)

    @property
    def num_pages(self) -> int:
        return self.page_bases.shape[1]


@runtime_checkable
class CsfScaleConsumer(Protocol):
    """MoE experts that serve csf_32x2_v1 expert scales from compressed planes."""

    def attach_csf_scales(self, w13: CsfExpertScales, w2: CsfExpertScales) -> None:
        """Take the layer's planes; called before process_weights_after_loading."""


# csf32 part name -> CsfExpertScales field; exceptions are concatenated per layer.
_FIXED_PARTS = ("bases", "codes", "page_bases", "offsets")


class CsfExpertScaleLoader:
    """Collects one MoE layer's csf32 parts straight into GPU planes.

    The fixed-size parts land in preallocated stacks as they arrive; the
    variable exception records are concatenated by `finish`.
    """

    def __init__(
        self,
        num_experts: int,
        projections: int,
        rows: int,
        groups: int,
        device: torch.device,
    ):
        if rows % 128 or groups % 4:
            raise ValueError(
                f"csf_32x2_v1 expert scales need rows % 128 and groups % 4, "
                f"got {rows}x{groups}"
            )
        self.num_experts, self.projections = num_experts, projections
        self.rows, self.groups = rows, groups
        blocks = rows // 32 * (groups // 2)
        pages = blocks // (8 * 32) + (blocks % (8 * 32) > 0)
        matrices = num_experts * projections
        self.parts = {
            "bases": torch.empty(matrices, rows, dtype=torch.uint8, device=device),
            "codes": torch.empty(
                matrices, rows * groups // 2, dtype=torch.uint8, device=device
            ),
            "page_bases": torch.empty(
                matrices, pages, dtype=torch.int32, device=device
            ),
            "offsets": torch.empty(
                matrices, blocks + pages, dtype=torch.int32, device=device
            ),
        }
        self.exceptions: dict[int, torch.Tensor] = {}
        self.seen = torch.zeros(matrices, len(_FIXED_PARTS) + 1, dtype=torch.bool)

    def load(self, expert: int, projection: int, part: str, tensor: torch.Tensor):
        matrix = expert * self.projections + projection
        if part == "exceptions":
            self.exceptions[matrix] = _widen_u16(tensor).to(self.parts["codes"].device)
            self.seen[matrix, -1] = True
            return
        target = self.parts[part][matrix]
        if part == "offsets":
            tensor = _widen_u16(tensor)
        elif part == "page_bases":
            tensor = tensor.view(torch.int32)
        if tensor.numel() != target.numel():
            raise ValueError(
                f"csf_32x2_v1 {part} of expert {expert} has {tensor.numel()} "
                f"entries, expected {target.numel()}"
            )
        target.copy_(tensor.reshape(-1))
        self.seen[matrix, _FIXED_PARTS.index(part)] = True

    def finish(self) -> CsfExpertScales:
        missing = (~self.seen).nonzero()
        if missing.numel():
            matrix, part = missing[0].tolist()
            name = (*_FIXED_PARTS, "exceptions")[part]
            raise ValueError(
                f"csf_32x2_v1 {name} missing for expert "
                f"{matrix // self.projections} ({missing.shape[0]} parts missing)"
            )
        ordered = [self.exceptions[matrix] for matrix in range(self.seen.shape[0])]
        counts = torch.tensor([t.numel() for t in ordered], dtype=torch.int64)
        starts = torch.cumsum(counts, 0) - counts
        device = self.parts["codes"].device
        exceptions = torch.cat(ordered)
        self.exceptions = {}
        return CsfExpertScales(
            rows=self.rows,
            groups=self.groups,
            projections=self.projections,
            exceptions=exceptions,
            exception_start=starts.to(device),
            exception_count=counts.to(torch.int32).to(device),
            **self.parts,
        )


def _widen_u16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int16).reshape(-1).to(torch.int32) & 0xFFFF


_SHARED_RASTERS: dict[tuple, torch.Tensor] = {}


def shared_scale_raster(
    device: torch.device, num_experts: int, rows: int, groups: int
) -> torch.Tensor:
    """The fp8-e4m3 [experts, rows, groups] raster every layer of a shape shares.

    Layers run one after another on one stream, and each expands its routed
    experts right before reading them, so one raster per device and shape
    serves all of them.
    """
    key = (device, num_experts, rows, groups)
    raster = _SHARED_RASTERS.get(key)
    if raster is None:
        raster = torch.zeros(
            num_experts, rows, groups, dtype=torch.float8_e4m3fn, device=device
        )
        _SHARED_RASTERS[key] = raster
    return raster


@triton.jit
def _qpn_lane(r):
    # Inverse of qpn_prepack's column order
    # col(lane) = 8*((lane>>2)&3) + (lane&3) + 4*(lane>>4).
    return (r & 3) | ((r >> 3) << 2) | (((r >> 2) & 1) << 4)


@triton.jit
def _csf32_expand_kernel(
    out_ptr,
    out_expert_stride,
    gids_ptr,
    goff_ptr,
    num_slots,
    bases_ptr,
    codes_ptr,
    ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
    PROJECTIONS: tl.constexpr,
):
    slot = tl.program_id(0)
    if tl.load(goff_ptr + slot) >= num_slots:
        return
    expert = tl.load(gids_ptr + slot).to(tl.int64)
    tile_id = tl.program_id(1)
    tiles_per_matrix: tl.constexpr = (ROWS // 128) * (GROUPS // 4)
    projection = tile_id // tiles_per_matrix
    tile = tile_id % tiles_per_matrix
    matrix = expert * PROJECTIONS + projection
    tile_pairs: tl.constexpr = GROUPS // 4

    e = tl.arange(0, 512)
    at = e // 64
    field = e % 64
    r = field // 2
    c = field % 2
    block = tile * 8 + at
    block_row = (tile // tile_pairs) * 4 + at % 4
    pair = (tile % tile_pairs) * 2 + at // 4
    row = block_row * 32 + r
    code = tl.load(codes_ptr + matrix * (ROWS * GROUPS // 2) + block * 32 + field // 2)
    nibble = (code.to(tl.int32) >> (4 * (field % 2))) & 0xF
    base = tl.load(bases_ptr + matrix * ROWS + row).to(tl.int32)
    value = (base + nibble).to(tl.uint8)

    out_tile = projection * (ROWS // 32) + block_row
    index = (out_tile * GROUPS + pair * 2 + c) * 32 + _qpn_lane(r)
    tl.store(out_ptr + expert * out_expert_stride + index, value)


@triton.jit
def _csf32_exceptions_kernel(
    out_ptr,
    out_expert_stride,
    gids_ptr,
    goff_ptr,
    num_slots,
    page_bases_ptr,
    offsets_ptr,
    exceptions_ptr,
    exception_start_ptr,
    exception_count_ptr,
    ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
    PROJECTIONS: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    STEP: tl.constexpr,
):
    slot = tl.program_id(0)
    if tl.load(goff_ptr + slot) >= num_slots:
        return
    expert = tl.load(gids_ptr + slot).to(tl.int64)
    projection = tl.program_id(1)
    matrix = expert * PROJECTIONS + projection
    num_blocks: tl.constexpr = (ROWS // 32) * (GROUPS // 2)
    tile_pairs: tl.constexpr = GROUPS // 4
    start = tl.load(exception_start_ptr + matrix)
    count = tl.load(exception_count_ptr + matrix)
    offsets_row = offsets_ptr + matrix * (num_blocks + NUM_PAGES)
    pages_row = page_bases_ptr + matrix * NUM_PAGES

    for first in range(0, count, STEP):
        k = first + tl.arange(0, STEP)
        valid = k < count
        record = tl.load(exceptions_ptr + start + k, mask=valid, other=0)
        # The block owning record k: the last block whose first record is <= k.
        low = tl.zeros([STEP], dtype=tl.int32)
        high = tl.full([STEP], num_blocks, dtype=tl.int32)
        for _ in range(SEARCH_STEPS):
            mid = (low + high) // 2
            page = mid // (8 * 32)
            first_record = tl.load(pages_row + page) // 2 + tl.load(
                offsets_row + mid + page
            )
            go_right = first_record <= k
            low = tl.where(go_right, mid, low)
            high = tl.where(go_right, high, mid)
        block = low
        tile = block // 8
        at = block % 8
        block_row = (tile // tile_pairs) * 4 + at % 4
        pair = (tile % tile_pairs) * 2 + at // 4
        position = record & 0xFF
        r = position // 2
        c = position % 2
        out_tile = projection * (ROWS // 32) + block_row
        index = (out_tile * GROUPS + pair * 2 + c) * 32 + _qpn_lane(r)
        tl.store(
            out_ptr + expert * out_expert_stride + index,
            (record >> 8).to(tl.uint8),
            mask=valid,
        )


def expand_csf32_experts(
    out: torch.Tensor,
    planes: CsfExpertScales,
    gids: torch.Tensor,
    goff: torch.Tensor,
) -> None:
    """Write the QPN scale rasters of the experts named by non-empty groups.

    Args:
        out: u8 [num_experts, projections * rows * groups] shared raster.
        planes: the layer's compressed planes.
        gids: int32 [S] expert of each routing group.
        goff: int32 [S + 1] group offsets; empty groups hold S.
    """
    if planes.rows % 128 or planes.groups % 4:
        raise ValueError(
            "CSF expansion supports full tiles only (rows % 128, groups % 4)"
        )
    num_slots = gids.shape[0]
    tiles = planes.projections * (planes.rows // 128) * (planes.groups // 4)
    _csf32_expand_kernel[(num_slots, tiles)](
        out,
        out.stride(0),
        gids,
        goff,
        num_slots,
        planes.bases,
        planes.codes,
        ROWS=planes.rows,
        GROUPS=planes.groups,
        PROJECTIONS=planes.projections,
    )
    _csf32_exceptions_kernel[(num_slots, planes.projections)](
        out,
        out.stride(0),
        gids,
        goff,
        num_slots,
        planes.page_bases,
        planes.offsets,
        planes.exceptions,
        planes.exception_start,
        planes.exception_count,
        ROWS=planes.rows,
        GROUPS=planes.groups,
        PROJECTIONS=planes.projections,
        NUM_PAGES=planes.num_pages,
        SEARCH_STEPS=max(1, (planes.num_blocks - 1).bit_length()),
        STEP=_RECORDS_PER_STEP,
    )
