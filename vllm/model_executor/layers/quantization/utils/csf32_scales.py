# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt ``csf_32x2_v1`` NVFP4 block scales: lossless decode and encode.

A scale matrix of R rows by C groups (C = K / 16) is stored as five tensors:
``bases`` (u8 [R]), ``codes`` (u8 [R*C/2], four-bit offsets), ``page_bases``
(u32 [P]), ``offsets`` (u16 [B+P]) and ``exceptions`` (u16 records). A block
is 32 rows by one column pair, a tile 4 block rows by 2 column pairs, a page
32 tiles. Blocks are streamed tile by tile (tiles row-major); inside a tile
the first column pair's block rows come first. A scale byte is its row's base
plus its offset unless an exception record (value << 8 | 2*row + column within
the block) replaces it. Each block owns the records ``offsets[e]..offsets[e+1]``
counted from its page's origin ``page_bases[page] / 2``; every page adds one
terminal offset.

Reference: local-inference-lab ``docs/csf-scale-blocks.md`` as implemented by
TensorFold (Apache-2.0),
https://github.com/ashhart/TensorFold/blob/main/zig/src/core/nvfp4_csf32.zig
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

BLOCK_ROWS = 32
TILE_BLOCK_ROWS = 4
TILE_PAIRS = 2
PAGE_TILES = 32
MAX_BASE = 240


@dataclass(frozen=True)
class Csf32Layout:
    """Stream order of a scale matrix's blocks, shared by every matrix of a shape.

    Arrays are indexed by the block's position in the stream.
    """

    rows: int
    cols: int
    block_row: np.ndarray  # first row / 32
    block_pair: np.ndarray  # column pair
    block_width: np.ndarray  # 2, or 1 for the last pair of an odd column count
    block_page: np.ndarray
    field_start: np.ndarray  # first four-bit field of the block in `codes`

    @property
    def num_blocks(self) -> int:
        return int(self.block_row.shape[0])

    @property
    def num_pages(self) -> int:
        return int(self.block_page[-1]) + 1


@lru_cache(maxsize=16)
def csf32_layout(rows: int, cols: int) -> Csf32Layout:
    if rows <= 0 or rows % BLOCK_ROWS or cols <= 0:
        raise ValueError(
            f"csf_32x2_v1 needs rows % 32 == 0 and cols > 0, got {rows}x{cols}"
        )
    block_rows = rows // BLOCK_ROWS
    pairs = (cols + 1) // 2
    order_row, order_pair, order_tile = [], [], []
    tile = 0
    for tile_row in range(0, block_rows, TILE_BLOCK_ROWS):
        height = min(TILE_BLOCK_ROWS, block_rows - tile_row)
        for tile_pair in range(0, pairs, TILE_PAIRS):
            width = min(TILE_PAIRS, pairs - tile_pair)
            for at in range(height * width):
                order_row.append(tile_row + at % height)
                order_pair.append(tile_pair + at // height)
                order_tile.append(tile)
            tile += 1
    block_row = np.asarray(order_row, dtype=np.int64)
    block_pair = np.asarray(order_pair, dtype=np.int64)
    block_width = np.minimum(2, cols - 2 * block_pair)
    fields = BLOCK_ROWS * block_width
    field_start = np.concatenate(([0], np.cumsum(fields)[:-1]))
    return Csf32Layout(
        rows,
        cols,
        block_row,
        block_pair,
        block_width,
        np.asarray(order_tile, dtype=np.int64) // PAGE_TILES,
        field_start,
    )


def _element_fields(layout: Csf32Layout) -> np.ndarray:
    """Four-bit field index of every scale, row-major [R, C]."""
    block_index = np.empty(
        (layout.rows // BLOCK_ROWS, (layout.cols + 1) // 2), np.int64
    )
    block_index[layout.block_row, layout.block_pair] = np.arange(layout.num_blocks)
    r = np.arange(layout.rows)[:, None]
    c = np.arange(layout.cols)[None, :]
    b = block_index[r // BLOCK_ROWS, c // 2]
    return layout.field_start[b] + (r % BLOCK_ROWS) * layout.block_width[b] + c % 2


@lru_cache(maxsize=16)
def _element_fields_cached(rows: int, cols: int) -> np.ndarray:
    return _element_fields(csf32_layout(rows, cols))


def decode_csf32(
    bases: torch.Tensor,
    codes: torch.Tensor,
    page_bases: torch.Tensor,
    offsets: torch.Tensor,
    exceptions: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Scale bytes [rows, cols] (row-major uint8) of one matrix, on its device.

    Raises ValueError when the parts disagree with the geometry or index.
    """
    layout = csf32_layout(rows, cols)
    device = codes.device
    bases = bases.view(torch.uint8).reshape(-1)
    codes = codes.view(torch.uint8).reshape(-1)
    page_bases = page_bases.view(torch.int32).reshape(-1).long()
    offsets = offsets.view(torch.int16).reshape(-1).long() & 0xFFFF
    records = exceptions.view(torch.int16).reshape(-1).long() & 0xFFFF
    if (
        bases.numel() != rows
        or codes.numel() != rows * cols // 2
        or page_bases.numel() != layout.num_pages
        or offsets.numel() != layout.num_blocks + layout.num_pages
    ):
        raise ValueError(f"csf_32x2_v1 parts disagree with a {rows}x{cols} matrix")
    if int(bases.max()) > MAX_BASE:
        raise ValueError("csf_32x2_v1 row base above 240")

    fields = torch.from_numpy(_element_fields_cached(rows, cols)).to(device)
    nibbles = (codes[fields // 2].long() >> (4 * (fields % 2))) & 0xF
    out = (bases.long()[:, None] + nibbles).to(torch.uint8)

    block_page = torch.from_numpy(layout.block_page).to(device)
    entry = torch.arange(layout.num_blocks, device=device) + block_page
    low, high = offsets[entry], offsets[entry + 1]
    counts = high - low
    if bool((counts < 0).any()):
        raise ValueError("csf_32x2_v1 block prefixes decrease")
    origin = page_bases[block_page] // 2
    expected_start = torch.cumsum(counts, 0) - counts
    if (
        not torch.equal(origin + low, expected_start)
        or int(counts.sum()) != records.numel()
    ):
        raise ValueError(
            "csf_32x2_v1 exception index does not cover the records in order"
        )
    if records.numel() == 0:
        return out
    record_block = torch.repeat_interleave(
        torch.arange(layout.num_blocks, device=device), counts
    )
    at = records & 0xFF
    width = torch.from_numpy(layout.block_width).to(device)[record_block]
    if bool(((at // 2 >= BLOCK_ROWS) | (at % 2 >= width)).any()):
        raise ValueError("csf_32x2_v1 exception outside its block")
    row = (
        torch.from_numpy(layout.block_row).to(device)[record_block] * BLOCK_ROWS
        + at // 2
    )
    col = torch.from_numpy(layout.block_pair).to(device)[record_block] * 2 + at % 2
    out[row, col] = (records >> 8).to(torch.uint8)
    return out


def encode_csf32(scale: np.ndarray) -> dict[str, np.ndarray]:
    """Parts of a uint8 [R, C] scale matrix, byte-identical to the reference encoder.

    A row's base starts its most covering 16-wide window, the lowest on a tie.
    Used by the tests; checkpoints arrive encoded.
    """
    rows, cols = scale.shape
    layout = csf32_layout(rows, cols)
    bases = np.zeros(rows, dtype=np.uint8)
    for r in range(rows):
        hist = np.bincount(scale[r], minlength=256)
        cover = np.convolve(hist, np.ones(16, dtype=np.int64))[15 : 15 + MAX_BASE + 1]
        bases[r] = int(np.argmax(cover))
    codes = np.zeros(rows * cols // 2 + (rows * cols) % 2, dtype=np.uint8)
    page_bases = np.zeros(layout.num_pages, dtype=np.uint32)
    offsets = np.zeros(layout.num_blocks + layout.num_pages, dtype=np.uint16)
    exceptions: list[int] = []
    entry = 0
    origin = 0
    previous_page = -1
    for b in range(layout.num_blocks):
        page = int(layout.block_page[b])
        if page != previous_page:
            if previous_page >= 0:
                entry += 1
            origin = len(exceptions)
            page_bases[page] = 2 * origin
            offsets[entry] = 0
            previous_page = page
        row0 = int(layout.block_row[b]) * BLOCK_ROWS
        col0 = int(layout.block_pair[b]) * 2
        width = int(layout.block_width[b])
        field = int(layout.field_start[b])
        for r in range(BLOCK_ROWS):
            base = int(bases[row0 + r])
            for c in range(width):
                value = int(scale[row0 + r, col0 + c])
                if base <= value < base + 16:
                    offset = value - base
                else:
                    exceptions.append(value << 8 | (2 * r + c))
                    offset = 0
                codes[field // 2] |= offset << (4 * (field % 2))
                field += 1
        entry += 1
        offsets[entry] = len(exceptions) - origin
    return {
        "bases": bases,
        "codes": codes,
        "page_bases": page_bases,
        "offsets": offsets,
        "exceptions": np.asarray(exceptions, dtype=np.uint16),
    }
