# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU layout checks only; these do NOT prove CUDA arithmetic or performance."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hc_batch_reuse_benchmark",
    ROOT / "benchmarks/kernels/benchmark_sm70_hc_batch_reuse.py",
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


@pytest.mark.parametrize("hidden", [640, 2560])
def test_weight_packing_preserves_every_element(hidden):
    # Use int32 so each source position is distinct, unlike a half arange.
    weight = torch.arange(4 * hidden * 320, dtype=torch.int32).reshape(4, hidden, 320)
    packed = BENCH.pack_weight(weight)
    assert packed.is_contiguous()
    assert packed.dtype == weight.dtype
    restored = packed.permute(3, 0, 4, 1, 2, 5).reshape_as(weight)
    assert torch.equal(restored, weight)
    for tile in (0, hidden // 8 - 1):
        for group in (0, 7, 19):
            for lane in range(32):
                r = (lane & 3) + (4 if lane & 16 else 0)
                branch = (lane >> 2) & 3
                for khalf in (0, 1):
                    assert torch.equal(
                        packed[tile, group, khalf, branch, r],
                        weight[
                            branch,
                            tile * 8 + r,
                            group * 16 + khalf * 8 : group * 16 + (khalf + 1) * 8,
                        ],
                    )


@pytest.mark.parametrize("rows", [2, 4, 8, 16])
@pytest.mark.parametrize("paired", [False, True])
def test_fragment_mapping_has_one_writer_per_output(rows, paired):
    actual = []
    groups = [0] if paired else range((rows + 7) // 8)
    for group in groups:
        for p in range(2 if paired else 1):
            for lane in range(32):
                branch = (lane >> 2) & 3
                for i in range(8):
                    row = (group + p) * 8 + (
                        (i & 2) | (4 if lane & 16 else 0) | (lane & 1)
                    )
                    col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2)
                    if row < rows:
                        actual.append((row, branch, col))
                    # Warp shuffles preserve both coordinates across branches.
                    for source_branch in range(4):
                        source = (lane & ~12) | (source_branch << 2)
                        assert source & 19 == lane & 19
    expected = {(r, b, c) for r in range(rows) for b in range(4) for c in range(8)}
    assert len(actual) == len(expected)
    assert set(actual) == expected


def test_invalid_packing_rejected():
    with pytest.raises(ValueError, match="shape"):
        BENCH.pack_weight(torch.empty(2560, 320))
    with pytest.raises(ValueError, match="hidden"):
        BENCH.pack_weight(torch.empty(4, 1280, 320))


@pytest.mark.parametrize("tile_n", [16, 32])
@pytest.mark.parametrize("columns", [88, 336])
def test_down_weight_packing_preserves_every_element(tile_n, columns):
    weight = torch.arange(columns * 10240, dtype=torch.int32).reshape(columns, 10240)
    padded_n = 96 if columns == 88 else 352
    packed = BENCH.pack_down_weight(weight, tile_n)
    assert packed.is_contiguous()
    assert packed.dtype == weight.dtype
    restored = packed.permute(0, 3, 1, 2, 4).reshape(padded_n, 10240)
    assert torch.equal(restored[:columns], weight)
    assert torch.count_nonzero(restored[columns:]) == 0
    for tile in (0, padded_n // tile_n - 2, padded_n // tile_n - 1):
        for split in (0, 9, 19):
            for lane in range(32):
                r = (lane & 3) + (4 if lane & 16 else 0)
                quad = ((lane >> 2) & 3) % (tile_n // 8)
                for khalf in (0, 1):
                    group = split * 32
                    start = group * 16 + khalf * 8
                    assert torch.equal(
                        packed[tile, group, khalf, quad * 8 + r],
                        restored[tile * tile_n + quad * 8 + r, start : start + 8],
                    )


def test_down_weight_packing_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        BENCH.pack_down_weight(torch.empty(320, 10240))
    with pytest.raises(ValueError, match="columns"):
        BENCH.pack_down_weight(torch.empty(336, 10240), 8)


@pytest.mark.parametrize("rows", [2, 4, 8, 16])
def test_down_m16_fragment_mapping_has_one_writer_per_output(rows):
    coordinates = []
    for lane in range(32):
        quad = (lane >> 2) & 3
        for i in range(8):
            row = (quad // 2) * 8 + ((i & 2) | (4 if lane & 16 else 0) | (lane & 1))
            col = (quad % 2) * 8 + (
                (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2)
            )
            if row < rows:
                coordinates.append((row, col))
    assert len(coordinates) == rows * 16
    assert set(coordinates) == {(r, c) for r in range(rows) for c in range(16)}


def test_tp4_down_shards_preserve_lora_and_rank3_injection_ownership():
    weight = torch.arange(336 * 10240, dtype=torch.int32).reshape(336, 10240)
    shards = []
    for rank in range(4):
        packed = BENCH.pack_down_weight(weight[rank * 80 : rank * 80 + 88])
        shards.append(packed.permute(0, 3, 1, 2, 4).reshape(96, 10240))
    assert torch.equal(torch.cat([s[:80] for s in shards]), weight[:320])
    assert torch.equal(shards[3][80:84], weight[320:324])
