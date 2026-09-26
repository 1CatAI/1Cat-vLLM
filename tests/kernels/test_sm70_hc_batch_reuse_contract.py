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
