# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.model_executor.layers.quantization.gguf_shared_layout import (
    SharedExpertBudget,
    SharedExpertLayout,
)


@pytest.mark.parametrize("kind", [16, 17, 29])
@pytest.mark.parametrize("role", ["w1", "w3", "w2"])
def test_v41_roundtrip_all_ranks_have_exactly_one_writer(kind, role):
    random = np.random.default_rng(41)
    first = SharedExpertLayout.for_type(5120, 2304, 4, 0, role, kind)
    shape = (first.source_rows, first.source_row_bytes)
    original = random.integers(0, 256, shape, dtype=np.uint8)
    restored = np.zeros_like(original)
    coverage = np.zeros_like(original)
    for rank in range(4):
        layout = SharedExpertLayout.for_type(5120, 2304, 4, rank, role, kind)
        packed = layout.pack(original)
        mask = np.zeros_like(original)
        layout.writeback(mask, np.ones_like(packed))
        coverage += mask
        # Bytes carried for another rank must never overwrite that rank's data.
        owned = (layout.write_end_block - layout.first_block) * layout.block_bytes
        packed[:, owned:] ^= 0xFF
        layout.writeback(restored, packed)
        if role == "w2":
            assert layout.physical_k == 768
            assert layout.logical_k_offset == rank * 64
            assert layout.guard_bytes == layout.block_bytes
        else:
            assert layout.physical_k == 5120 and layout.guard_bytes == 0
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))
    np.testing.assert_array_equal(restored, original)


def test_one_source_block_shared_by_all_four_ranks_still_has_one_writer():
    original = np.arange(32 * 66, dtype=np.uint8).reshape(32, 66)
    restored = np.zeros_like(original)
    writes = []
    for rank in range(4):
        layout = SharedExpertLayout.for_type(32, 256, 4, rank, "w2", 16)
        writes.append(layout.write_end_block - layout.first_block)
        layout.writeback(restored, layout.pack(original))
    assert writes == [0, 0, 0, 1]
    np.testing.assert_array_equal(restored, original)


@pytest.mark.parametrize(
    "hidden,intermediate,tp,rank,role",
    [
        (5120, 2303, 4, 0, "w2"),
        (5120, 2304, 4, 4, "w1"),
        (5121, 2304, 4, 0, "w3"),
        (5120, 2304, 4, 0, "unknown"),
    ],
)
def test_invalid_layout_is_rejected(hidden, intermediate, tp, rank, role):
    with pytest.raises(ValueError):
        SharedExpertLayout.for_type(hidden, intermediate, tp, rank, role, 16)


@pytest.mark.parametrize(
    "fraction,cold,slots,bytes_",
    [
        (0.30, 115, 277, 39_553_228_800),
        (0.25, 96, 296, 33_018_347_520),
        (0.20, 77, 315, 26_483_466_240),
    ],
)
def test_offload_budget_counts_original_bytes_and_separate_spares(
    fraction, cold, slots, bytes_
):
    # Synthetic layer distribution with the checkpoint's total expert bytes;
    # exact per-layer types remain the loader's responsibility.
    smallest, extra = divmod(343_941_120, 40)
    layers = (smallest + 1,) * extra + (smallest,) * (40 - extra)
    budget = SharedExpertBudget(384, fraction, layers)
    assert budget.cold_experts == cold
    assert budget.gpu_slots == slots
    assert budget.hot_experts + cold == 384
    assert budget.cold_bytes == bytes_
    assert budget.arena_bytes == bytes_ + 512 * 1024**2


@pytest.mark.parametrize(
    "updates",
    [
        {"offload_fraction": float("nan")},
        {"offload_fraction": 0.0},
        {"offload_fraction": 1.0},
        {"offload_fraction": 0.001},  # no cold experts after rounding
        {"offload_fraction": 0.01},  # cold set smaller than spare allocation
        {"offload_fraction": 0.95},  # cannot hold M6's routed set
        {"exchange_bytes": 44 * 10_229_760 - 1},
        {"expert_bytes": ()},
        {"expert_bytes": (0,)},
        {"spare_slots": 0},
    ],
)
def test_invalid_shared_capacity_fails_before_allocating(updates):
    values: dict = dict(
        num_experts=384, offload_fraction=0.3, expert_bytes=(10_229_760,) * 40
    )
    values.update(updates)
    with pytest.raises(ValueError):
        SharedExpertBudget(**values)
