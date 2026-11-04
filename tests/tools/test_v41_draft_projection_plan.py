# SPDX-License-Identifier: Apache-2.0
"""Arithmetic acceptance for output-row sharding; no GPU or replacement op."""

import pytest
import torch

from vllm.models.deepseek_v41 import draft_projection as projection


def projection_weights():
    return dict(
        weight=torch.empty(5120, 15360, dtype=torch.float8_e4m3fn, device="meta"),
        scales=torch.empty(160, 480, dtype=torch.float8_e8m0fnu, device="meta"),
    )


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_shards_cover_full_weight_once_and_keep_full_call_split(world):
    plans = [
        projection.projection_plan(5120, 15360, 6, world, rank) for rank in range(world)
    ]
    assert [(p.first, p.last) for p in plans] == [
        (r * (5120 // world), (r + 1) * (5120 // world)) for r in range(world)
    ]
    assert all(p.split == 2 for p in plans)  # ceil(240 / ceil(5120 / 32)) = 2


@pytest.mark.parametrize(
    "args",
    [
        (5120, 15360, 6, 4, -1),
        (5120, 15360, 6, 4, 4),
        (5120, 15360, 0, 4, 0),
        (5120, 15360, 9, 4, 0),
        (5120, 15360, 6, 3, 0),
        (5120, 15359, 6, 4, 0),
    ],
)
def test_invalid_geometry_refuses(args):
    with pytest.raises(ValueError):
        projection.projection_plan(*args)


@pytest.mark.parametrize(
    "world, gemv_ready, biased, message",
    [(1, True, False, "TP"), (4, False, False, "GEMV"), (4, True, True, "bias-free")],
)
def test_construction_admission_refuses(world, gemv_ready, biased, message):
    with pytest.raises(ValueError, match=message):
        projection.admit_projection(
            world, gemv_ready, biased, 5120, 15360, **projection_weights()
        )


def test_construction_admission_accepts_supported_projection():
    projection.admit_projection(4, True, False, 5120, 15360, **projection_weights())
