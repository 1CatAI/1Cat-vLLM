# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V4.1 pipeline split admits only boundaries that ship nothing but HC."""

from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v41.contract import pp_split_problem


def _target():
    return SimpleNamespace(
        num_hidden_layers=40,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        engram_layer_ids=[1, 14],
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
    )


def test_only_layer_20_splits_the_checkpoint():
    config = _target()
    assert [s for s in range(41) if pp_split_problem(config, s) is None] == [20]


@pytest.mark.parametrize(
    "split,reason",
    [
        (0, "inside"),
        (40, "inside"),
        (19, "KV source"),
        (21, "KV source"),
        (24, "KV source"),
        (14, "Engram"),
        (8, "Engram"),
        (2, "Engram"),
    ],
)
def test_straddling_boundaries_name_their_reason(split, reason):
    assert reason in pp_split_problem(_target(), split)


def test_ratio_two_state_must_stay_on_the_first_stage():
    config = _target()
    config.engram_layer_ids = [1]
    # 14 is a KV and index source with Engram moved below it, but layers
    # 14-19 still commit ratio-2 compressor state.
    assert "ratio-2" in pp_split_problem(config, 14)


def test_staged_state_admits_balanced_pp4_10_10_11_9():
    config = _target()
    # Under native staged_state=True, cuts inside both ratio-2 (layer 10 -> mirror 8)
    # and ratio-1 (layer 31 -> mirror 20) compressed sources are admissible, while
    # layers 0 and 1 (uncompressed window-only) are rejected.
    assert pp_split_problem(config, 1, staged_state=True) is not None
    for split in (8, 10, 14, 20, 31, 32):
        assert pp_split_problem(config, split, staged_state=True) is None


def test_staged_state_admits_11_11_11_7():
    config = _target()
    # Balanced 11,11,11,7 boundaries (11, 22, 33) offload 2 target layers from
    # Stage 3 while keeping DSpark target tap layers (37, 38, 39) on Stage 3.
    for split in (11, 22, 33):
        assert pp_split_problem(config, split, staged_state=True) is None
