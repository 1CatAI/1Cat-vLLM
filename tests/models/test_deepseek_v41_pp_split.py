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
