# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter

import pytest

from benchmarks.build_sm70_mtp_draft_vocab import mix_target_counts, rank_vocab


def test_heldout_counts_do_not_change_training_ranking():
    counts = {
        ("en", "train"): Counter({0: 20, 1: 10, 2: 1}),
        ("zh", "train"): Counter({3: 5, 4: 1}),
        ("zh", "heldout"): Counter({2: 1000000}),
    }
    assert rank_vocab(counts, 3, [4], {"en": 0.5, "zh": 0.5}) == [0, 3, 4]
    counts["zh", "heldout"] = Counter({1: 1})
    assert rank_vocab(counts, 3, [4], {"en": 0.5, "zh": 0.5}) == [0, 3, 4]


def test_independent_target_outputs_change_priority_without_changing_corpus():
    source = {
        ("en", "train"): Counter({0: 20, 1: 10, 2: 1}),
        ("en", "heldout"): Counter({1: 1000000}),
    }
    mixed = mix_target_counts(
        source,
        [{"id": "vocab-train/en/0", "language": "en", "token_ids": [2, 2, 2]}],
    )
    assert rank_vocab(mixed, 1, [], {"en": 1}) == [2]
    assert source["en", "train"] == Counter({0: 20, 1: 10, 2: 1})
    assert mixed["en", "heldout"] == source["en", "heldout"]


def test_quality_outputs_cannot_be_used_as_training():
    with pytest.raises(ValueError, match="independent"):
        mix_target_counts(
            {("en", "train"): Counter({0: 1})},
            [{"id": "gsm8k/gsm8k_588", "language": "en", "token_ids": [0]}],
        )
