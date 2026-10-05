# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter

from benchmarks.build_sm70_mtp_draft_vocab import rank_vocab


def test_heldout_counts_do_not_change_training_ranking():
    counts = {
        ("en", "train"): Counter({0: 20, 1: 10, 2: 1}),
        ("zh", "train"): Counter({3: 5, 4: 1}),
        ("zh", "heldout"): Counter({2: 1000000}),
    }
    assert rank_vocab(counts, 3, [4], {"en": 0.5, "zh": 0.5}) == [0, 3, 4]
    counts["zh", "heldout"] = Counter({1: 1})
    assert rank_vocab(counts, 3, [4], {"en": 0.5, "zh": 0.5}) == [0, 3, 4]
