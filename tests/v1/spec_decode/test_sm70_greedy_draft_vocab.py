# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

import vllm.v1.spec_decode.sm70_greedy_draft_vocab as vocab_module
from vllm.v1.spec_decode.sm70_greedy_draft_vocab import (
    Sm70GreedyDraftVocab,
    select_local_pair,
)


def test_local_argmax_uses_original_ids_for_shuffled_ties():
    logits = torch.tensor([[3.0, 3.0, 1.0], [1.0, 2.0, 2.0]])
    ids = torch.tensor([7, 2, 5])
    assert select_local_pair(logits, ids).tolist() == [[3.0, 2.0], [2.0, 2.0]]


def test_batched_tp_pairs_preserve_original_global_tie_break():
    logits = torch.tensor([[2.0, 2.0, 1.0, 0.0], [1.0, 0.0, 4.0, 4.0]])
    ids = torch.tensor([[30, 10, 20, 0], [3, 2, 1, 0]])
    assert select_local_pair(logits, ids).tolist() == [[2.0, 10.0], [4.0, 0.0]]


def test_bootstrap_keeps_highest_priority_and_skips_nonfinite_ids():
    updated = []
    bridge = Sm70GreedyDraftVocab.__new__(Sm70GreedyDraftVocab)
    bridge.runtime = SimpleNamespace(
        update=lambda candidates: updated.append(candidates)
    )
    bridge.observe_target_logits(
        torch.tensor([[0.0, 8.0, -torch.inf, 4.0]]), prefill=True
    )
    assert updated[0].tolist() == [[-1, 0, 3, 1]]


def test_greedy_head_merges_base_and_shard_tail_without_dense_gather(monkeypatch):
    head = SimpleNamespace(
        quant_method=SimpleNamespace(apply=lambda *_: torch.tensor([[4.0, 4.0]]))
    )
    bridge = Sm70GreedyDraftVocab.__new__(Sm70GreedyDraftVocab)
    bridge.runtime = SimpleNamespace(
        lm_head=head,
        local_tail_weight=torch.tensor([[4.0, 0.0], [0.0, 8.0], [99.0, 99.0]]),
        local_tail_token_ids=torch.tensor([3, 20, -1]),
        token_id_map=torch.tensor([9, 2]),
    )
    bridge._base_start = 0
    bridge._base_end = 2
    observed = []

    def gather(pair, dim):
        observed.append(pair.clone())
        assert pair.shape == (1, 2)
        # Remote rank wins a tie using lower original ID.
        return torch.cat((pair, torch.tensor([[8.0, 10.0]])), dim=dim)

    monkeypatch.setattr(vocab_module, "tensor_model_parallel_all_gather", gather)
    assert bridge.get_top_tokens(torch.tensor([[1.0, 1.0]])).tolist() == [10]
    assert observed[0].tolist() == [[8.0, 20.0]]
