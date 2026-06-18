# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode import llm_base_proposer
from vllm.v1.spec_decode.llm_base_proposer import (
    _sync_draft_token_ids_across_tp,
    compute_probs_and_sample_next_token,
)


class FakeTPGroup:
    def __init__(self, source_token_ids: torch.Tensor):
        self.world_size = 2
        self.source_token_ids = source_token_ids
        self.broadcast_calls: list[tuple[torch.Tensor, int]] = []

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        self.broadcast_calls.append((input_.clone(), src))
        input_.copy_(self.source_token_ids.to(input_.device))
        return input_


def _random_sampling_metadata(batch_size: int) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.ones(batch_size),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0),
        presence_penalties=torch.empty(0),
        repetition_penalties=torch.empty(0),
        output_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_same_probs_different_random_numbers_can_diverge():
    probs = torch.tensor([[0.49, 0.51]], dtype=torch.float32)

    rank0_q = torch.tensor([[0.90, 0.10]], dtype=torch.float32)
    rank1_q = torch.tensor([[0.10, 0.90]], dtype=torch.float32)

    rank0_token = probs.div(rank0_q).argmax(dim=-1)
    rank1_token = probs.div(rank1_q).argmax(dim=-1)

    assert rank0_token.item() == 1
    assert rank1_token.item() == 0


def test_tp_sync_broadcasts_source_draft_token_without_rewriting_probs():
    probs = torch.tensor([[0.49, 0.51]], dtype=torch.float32)
    source_token_ids = torch.tensor([1], dtype=torch.int64)
    local_token_ids = torch.tensor([0], dtype=torch.int64)
    tp_group = FakeTPGroup(source_token_ids)

    synced_token_ids = _sync_draft_token_ids_across_tp(local_token_ids, tp_group)

    assert torch.equal(synced_token_ids, source_token_ids)
    assert len(tp_group.broadcast_calls) == 1
    assert tp_group.broadcast_calls[0][1] == 0
    assert torch.equal(probs, torch.tensor([[0.49, 0.51]], dtype=torch.float32))
    assert probs.gather(1, synced_token_ids.view(-1, 1)).item() == probs[0, 1].item()


def test_compute_probs_and_sample_next_token_applies_tp_sync(monkeypatch):
    forced_token_ids = torch.tensor([2], dtype=torch.int64)
    calls: list[torch.Tensor] = []

    def fake_sync(
        next_token_ids: torch.Tensor,
        tp_group=None,
    ) -> torch.Tensor:
        calls.append(next_token_ids.clone())
        return forced_token_ids.to(next_token_ids.device)

    monkeypatch.setattr(
        llm_base_proposer,
        "_sync_draft_token_ids_across_tp",
        fake_sync,
    )

    logits = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    token_ids, probs = compute_probs_and_sample_next_token(
        logits,
        _random_sampling_metadata(batch_size=1),
    )

    assert len(calls) == 1
    assert torch.equal(token_ids, forced_token_ids)
    assert probs.shape == logits.shape
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_compute_probs_uses_top_k_only_proposal_when_top_k_present(monkeypatch):
    calls = []

    def fake_apply_top_k_top_p(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        calls.append((k.clone() if k is not None else None,
                      p.clone() if p is not None else None))
        return logits

    monkeypatch.setattr(
        llm_base_proposer,
        "apply_top_k_top_p",
        fake_apply_top_k_top_p,
    )
    monkeypatch.setattr(
        llm_base_proposer,
        "_sync_draft_token_ids_across_tp",
        lambda token_ids, tp_group=None: token_ids,
    )

    metadata = _random_sampling_metadata(batch_size=1)
    metadata.top_k = torch.tensor([20], dtype=torch.int32)
    metadata.top_p = torch.tensor([0.95], dtype=torch.float32)

    compute_probs_and_sample_next_token(
        torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        metadata,
    )

    assert len(calls) == 1
    k, p = calls[0]
    assert k is not None
    assert torch.equal(k, metadata.top_k)
    assert p is None
