# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark5 Markov proposals and real rejection sampling against a known PDF.

This isolates the sampling law from checkpoint numerics. It is not a claim
about full-model stochastic quality or independent requests' exact token IDs.
"""

from types import MethodType, SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from tests.v1.sample.test_rejection_sampler import (
    create_sampling_metadata,
    create_spec_decode_metadata,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler, SamplerOutput
from vllm.v1.spec_decode.dspark import DSparkProposer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Markov:
    def compute_logits(self, hidden):
        return torch.linspace(-0.3, 0.3, 16, device=hidden.device).expand(
            hidden.shape[0], -1
        )

    def markov_embed(self, ids):
        return ids[:, None].float()

    def markov_bias(self, embedding):
        columns = torch.arange(16, device=embedding.device)[None]
        return torch.sin(columns * 0.7 + embedding * 0.4) * 0.25

    def map_draft_to_target(self, ids):
        return ids


@pytest.mark.parametrize("all_greedy", [False, True])
@pytest.mark.parametrize("cap", [0, 2, 5])
def test_greedy_graph_dispatch_keeps_caps_and_random_sampling_path(
    all_greedy, cap, monkeypatch
):
    proposer = object.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer.model = _Markov()
    proposer.collect_confidence_logits = False
    proposer.confidence_scheduling_enabled = cap < 5
    proposer._max_verification_tokens = cap
    proposer._static_draft_vocab = None
    proposer._anchor_indices = torch.zeros(1, device="cuda", dtype=torch.int64)
    proposer.input_ids = torch.tensor([9, 0, 0, 0, 0], device="cuda", dtype=torch.int32)
    called = []

    def graph(hidden, anchor):
        called.append(anchor.clone())
        return torch.arange(5, device=hidden.device, dtype=torch.int64)

    monkeypatch.setattr(proposer.model, "greedy_draft_tokens", graph, raising=False)
    monkeypatch.setattr(
        proposer,
        "_sample_from_logits",
        lambda logits, metadata: (logits.argmax(-1), None),
    )
    metadata = cast(SamplingMetadata, SimpleNamespace(all_greedy=all_greedy))
    ids, probabilities = proposer._sample_draft_tokens(
        torch.zeros(5, 2, device="cuda"),
        metadata,
    )
    assert len(called) == int(all_greedy)
    assert probabilities is None and ids.shape == (5,)
    if all_greedy:
        assert called[0].item() == 9
        torch.testing.assert_close(ids, torch.arange(5, device="cuda"))
    if cap < 5:
        lengths = proposer.take_last_verification_lengths()
        assert lengths is not None and lengths.item() == cap
    else:
        assert proposer.take_last_verification_lengths() is None


@pytest.mark.parametrize("probabilistic", [False, True])
def test_dspark5_rejection_distribution_with_conditional_markov_proposals(
    probabilistic,
    monkeypatch,
):
    batch, width, vocab = 32768, 5, 16
    generator = torch.Generator(device="cuda").manual_seed(471)
    proposer = object.__new__(DSparkProposer)
    proposer.num_speculative_tokens = width
    proposer.model = _Markov()
    proposer.collect_confidence_logits = proposer.confidence_scheduling_enabled = False
    proposer._anchor_indices = torch.arange(batch, device="cuda") * width
    proposer.input_ids = torch.zeros(batch * width, device="cuda", dtype=torch.int32)
    proposer.input_ids[proposer._anchor_indices] = (
        torch.arange(batch, device="cuda").int() % vocab
    )

    def sample(self, logits, metadata):
        del self, metadata
        if not probabilistic:
            return logits.argmax(-1), None
        probabilities = logits.softmax(-1)
        return torch.multinomial(probabilities, 1, generator=generator)[
            :, 0
        ], probabilities

    monkeypatch.setattr(proposer, "_sample_from_logits", MethodType(sample, proposer))
    ids, draft_probs = proposer._sample_draft_tokens(
        torch.zeros(batch * width, 2, device="cuda", dtype=torch.float16),
        cast(SamplingMetadata, None),
    )
    target = torch.linspace(0.3, -0.3, vocab, device="cuda")
    probabilities = target.softmax(-1)
    logits = target.expand(batch * width, -1).contiguous()
    metadata = create_spec_decode_metadata(ids.view(batch, width).tolist(), logits)
    sampling = create_sampling_metadata(
        all_greedy=False, temperature=torch.ones(batch, device="cuda")
    )
    sampler = Mock(spec=Sampler)
    sampler.logprobs_mode = "raw_logprobs"
    sampler.return_value = SamplerOutput(
        sampled_token_ids=torch.multinomial(
            probabilities.expand(batch, -1), 1, generator=generator
        ),
        logprobs_tensors=None,
    )
    output = RejectionSampler(sampler)(
        metadata, draft_probs, logits, sampling
    ).sampled_token_ids
    observed_positions = 0
    for position in range(width + 1):
        values = output[:, position]
        values = values[values >= 0].long()
        if values.numel() < 500:
            continue
        observed_positions += 1
        empirical = torch.bincount(values, minlength=vocab) / values.numel()
        # Fixed six-sigma binomial bounds, with no fitted tolerance. The
        # independent target law is the same at every reached position.
        bound = 6 * torch.sqrt(probabilities * (1 - probabilities) / values.numel())
        bound += 1 / values.numel()
        assert ((empirical - probabilities).abs() <= bound).all(), (
            probabilistic,
            position,
            empirical.tolist(),
            probabilities.tolist(),
        )
    assert observed_positions == (6 if probabilistic else 2)
