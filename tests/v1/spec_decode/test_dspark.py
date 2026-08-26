# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM,
    DSparkDeepseekV4Model,
    DSparkMarkovHead,
)
from vllm.v1.spec_decode.dspark import DSparkProposer


def test_deepseek_v4_dspark_checkpoint_name_mapping() -> None:
    remap = DSparkDeepseekV4ForCausalLM._remap_dspark_name

    assert remap("model.layers.0.attn.wq_a.weight") is None
    assert remap("mtp.0.main_proj.weight") == "model.main_proj.weight"
    assert remap("mtp.0.main_norm.weight") == "model.main_norm.weight"
    assert remap("mtp.1.ffn.experts.7.w2.weight") == (
        "model.layers.1.ffn.experts.7.w2.weight"
    )
    assert remap("mtp.2.markov_head.markov_w1.weight") == (
        "model.markov_head.markov_w1.weight"
    )
    assert remap("mtp.2.confidence_head.proj.weight") is None


class _FakeDSparkModel:
    vocab_size = 6

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            hidden_states.shape[0], self.vocab_size, dtype=hidden_states.dtype
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.unsqueeze(-1)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        previous = markov_embed.squeeze(-1).to(torch.long)
        bias = torch.zeros(previous.shape[0], self.vocab_size)
        bias.scatter_(1, ((previous + 1) % self.vocab_size).unsqueeze(1), 10.0)
        return bias

    @staticmethod
    def map_draft_to_target(token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids


@pytest.mark.parametrize("num_speculative_tokens", [4, 7])
def test_dspark_markov_sampling_is_sequential(
    num_speculative_tokens: int,
) -> None:
    proposer = object.__new__(DSparkProposer)
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.model = _FakeDSparkModel()
    proposer._enable_probabilistic_draft_probs = False
    proposer._static_draft_vocab = None
    proposer.input_ids = torch.zeros(2 * num_speculative_tokens, dtype=torch.int32)
    proposer.input_ids[0] = 1
    proposer.input_ids[num_speculative_tokens] = 4
    proposer._anchor_indices = torch.tensor(
        [0, num_speculative_tokens], dtype=torch.int64
    )
    hidden_states = torch.zeros(2 * num_speculative_tokens, 2)

    output, draft_probs = proposer._sample_draft_tokens(
        hidden_states,
        sampling_metadata=None,  # type: ignore[arg-type]
    )

    assert draft_probs is None
    expected = [
        [
            (anchor + step + 1) % _FakeDSparkModel.vocab_size
            for step in range(num_speculative_tokens)
        ]
        for anchor in (1, 4)
    ]
    assert output.view(2, num_speculative_tokens).tolist() == expected


@pytest.mark.parametrize("num_speculative_tokens", [5, 7])
def test_dspark_probabilistic_sampling_returns_sequential_probs(
    num_speculative_tokens: int,
) -> None:
    proposer = object.__new__(DSparkProposer)
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.model = _FakeDSparkModel()
    proposer._enable_probabilistic_draft_probs = True
    proposer._static_draft_vocab = None
    proposer.input_ids = torch.zeros(2 * num_speculative_tokens, dtype=torch.int32)
    proposer.input_ids[0] = 1
    proposer.input_ids[num_speculative_tokens] = 4
    proposer._anchor_indices = torch.tensor(
        [0, num_speculative_tokens], dtype=torch.int64
    )

    def sample_from_logits(
        self: DSparkProposer,
        logits: torch.Tensor,
        sampling_metadata: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del self, sampling_metadata
        return logits.argmax(dim=-1), logits.softmax(dim=-1)

    proposer._sample_from_logits = MethodType(sample_from_logits, proposer)
    hidden_states = torch.zeros(2 * num_speculative_tokens, 2)
    metadata = SimpleNamespace(all_greedy=False)

    output, draft_probs = proposer._sample_draft_tokens(
        hidden_states,
        sampling_metadata=metadata,  # type: ignore[arg-type]
    )

    assert draft_probs is not None
    assert draft_probs.shape == (
        2 * num_speculative_tokens,
        _FakeDSparkModel.vocab_size,
    )
    assert torch.equal(draft_probs.argmax(dim=-1), output)
    expected = [
        [
            (anchor + step + 1) % _FakeDSparkModel.vocab_size
            for step in range(num_speculative_tokens)
        ]
        for anchor in (1, 4)
    ]
    assert output.view(2, num_speculative_tokens).tolist() == expected


def test_dspark_replicated_linears_return_tensors() -> None:
    draft_model = object.__new__(DSparkDeepseekV4Model)
    nn.Module.__init__(draft_model)
    draft_model.main_proj = nn.Identity()
    draft_model.main_norm = nn.Identity()
    draft_model.main_proj_input_scale = 2.0**-6

    hidden_states = torch.randn(3, 4)
    torch.testing.assert_close(
        draft_model.combine_hidden_states(hidden_states),
        hidden_states * (2.0**-6),
        rtol=0.0,
        atol=0.0,
    )

    markov_head = object.__new__(DSparkMarkovHead)
    nn.Module.__init__(markov_head)
    markov_head.markov_w2 = nn.Identity()
    markov_embed = torch.randn(3, 4)
    assert torch.equal(markov_head.bias(markov_embed), markov_embed)
