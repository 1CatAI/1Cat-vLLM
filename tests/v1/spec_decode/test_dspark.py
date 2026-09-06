# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.models.deepseek_v4.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM,
    DSparkDeepseekV4Model,
    DSparkMarkovHead,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.v1.spec_decode.dspark import DSparkProposer


def _make_dspark_speculative_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pp_size: int,
    tp_size: int,
) -> tuple[ParallelConfig, SpeculativeConfig]:
    monkeypatch.setattr(current_platform, "device_count", lambda: 8)
    DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=4096,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        vocab_size=129280,
        num_nextn_predict_layers=1,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_markov_rank=256,
    ).save_pretrained(tmp_path)
    target_model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        runner="generate",
        dtype=torch.float16,
        max_model_len=2048,
    )
    target_parallel_config = ParallelConfig(
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=tp_size,
    )
    speculative_config = SpeculativeConfig(
        target_model_config=target_model_config,
        target_parallel_config=target_parallel_config,
        method="dspark",
        num_speculative_tokens=7,
    )
    return target_parallel_config, speculative_config


@pytest.mark.parametrize(("pp_size", "tp_size"), [(2, 4), (4, 2)])
def test_dspark_draft_is_local_to_last_pipeline_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pp_size: int,
    tp_size: int,
) -> None:
    target_parallel_config, speculative_config = _make_dspark_speculative_config(
        tmp_path,
        monkeypatch,
        pp_size=pp_size,
        tp_size=tp_size,
    )

    assert target_parallel_config.pipeline_parallel_size == pp_size
    assert speculative_config.draft_parallel_config.pipeline_parallel_size == 1
    assert speculative_config.draft_parallel_config.tensor_parallel_size == tp_size
    assert speculative_config.draft_model_config.architectures == ["DSparkDraftModel"]


def test_dspark_rejects_target_layers_outside_final_pipeline_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "42,1")

    with pytest.raises(ValueError, match="final target pipeline stage"):
        _make_dspark_speculative_config(
            tmp_path,
            monkeypatch,
            pp_size=2,
            tp_size=4,
        )


def test_deepseek_v4_dspark_checkpoint_name_mapping() -> None:
    remap = DSparkDeepseekV4ForCausalLM._remap_dspark_name

    assert remap("model.layers.0.attn.wq_a.weight") is None
    # The target's embedding also fills the drafter's own table: under
    # pipeline parallelism the drafter (last stage) cannot share the target
    # embedding (first stage).
    assert remap("embed.weight") == "model.embed_tokens.weight"
    assert remap("mtp.0.main_proj.weight") == "model.main_proj.weight"
    assert remap("mtp.0.main_norm.weight") == "model.main_norm.weight"
    assert remap("mtp.1.ffn.experts.7.w2.weight") == (
        "model.layers.1.ffn.experts.7.w2.weight"
    )
    assert remap("mtp.2.markov_head.markov_w1.weight") == (
        "model.markov_head.markov_w1.weight"
    )
    assert remap("mtp.2.confidence_head.proj.weight") == (
        "model.confidence_head.proj.weight"
    )


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
    def confidence_logits(
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states.sum(dim=-1) + markov_embeds.sum(dim=-1)

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
    proposer.collect_confidence_logits = False
    proposer.confidence_scheduling_enabled = False
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
    assert proposer._last_confidence_logits is None
    assert proposer._last_verification_lengths is None


@pytest.mark.parametrize("num_speculative_tokens", [5, 7])
def test_dspark_probabilistic_sampling_returns_sequential_probs(
    num_speculative_tokens: int,
) -> None:
    proposer = object.__new__(DSparkProposer)
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.model = _FakeDSparkModel()
    proposer._enable_probabilistic_draft_probs = True
    proposer._static_draft_vocab = None
    proposer.collect_confidence_logits = True
    proposer.confidence_scheduling_enabled = True
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
    assert proposer._last_confidence_logits.shape == (2, num_speculative_tokens)
    assert proposer._last_verification_lengths.shape == (2,)
    confidence_logits = proposer.take_last_confidence_logits()
    verification_lengths = proposer.take_last_verification_lengths()
    assert confidence_logits is not None
    assert verification_lengths is not None
    assert proposer.take_last_confidence_logits() is None
    assert proposer.take_last_verification_lengths() is None


def test_dspark_static_verification_cap_skips_confidence_projection() -> None:
    proposer = object.__new__(DSparkProposer)
    proposer.num_speculative_tokens = 5
    proposer.model = _FakeDSparkModel()
    proposer._enable_probabilistic_draft_probs = False
    proposer._static_draft_vocab = None
    proposer.collect_confidence_logits = False
    proposer.confidence_scheduling_enabled = True
    proposer._max_verification_tokens = 2
    proposer.input_ids = torch.zeros(5, dtype=torch.int32)
    proposer._anchor_indices = torch.tensor([0], dtype=torch.int64)

    output, draft_probs = proposer._sample_draft_tokens(
        torch.zeros(5, 2),
        sampling_metadata=None,  # type: ignore[arg-type]
    )

    assert output.shape == (5,)
    assert draft_probs is None
    assert proposer._last_confidence_logits is None
    assert proposer._last_verification_lengths.tolist() == [2]


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


def test_dspark_confidence_projection_rejects_missing_weight() -> None:
    draft_model = object.__new__(DSparkDeepseekV4ForCausalLM)
    nn.Module.__init__(draft_model)
    draft_model._confidence_head_loaded = False

    with pytest.raises(RuntimeError, match="did not provide confidence_head"):
        draft_model.confidence_logits(torch.empty(1, 1), torch.empty(1, 1))


def test_dspark_confidence_scheduler_keeps_only_confident_prefix() -> None:
    proposer = object.__new__(DSparkProposer)
    proposer._confidence_temperatures = torch.tensor([[1.0, 2.0, 1.0, 1.0, 1.0]])
    proposer._confidence_threshold = 0.7
    proposer._max_verification_tokens = 4
    logits = torch.tensor(
        [
            [2.0, 2.0, 0.0, 4.0, 4.0],
            [2.0, 4.0, 4.0, 4.0, 4.0],
            [0.0, 4.0, 4.0, 4.0, 4.0],
        ]
    )

    lengths = proposer._select_verification_lengths(logits)

    assert lengths.dtype == torch.int32
    assert lengths.tolist() == [2, 4, 0]


def test_dspark_confidence_scheduler_zero_threshold_uses_cap() -> None:
    proposer = object.__new__(DSparkProposer)
    proposer._confidence_temperatures = torch.ones((1, 5))
    proposer._confidence_threshold = 0.0
    proposer._max_verification_tokens = 2

    lengths = proposer._select_verification_lengths(torch.randn(3, 5))

    assert lengths.tolist() == [2, 2, 2]
