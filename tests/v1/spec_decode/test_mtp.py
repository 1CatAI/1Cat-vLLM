# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    try_get_attention_backend,
)
from vllm import envs
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.load import LoadConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.llm_base_proposer import (
    SpecDecodeBaseProposer,
    _sm70_mtp_hotpath_warmup_batch_sizes,
    _sm70_mtp_moe_warmup_sizes,
)

mimo_7b_dir = "XiaomiMiMo/MiMo-7B-Base"
DEVICE_TYPE = current_platform.device_type


def test_sm70_mtp_moe_warmup_sizes():
    assert _sm70_mtp_moe_warmup_sizes(256, 8, 8192) == (9, 33, 257)
    assert _sm70_mtp_moe_warmup_sizes(256, 8, 32) == (9,)
    assert _sm70_mtp_moe_warmup_sizes(256, 8, 8192, 16) == (
        *range(1, 17),
        33,
        257,
    )
    assert _sm70_mtp_moe_warmup_sizes(0, 8, 8192) == ()


def test_sm70_mtp_hotpath_warmup_batch_sizes():
    assert _sm70_mtp_hotpath_warmup_batch_sizes(1) == (1,)
    assert _sm70_mtp_hotpath_warmup_batch_sizes(4) == (4,)
    assert _sm70_mtp_hotpath_warmup_batch_sizes(4, include_alternate=True) == (1, 4)
    assert _sm70_mtp_hotpath_warmup_batch_sizes(64, include_alternate=True) == (1, 4)


def test_sm70_mtp_concurrency_warmup_is_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_MTP_CONCURRENCY_WARMUP", raising=False)
    envs.disable_envs_cache()
    try:
        assert not envs.VLLM_SM70_MTP_CONCURRENCY_WARMUP
    finally:
        envs.disable_envs_cache()


def test_sm70_mtp_moe_warmup_runs_each_shape_once():
    dummy_run = mock.Mock()
    draft_model_config = SimpleNamespace(
        is_moe=True,
        hf_text_config=SimpleNamespace(
            num_experts_per_tok=8,
            moe_intermediate_size=512,
        ),
        get_num_experts=lambda: 256,
        get_hidden_size=lambda: 4096,
    )
    proposer = SimpleNamespace(
        method="mtp",
        device=torch.device("cuda"),
        draft_model_config=draft_model_config,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=4)
        ),
        max_num_tokens=8192,
        dummy_run=dummy_run,
        _sm70_mtp_moe_warmed=False,
    )

    with (
        mock.patch.object(current_platform, "is_device_capability", return_value=True),
        mock.patch.object(torch.accelerator, "synchronize"),
    ):
        assert SpecDecodeBaseProposer.warmup_sm70_mtp_moe_kernels(proposer) == (
            "mtp_draft_moe",
        )

    assert [call.args[0] for call in dummy_run.call_args_list] == [9, 33, 257]
    assert all(call.kwargs["spec_step_idx"] == 0 for call in dummy_run.call_args_list)


def test_sm70_mtp_moe_warmup_covers_opted_in_exact_tiles():
    dummy_run = mock.Mock()
    draft_model_config = SimpleNamespace(
        is_moe=True,
        hf_text_config=SimpleNamespace(
            num_experts_per_tok=8,
            moe_intermediate_size=512,
        ),
        get_num_experts=lambda: 256,
        get_hidden_size=lambda: 2048,
    )
    proposer = SimpleNamespace(
        method="mtp",
        device=torch.device("cuda"),
        draft_model_config=draft_model_config,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=4)
        ),
        max_num_tokens=8192,
        dummy_run=dummy_run,
        _sm70_mtp_moe_warmed=False,
    )

    with (
        mock.patch.object(current_platform, "is_device_capability", return_value=True),
        mock.patch.object(torch.accelerator, "synchronize"),
        mock.patch.object(envs, "VLLM_SM70_MTP_MOE_TUNED_CONFIG", True),
    ):
        assert SpecDecodeBaseProposer.warmup_sm70_mtp_moe_kernels(proposer) == (
            "mtp_draft_moe",
        )

    assert [call.args[0] for call in dummy_run.call_args_list] == [
        *range(1, 17),
        33,
        257,
        1,
        2,
        9,
        10,
    ]


def test_sm70_mtp_moe_warmup_covers_qwen38_exact_tiles():
    dummy_run = mock.Mock()
    draft_model_config = SimpleNamespace(
        is_moe=True,
        hf_text_config=SimpleNamespace(
            num_experts_per_tok=10,
            moe_intermediate_size=640,
        ),
        get_num_experts=lambda: 512,
        get_hidden_size=lambda: 2560,
    )
    proposer = SimpleNamespace(
        method="mtp",
        device=torch.device("cuda"),
        draft_model_config=draft_model_config,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=4)
        ),
        max_num_tokens=8192,
        dummy_run=dummy_run,
        _sm70_mtp_moe_warmed=False,
    )

    with (
        mock.patch.object(current_platform, "is_device_capability", return_value=True),
        mock.patch.object(torch.accelerator, "synchronize"),
        mock.patch.object(envs, "VLLM_SM70_MTP_MOE_TUNED_CONFIG", True),
    ):
        assert SpecDecodeBaseProposer.warmup_sm70_mtp_moe_kernels(proposer) == (
            "mtp_draft_moe",
        )

    assert [call.args[0] for call in dummy_run.call_args_list] == [
        *range(1, 6),
        13,
        52,
        513,
    ]


def _create_mtp_proposer(num_speculative_tokens: int) -> EagleProposer:
    """Create an MTP proposer with unified model configuration."""
    model_config = ModelConfig(
        model=mimo_7b_dir, runner="generate", max_model_len=100, trust_remote_code=True
    )

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=mimo_7b_dir,
        method="mtp",
        num_speculative_tokens=num_speculative_tokens,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=DEVICE_TYPE),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )

    return EagleProposer(vllm_config=vllm_config, device=DEVICE_TYPE)


@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_pp_group")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_model")
def test_mtp_load_model_unified(mock_get_model, mock_get_layers, mock_get_pp_group):
    """Test MTP-specific model loading with unified model approach."""

    # Setup mocks
    mock_model = mock.MagicMock()
    mock_model.model.embed_tokens.weight.shape = (131072, 4096)
    mock_get_model.return_value = mock_model
    # MTP does not have its own embed_tokens or lm_head
    # so it should share them with the target model
    mock_model.has_own_embed_tokens = False
    mock_model.has_own_lm_head = False

    target_attn_layers = {"target_attn_1": mock.MagicMock()}
    all_attn_layers = {**target_attn_layers, "draft_attn_1": mock.MagicMock()}
    target_indexer_layers: dict = {}
    all_indexer_layers: dict = {}

    mock_get_layers.side_effect = [
        target_attn_layers,
        target_indexer_layers,
        all_attn_layers,
        all_indexer_layers,
    ]

    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    mock_get_pp_group.return_value = mock_pp_group

    # Create target model
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)
    target_model.lm_head = mock.MagicMock()

    # Create MTP proposer
    proposer = _create_mtp_proposer(num_speculative_tokens=4)
    proposer.load_model(target_model)

    # Verify MTP-specific behavior:
    # Model is loaded
    mock_get_model.assert_called_once()
    # MTP shares lm_head with target model
    assert proposer.model.lm_head == target_model.lm_head
    # MTP shares embed_tokens with target model
    assert proposer.model.model.embed_tokens == target_model.model.embed_tokens


@pytest.mark.parametrize("num_speculative_tokens", [1])
def test_mtp_propose(num_speculative_tokens, monkeypatch):
    """Test that MTP's forward method returns hidden states directly"""

    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 100

    proposer = _create_mtp_proposer(num_speculative_tokens)
    hidden_size = proposer.hidden_size

    # Mock the MTP model to verify it returns hidden states directly
    model_mock = mock.MagicMock()

    # MTP returns hidden states directly
    if num_speculative_tokens == 1:
        model_mock.return_value = torch.zeros(total_tokens, hidden_size, device=device)
    else:
        # Multiple forward passes for multi-token speculation
        forward_returns = []
        for i in range(num_speculative_tokens):
            if i == 0:
                h_states = torch.zeros(total_tokens, hidden_size, device=device)
            else:
                h_states = torch.zeros(batch_size, hidden_size, device=device)
            forward_returns.append(h_states)
        model_mock.side_effect = forward_returns

    # Mock compute_logits
    def create_deterministic_logits(batch_size, vocab_size, token_offset):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        logits[:, token_offset] = 100.0
        return logits

    if num_speculative_tokens == 1:
        model_mock.compute_logits.return_value = create_deterministic_logits(
            batch_size, vocab_size, 42
        )
    else:
        logits_returns = [
            create_deterministic_logits(batch_size, vocab_size, 42 + i)
            for i in range(num_speculative_tokens)
        ]
        model_mock.compute_logits.side_effect = logits_returns

    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    # Prepare inputs
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [
            torch.arange(seq_lens[0], device=device),
            torch.arange(seq_lens[1], device=device),
        ]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    # Setup attention metadata
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )

    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )

    proposer.runner = mock.MagicMock()
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    # Run propose
    result = proposer.propose(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    # Verify the model was called correctly
    assert model_mock.called
    # Verify output shape
    assert result.shape == (batch_size, num_speculative_tokens)
