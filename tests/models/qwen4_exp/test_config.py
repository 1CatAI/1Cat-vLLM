# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)


def test_sm70_v2_route_accepts_prefix_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Qwen3_5ForConditionalGenerationConfig,
        "verify_and_update_config",
        lambda _config: None,
    )
    text_config = SimpleNamespace(
        hc_count=4,
        ple_layer_ids=[2],
        indexer_n_heads=4,
        rope_parameters={"mrope_section": [11, 11, 10]},
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(rope_parameters={"mrope_interleaved": True}),
            hf_text_config=text_config,
            multimodal_config=SimpleNamespace(language_model_only=True),
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        parallel_config=SimpleNamespace(enable_dbo=False, ubatch_size=1),
        speculative_config=None,
        use_v2_model_runner=True,
    )

    Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)
    assert "mrope_section" not in text_config.rope_parameters
    assert "mrope_interleaved" not in vllm_config.model_config.hf_config.rope_parameters


def test_initial_sm70_route_rejects_multimodal_tower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Qwen3_5ForConditionalGenerationConfig,
        "verify_and_update_config",
        lambda _config: None,
    )
    text_config = SimpleNamespace(
        hc_count=4,
        ple_layer_ids=[2],
        indexer_n_heads=4,
        rope_parameters={"mrope_section": [11, 11, 10]},
    )
    multimodal_config = SimpleNamespace(language_model_only=False)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(rope_parameters={}),
            hf_text_config=text_config,
            multimodal_config=multimodal_config,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(enable_dbo=False, ubatch_size=1),
        speculative_config=None,
    )

    with pytest.raises(NotImplementedError, match="--language-model-only"):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)


@pytest.mark.parametrize(
    "architecture",
    ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"],
)
def test_qwen4_exp_defaults_to_v2_even_when_quantized_moe(
    architecture: str,
) -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="generate",
            architectures=[architecture],
            is_moe=True,
            is_quantized=True,
        )
    )

    assert VllmConfig._is_default_v2_model_runner_model(vllm_config)


def test_initial_sm70_v2_route_accepts_native_mtp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Qwen3_5ForConditionalGenerationConfig,
        "verify_and_update_config",
        lambda _config: None,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hc_count=4,
                ple_layer_ids=[2],
                indexer_n_heads=4,
            ),
            multimodal_config=None,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(enable_dbo=False, ubatch_size=1),
        speculative_config=SimpleNamespace(method="mtp"),
    )

    Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)


def test_initial_sm70_v2_route_rejects_unvalidated_speculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Qwen3_5ForConditionalGenerationConfig,
        "verify_and_update_config",
        lambda _config: None,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hc_count=4,
                ple_layer_ids=[2],
                indexer_n_heads=4,
            ),
            multimodal_config=None,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(enable_dbo=False, ubatch_size=1),
        speculative_config=SimpleNamespace(method="dflash"),
    )

    with pytest.raises(NotImplementedError, match="supports only its native MTP"):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)


def test_qwen4_exp_registers_v2_model_state() -> None:
    from vllm.models.qwen4_exp.nvidia.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )

    assert Qwen4ExpForCausalLM.get_model_state_cls() is Qwen4ExpModelState
    assert Qwen4ExpForConditionalGeneration.get_model_state_cls() is Qwen4ExpModelState


def test_qwen4_exp_v2_model_state_uses_committed_ngram_context() -> None:
    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((4, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(5, dtype=torch.int32)

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    torch.testing.assert_close(
        model_inputs["query_start_loc"],
        torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        model_inputs["ngram_context"],
        torch.tensor([[99, 99, 20], [1, 2, 3], [99, 99, 99]], dtype=torch.int32),
    )


def test_qwen4_exp_ple_builder_receives_v2_decode_metadata() -> None:
    num_accepted_tokens = torch.tensor([1, 2], dtype=torch.int32)
    num_decode_draft_tokens_cpu = torch.tensor([-1, 2], dtype=torch.int32)
    metadata = MambaHybridAttnMetadata(
        is_prefilling=torch.tensor([False, False]),
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )
    builder = PleShortConvAttentionMetadataBuilder.__new__(
        PleShortConvAttentionMetadataBuilder
    )

    kwargs = metadata.get_extra_attn_kwargs(builder, num_reqs=2)

    torch.testing.assert_close(kwargs["num_accepted_tokens"], num_accepted_tokens)
    torch.testing.assert_close(
        kwargs["num_decode_draft_tokens_cpu"], num_decode_draft_tokens_cpu
    )


def test_qwen4_exp_qsa_owns_qkv_projection_for_private_qwen3_api() -> None:
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    class AddRotaryOffset(torch.nn.Module):
        def forward(self, positions, query, key):
            del positions
            return query + 1, key + 2

    layer = object.__new__(Qwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.q_size = 4
    layer.kv_size = 2
    layer.num_heads = 2
    layer.num_kv_heads = 1
    layer.head_dim = 2
    layer.q_norm = torch.nn.Identity()
    layer.k_norm = torch.nn.Identity()
    layer.rotary_emb = AddRotaryOffset()
    qkv = torch.arange(12, dtype=torch.float32).reshape(1, 12)

    query, key, value, gate = layer._project_qkv_gate(
        qkv, torch.tensor([0], dtype=torch.int64)
    )

    torch.testing.assert_close(query, torch.tensor([[1, 2, 5, 6.0]]))
    torch.testing.assert_close(gate, torch.tensor([[2, 3, 6, 7.0]]))
    torch.testing.assert_close(key, torch.tensor([[10, 11.0]]))
    torch.testing.assert_close(value, torch.tensor([[10, 11.0]]))


def test_qwen4_exp_qsa_uses_private_mrope_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.qwen4_exp.nvidia import indexer_qsa

    calls = []

    def private_triton_mrope(
        query,
        key,
        cos,
        sin,
        mrope_section,
        head_size,
        rotary_dim,
        mrope_interleaved,
    ):
        calls.append(
            (
                cos.shape,
                sin.shape,
                mrope_section,
                head_size,
                rotary_dim,
                mrope_interleaved,
            )
        )
        return query, key

    monkeypatch.setattr(indexer_qsa, "triton_mrope", private_triton_mrope)
    cache = torch.zeros(16, 4)
    rotary_emb = SimpleNamespace(
        rotary_dim=2,
        mrope_section=[1, 0, 0],
        mrope_interleaved=True,
        _match_cos_sin_cache_dtype=lambda _tensor: cache,
    )
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    positions = torch.tensor([[0, 1], [2, 3], [4, 5]])

    output = indexer_qsa.apply_qsa_rope(rotary_emb, positions, tensor)

    torch.testing.assert_close(output, tensor)
    assert calls == [
        (torch.Size([3, 2, 2]), torch.Size([3, 2, 2]), [1, 0, 0], 4, 2, True)
    ]
