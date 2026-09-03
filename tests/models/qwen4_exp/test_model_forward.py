# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import model as qwen4_exp_model
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpModel,
)


class _AttentionHyperConnection:
    def mix(self, hidden_states: torch.Tensor):
        return hidden_states, hidden_states, torch.zeros_like(hidden_states)


class _MlpHyperConnection:
    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ):
        return hidden_states, block_output, injection


class _OutputOnlyGdn:
    def __init__(self) -> None:
        self.output: torch.Tensor | None = None

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        assert output is not None
        self.output = output
        output.copy_(hidden_states + 1)
        return hidden_states + 100


def test_linear_attention_forwards_preallocated_output_buffer() -> None:
    layer = object.__new__(Qwen4ExpDecoderLayer)
    object.__setattr__(layer, "ple", None)
    object.__setattr__(layer, "layer_type", "linear_attention")
    object.__setattr__(layer, "attn_hyper_connection", _AttentionHyperConnection())
    object.__setattr__(layer, "mlp_hyper_connection", _MlpHyperConnection())
    gdn = _OutputOnlyGdn()
    object.__setattr__(layer, "linear_attn", gdn)
    object.__setattr__(layer, "mlp", lambda hidden_states: hidden_states)
    hidden_states = torch.arange(6, dtype=torch.float32).view(2, 3)

    _, mlp_out, _ = Qwen4ExpDecoderLayer.forward(
        layer,
        hidden_states,
        None,
        None,
        torch.arange(2),
        input_ids=None,
        query_start_loc=None,
        ngram_context=None,
    )

    assert gdn.output is not None
    torch.testing.assert_close(mlp_out, hidden_states + 1)


def test_qsa_model_shares_one_topk_indices_buffer(monkeypatch) -> None:
    class FakeEmbedding(nn.Module):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()

    class FakeDecoderLayer(nn.Module):
        def __init__(
            self,
            _vllm_config,
            layer_type: str,
            prefix: str = "",
            topk_indices_buffer: torch.Tensor | None = None,
        ) -> None:
            super().__init__()
            self.layer_type = layer_type
            self.prefix = prefix
            self.topk_indices_buffer = topk_indices_buffer

    def fake_make_layers(num_layers, get_layer, prefix):
        layers = nn.ModuleList(
            get_layer(f"{prefix}.{layer_idx}") for layer_idx in range(num_layers)
        )
        return 0, num_layers, layers

    monkeypatch.setattr(qwen4_exp_model, "VocabParallelEmbedding", FakeEmbedding)
    monkeypatch.setattr(qwen4_exp_model, "Qwen4ExpDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(qwen4_exp_model, "make_layers", fake_make_layers)
    monkeypatch.setattr(
        qwen4_exp_model,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        qwen4_exp_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )

    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        hc_count=2,
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        indexer_n_heads=4,
        indexer_budget=8,
        indexer_compress_ratio=4,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=config,
            dtype=torch.float16,
        ),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        compilation_config=SimpleNamespace(mode=0),
        speculative_config=None,
    )

    model = Qwen4ExpModel(vllm_config=vllm_config)

    assert model.topk_indices_buffer.shape == (16, 11)
    assert model.topk_indices_buffer.dtype == torch.int32
    qsa_layers = [
        layer for layer in model.layers if layer.layer_type == "full_attention"
    ]
    assert len(qsa_layers) == 2
    assert all(
        layer.topk_indices_buffer is model.topk_indices_buffer for layer in qsa_layers
    )
