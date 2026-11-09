# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import model as qwen4_exp_model
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpForCausalLM,
    Qwen4ExpModel,
    _Qwen4ExpPrefillSequenceModel,
)


@pytest.mark.parametrize(
    "rows,expected",
    [
        (0, False),
        (5, False),
        (4095, False),
        (4096, True),
        (4097, True),
        (4098, True),
        (4099, True),
        (7344, True),
        (9216, True),
        (16384, True),
        (16383, True),
        (16385, False),
    ],
)
def test_hc_sequence_prefill_entry_boundaries(monkeypatch, rows, expected):
    from vllm.config import CUDAGraphMode

    backend = torch.backends.cuda.matmul
    monkeypatch.setattr(backend, "allow_fp16_reduced_precision_reduction", False)
    monkeypatch.setattr(backend, "allow_fp16_accumulation", False)
    owner = SimpleNamespace(_hc_prefill_sequence_qualified=True)
    monkeypatch.setattr(qwen4_exp_model, "is_forward_context_available", lambda: True)
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(qwen4_exp_model, "get_forward_context", lambda: context)
    ids = torch.zeros(rows, dtype=torch.int32)
    select = Qwen4ExpForCausalLM._use_hc_prefill_sequence_rows
    assert select(owner, ids, None, None, {}) is expected
    context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    assert not select(owner, ids, None, None, {})
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    assert not select(owner, ids, None, object(), {})
    assert not select(owner, ids, None, None, {"deepstack_input_embeds": object()})
    owner._hc_prefill_sequence_qualified = False
    assert not select(owner, ids, None, None, {})


def test_hc_sequence_prefill_embedding_dtype_and_missing_context(monkeypatch):
    from vllm.config import CUDAGraphMode

    backend = torch.backends.cuda.matmul
    monkeypatch.setattr(backend, "allow_fp16_reduced_precision_reduction", False)
    monkeypatch.setattr(backend, "allow_fp16_accumulation", False)
    owner = SimpleNamespace(_hc_prefill_sequence_qualified=True)
    select = Qwen4ExpForCausalLM._use_hc_prefill_sequence_rows
    monkeypatch.setattr(qwen4_exp_model, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        qwen4_exp_model,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE),
    )
    assert select(owner, None, torch.empty(4096, 2, dtype=torch.half), None, {})
    assert not select(owner, None, torch.empty(4096, 2, dtype=torch.bfloat16), None, {})
    monkeypatch.setattr(qwen4_exp_model, "is_forward_context_available", lambda: False)
    assert not select(owner, torch.zeros(4096), None, None, {})


@pytest.mark.parametrize("reduced,accumulation", [(False, True), (True, False)])
def test_hc_sequence_prefill_other_accumulation_policies_fall_back(
    monkeypatch, reduced, accumulation
):
    backend = torch.backends.cuda.matmul
    monkeypatch.setattr(backend, "allow_fp16_reduced_precision_reduction", reduced)
    monkeypatch.setattr(backend, "allow_fp16_accumulation", accumulation)
    owner = SimpleNamespace(_hc_prefill_sequence_qualified=True)
    assert not Qwen4ExpForCausalLM._use_hc_prefill_sequence_rows(
        owner, torch.zeros(4096), None, None, {}
    )


def test_hc_sequence_prefill_view_borrows_parameters_and_state():
    class Owner(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer("state", torch.zeros(1))

        def forward(self, ids, *args, hc_sequence_parallel=False):
            assert hc_sequence_parallel
            self.state.add_(1)
            return ids * self.weight

    owner = Owner()
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(mode=0))
    view = _Qwen4ExpPrefillSequenceModel(target_model=owner, vllm_config=cfg)
    assert list(view.parameters()) == [] and list(view.buffers()) == []
    ids = torch.ones(1)
    assert torch.equal(view(ids, torch.zeros(1)), ids)
    assert torch.equal(owner.state, torch.ones(1))
    assert view._target_model is owner


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
            dcp_local_indices_buffer: torch.Tensor | None = None,
            dcp_partial_output_buffer: torch.Tensor | None = None,
            dcp_partial_lse_buffer: torch.Tensor | None = None,
        ) -> None:
            super().__init__()
            self.layer_type = layer_type
            self.prefix = prefix
            self.topk_indices_buffer = topk_indices_buffer
            self.dcp_local_indices_buffer = dcp_local_indices_buffer
            self.dcp_partial_output_buffer = dcp_partial_output_buffer
            self.dcp_partial_lse_buffer = dcp_partial_lse_buffer

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
        num_attention_heads=8,
        head_dim=16,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=config,
            dtype=torch.float16,
            quantization=None,
        ),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            decode_context_parallel_size=1,
            tensor_parallel_size=4,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        cache_config=SimpleNamespace(cache_dtype="float16"),
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

    vllm_config.parallel_config.decode_context_parallel_size = 2
    dcp_model = Qwen4ExpModel(vllm_config=vllm_config)
    dcp_layers = [
        layer for layer in dcp_model.layers if layer.layer_type == "full_attention"
    ]
    assert dcp_model.dcp_local_indices_buffer.shape == (16, 11)
    assert dcp_model.dcp_partial_output_buffer.shape == (16, 4, 16)
    assert dcp_model.dcp_partial_lse_buffer.shape == (16, 4)
    assert all(
        layer.dcp_local_indices_buffer is dcp_model.dcp_local_indices_buffer
        and layer.dcp_partial_output_buffer is dcp_model.dcp_partial_output_buffer
        and layer.dcp_partial_lse_buffer is dcp_model.dcp_partial_lse_buffer
        for layer in dcp_layers
    )
