# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.compilation.sm70_decode_graph import (
    is_sm70_decode_graph_compiling,
    sm70_decode_graph_compilation,
    use_sm70_decode_graph_semantics,
)
from vllm.config.vllm import _is_sm70_qwen38_nomtp_dual_compile_contract


def test_sm70_decode_graph_compilation_context(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_QWEN38_DUAL_COMPILE", "1")

    assert not is_sm70_decode_graph_compiling()
    assert not use_sm70_decode_graph_semantics()
    with sm70_decode_graph_compilation():
        assert is_sm70_decode_graph_compiling()
        assert use_sm70_decode_graph_semantics()
    assert not is_sm70_decode_graph_compiling()


def test_sm70_decode_graph_legacy_semantics(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_QWEN38_DUAL_COMPILE", "0")
    assert use_sm70_decode_graph_semantics()


def test_qwen38_nomtp_dual_compile_contract() -> None:
    text_config = SimpleNamespace(
        hidden_size=2560,
        num_hidden_layers=48,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        hc_count=4,
        hc_lowrank=320,
        num_attention_heads=24,
        num_key_value_heads=2,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    model_config = SimpleNamespace(
        architectures=("Qwen4ExpForCausalLM",),
        dtype=torch.float16,
        hf_text_config=text_config,
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
    )

    assert _is_sm70_qwen38_nomtp_dual_compile_contract(
        model_config, None, parallel_config
    )
    assert not _is_sm70_qwen38_nomtp_dual_compile_contract(
        model_config, SimpleNamespace(method="mtp"), parallel_config
    )
    parallel_config.tensor_parallel_size = 2
    assert not _is_sm70_qwen38_nomtp_dual_compile_contract(
        model_config, None, parallel_config
    )


def test_qwen38_hybrid_ple_decode_uses_local_module(monkeypatch) -> None:
    from vllm.model_executor.layers.ple_offload_layer import PleOffloadLayer

    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "1")
    monkeypatch.setenv("VLLM_SM70_QWEN38_HYBRID_PLE", "1")
    monkeypatch.setenv("VLLM_SM70_QWEN38_DUAL_COMPILE", "1")

    class ToyPle(PleOffloadLayer):
        def __init__(self) -> None:
            super().__init__()
            self.initialized_locally = True
            self._is_cpu_offloaded = True

        def forward_impl(
            self,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> torch.Tensor:
            return hidden_states + input_ids

    layer = ToyPle()
    with sm70_decode_graph_compilation():
        output = layer(torch.tensor([2]), torch.tensor([3]))

    assert layer.initialized_locally
    torch.testing.assert_close(output, torch.tensor([5]))


def test_qwen38_hybrid_ple_skips_decode_offload_request(monkeypatch) -> None:
    from vllm.v1.ple_offload.connector import PleOffloadConnector

    monkeypatch.setenv("VLLM_SM70_QWEN38_HYBRID_PLE", "1")
    launches: list[tuple[int, int]] = []
    connector = SimpleNamespace(
        _launch=lambda num_reqs, num_tokens: launches.append(
            (num_reqs, num_tokens)
        )
    )

    PleOffloadConnector.prepare_forward(
        connector, 1, 1, dummy_run=False, use_local_model=True
    )
    PleOffloadConnector.prepare_forward(
        connector, 1, 8192, dummy_run=False, use_local_model=False
    )

    assert launches == [(1, 8192)]
