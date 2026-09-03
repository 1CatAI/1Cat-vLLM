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


def test_qwen38_pinned_ple_cpu_gather_preserves_rows() -> None:
    from vllm.models.qwen4_exp.nvidia.ple_layer import (
        _gather_pinned_ple_rows_cpu,
    )

    weight = torch.arange(48, dtype=torch.uint8).reshape(8, 6)
    input_ids = torch.tensor([5, 1, 5, 0, 7, 1], dtype=torch.int64)
    output = torch.empty((input_ids.numel(), weight.shape[1]), dtype=torch.uint8)

    unique_rows = _gather_pinned_ple_rows_cpu(weight, input_ids, output)

    assert unique_rows == 4
    torch.testing.assert_close(output, weight[input_ids])
