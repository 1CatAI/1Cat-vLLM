# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.compilation.sm70_decode_graph import (
    is_sm70_decode_graph_compiling,
    sm70_decode_graph_compilation,
    use_sm70_decode_graph_semantics,
)
from vllm.config.parallel import ParallelConfig
from vllm.config.vllm import (
    _apply_sm70_qwen38_hybrid_ple_defaults,
    _apply_sm70_qwen38_nomtp_defaults,
    _is_sm70_qwen38_nomtp_dual_compile_contract,
)


def _qwen38_model_config(
    architecture: str,
    *,
    language_model_only: bool | None = None,
    quantization: str | None = None,
) -> SimpleNamespace:
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
    multimodal_config = (
        None
        if language_model_only is None
        else SimpleNamespace(language_model_only=language_model_only)
    )
    return SimpleNamespace(
        architectures=(architecture,),
        dtype=torch.float16,
        hf_text_config=text_config,
        multimodal_config=multimodal_config,
        quantization=quantization,
    )


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
    model_config = _qwen38_model_config("Qwen4ExpForCausalLM")
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


def test_qwen38_nomtp_dual_compile_contract_accepts_awq_lm_only_wrapper() -> None:
    model_config = _qwen38_model_config(
        "Qwen4ExpForConditionalGeneration",
        language_model_only=True,
        quantization="awq",
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
    )

    assert _is_sm70_qwen38_nomtp_dual_compile_contract(
        model_config, None, parallel_config
    )

    model_config.multimodal_config.language_model_only = False
    assert not _is_sm70_qwen38_nomtp_dual_compile_contract(
        model_config, None, parallel_config
    )

    model_config.multimodal_config = None
    assert not _is_sm70_qwen38_nomtp_dual_compile_contract(
        model_config, None, parallel_config
    )


def _nomtp_default_config():
    return SimpleNamespace(
        model_config=_qwen38_model_config(
            "Qwen4ExpForConditionalGeneration",
            language_model_only=True,
            quantization="modelopt_fp4",
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            enable_dbo=False,
            data_parallel_size=1,
            nnodes_within_dp=1,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="float16", mamba_ssm_cache_dtype="float32"
        ),
        lora_config=None,
    )


def test_qwen38_nomtp_defaults_preserve_overrides(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    cfg = _nomtp_default_config()
    disabled = "VLLM_SM70_QWEN38_FP16_GEMV"
    os.environ[disabled] = "0"
    applied = _apply_sm70_qwen38_nomtp_defaults(cfg, is_sm70=True)
    assert disabled not in applied and os.environ[disabled] == "0"
    assert len(applied) == 4
    assert os.environ["VLLM_SM70_QWEN38_FUSED_HC_FP16"] == "1"
    assert _apply_sm70_qwen38_nomtp_defaults(cfg, is_sm70=True) == ()
    assert "VLLM_SM70_QWEN4_EXP_ONLINE_QPN8" not in os.environ
    assert "VLLM_SM70_NVFP4_QPN2" not in os.environ


@pytest.mark.parametrize(
    "mismatch",
    [
        "device",
        "model",
        "dtype",
        "quantization",
        "mtp",
        "tp",
        "pp",
        "ep",
        "dbo",
        "dp",
        "nodes",
        "kv",
        "ssm",
        "lora",
        "multimodal",
    ],
)
def test_qwen38_nomtp_defaults_reject_unqualified_contract(monkeypatch, mismatch):
    monkeypatch.setattr(os, "environ", {})
    cfg = _nomtp_default_config()
    if mismatch == "model":
        cfg.model_config.hf_text_config.hidden_size = 5120
    elif mismatch == "dtype":
        cfg.model_config.dtype = torch.bfloat16
    elif mismatch == "quantization":
        cfg.model_config.quantization = "awq"
    elif mismatch == "mtp":
        cfg.speculative_config = SimpleNamespace(method="mtp")
    elif mismatch in ("tp", "pp", "dp", "nodes"):
        field = {
            "tp": "tensor_parallel_size",
            "pp": "pipeline_parallel_size",
            "dp": "data_parallel_size",
            "nodes": "nnodes_within_dp",
        }[mismatch]
        setattr(cfg.parallel_config, field, 2)
    elif mismatch in ("ep", "dbo"):
        setattr(
            cfg.parallel_config,
            "enable_expert_parallel" if mismatch == "ep" else "enable_dbo",
            True,
        )
    elif mismatch == "kv":
        cfg.cache_config.cache_dtype = "fp8_e4m3"
    elif mismatch == "ssm":
        cfg.cache_config.mamba_ssm_cache_dtype = "float16"
    elif mismatch == "lora":
        cfg.lora_config = SimpleNamespace()
    elif mismatch == "multimodal":
        cfg.model_config.multimodal_config.language_model_only = False
    assert _apply_sm70_qwen38_nomtp_defaults(cfg, is_sm70=mismatch != "device") == ()
    assert not os.environ


def test_parallel_config_initializes_ple_ipc_after_late_auto_enable(
    monkeypatch,
) -> None:
    for env_name in (
        "VLLM_SM70_QWEN38_HYBRID_PLE",
        "VLLM_PLE_CPU_OFFLOAD",
        "VLLM_PLE_DISK_OFFLOAD",
    ):
        monkeypatch.delenv(env_name, raising=False)
    parallel_config = ParallelConfig()
    assert parallel_config._ple_offload_ipc_path == ""

    _apply_sm70_qwen38_hybrid_ple_defaults(parallel_config)
    ipc_path = parallel_config._ple_offload_ipc_path

    assert os.environ["VLLM_SM70_QWEN38_HYBRID_PLE"] == "1"
    assert os.environ["VLLM_PLE_CPU_OFFLOAD"] == "1"
    assert os.environ["VLLM_PLE_DISK_OFFLOAD"] == "1"
    assert ipc_path.startswith("ipc://")
    parallel_config.ensure_ple_offload_ipc_path()
    assert parallel_config._ple_offload_ipc_path == ipc_path


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
        _launch=lambda num_reqs, num_tokens: launches.append((num_reqs, num_tokens))
    )

    PleOffloadConnector.prepare_forward(
        connector, 1, 1, dummy_run=False, use_local_model=True
    )
    PleOffloadConnector.prepare_forward(
        connector, 1, 8192, dummy_run=False, use_local_model=False
    )

    assert launches == [(1, 8192)]
