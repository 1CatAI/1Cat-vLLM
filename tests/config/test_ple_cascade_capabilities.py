# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config import vllm as config_module
from vllm.model_executor.layers.ple_offload_layer import ple_offload_enabled
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.models.qwen4_exp.common.ple import ple_cascade_configured


@pytest.fixture
def config(monkeypatch):
    for name in (
        "VLLM_SM70_QWEN38_HYBRID_PLE",
        "VLLM_PLE_DISK_OFFLOAD",
        "VLLM_PLE_CPU_OFFLOAD",
    ):
        monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    cfg = VllmConfig()
    cfg.model_config = NS(
        dtype=torch.float16,
        hf_text_config=NS(
            ple_layer_ids=[0, 1],
            ple_embedding_dtype="float8_e4m3fn",
            num_hidden_layers=8,
        ),
    )
    cfg.quant_config = Fp8Config(is_checkpoint_fp8_serialized=True)
    cfg.parallel_config.pipeline_parallel_size = 4
    monkeypatch.setattr(config_module, "_ple_disk_cascade_cuda_supported", lambda: True)
    yield cfg
    envs.disable_envs_cache()


def test_default_admission_and_engine_local_activation(config):
    before = dict(os.environ)
    assert config_module._qwen4exp_ple_cascade_requested(config)
    config_module._apply_qwen4exp_ple_cascade_defaults(config.parallel_config)
    assert dict(os.environ) == before
    assert config.parallel_config._ple_offload_ipc_path
    with set_current_vllm_config(config):
        assert ple_cascade_configured()
        assert ple_offload_enabled()
    assert not ple_offload_enabled()


@pytest.mark.parametrize(
    "case,reason",
    [
        ("disabled", "KernelConfig"),
        ("dtype", "FP16"),
        ("storage", "E4M3"),
        ("format", "safetensors"),
        ("missing", "no PLE"),
        ("dcp", "context-parallel"),
    ],
)
def test_capability_rejections(config, case, reason):
    if case == "disabled":
        config.kernel_config.ple_disk_cascade = False
    elif case == "dtype":
        config.model_config.dtype = torch.bfloat16
    elif case == "storage":
        config.model_config.hf_text_config.ple_embedding_dtype = ""
        config.quant_config = None
    elif case == "format":
        config.load_config.load_format = "pt"
    elif case == "missing":
        config.model_config.hf_text_config.ple_layer_ids = []
    else:
        config.parallel_config.decode_context_parallel_size = 2
    assert not config_module._qwen4exp_ple_cascade_requested(config)
    assert reason in config.kernel_config.ple_disk_cascade_reason


def test_pp_admission_uses_layer_layout(config):
    config.model_config.hf_text_config.ple_layer_ids = [3]
    assert not config_module._qwen4exp_ple_cascade_requested(config)
    assert "first" in config.kernel_config.ple_disk_cascade_reason


def test_existing_explicit_placement_takes_precedence(config, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QWEN38_HYBRID_PLE", "1")
    assert not config_module._qwen4exp_ple_cascade_requested(config)
    assert "precedence" in config.kernel_config.ple_disk_cascade_reason


def test_multiple_engines_keep_independent_policy(config):
    other = copy.deepcopy(config)
    other.kernel_config.ple_disk_cascade = False
    assert config_module._qwen4exp_ple_cascade_requested(config)
    assert not config_module._qwen4exp_ple_cascade_requested(other)
    assert config.kernel_config.ple_disk_cascade_active
    with set_current_vllm_config(other):
        assert not ple_offload_enabled()
    with set_current_vllm_config(config):
        assert ple_offload_enabled()


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("policy", ["ple", "sparse"])
def test_hf_config_clone_preserves_policy_ownership(tmp_path, active, policy):
    from transformers import Qwen3Config

    hf = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        vocab_size=64,
    )
    hf.save_pretrained(tmp_path)
    model = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        dtype="half",
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=64,
    )
    parent = VllmConfig(model_config=model)
    if policy == "ple":
        parent.kernel_config.ple_disk_cascade_active = active
    else:
        parent.kernel_config.sm70_sparse.active = active
    child = parent.with_hf_config(copy.deepcopy(hf))
    assert (child.kernel_config is parent.kernel_config) is not active
    if policy == "ple":
        assert parent.kernel_config.ple_disk_cascade_active is active
    else:
        assert parent.kernel_config.sm70_sparse.active is active
        assert not child.kernel_config.sm70_sparse.active
    assert not child.kernel_config.ple_disk_cascade_active
