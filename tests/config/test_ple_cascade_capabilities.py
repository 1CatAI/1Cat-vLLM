# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import os
from types import SimpleNamespace as NS

import pytest
import torch

from tests.config.runtime_policy_utils import make_policy_defaults
from vllm import envs
from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.ple_offload_layer import ple_offload_enabled
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.gguf import GGUFConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.models.qwen4_exp.common.ple import ple_cascade_configured
from vllm.platforms import runtime_defaults as config_module


@pytest.fixture
def config(monkeypatch):
    for name in (
        "VLLM_SM70_QWEN38_HYBRID_PLE",
        "VLLM_PLE_DISK_OFFLOAD",
        "VLLM_PLE_CPU_OFFLOAD",
    ):
        monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
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
    defaults = make_policy_defaults()
    assert config_module._qwen4exp_ple_cascade_requested(config, defaults=defaults)
    config_module._apply_qwen4exp_ple_cascade_defaults(config.parallel_config)
    assert dict(os.environ) == before
    assert config.parallel_config._ple_offload_ipc_path
    with set_current_vllm_config(config):
        assert ple_cascade_configured()
        assert ple_offload_enabled()
    assert not ple_offload_enabled()


def test_packed_gguf_admission_without_fp8_storage_hint(config):
    config.load_config.load_format = "gguf"
    config.quant_config = GGUFConfig()
    config.model_config.hf_text_config.ple_embedding_dtype = ""
    defaults = make_policy_defaults()
    assert config_module._qwen4exp_ple_cascade_requested(config, defaults=defaults)
    assert config.kernel_config.ple_disk_cascade_reason is None
    with set_current_vllm_config(config):
        assert ple_cascade_configured()
        assert ple_offload_enabled()


def test_gguf_format_requires_supported_embedding_method(config):
    config.load_config.load_format = "gguf"
    config.quant_config = None
    config.model_config.hf_text_config.ple_embedding_dtype = ""
    defaults = make_policy_defaults()
    assert not config_module._qwen4exp_ple_cascade_requested(config, defaults=defaults)
    assert "packed GGUF" in config.kernel_config.ple_disk_cascade_reason


@pytest.mark.parametrize(
    "case,reason",
    [
        ("disabled", "KernelConfig"),
        ("dtype", "FP16"),
        ("storage", "E4M3"),
        ("format", "safetensors"),
        ("missing", "no PLE"),
        ("pcp", "prefill context-parallel"),
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
        config.parallel_config.prefill_context_parallel_size = 2
    defaults = make_policy_defaults()
    assert not config_module._qwen4exp_ple_cascade_requested(config, defaults=defaults)
    assert reason in config.kernel_config.ple_disk_cascade_reason


@pytest.mark.parametrize("dcp_size", [1, 2, 4])
def test_decode_context_parallel_preserves_cascade_admission(config, dcp_size):
    config.parallel_config.tensor_parallel_size = 4
    config.parallel_config.decode_context_parallel_size = dcp_size
    before = dict(os.environ)
    defaults = make_policy_defaults()
    assert config_module._qwen4exp_ple_cascade_requested(config, defaults=defaults)
    assert config.kernel_config.ple_disk_cascade_active
    assert config.parallel_config.decode_context_parallel_size == dcp_size
    assert dict(os.environ) == before


@pytest.mark.parametrize("table_layer,admitted", [(1, True), (2, False)])
def test_modelopt_mixed_storage_uses_checkpoint_layer_names(
    config, table_layer, admitted
):
    # ModelOpt mixed-precision checkpoints declare the FP8 PLE table per layer
    # under checkpoint names instead of setting ple_embedding_dtype. PLE id 2
    # is the PLE module of decoder layer 1.
    config.model_config.hf_text_config.ple_layer_ids = [2]
    config.model_config.hf_text_config.ple_embedding_dtype = ""
    config.quant_config = ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "ignore": ["lm_head", "model.language_model.embed_tokens"],
            "quantized_layers": {
                f"model.language_model.layers.{table_layer}"
                ".ple.ple_embedding.ngram_embedding": {"quant_algo": "FP8"},
            },
        }
    )
    defaults = make_policy_defaults()
    assert (
        config_module._qwen4exp_ple_cascade_requested(config, defaults=defaults)
        is admitted
    )
    if not admitted:
        assert "E4M3" in config.kernel_config.ple_disk_cascade_reason


def test_pp_admission_uses_layer_layout(config):
    config.model_config.hf_text_config.ple_layer_ids = [3]
    defaults = make_policy_defaults()
    assert not config_module._qwen4exp_ple_cascade_requested(config, defaults=defaults)
    assert "first" in config.kernel_config.ple_disk_cascade_reason


def test_existing_explicit_placement_takes_precedence(config, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QWEN38_HYBRID_PLE", "1")
    defaults = make_policy_defaults()
    assert not config_module._qwen4exp_ple_cascade_requested(config, defaults=defaults)
    assert "precedence" in config.kernel_config.ple_disk_cascade_reason


def test_multiple_engines_keep_independent_policy(config):
    other = copy.deepcopy(config)
    other.kernel_config.ple_disk_cascade = False
    defaults = make_policy_defaults()
    assert config_module._qwen4exp_ple_cascade_requested(config, defaults=defaults)
    assert not config_module._qwen4exp_ple_cascade_requested(other, defaults=defaults)
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
    parent = VllmConfig(model_config=model, device_config=DeviceConfig(device="cpu"))
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
